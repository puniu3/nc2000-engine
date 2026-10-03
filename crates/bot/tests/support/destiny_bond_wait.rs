use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde_json::json;

fn set(species: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(json!({
        "species":species,"name":species,"level":50,"moves":moves,
        "item":"","ability":"No Ability","happiness":255,
        "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}
    }))
    .unwrap()
}

fn step(b: &mut Battle, dex: &Dex, inputs: [&str; 2]) {
    let actions = std::array::from_fn(|s| {
        Some(
            b.legal_choices(dex, s)
                .into_iter()
                .find(|a| a.to_input(dex) == inputs[s])
                .unwrap_or_else(|| panic!("illegal {}", inputs[s])),
        )
    });
    b.apply_choices(dex, actions).unwrap();
}

pub fn position(dex: &Dex, plain: bool, fast: bool) -> (Battle, Battle) {
    let mut own = set(
        if fast { "Gengar" } else { "Misdreavus" },
        &[
            "Mean Look",
            if plain { "Psychic" } else { "Curse" },
            "Shadow Ball",
            "Return",
        ],
    );
    own.level = 55;
    own.item = "Leftovers".into();
    let mut foe = set("Gengar", &["Destiny Bond", "Haze"]);
    if !fast {
        foe.evs.as_mut().unwrap().insert("spe".into(), 0);
    }
    let preview = Battle::from_fixture(
        dex,
        "1,2,3,4",
        &[set("Snorlax", &["Self-Destruct"]), own],
        &[foe, set("Magikarp", &["Splash"])],
    )
    .unwrap();
    let mut b = preview.clone();
    b.choose(dex, 0, "team 1,2").unwrap();
    b.choose(dex, 1, "team 1,2").unwrap();
    step(&mut b, dex, ["move selfdestruct", "move haze"]);
    let replacement = b.legal_choices(dex, 0)[0];
    b.apply_choices(dex, [Some(replacement), None]).unwrap();
    if plain {
        step(&mut b, dex, ["move psychic", "move haze"]);
        for _ in 0..8 {
            step(&mut b, dex, ["move meanlook", "move destinybond"]);
        }
    } else {
        for _ in 0..7 {
            step(&mut b, dex, ["move meanlook", "move destinybond"]);
        }
        step(&mut b, dex, ["move meanlook", "move haze"]);
        step(&mut b, dex, ["move curse", "move haze"]);
        step(&mut b, dex, ["move return", "move haze"]);
        step(&mut b, dex, ["move return", "move destinybond"]);
    }
    let own_id = b.active_id(0).unwrap();
    let foe_id = b.active_id(1).unwrap();
    assert!(b.poke(own_id).speed > b.poke(foe_id).speed);
    assert!(b.poke(foe_id).trapped);
    assert!(b
        .poke(foe_id)
        .has_volatile(dex.conds_id("destinybond").unwrap()));
    assert!(!b
        .poke(foe_id)
        .has_volatile(dex.conds_id("perishsong").unwrap()));
    assert_eq!(b.sides[0].pokemon_left, 1);
    assert_eq!(b.sides[1].pokemon_left, 2);
    (preview, b)
}
