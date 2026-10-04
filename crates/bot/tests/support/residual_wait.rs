use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde_json::json;

fn set(species: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(json!({"species":species,"name":species,"level":50,
        "moves":moves,"item":"","happiness":255,
        "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}}))
    .unwrap()
}

fn step(b: &mut Battle, dex: &Dex, inputs: [&str; 2]) {
    let joint = std::array::from_fn(|s| {
        Some(
            b.legal_choices(dex, s)
                .into_iter()
                .find(|a| a.to_input(dex) == inputs[s])
                .unwrap(),
        )
    });
    b.apply_choices(dex, joint).unwrap();
}

pub fn position(dex: &Dex) -> (Battle, Battle, [Vec<PokemonSet>; 2]) {
    let mine = [set(
        "Jolteon",
        &["Return", "Thunder Shock", "Toxic", "Thunder Wave"],
    )];
    let mut theirs = [set(
        "Wobbuffet",
        &["Counter", "Mirror Coat", "Safeguard", "Destiny Bond"],
    )];
    theirs[0].item = "Leftovers".into();
    let preview = Battle::from_fixture(dex, "1,2,3,1", &mine, &theirs).unwrap();
    let mut b = preview.clone();
    b.choose(dex, 0, "team 1").unwrap();
    b.choose(dex, 1, "team 1").unwrap();
    for pair in [
        ["move return", "move counter"],
        ["move thundershock", "move mirrorcoat"],
        ["move toxic", "move mirrorcoat"],
        ["move thunderwave", "move mirrorcoat"],
        ["move thunderwave", "move mirrorcoat"],
        ["move thunderwave", "move mirrorcoat"],
        ["move thunderwave", "move mirrorcoat"],
    ] {
        step(&mut b, dex, pair);
    }
    (preview, b, [mine.to_vec(), theirs.to_vec()])
}
