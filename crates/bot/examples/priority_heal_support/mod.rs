use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};

pub fn position(dex: &Dex) -> (Battle, Battle) {
    let set = |species: &str, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "species":species,"name":species,"level":50,"moves":moves,
            "item":"","ability":"No Ability","happiness":255,
            "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
            "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}
        }))
        .unwrap()
    };
    let preview = Battle::from_fixture(
        dex,
        "1,2,3,4",
        &[set("Alakazam", &["Recover", "Psychic"])],
        &[set("Scizor", &["Quick Attack", "Return"])],
    )
    .unwrap();
    let mut b = preview.clone();
    b.choose(dex, 0, "team 1").unwrap();
    b.choose(dex, 1, "team 1").unwrap();
    (preview, b)
}
