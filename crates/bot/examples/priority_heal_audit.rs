use conformance::load_dex;
use nc2000_bot::smmcts::dominated_actions;
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::json;

fn main() {
    let dex = load_dex();
    let set = |species: &str, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(json!({
            "species":species,"name":species,"level":50,"moves":moves,
            "item":"","ability":"No Ability","happiness":255,
            "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
            "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}
        }))
        .unwrap()
    };
    let own = [set("Alakazam", &["Recover", "Psychic"])];
    let foe = [set("Scizor", &["Quick Attack", "Return"])];
    for seed in ["1,2,3,4", "5,6,7,8", "9,10,11,12", "13,14,15,16"] {
        let mut start = Battle::from_fixture(&dex, seed, &own, &foe).unwrap();
        start.choose(&dex, 0, "team 1").unwrap();
        start.choose(&dex, 1, "team 1").unwrap();
        let id = start.active_id(0).unwrap();
        let enemy = start.active_id(1).unwrap();
        assert!(
            start.get_pokemon_action_speed(&dex, id) > start.get_pokemon_action_speed(&dex, enemy)
        );
        let masked = dominated_actions(&start, &dex, 0)
            .into_iter()
            .find(|(c, _)| c.to_input(&dex) == "move recover")
            .unwrap()
            .1;
        for reply in ["quickattack", "return"] {
            let mut b = start.clone();
            b.log.clear();
            b.choose(&dex, 0, "move recover").unwrap();
            b.choose(&dex, 1, &format!("move {reply}")).unwrap();
            let healed = b.log.iter().any(|line| line.starts_with("|-heal|p1a:"));
            assert_eq!(healed, reply == "quickattack");
            println!(
                "{}",
                json!({"seed":seed,"opponent_move":reply,
                "excluded":"move recover","reason":masked,"healed":healed,
                "hp_before":start.poke(id).hp,"hp_after":b.poke(id).hp,
                "scope":"Constructed one-turn correctness probe; no strength or prevalence claim",
                "log":b.log})
            );
        }
    }
}
