use super::*;

pub(super) fn teams() -> [Vec<PokemonSet>; 2] {
    let mut snorlax = set("Snorlax", &["Earthquake", "Curse"], true);
    snorlax.evs.as_mut().unwrap().insert("hp".into(), 120);
    [
        vec![
            set("Electrode", &["Explosion"], true),
            set("Raikou", &["Quick Attack"], false),
            snorlax,
            set("Starmie", &["Surf"], true),
            set("Marowak", &["Earthquake"], true),
            set("Heracross", &["Megahorn"], true),
        ],
        vec![
            set("Raikou", &["Thunderbolt"], false),
            set("Gengar", &["Thunderbolt", "Destiny Bond"], true),
            set("Magikarp", &["Splash"], true),
            set("Snorlax", &["Body Slam"], true),
            set("Marowak", &["Earthquake"], true),
            set("Heracross", &["Megahorn"], true),
        ],
    ]
}

pub(super) fn prefix(
    dex: &Dex,
    b: &mut Battle,
    obs: &mut [Observer; 2],
    beliefs: &mut [Belief; 2],
    records: &mut Vec<Value>,
) {
    let mv = |name: &str| Some(SearchChoice::Move(dex.moves.id(name).unwrap()));
    step(
        dex,
        b,
        obs,
        beliefs,
        [Some(SearchChoice::Team([1, 2, 3])); 2],
        records,
    );
    step(
        dex,
        b,
        obs,
        beliefs,
        [mv("explosion"), mv("thunderbolt")],
        records,
    );
    step(
        dex,
        b,
        obs,
        beliefs,
        [Some(SearchChoice::Switch(2)); 2],
        records,
    );
    step(
        dex,
        b,
        obs,
        beliefs,
        [mv("quickattack"), Some(SearchChoice::Switch(3))],
        records,
    );
    while b.poke(b.active_id(1).unwrap()).hp > 40 {
        step(
            dex,
            b,
            obs,
            beliefs,
            [mv("quickattack"), mv("splash")],
            records,
        );
    }
    let back = b
        .legal_choices(dex, 1)
        .into_iter()
        .find(|a| match a {
            SearchChoice::Switch(pos) => {
                dex.species
                    .key(b.sides[1].roster[b.sides[1].party[(*pos - 1) as usize] as usize].species)
                    == "gengar"
            }
            _ => false,
        })
        .unwrap();
    step(
        dex,
        b,
        obs,
        beliefs,
        [mv("quickattack"), Some(back)],
        records,
    );
    while b.needs_choice() == [true, true] {
        step(
            dex,
            b,
            obs,
            beliefs,
            [mv("quickattack"), mv("thunderbolt")],
            records,
        );
    }
    step(
        dex,
        b,
        obs,
        beliefs,
        [Some(SearchChoice::Switch(3)), None],
        records,
    );
    step(
        dex,
        b,
        obs,
        beliefs,
        [mv("earthquake"), mv("thunderbolt")],
        records,
    );
    for _ in 0..3 {
        step(
            dex,
            b,
            obs,
            beliefs,
            [mv("curse"), mv("thunderbolt")],
            records,
        );
    }
    step(
        dex,
        b,
        obs,
        beliefs,
        [mv("curse"), mv("destinybond")],
        records,
    );
}
