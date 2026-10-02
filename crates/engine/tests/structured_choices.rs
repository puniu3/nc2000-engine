use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn setup() -> (Dex, Battle) {
    let dex = Dex::from_json(include_str!("../../../data/gen2stadium2.json")).unwrap();
    let team: Vec<PokemonSet> = (0..3).map(|_| serde_json::from_value(serde_json::json!({
        "species":"Snorlax", "name":"Snorlax", "moves":["Hyper Beam", "Outrage", "Rest", "Tackle"],
        "level":50,"item":"","ability":"No Ability"
    })).unwrap()).collect();
    let b = Battle::from_fixture(&dex,"1,2,3,4",&team,&team).unwrap();
    (dex,b)
}

fn compare(b: &Battle, dex: &Dex, side: usize, choice: SearchChoice) {
    let mut direct = b.clone();
    let mut text = b.clone();
    let a = direct.apply_choice(dex,side,choice);
    let c = text.choose(dex,side,&choice.to_input(dex));
    assert_eq!(format!("{a:?}"),format!("{c:?}"),"{choice:?}");
    assert_eq!(format!("{direct:?}"),format!("{text:?}"),"{choice:?}");
}

#[test]
fn malformed_and_partial_teams_and_wrong_requests_match() {
    let (dex,mut b)=setup();
    for choice in [SearchChoice::Team([0,0,0]),SearchChoice::Team([1,0,2]),SearchChoice::Team([1,1,2]),SearchChoice::Team([7,2,3]),SearchChoice::Move(dex.moves.id("tackle").unwrap()),SearchChoice::Pass] {
        compare(&b,&dex,0,choice);
    }
    b.choose(&dex,0,"team 1,2,3").unwrap();
    b.choose(&dex,1,"team 1,2,3").unwrap();
    for choice in [SearchChoice::Team([1,2,3]),SearchChoice::Switch(1),SearchChoice::Switch(7),SearchChoice::Move(dex.moves.id("thunderbolt").unwrap()),SearchChoice::Pass] {
        compare(&b,&dex,0,choice);
    }
    b.ended = true;
    b.sides[0].request = None;
    compare(&b,&dex,0,SearchChoice::Move(dex.moves.id("tackle").unwrap()));
}

#[test]
fn disabled_struggle_recharge_and_lock_match() {
    let (dex,mut b)=setup();
    b.choose(&dex,0,"team 1,2,3").unwrap();
    b.choose(&dex,1,"team 1,2,3").unwrap();
    let active=b.active_id(0).unwrap();
    b.poke_mut(active).move_slots[3].disabled=true;
    compare(&b,&dex,0,SearchChoice::Move(dex.moves.id("tackle").unwrap()));
    let mut empty=b.clone();
    for slot in &mut empty.poke_mut(active).move_slots { slot.pp=0; }
    compare(&empty,&dex,0,SearchChoice::Move(dex.moves.id("struggle").unwrap()));
    compare(&empty,&dex,0,SearchChoice::Move(dex.moves.id("thunderbolt").unwrap()));
    for name in ["hyperbeam","outrage"] {
        let expected = if name == "hyperbeam" { "recharge" } else { "outrage" };
        let mut locked = (1..=64).find_map(|seed| {
            let mut state=b.clone();
            state.reseed(seed);
            state.choose(&dex,0,&format!("move {name}")).unwrap();
            state.choose(&dex,1,"move rest").unwrap();
            (state.legal_choices(&dex,0) == vec![SearchChoice::Move(dex.moves.id(expected).unwrap())]).then_some(state)
        }).expect("a successful hit must create a forced move");
        for side in 0..2 {
            for c in locked.legal_choices(&dex,side) { compare(&locked,&dex,side,c); }
        }
    }
}
