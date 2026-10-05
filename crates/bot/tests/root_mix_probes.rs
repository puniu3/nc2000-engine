use conformance::load_dex;
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    state::Battle,
};
use serde_json::json;

#[test]
fn joint_probes_force_legal_replies_and_preserve_the_unavailable_reply_fallback() {
    let dex = load_dex();
    let set: PokemonSet = serde_json::from_value(json!({"name":"Starmie","species":"Starmie",
        "level":50,"ability":"","moves":["Return","Surf"]}))
    .unwrap();
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", &[set.clone()], &[set]).unwrap();
    let mut obs = Observer::new(&b, 0);
    let mut belief = Belief::pinned_from_battle(&b, &obs);
    b.choose(&dex, 0, "team 1").unwrap();
    b.choose(&dex, 1, "team 1").unwrap();
    obs.observe(&b, &dex);
    belief.sync(&dex, &obs);
    let mut ordinary = BlindSearch::new(&b, &dex, RmConfig::default(), 0, 10);
    let mut unavailable = BlindSearch::new(&b, &dex, RmConfig::default(), 0, 10);
    let missing = SearchChoice::Move(dex.moves.id("splash").unwrap());
    for i in 0..20 {
        let own = i % ordinary.actions().len();
        assert_eq!(
            ordinary.step_forced(&dex, &belief, &obs, own),
            unavailable.step_forced_joint(&dex, &belief, &obs, own, missing)
        );
    }
    assert_eq!(ordinary.visits(), unavailable.visits());
    assert_eq!(ordinary.means(), unavailable.means());
    assert_eq!(ordinary.root_matrix(), unavailable.root_matrix());
    let mut forced = BlindSearch::new(&b, &dex, RmConfig::default(), 0, 10);
    let surf = SearchChoice::Move(dex.moves.id("surf").unwrap());
    for _ in 0..20 {
        forced.step_forced_joint(&dex, &belief, &obs, 0, surf);
    }
    assert_eq!(forced.iterations(), 20);
    assert_eq!(forced.visits(), &[20, 0]);
    assert!(forced
        .root_matrix()
        .iter()
        .all(|&(own, reply, _, _)| own == 0 && reply == surf));
    assert_eq!(forced.root_matrix().iter().map(|c| c.2).sum::<u32>(), 20);
}
