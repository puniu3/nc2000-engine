use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::json;

#[test]
fn unseen_claw_cannot_change_the_root_mask_or_sampled_search() {
    let dex = conformance::load_dex();
    let mon = |item: &str| -> PokemonSet {
        serde_json::from_value(json!({"species":"Snorlax","name":"Snorlax","level":50,
            "item":item,"moves":["Rest","Swagger"]}))
        .unwrap()
    };
    let mut root = Battle::from_fixture(&dex, "1,2,3,4", &[mon("Quick Claw")], &[mon("")]).unwrap();
    let mut observer = Observer::new(&root, 0);
    let mut belief = Belief::pinned_from_battle(&root, &observer);
    root.choose(&dex, 0, "team 1").unwrap();
    root.choose(&dex, 1, "team 1").unwrap();
    observer.observe(&root, &dex);
    belief.sync(&dex, &observer);
    let mut roots = [root.clone(), root];
    for (coin, root) in roots.iter_mut().enumerate() {
        root.quick_claw_roll = coin == 1;
        root.update_all_speeds(&dex);
    }
    assert_eq!(roots[0].log, roots[1].log);
    let baseline = roots
        .each_ref()
        .map(|b| BlindSearch::new(b, &dex, RmConfig::default(), 0, 97));
    assert_ne!(baseline[0].dominated(), baseline[1].dominated());
    let mut corrected = roots.each_ref().map(|b| {
        BlindSearch::new(
            b,
            &dex,
            RmConfig {
                unseen_claw_root: true,
                horizon: 1,
                ..Default::default()
            },
            0,
            97,
        )
    });
    assert_eq!(corrected[0].dominated(), corrected[1].dominated());
    for search in &mut corrected {
        search.step(&dex, &belief, &observer, 512);
    }
    assert_eq!(corrected[0].visits(), corrected[1].visits());
    assert_eq!(corrected[0].means(), corrected[1].means());
    assert_eq!(corrected[0].best(), corrected[1].best());
    assert_eq!(corrected[0].node_count(), corrected[1].node_count());
}
