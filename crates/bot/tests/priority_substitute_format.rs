use conformance::load_dex;
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};

#[allow(dead_code)]
mod fixture {
    include!("../../../findings/priority-substitute-format-v1/priority_substitute_choice.rs");
    pub fn case(dex: &Dex, side: usize) -> (Battle, Battle) { position(dex, side) }
}

#[test]
fn full_format_choice_keeps_the_priority_pp_stall_found_by_search() {
    let dex = load_dex();
    for side in 0..2 {
        let (preview, battle) = fixture::case(&dex, side);
        let mut obs = Observer::new(&preview, side);
        let mut belief = Belief::pinned_from_battle(&preview, &obs);
        obs.observe(&battle, &dex);
        belief.sync_checked(&dex, &obs).unwrap();
        let mut search = BlindSearch::new(
            &battle, &dex, RmConfig { c: 0.4, ..Default::default() }, side, 2026100807,
        );
        search.step(&dex, &belief, &obs, 27000);
        let most_visited = (0..search.actions().len()).max_by_key(|&i| search.visits()[i]).unwrap();
        assert_eq!(search.actions()[most_visited].to_input(&dex), "move substitute");
        assert_eq!(search.best(), Some(search.actions()[most_visited]));
        assert!(search.visits()[most_visited] > 26000);
    }
}
