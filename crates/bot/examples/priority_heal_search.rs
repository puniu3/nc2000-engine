use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    preview::load_meta_pool,
    smmcts::{MaskRules, SelRule},
    Belief, BlindSearch, Observer, RmConfig,
};
use serde_json::json;

#[path = "priority_heal_support/mod.rs"]
mod fixture;

fn main() {
    let dex = load_dex();
    let (preview, b) = fixture::position(&dex);
    let pool = load_meta_pool(&repo_root().join("data/belief-pool-v3/belief-pool.json"));
    for blind in [false, true] {
        let mut obs = Observer::new(&preview, 0);
        let mut belief = if blind {
            Belief::new(&dex, &pool, &obs)
        } else {
            Belief::pinned_from_battle(&preview, &obs)
        };
        obs.observe(&b, &dex);
        belief.sync(&dex, &obs);
        for seed in 98001..98009 {
            let mut previous = None;
            for enabled in [false, true] {
                let cfg = RmConfig {
                    iterations: 27000,
                    c: 0.4,
                    rule: SelRule::Ucb,
                    mask_rules: MaskRules {
                        priority_heal: enabled,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut search = BlindSearch::new(&b, &dex, cfg, 0, seed);
                search.step(&dex, &belief, &obs, 27000);
                let stats = (search.visits().to_vec(), search.means());
                if let Some(old) = &previous {
                    assert_eq!(&stats, old);
                }
                previous = Some(stats);
                let raw = (0..search.actions().len())
                    .max_by_key(|&i| search.visits()[i])
                    .unwrap();
                println!(
                    "{}",
                    json!({"blind":blind,"seed":seed,"enabled":enabled,
                    "choice":search.best().unwrap().to_input(&dex),"raw":search.actions()[raw].to_input(&dex),
                    "actions":search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i],"excluded":search.dominated()[i]})).collect::<Vec<_>>() })
                );
            }
        }
    }
}
