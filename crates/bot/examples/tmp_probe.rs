use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::belief::Belief;
use nc2000_bot::import::{ProtocolTracker, Request};
use nc2000_bot::observe::Observer;
use nc2000_bot::position::PositionSpec;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::rng::SplitMix64;

fn main() {
    let dex = load_dex();
    let path = std::env::args().nth(1).unwrap();
    let spec = PositionSpec::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let pool = load_meta_pool(&repo_root().join("data/belief-pool-v1/belief-pool.json"));
    let tracker = ProtocolTracker::from_spec(&dex, &spec).unwrap();
    let obs = Observer::from_position(&dex, &spec).unwrap();
    let mut belief = Belief::new(&dex, &pool, &obs);
    belief.sync_checked(&dex, &obs).unwrap();
    println!("belief alive = {:?}", belief.alive());
    let mut rng = SplitMix64::new(7);
    let _ = (&tracker, &Request::parse(&dex, &spec.request_json(&dex).unwrap()).unwrap());
    let base = nc2000_bot::position::synthesize_spec(&dex, &spec, &pool, None, 7).unwrap();
    for i in 0..8 {
        let mut sim = belief.determinize(&dex, &base, &obs, &mut rng);
        let ch = sim.legal_choices(&dex, 1);
        let roster: Vec<String> = sim.sides[1]
            .roster
            .iter()
            .map(|p| format!("{}{}", dex.species.key(p.species), if p.fainted { "!" } else { "" }))
            .collect();
        println!(
            "det {i}: left={} party={:?} roster={:?}\n        legal={:?}",
            sim.sides[1].pokemon_left,
            sim.sides[1].party,
            roster,
            ch.iter().map(|c| c.to_input(&dex)).collect::<Vec<_>>()
        );
    }
}
