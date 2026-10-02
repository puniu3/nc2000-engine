//! Per-leaf cost split of the value pipeline: engine step, encoding, forward.
//!   fvi_bench MODEL.json
use std::time::Instant;

use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::value::{legal_actions, ValueCodec, ValueNet};
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let codec = ValueCodec::new(&dex);
    let net = ValueNet::load(&root.join(&args[1])).unwrap();
    let teams: Vec<Vec<PokemonSet>> = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"))
        .teams.into_iter().map(|t| t.sets).collect();
    let mut rng = SplitMix64::new(1);
    let mut states = Vec::new();
    while states.len() < 2000 {
        let (a, b) = (rng.below(teams.len()), rng.below(teams.len()));
        let mut bt = Battle::from_fixture(&dex, &rng.battle_seed(), &teams[a], &teams[b]).unwrap();
        bt.set_log_enabled(false);
        while bt.outcome().is_none() && bt.turn < 100 {
            let acts = legal_actions(&bt, &dex);
            if bt.turn > 0 {
                states.push((bt.clone(), acts.clone()));
            }
            let c = [acts[0][rng.below(acts[0].len())], acts[1][rng.below(acts[1].len())]];
            bt.apply_choices(&dex, c).unwrap();
        }
    }
    let t = Instant::now();
    let mut succ = Vec::new();
    for (b, acts) in &states {
        let mut c = b.clone();
        c.reseed(rng.next());
        c.apply_choices(&dex, [acts[0][0], acts[1][0]]).unwrap();
        succ.push(c);
    }
    let step = t.elapsed().as_secs_f64() / states.len() as f64;
    let t = Instant::now();
    let mut recs = Vec::new();
    for c in &succ {
        codec.encode(c, &dex, &mut recs);
    }
    let enc = t.elapsed().as_secs_f64() / succ.len() as f64;
    let t = Instant::now();
    let mut acc = 0.0;
    for r in recs.chunks_exact(nc2000_bot::value::RECORD) {
        acc += net.value(&codec, r);
    }
    let fwd = t.elapsed().as_secs_f64() / succ.len() as f64;
    println!("per leaf: clone+step {:.1} us, encode {:.1} us, forward {:.1} us ({acc:.1})", step * 1e6, enc * 1e6, fwd * 1e6);
}
