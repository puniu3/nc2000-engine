use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    import::ProtocolAgent, preview::load_meta_pool, Belief, FallbackPolicy, RmConfig, SplitMix64,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let input = std::fs::read_to_string(&args[1]).unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
    let dex = load_dex();
    let pool = load_meta_pool(&repo_root().join("data/belief-pool-v3/belief-pool.json"));
    for line in input.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let side = row["case"]["side"].as_u64().unwrap() as usize;
        let spec = serde_json::from_value(row["position"].clone()).unwrap();
        let mut agent = ProtocolAgent::new(&dex, side, pool.clone(), RmConfig::default(), 1);
        agent.set_position(&dex, &spec).unwrap();
        let battle = agent.battle().unwrap();
        let obs = agent.observer().unwrap();
        let candidate =
            Belief::with_fallback_policy(&dex, &pool, obs, FallbackPolicy::RetainPerish);
        let mut ra = SplitMix64::new(771);
        let mut rb = ra.clone();
        let mut a_kits = BTreeMap::<Vec<String>, usize>::new();
        let mut b_kits = a_kits.clone();
        let mut changed = 0;
        for _ in 0..64 {
            let a = agent
                .belief()
                .unwrap()
                .determinize(&dex, battle, obs, &mut ra);
            let mut b = candidate.determinize(&dex, battle, obs, &mut rb);
            assert_eq!(ra.0, rb.0);
            let kit = |state: &nc2000_engine::state::Battle| {
                state
                    .poke(state.active_id(1 - side).unwrap())
                    .move_slots
                    .iter()
                    .map(|m| dex.moves.key(m.id).to_string())
                    .collect::<Vec<_>>()
            };
            *a_kits.entry(kit(&a)).or_default() += 1;
            *b_kits.entry(kit(&b)).or_default() += 1;
            changed += usize::from(a.state_key128() != b.state_key128());
            for (pa, pb) in a.sides[1 - side]
                .roster
                .iter()
                .zip(&mut b.sides[1 - side].roster)
            {
                pb.move_slots = pa.move_slots;
                pb.base_move_slots = pa.base_move_slots;
            }
            assert_eq!(a.state_key128(), b.state_key128());
        }
        let kits = |map: BTreeMap<Vec<String>, usize>| {
            map.into_iter()
                .map(|(moves, n)| json!({"moves": moves, "n": n}))
                .collect::<Vec<_>>()
        };
        let baseline = kits(a_kits);
        assert_eq!(json!(baseline), row["model_kits"]);
        let mut case = row["case"].clone();
        let log = case.as_object_mut().unwrap().remove("log").unwrap();
        case["battle"] = json!(std::path::Path::new(log.as_str().unwrap())
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_end_matches(".raw.log"));
        writeln!(
            out,
            "{}",
            json!({"case": case, "samples": 64,
            "changed_samples": changed, "baseline": baseline, "candidate": kits(b_kits)})
        )
        .unwrap();
    }
}
