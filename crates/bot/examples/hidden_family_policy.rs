use nc2000_bot::{rng, smmcts::SelRule, Belief, BlindSearch, RmConfig};
use serde_json::{json, Value};

#[allow(dead_code)]
mod source {
    include!("hidden_family_search.rs");
    pub fn model(dex: &Dex, game: &Value, row: &Value, alternate: bool) -> (Battle, Observer) {
        replay(dex, game, row, alternate)
    }
}

#[path = "../src/exp3.rs"]
mod exp3;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus search-records output");
    let records: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let cases = records["cases"].as_array().unwrap();
    let game: Value = std::fs::read_to_string(&args[1]).unwrap().lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap()).find(|g| g["game"] == cases[0]["game"]).unwrap();
    let dex = conformance::load_dex();
    let side = cases[0]["side"].as_u64().unwrap() as usize;
    let mut worlds = vec![source::model(&dex, &game, &cases[0], false)];
    for case in cases { worlds.push(source::model(&dex, &game, case, true)); }
    let beliefs: Vec<_> = worlds.iter().map(|(b, obs)| Belief::pinned_from_battle(b, obs)).collect();
    let arms: Vec<_> = worlds.iter_mut().enumerate().flat_map(|(i, (b, _))|
        b.legal_choices(&dex, 1 - side).into_iter().map(move |reply| (i, reply))).collect();
    let mut rows = Vec::new();
    for record in records["rows"].as_array().unwrap().iter().filter(|r|
        r["mode"] == "robust" && r["timed"] == true && r["template"] == 0) {
        let seed = record["seed"].as_u64().unwrap();
        let mut search = BlindSearch::new(&worlds[0].0, &dex,
            RmConfig { c: 0.4, rule: SelRule::Ucb, ..Default::default() }, side, seed);
        search.enable_compact_history_continuation(true);
        let mut own = exp3::Exp3Ix::new(search.actions().len());
        let mut adversary = exp3::Exp3Ix::new(arms.len());
        let mut own_rng = rng::SplitMix64::new(seed ^ 0xb1e54f83902ca761);
        let mut world_rng = rng::SplitMix64::new(seed ^ 0x54bc170a2998d6e3);
        for _ in 0..record["iterations"].as_u64().unwrap() {
            let x = own.sample(&mut own_rng);
            let y = adversary.sample(&mut world_rng);
            let (w, reply) = arms[y.action];
            let r = search.step_forced_joint(&dex, &beliefs[w], &worlds[0].1, x.action, reply);
            let value = if side == 0 { r } else { 1. - r };
            own.observe(x, 1. - value);
            adversary.observe(y, value);
        }
        let snapshot = json!({"best":search.best().unwrap().to_input(&dex),"visits":search.visits(),
            "means":search.means(),"rng":search.search_rng_state(),"nodes":search.node_count(),
            "own_rng":own_rng.0,"world_rng":world_rng.0});
        for (name, value) in snapshot.as_object().unwrap() { assert_eq!(value, &record[name]); }
        let policy = search.history_continuation().unwrap();
        let histories: Vec<_> = policy.histories().iter().map(|h| json!({"parent":h.parent,
            "own_action":h.own_action.map(|a|a.to_input(&dex)),"request":h.request.as_str(),
            "log":h.log,"depth":h.depth})).collect();
        let groups: Vec<_> = policy.groups().iter().map(|g| {
            let best = g.visits.iter().enumerate().max_by_key(|&(i,n)| (n,i)).unwrap().0;
            json!({"history":g.history,"actions":g.actions.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),
                "visits":g.visits,"rewards":g.rewards,"best":g.actions[best].to_input(&dex)})
        }).collect();
        eprintln!("seed={seed} histories={} groups={}", histories.len(), groups.len());
        rows.push(json!({"seed":seed,"iterations":record["iterations"],"snapshot":snapshot,"histories":histories,"groups":groups}));
        std::fs::write(&args[3],serde_json::to_vec(&json!({"rows":rows})).unwrap()).unwrap();
    }
}
