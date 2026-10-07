use nc2000_bot::rng;
use nc2000_engine::prng::BattleRng;
use serde_json::{json, Value};

#[allow(dead_code)]
mod source {
    include!("hidden_family_search.rs");
    pub fn model(dex: &Dex, game: &Value, row: &Value, alternate: bool) -> (Battle, Observer) {
        replay(dex, game, row, alternate)
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus case output");
    let case: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let game: Value = std::fs::read_to_string(&args[1]).unwrap().lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap()).find(|g| g["game"] == case["game"]).unwrap();
    let dex = conformance::load_dex();
    let side = case["side"].as_u64().unwrap() as usize;
    let (mut base, _) = source::model(&dex, &game, &case, true);
    let choices = [0, 1].map(|s| base.legal_choices(&dex, s).into_iter().find(|a|
        a.to_input(&dex) == if s == side { "move rockslide" } else { "move sleeptalk" }));
    assert!(choices.iter().all(Option::is_some));
    let mut pending = vec![Vec::<usize>::new()];
    let mut runs = 0;
    let mut leaves = Vec::new();
    while let Some(script) = pending.pop() {
        let mut b = base.clone();
        b.set_log_enabled(false);
        b.set_log_enabled(true);
        b.prng = BattleRng::enumerating(script.clone());
        b.apply_choices(&dex, choices).unwrap();
        runs += 1;
        assert!(runs <= 200000);
        let trace = b.prng.oracle.take().unwrap().trace;
        assert!(script.len() <= trace.len());
        if script.len() < trace.len() {
            for (i, &count) in trace[script.len()].counts.iter().enumerate().rev() {
                if count > 0 {
                    let mut next = script.clone();
                    next.push(i);
                    pending.push(next);
                }
            }
        } else {
            let damage = !b.damage_bookkeeping_observable(&dex);
            let claw = !b.quick_claw_roll_observable(&dex);
            let key = if claw { b.state_key128_without_quick_claw_roll(&dex, damage) }
                else if damage { b.state_key128_without_damage_bookkeeping() } else { b.state_key128() };
            let legal = b.legal_choices(&dex, side).iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>();
            leaves.push(json!({"script":script,"trace":trace.iter().map(|d|
                json!({"label":d.label,"counts":d.counts,"chosen":d.chosen})).collect::<Vec<_>>(),
                "cache_key":[damage,claw,key.to_string()],"log":b.log,"request":b.request_state.as_str(),"legal":legal}));
        }
    }
    std::fs::write(&args[3], serde_json::to_vec(&json!({"case":case,"runs":runs,"leaves":leaves})).unwrap()).unwrap();
}
