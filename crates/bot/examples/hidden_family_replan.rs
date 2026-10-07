use nc2000_bot::rng;
use nc2000_engine::prng::BattleRng;
use serde_json::{json, Value};

#[allow(dead_code)]
mod source {
    include!("hidden_family_search.rs");
    pub fn model(dex: &Dex, game: &Value, row: &Value) -> (Battle, Observer) {
        replay(dex, game, row, true)
    }
    pub fn checks(dex: &Dex, worlds: &[(Battle, Observer)], side: usize, reflect: usize) -> Value {
        check_model(dex, worlds, side, reflect)
    }
    pub fn search(dex: &Dex, worlds: &[(Battle, Observer)], pool: &MetaPool, side: usize,
        template: usize, mode: &str, seed: u64, timed: bool) -> Value {
        run(dex, worlds, pool, side, template, mode, seed, timed)
    }
}

#[allow(dead_code)]
mod certificate {
    include!("hidden_terminal.rs");
    pub fn solve(dex: &Dex, b: Battle, config: Value) -> Value {
        finite::solve(dex, b, config)
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus focus output");
    let focus: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let cases = focus["rows"].as_array().unwrap();
    let game: Value = std::fs::read_to_string(&args[1]).unwrap().lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap()).find(|g| g["game"] == cases[0]["case"]["game"]).unwrap();
    let dex = conformance::load_dex();
    let side = cases[0]["case"]["side"].as_u64().unwrap() as usize;
    let mut worlds: Vec<(nc2000_engine::state::Battle, nc2000_bot::Observer)> = Vec::new();
    let mut world_rows = Vec::new();
    for row in cases {
        let (mut b, mut obs) = source::model(&dex, &game, &row["case"]);
        let prefix = b.log.len();
        let choices = [0, 1].map(|s| b.legal_choices(&dex, s).into_iter().find(|a|
            a.to_input(&dex) == if s == side { "move rockslide" } else { "move sleeptalk" }));
        assert!(choices.iter().all(Option::is_some));
        let script: Vec<usize> = serde_json::from_value(row["leaf"]["script"].clone()).unwrap();
        b.prng = BattleRng::enumerating(script);
        b.apply_choices(&dex, choices).unwrap();
        obs.observe(&b, &dex);
        let trace = b.prng.oracle.take().unwrap().trace;
        let draws: Vec<_> = trace.iter().map(|d| json!({"label":d.label,"counts":d.counts,"chosen":d.chosen})).collect();
        assert_eq!(json!(draws), row["leaf"]["trace"]);
        assert_eq!(json!(&b.log[prefix..]), row["leaf"]["log"]);
        assert!(!b.quick_claw_roll_observable(&dex) && !b.damage_bookkeeping_observable(&dex));
        let key = json!([true,true,b.state_key128_without_quick_claw_roll(&dex,true).to_string()]);
        assert_eq!(key, row["leaf"]["cache_key"]);
        if let Some((other, observer)) = worlds.first() {
            assert_eq!(b.log, other.log);
            assert_eq!(format!("{:?}",obs.mons()),format!("{:?}",observer.mons()));
        }
        world_rows.push(json!({"move":row["case"]["replacement"],"cache_key":key,"state":b.essence(&dex)}));
        worlds.push((b,obs));
    }
    assert_eq!(worlds.len(),39);
    let reflect = world_rows.iter().position(|w|w["move"]=="reflect").unwrap();
    let checks = source::checks(&dex,&worlds,side,reflect);
    let snore = world_rows.iter().position(|w|w["move"]=="snore").unwrap();
    let config=json!({"game":cases[0]["case"]["game"],"decision":cases[0]["case"]["decision"],
        "target_side":side,"policy_side":side,"first_choice":"move rockslide","priority":["move rockslide"],
        "rest_at_percent":null,"horizon":2,"work_budget":200000,"cell_cap":10000,
        "guarded_keys":true,"support_certificate":true});
    let snore_certificate = certificate::solve(&dex,worlds[snore].0.clone(),config);
    eprintln!("snore runs={} bounds={}",snore_certificate["runs"],snore_certificate["bounds"]);
    let pool=nc2000_bot::preview::load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut rows=Vec::new();
    for timed in [false,true] {
        for index in 0..16 {
            let order=if index%2==0 {["catalog","uniform","robust"]} else {["robust","uniform","catalog"]};
            for mode in order {
                for template in if timed {vec![0]} else {vec![0,reflect]} {
                    let row=source::search(&dex,&worlds,&pool,side,template,mode,0x2026110401+index,timed);
                    eprintln!("{mode} seed={index} timed={timed} template={template} best={} iterations={}",row["best"],row["iterations"]);
                    rows.push(row);
                    std::fs::write(&args[3],serde_json::to_vec(&json!({"focus":focus,"worlds":world_rows,
                        "checks":checks,"snore_certificate":snore_certificate,"rows":rows})).unwrap()).unwrap();
                }
            }
        }
    }
}
