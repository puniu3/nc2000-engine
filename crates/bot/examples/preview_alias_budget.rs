use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{battle::SearchChoice, state::Battle};
use serde_json::json;
use std::{path::Path, time::Instant};

fn main() {
    let output = std::env::args().nth(1).expect("output path");
    let dex = conformance::load_dex();
    let catalog = load_meta_pool(Path::new("data/team-pool-v2/team-pool.json"));
    let prior = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    assert_eq!(catalog.teams.len(), 24);
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut rows = Vec::new();
    for i in 0..catalog.teams.len() {
        let j = (i + 1) % catalog.teams.len();
        let side = i % 2;
        let teams = if side == 0 { [i, j] } else { [j, i] };
        let b = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &catalog.teams[teams[0]].sets,
            &catalog.teams[teams[1]].sets,
        )
        .unwrap();
        let obs = Observer::new(&b, side);
        let belief = Belief::new(&dex, &prior, &obs);
        let seed = 0x2026102500 ^ i as u64;
        let mut warm = BlindSearch::new(&b, &dex, cfg.clone(), side, seed);
        let actions: Vec<_> = warm
            .actions()
            .iter()
            .map(|a| {
                let SearchChoice::Team(t) = a else {
                    panic!("preview")
                };
                *t
            })
            .collect();
        warm.step(&dex, &belief, &obs, 1000);
        drop(warm);
        let mut runs = Vec::new();
        for repeat in 0..3 {
            for enabled in if (i + repeat) % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let mut search = BlindSearch::new(&b, &dex, cfg.clone(), side, seed);
                let removed = if enabled {
                    search.prune_preview_aliases(&dex)
                } else {
                    0
                };
                search.step(&dex, &belief, &obs, 27000);
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                let SearchChoice::Team(mut best) = search.best().unwrap() else {
                    panic!("preview")
                };
                best[1..].sort_unstable();
                runs.push(json!({"enabled":enabled,"repeat":repeat,"elapsed_ms":elapsed_ms,"removed":removed,
                    "best_canonical":best,"best":search.best().unwrap().to_input(&dex),
                    "visits":search.visits(),"means":search.means(),"nodes":search.node_count()}));
            }
        }
        eprintln!("preview {}/24", i + 1);
        rows.push(
            json!({"index":i,"own":catalog.teams[i].id,"opponent":catalog.teams[j].id,
            "side":side,"seed":seed,"state_key":b.state_key128().to_string(),"actions":actions,"runs":runs}),
        );
    }
    std::fs::write(
        output,
        serde_json::to_vec(&json!({"iterations":27000,"rows":rows})).unwrap(),
    )
    .unwrap();
}
