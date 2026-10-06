use nc2000_bot::{
    preview::load_meta_pool,
    smmcts::{RmConfig, SelRule},
    Belief, BlindSearch, Observer,
};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::{json, Value};
use std::{
    io::BufRead,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};

#[derive(Clone)]
struct Case {
    battle: Battle,
    observer: Observer,
    game: Value,
    decision: usize,
    side: usize,
    exposed: bool,
    order: u128,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "corpus output");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut cases = Vec::new();
    let mut decisions = 0;
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        b.set_log_enabled(true);
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
            for side in [0, 1] {
                if !b.needs_choice()[side] || legal[side].len() < 2 || b.active_id(side).is_none() {
                    continue;
                }
                decisions += 1;
                let search = BlindSearch::new(&b, &dex, cfg.clone(), side, 0);
                let exposed =
                    search.dominated().iter().any(|&x| x) && search.dominated().iter().any(|&x| !x);
                let order = b
                    .state_key128()
                    .wrapping_mul(0x9e3779b97f4a7c15u128)
                    .wrapping_add(side as u128);
                cases.push(Case {
                    battle: b.clone(),
                    observer: observers[side].clone(),
                    game: game["game"].clone(),
                    decision,
                    side,
                    exposed,
                    order,
                });
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| legal[s][i as usize])),
            )
            .unwrap();
            for s in [0, 1] {
                observers[s].observe(&b, &dex);
            }
        }
    }
    let exposed = cases.iter().filter(|c| c.exposed).count();
    cases.sort_by_key(|c| c.order);
    let mut quotas = [8usize, 48];
    cases.retain(|c| {
        let i = usize::from(c.exposed);
        if quotas[i] == 0 {
            false
        } else {
            quotas[i] -= 1;
            true
        }
    });
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(||loop {
        let i=next.fetch_add(1,Ordering::Relaxed);if i>=cases.len(){break;}let c=&cases[i];let seed=0x20261007u64^(c.order as u64);
        let belief=Belief::new(&dex,&pool,&c.observer);let mut arms=Vec::new();
        for prune in [false,true] {
            let mut search=BlindSearch::new(&c.battle,&dex,cfg.clone(),c.side,seed);if prune {search.prune_dominated();}
            let started=Instant::now();let mut checkpoints=Vec::new();
            for budget in [3000,27000] {
                search.step(&dex,&belief,&c.observer,budget-search.iterations());
                checkpoints.push(json!({"iterations":budget,"best":search.best().unwrap().to_input(&dex),"visits":search.visits(),"means":search.means(),"nodes":search.node_count(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.}));
            }
            if prune {assert!(search.visits().iter().zip(search.dominated()).all(|(&n,&masked)|!masked || n==0));}
            arms.push(json!({"prune":prune,"dominated":search.dominated(),"actions":search.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"checkpoints":checkpoints}));
        }
        if !c.exposed {for j in 0..2 {let a=&arms[0]["checkpoints"][j];let b=&arms[1]["checkpoints"][j];for k in ["iterations","best","visits","means","nodes"] {assert_eq!(a[k],b[k]);}}}
        rows.lock().unwrap().push(json!({"index":i,"game":c.game,"decision":c.decision,"side":c.side,"exposed":c.exposed,"state_key":c.battle.state_key128().to_string(),"seed":seed,"arms":arms}));
        eprintln!("root {}/{} complete",i+1,cases.len());
    });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|r| r["index"].as_u64().unwrap());
    let output = json!({"decisions":decisions,"exposed":exposed,"sampled":rows.len(),"rows":rows});
    std::fs::write(&args[2], serde_json::to_vec(&output).unwrap()).unwrap();
}
