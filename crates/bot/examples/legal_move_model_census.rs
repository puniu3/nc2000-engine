use nc2000_bot::{preview::load_meta_pool, Observer};
use nc2000_engine::{battle::PokemonSet, state::Battle, validate::Learnsets};
use serde_json::{json, Value};

mod legal_move_models;

fn main() {
    let args: Vec<_>=std::env::args().collect();
    assert_eq!(args.len(),3,"corpus output");
    let dex=conformance::load_dex();
    let learnsets=Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap()).unwrap();
    let routes: Value=serde_json::from_slice(&std::fs::read("data/legal-move-sources-gen2.json").unwrap()).unwrap();
    let pool=load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut rows=Vec::new();let mut boundaries=0;
    for line in std::fs::read_to_string(&args[1]).unwrap().lines() {
        let game: Value=serde_json::from_str(line).unwrap();
        let teams: [Vec<PokemonSet>;2]=serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b=Battle::from_fixture(&dex,game["seed"].as_str().unwrap(),&teams[0],&teams[1]).unwrap();
        b.set_log_enabled(true);
        let mut obs=[0,1].map(|s|Observer::new(&b,s));
        for (decision,d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(),d["key"].as_str().unwrap());
            boundaries+=1;
            for side in 0..2 {
                if !b.needs_choice()[side] || b.legal_choices(&dex,side).len()<2 {continue;}
                if let Some(models)=legal_move_models::generate(&dex,&learnsets,&routes,&pool,&b,&obs[side],0) {
                    let active=b.active_id(1-side).unwrap();
                    let known=&obs[side].mons()[active.slot as usize].revealed_moves;
                    for world in &models.worlds {
                        let p=world.poke(active);
                        assert!((3..=4).contains(&p.base_move_slots.len()));
                        assert!(known.iter().all(|&r|p.base_move_slots.iter().any(|m|nc2000_bot::observe::move_matches(&dex,m.id,r))));
                        assert_eq!(world.clone().legal_choices(&dex,side),b.legal_choices(&dex,side));
                    }
                    rows.push(json!({"game":game["game"],"decision":decision,"side":side,"key":d["key"],
                        "species":dex.species.key(b.poke(active).species),"level":b.poke(active).level,
                        "known":known.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>(),
                        "candidates":models.names,"excluded":models.excluded}));
                }
            }
            let actions=[0,1].map(|s|b.legal_choices(&dex,s));
            let choices=[0,1].map(|s|d["choices"][s].as_u64().map(|i|actions[s][i as usize]));
            b.apply_choices(&dex,choices).unwrap();
            for observer in &mut obs {observer.observe(&b,&dex);}
        }
    }
    std::fs::write(&args[2],serde_json::to_vec(&json!({"boundaries":boundaries,"rows":rows})).unwrap()).unwrap();
}
