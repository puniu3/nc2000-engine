use nc2000_bot::{
    preview::load_meta_pool,
    smmcts::{SearchTrace, SelRule},
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, PokemonSet},
    dex::Dex,
    state::{Battle, DK},
};
use serde_json::{json, Value};
use std::{
    io::BufRead,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

fn state(b: &Battle, dex: &Dex) -> Value {
    json!({"turn":b.turn,"outcome":format!("{:?}",b.outcome()),"sides":([0,1].map(|s| {
        json!({"alive":b.sides[s].pokemon_left,"active":b.active_id(s).map(|id| {
            let p=b.poke(id);
            json!({"species":dex.species.key(p.species),"hp":p.hp,"maxhp":p.maxhp,
                "status":p.status.as_str(),"sleep":p.status_state.get_int(DK::Time),
                "boosts":format!("{:?}",p.boosts),"moves":p.move_slots.iter().map(|m|json!({"move":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>()})
        })})
    }))})
}

fn snapshot(search: &BlindSearch, dex: &Dex) -> Value {
    json!({"iterations":search.iterations(),"best":search.best().unwrap().to_input(dex),
        "visits":search.visits(),"means":search.means(),"nodes":search.node_count()})
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        args.len() == 3
            || (args.len() == 4
                && matches!(args[3].as_str(), "--revisit" | "--revisit-integrated")),
        "corpus output [--revisit | --revisit-integrated]"
    );
    let revisit = args.len() == 4;
    let integrated = args.get(3).is_some_and(|a| a == "--revisit-integrated");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut cases = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let g: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let game = g["game"].as_u64().unwrap();
        let decision = match game {
            21 => 22,
            23 => 12,
            _ => continue,
        };
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(g["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, g["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
        let mut obs = Observer::new(&b, 0);
        for (i, d) in g["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            if i == decision {
                cases.push((game, decision, b.clone(), obs.clone()));
                break;
            }
            let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|a| acts[s][a as usize])),
            )
            .unwrap();
            obs.observe(&b, &dex);
        }
    }
    assert_eq!(cases.len(), 2);
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(||loop {
                let task=next.fetch_add(1,Ordering::Relaxed);
                if task>=34 {break;}
                let (game,decision,b,obs)=&cases[task/17];
                let index=task%17;
                let original=0x20261008u64^(b.state_key128().wrapping_mul(0x9e3779b97f4a7c15) as u64);
                let seed=if index==0 {original} else {0x2026100900+index as u64};
                let belief=Belief::new(&dex,&pool,obs);
                let mut arms=Vec::new();
                for mode in if revisit {vec!["revisit"]} else {vec!["baseline","merged","equal"]} {
                    let mut search=BlindSearch::new(b,&dex,RmConfig {root_revisit_interval:if integrated {64} else {0},..cfg.clone()},0,seed);
                    if mode=="merged" {assert_eq!(search.prune_rest_aliases(&dex),2);}
                    let actions=search.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>();
                    let mut traces=Vec::new();
                    let mut checkpoints=Vec::new();
                    for budget in if index==0 && !revisit {vec![27000,270000]} else {vec![27000]} {
                        if integrated {search.step(&dex,&belief,obs,budget-search.iterations());}
                        else if mode=="revisit" {
                            while search.iterations()<budget {
                                if search.iterations()>0 && search.iterations()%64==0 {
                                    let action=search.visits().iter().enumerate().min_by_key(|(_,n)|**n).unwrap().0;
                                    search.step_forced(&dex,&belief,obs,action);
                                } else {search.step(&dex,&belief,obs,1);}
                            }
                        } else if mode=="equal" {
                            while search.iterations()<budget {
                                let action=search.iterations() as usize%actions.len();
                                search.step_forced(&dex,&belief,obs,action);
                            }
                        } else if index==0 {
                            while search.iterations()<budget {
                                let iteration=search.iterations()+1;
                                let mut path=Vec::new();
                                let mut selected=false;
                                let mut root_seen=false;
                                search.step_observed(&dex,&belief,obs,1,&mut |trace| match trace {
                                    SearchTrace::Choice{battle,chosen,..}=> {
                                        if !root_seen {selected=chosen[0].is_some_and(|a|a.to_input(&dex)=="move sleeptalk");root_seen=true;}
                                        if selected && traces.len()<25 {path.push(json!({"state":state(battle,&dex),"chosen":chosen.map(|a|a.map(|a|a.to_input(&dex)))}));}
                                    },
                                    SearchTrace::Leaf{battle,rollout,..}=> {
                                        if selected && traces.len()<25 {path.push(json!({"leaf":state(battle,&dex),"rollout":rollout}));}
                                    },
                                    SearchTrace::Result{battle,reward0}=> {
                                        if selected && traces.len()<25 {traces.push(json!({"iteration":iteration,"path":path,"result":state(battle,&dex),"reward":reward0}));}
                                    }
                                });
                            }
                        } else {search.step(&dex,&belief,obs,budget-search.iterations());}
                        checkpoints.push(snapshot(&search,&dex));
                    }
                    arms.push(json!({"mode":mode,"actions":actions,"checkpoints":checkpoints,"traces":traces}));
                }
                rows.lock().unwrap().push(json!({"game":game,"decision":decision,"seed_index":index,"seed":seed,"arms":arms}));
                eprintln!("focus {}/34 complete",task+1);
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|r| (r["game"].as_u64(), r["seed_index"].as_u64()));
    let mut details = Vec::new();
    for (game, decision, mut b, _) in cases.into_iter().filter(|_| !revisit) {
        let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
        let mut cells = Vec::new();
        for own in [
            acts[0][0],
            *acts[0]
                .iter()
                .find(|a| a.to_input(&dex) == "move sleeptalk")
                .unwrap(),
        ] {
            for &reply in &acts[1] {
                let step = enumerate_step_partial(&dex, &b, [Some(own), Some(reply)], 50000);
                let mut groups = std::collections::BTreeMap::<String, f64>::new();
                let mut terminal_mass = 0.0;
                for leaf in &step.leaves {
                    *groups
                        .entry(state(&leaf.battle, &dex).to_string())
                        .or_default() += leaf.prob;
                    if leaf.battle.outcome().is_some() {
                        terminal_mass += leaf.prob;
                    }
                }
                let pending = step.pending.iter().map(|p| p.mass).sum::<f64>();
                assert!((groups.values().sum::<f64>() + pending - 1.0).abs() < 1e-8);
                cells.push(json!({"own":own.to_input(&dex),"reply":reply.to_input(&dex),"runs":step.runs,
                    "pending_mass":pending,"terminal_mass":terminal_mass,"groups":groups.into_iter().map(|(s,p)|json!({"state":serde_json::from_str::<Value>(&s).unwrap(),"prob":p})).collect::<Vec<_>>()}));
            }
        }
        details.push(json!({"game":game,"decision":decision,"key":b.state_key128().to_string(),"state":state(&b,&dex),"cells":cells}));
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec(&json!({"rows":rows,"details":details})).unwrap(),
    )
    .unwrap();
}
