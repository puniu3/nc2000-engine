use nc2000_bot::{
    preview::load_meta_pool,
    smmcts::{SearchTrace, SelRule},
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::BufRead};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        args.len() == 3 || (args.len() == 4 && args[3] == "--shared"),
        "corpus output [--shared]"
    );
    let enabled = args.len() == 4;
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut cases = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
            if b.turn > 0 && actions.iter().all(|a| a.len() > 1) {
                let active_claw = [0, 1].iter().any(|&s| {
                    b.active_id(s).is_some_and(|id| {
                        b.poke(id).item.is_some() && b.poke(id).item == dex.known_items.quickclaw
                    })
                });
                if active_claw || !b.quick_claw_roll_observable(&dex) {
                    for side in 0..2 {
                        let order = b
                            .state_key128()
                            .wrapping_mul(0x9e3779b97f4a7c15)
                            .wrapping_add(side as u128);
                        cases.push((
                            order,
                            active_claw,
                            game["game"].clone(),
                            decision,
                            side,
                            b.clone(),
                            observers[side].clone(),
                        ));
                    }
                }
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize])),
            )
            .unwrap();
            for obs in &mut observers {
                obs.observe(&b, &dex);
            }
        }
    }
    let available = [false, true].map(|active| cases.iter().filter(|c| c.1 == active).count());
    cases.sort_by_key(|c| c.0);
    let mut retained = [0, 0];
    cases.retain(|c| {
        let i = usize::from(c.1);
        retained[i] += 1;
        retained[i] <= if c.1 { 8 } else { 4 }
    });
    assert_eq!(cases.len(), 12);
    let mut rows = Vec::new();
    for (_, active_claw, game, decision, side, battle, observer) in cases {
        let belief = Belief::new(&dex, &pool, &observer);
        for seed in [0x2026102101, 0x2026102102] {
            let mut search = BlindSearch::new(
                &battle,
                &dex,
                RmConfig {
                    c: 0.4,
                    unseen_claw_root: enabled,
                    rule: SelRule::Ucb,
                    ..Default::default()
                },
                side,
                seed,
            );
            let mut states = BTreeMap::<String, [Option<Value>; 2]>::new();
            let mut samples = BTreeMap::<String, [u32; 2]>::new();
            for iteration in 0..27000 {
                let mut root_seen = false;
                let mut pending = None;
                search.step_observed(&dex,&belief,&observer,1,&mut |trace| match trace {
                    SearchTrace::Choice {battle,node,actions,visits,rewards,chosen} if !root_seen => {
                        root_seen=true;
                        let coin=usize::from(battle.quick_claw_roll);
                        let mut normalized=battle.clone();
                        normalized.quick_claw_roll=false;
                        if normalized.is_idle_move_request() { normalized.update_all_speeds(&dex); }
                        let key=normalized.state_key128().to_string();
                        let n=samples.entry(key.clone()).or_default();n[coin]+=1;
                        let choices=actions[1-side].iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>();
                        let selected=chosen[1-side].and_then(|a|actions[1-side].iter().position(|&b|a==b));
                        pending=Some((key,coin,selected));
                        states.entry(pending.as_ref().unwrap().0.clone()).or_default()[coin]=Some(json!({
                            "node":node,"actions":choices,"visits":visits[1-side],"rewards":rewards[1-side],
                            "iteration":iteration+1,"observable":battle.quick_claw_roll_observable(&dex),"idle":battle.is_idle_move_request()}));
                    },
                    SearchTrace::Result {reward0,..} => {
                        if let Some((key,coin,selected))=pending.take() {
                            let state=states.get_mut(&key).unwrap()[coin].as_mut().unwrap();
                            if let Some(a)=selected.filter(|_|state["actions"].as_array().unwrap().len()>1) {
                                state["rewards"][a]=json!(state["rewards"][a].as_f64().unwrap()+if side==0 {1.-reward0} else {reward0});
                            }
                        }
                    },
                    _=>{}
                });
            }
            let paired=states.into_iter().map(|(key,coins)|json!({"samples":samples[&key],"key_without_coin":key,"coins":coins})).collect::<Vec<_>>();
            rows.push(json!({"game":game,"decision":decision,"side":side,"seed":seed,"active_claw":active_claw,
                "key":battle.state_key128().to_string(),"best":search.best().unwrap().to_input(&dex),
                "visits":search.visits(),"means":search.means(),"nodes":search.node_count(),"types":paired}));
            eprintln!("claw information {}/24 complete", rows.len());
        }
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec(
            &json!({"enabled":enabled,"available":available,"iterations":27000,"rows":rows}),
        )
        .unwrap(),
    )
    .unwrap();
}
