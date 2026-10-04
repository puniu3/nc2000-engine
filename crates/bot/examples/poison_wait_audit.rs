use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    smmcts::dominated_actions_with, Belief, BlindSearch, Observer, ProtocolAgent, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, Outcome},
    dex::Dex,
    state::Battle,
};
use serde_json::json;

#[path = "../tests/support/residual_wait.rs"]
mod fixture;

fn matrix(b: &Battle, dex: &Dex) -> serde_json::Value {
    let mut probe = b.clone();
    let acts = [probe.legal_choices(dex, 0), probe.legal_choices(dex, 1)];
    let rows: Vec<_> = acts[0]
        .iter()
        .map(|&a| {
            let cells: Vec<_> = acts[1]
                .iter()
                .map(|&reply| {
                    let result = enumerate_step(dex, b, [Some(a), Some(reply)], 100000).unwrap();
                    let mut scores = [0.0; 4];
                    for leaf in result.leaves {
                        let index = match leaf.battle.outcome() {
                            Some(Outcome::P1Win) => 0,
                            Some(Outcome::P2Win) => 1,
                            Some(Outcome::Tie) => 2,
                            None => 3,
                        };
                        scores[index] += leaf.prob;
                    }
                    assert!((scores.iter().sum::<f64>() - 1.0).abs() < 1e-8);
                    json!({"reply":reply.to_input(dex),"win":scores[0],"loss":scores[1],
                "tie":scores[2],"unresolved":scores[3],"runs":result.runs})
                })
                .collect();
            json!({"action":a.to_input(dex),"cells":cells})
        })
        .collect();
    json!(rows)
}

fn main() {
    let dex = load_dex();
    let (preview, b, [mine, theirs]) = fixture::position(&dex);
    let battle_seed = 1;
    let baseline = std::env::args().any(|arg| arg == "--baseline");
    let rules = nc2000_bot::smmcts::MaskRules {
        residual_damage_wait: !baseline,
        ..Default::default()
    };
    assert!(b.outcome().is_none());
    let m = matrix(&b, &dex);
    for row in m.as_array().unwrap() {
        let cells = row["cells"].as_array().unwrap();
        assert!(cells.iter().all(|c| c["unresolved"] == 0.0));
        if ["move toxic", "move thunderwave"].contains(&row["action"].as_str().unwrap()) {
            assert!(cells
                .iter()
                .all(|c| c["win"].as_f64().unwrap() > 1.0 - 1e-8));
        } else {
            assert!(cells.iter().any(|c| c["loss"].as_f64().unwrap() > 0.9));
        }
    }
    println!(
        "{}",
        json!({"type":"proof","battle_seed":[1,2,3,battle_seed],
        "teams":[mine,theirs],"log":b.log,"matrix":m,
        "hp":[b.poke(b.active_id(0).unwrap()).hp,b.poke(b.active_id(1).unwrap()).hp],
        "mask":dominated_actions_with(&b,&dex,0,rules).iter().map(|(a,r)|json!([a.to_input(&dex),r])).collect::<Vec<_>>()})
    );
    for blind in [false, true] {
        let mut observer = Observer::new(&preview, 0);
        let mut belief = if blind {
            Belief::new(
                &dex,
                &nc2000_bot::preview::load_meta_pool(
                    &repo_root().join("data/belief-pool-v3/belief-pool.json"),
                ),
                &observer,
            )
        } else {
            Belief::pinned_from_battle(&preview, &observer)
        };
        observer.observe(&b, &dex);
        belief.sync(&dex, &observer);
        for seed in 81001..81009 {
            let mut search = BlindSearch::new(
                &b,
                &dex,
                RmConfig {
                    c: 0.4,
                    rule: nc2000_bot::smmcts::SelRule::Ucb,
                    iterations: 27000,
                    mask_rules: rules,
                    ..RmConfig::default()
                },
                0,
                seed,
            );
            search.step(&dex, &belief, &observer, 27000);
            println!(
                "{}",
                json!({"type":"search","blind":blind,"seed":seed,
                "choice":search.best().unwrap().to_input(&dex),
                "unfiltered_choice":search.actions()[(0..search.actions().len()).max_by_key(|&i| search.visits()[i]).unwrap()].to_input(&dex),
                "actions":search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),
                    "visits":search.visits()[i],"mean":search.means()[i],"excluded":search.dominated()[i]})).collect::<Vec<_>>()})
            );
        }
    }
    for seed in 82001..82009 {
        let pool = nc2000_bot::preview::load_meta_pool(
            &repo_root().join("data/belief-pool-v3/belief-pool.json"),
        );
        let mut agent = ProtocolAgent::new(
            &dex,
            0,
            pool,
            RmConfig {
                c: 0.4,
                rule: nc2000_bot::smmcts::SelRule::Ucb,
                iterations: 27000,
                mask_rules: rules,
                ..RmConfig::default()
            },
            seed,
        );
        agent.set_own_team(mine.to_vec());
        let frame = nc2000_bot::player::PlayerChannel::new(0)
            .frame(&mut b.clone(), &dex)
            .unwrap();
        for line in frame.lines {
            agent.push_line(&dex, &line);
        }
        assert!(agent.on_request(&dex, &frame.request.to_string()).unwrap());
        agent.step(&dex, 27000).unwrap();
        let choice = agent.best(&dex).unwrap();
        let search = agent.search().unwrap();
        let unfiltered = (0..search.actions().len())
            .max_by_key(|&i| search.visits()[i])
            .unwrap();
        println!(
            "{}",
            json!({"type":"protocol_search","seed":seed,"choice":choice,
            "unfiltered_choice":search.actions()[unfiltered].to_input(&dex),
            "actions":search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),
                "visits":search.visits()[i],"mean":search.means()[i],"excluded":search.dominated()[i]})).collect::<Vec<_>>()})
        );
    }
    for control in ["no_poison", "attack_also_kills"] {
        let mut changed = b.clone();
        let foe = changed.active_id(1).unwrap();
        if control == "no_poison" {
            changed.poke_mut(foe).status = nc2000_engine::state::Status::None;
        } else {
            changed.poke_mut(foe).hp = 1;
        }
        let m = matrix(&changed, &dex);
        for row in m.as_array().unwrap() {
            for cell in row["cells"].as_array().unwrap() {
                if control == "attack_also_kills" {
                    assert!(cell["win"].as_f64().unwrap() > 1.0 - 1e-8);
                } else if row["action"] == "move thunderwave" || row["action"] == "move toxic" {
                    assert_eq!(cell["win"], 0.0);
                    assert_eq!(cell["unresolved"], 1.0);
                }
            }
        }
        println!("{}", json!({"type":"control","case":control,"matrix":m}));
    }
}
