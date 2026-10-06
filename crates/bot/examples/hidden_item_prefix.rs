use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig, SplitMix64,
};
use nc2000_engine::{
    battle::PokemonSet,
    state::Battle,
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use std::{io::BufRead, path::Path};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 4, "corpus item-plan output");
    let games: Vec<Value> = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap())
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    let plan: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut pairs = Vec::new();
    for r in plan["rows"].as_array().unwrap() {
        let game = games.iter().find(|g| g["game"] == r["game"]).unwrap();
        let side = r["side"].as_u64().unwrap() as usize;
        let slot = r["foe_slot"].as_u64().unwrap() as usize;
        let original: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut alternate = original.clone();
        assert_eq!(
            original[1 - side][slot].item,
            r["original_item"].as_str().unwrap()
        );
        alternate[1 - side][slot].item = r["alternate_item"].as_str().unwrap().to_string();
        if let Some(other) = r["swapped_slot"].as_u64() {
            alternate[1 - side][other as usize].item = original[1 - side][slot].item.clone();
        }
        assert_eq!(
            validate_team(
                &dex,
                &learnsets,
                &serde_json::to_string(&alternate[1 - side]).unwrap()
            )["ok"],
            true
        );
        let mut battles = [&original, &alternate].map(|t| {
            let mut b =
                Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &t[0], &t[1]).unwrap();
            b.set_log_enabled(true);
            b
        });
        let mut obs = battles.each_ref().map(|b| Observer::new(b, side));
        let mut rows = Vec::new();
        let mut stop =
            json!({"reason":"corpus_end","decision":game["decisions"].as_array().unwrap().len()});
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                battles[0].state_key128().to_string(),
                d["key"].as_str().unwrap()
            );
            if battles[0].log != battles[1].log {
                let mismatch = battles[0]
                    .log
                    .iter()
                    .zip(&battles[1].log)
                    .position(|(a, b)| a != b)
                    .unwrap_or(battles[0].log.len().min(battles[1].log.len()));
                stop = json!({"reason":"public_log","decision":decision,"line":mismatch,
                    "lines":battles.each_ref().map(|b|b.log.get(mismatch))});
                break;
            }
            let obs_equal = format!("{:?}", obs[0].mons()) == format!("{:?}", obs[1].mons());
            if !obs_equal {
                stop = json!({"reason":"observer","decision":decision});
                break;
            }
            let choices = battles
                .each_mut()
                .map(|b| [0, 1].map(|s| b.legal_choices(&dex, s)));
            if choices[0] != choices[1] {
                stop = json!({"reason":"legal_choices","decision":decision});
                break;
            }
            if choices[0][side].len() > 1
                && battles[0].is_idle_move_request()
                && battles[1].is_idle_move_request()
            {
                let roots = battles.each_ref().map(|b| {
                    let mut root = b.clone();
                    root.quick_claw_roll = false;
                    root.update_all_speeds(&dex);
                    root
                });
                let belief = Belief::new(&dex, &pool, &obs[0]);
                let mut mismatches = Vec::new();
                let seed = 0x2026102900 ^ battles[0].state_key128() as u64 ^ ((side as u64) << 32);
                for sample in 0..32 {
                    let projections = [0, 1].map(|i| {
                        belief.determinize(
                            &dex,
                            &roots[i],
                            &obs[i],
                            &mut SplitMix64::new(seed ^ sample),
                        )
                    });
                    if projections[0].state_key128() != projections[1].state_key128() {
                        mismatches.push(json!({"sample":sample,"keys":projections.each_ref().map(|b|b.state_key128().to_string())}));
                    }
                }
                let mut searches = roots
                    .each_ref()
                    .map(|b| BlindSearch::new(b, &dex, cfg.clone(), side, seed));
                let mask_changed = searches[0].dominated() != searches[1].dominated();
                let mut checkpoints = Vec::new();
                if mask_changed || !mismatches.is_empty() {
                    for target in [3000, 27000] {
                        for i in 0..2 {
                            let remaining = target - searches[i].iterations();
                            searches[i].step(&dex, &belief, &obs[i], remaining);
                        }
                        checkpoints.push(json!({"iterations":target,
                            "arms":searches.each_ref().map(|s|json!({"best":s.best().unwrap().to_input(&dex),"visits":s.visits(),"means":s.means(),"rng":s.search_rng_state()}))}));
                    }
                }
                rows.push(json!({"decision":decision,"turn":battles[0].turn,"key":d["key"],"seed":seed,
                    "public_log_and_observer_match":true,"actions":searches[0].actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),
                    "masks":searches.each_ref().map(|s|s.dominated()),"mask_changed":mask_changed,
                    "projection_checks":32,"projection_mismatches":mismatches,"checkpoints":checkpoints}));
            }
            for i in 0..2 {
                battles[i]
                    .apply_choices(
                        &dex,
                        [0, 1].map(|s| {
                            d["choices"][s]
                                .as_u64()
                                .map(|index| choices[i][s][index as usize])
                        }),
                    )
                    .unwrap();
                obs[i].observe(&battles[i], &dex);
            }
        }
        let end_logs_equal = battles[0].log == battles[1].log;
        if stop["reason"] == "corpus_end" && !end_logs_equal {
            stop["reason"] = json!("public_log_after_last_step");
        }
        pairs.push(json!({"game":r["game"],"side":side,"foe_slot":slot,"rows":rows,"stop":stop,
            "end_logs_equal":end_logs_equal,"outcomes":battles.each_ref().map(|b|format!("{:?}",b.outcome()))}));
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({"pairs":pairs})).unwrap(),
    )
    .unwrap();
}
