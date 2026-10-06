use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig, SplitMix64,
};
use nc2000_engine::{
    battle::PokemonSet,
    dex::Dex,
    state::Battle,
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use std::{io::BufRead, path::Path};

fn opening(
    dex: &Dex,
    teams: &[Vec<PokemonSet>; 2],
    game: &Value,
    side: usize,
) -> (Battle, Observer) {
    let mut b =
        Battle::from_fixture(dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(true);
    let mut obs = Observer::new(&b, side);
    let actions = [0, 1].map(|s| b.legal_choices(dex, s));
    b.apply_choices(
        dex,
        [0, 1].map(|s| {
            Some(actions[s][game["decisions"][0]["choices"][s].as_u64().unwrap() as usize])
        }),
    )
    .unwrap();
    obs.observe(&b, dex);
    b.quick_claw_roll = false;
    b.update_all_speeds(dex);
    (b, obs)
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 3, "corpus output");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let mut rows = Vec::new();
    let mut openings = 0;
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        for side in 0..2 {
            let (a, obs) = opening(&dex, &teams, &game, side);
            let foe = a.active_id(1 - side).unwrap();
            let slot = foe.slot as usize;
            if a.poke(foe).item.is_none() {
                continue;
            }
            assert!(obs.mons()[slot].item.current.is_none());
            let mut alternate = teams.clone();
            let roster = &mut alternate[1 - side];
            let claw = roster
                .iter()
                .position(|p| nc2000_engine::dex::toid(&p.item) == "quickclaw");
            let swap = if claw == Some(slot) {
                roster.iter().position(|p| {
                    !p.item.is_empty() && nc2000_engine::dex::toid(&p.item) != "quickclaw"
                })
            } else {
                claw
            };
            if let Some(other) = swap {
                let item = roster[slot].item.clone();
                roster[slot].item = roster[other].item.clone();
                roster[other].item = item;
            } else {
                assert_ne!(claw, Some(slot));
                roster[slot].item = "Quick Claw".into();
            }
            let validity = validate_team(
                &dex,
                &learnsets,
                &serde_json::to_string(&alternate[1 - side]).unwrap(),
            );
            assert_eq!(validity["ok"], true, "{validity}");
            let (b, obs_b) = opening(&dex, &alternate, &game, side);
            assert_eq!(a.log, b.log);
            assert_eq!(format!("{:?}", obs.mons()), format!("{:?}", obs_b.mons()));
            openings += 1;
            let belief = Belief::new(&dex, &pool, &obs);
            for sample in 0..32 {
                let seed = 0x2026102800 ^ sample;
                let x = belief.determinize(&dex, &a, &obs, &mut SplitMix64::new(seed));
                let y = belief.determinize(&dex, &b, &obs_b, &mut SplitMix64::new(seed));
                assert_eq!(x.state_key128(), y.state_key128());
            }
            let seed = 0x2026102800 ^ game["game"].as_u64().unwrap() ^ ((side as u64) << 32);
            let cfg = RmConfig {
                c: 0.4,
                rule: SelRule::Ucb,
                ..Default::default()
            };
            let mut searches =
                [&a, &b].map(|root| BlindSearch::new(root, &dex, cfg.clone(), side, seed));
            let mask_changed = searches[0].dominated() != searches[1].dominated();
            let mut checkpoints = Vec::new();
            if mask_changed {
                for target in [3000, 27000] {
                    for search in &mut searches {
                        search.step(&dex, &belief, &obs, target - search.iterations());
                    }
                    assert_eq!(searches[0].visits(), searches[1].visits());
                    assert_eq!(searches[0].means(), searches[1].means());
                    assert_eq!(
                        searches[0].search_rng_state(),
                        searches[1].search_rng_state()
                    );
                    checkpoints.push(json!({"iterations":target,"choices":searches.each_ref().map(|s|s.best().unwrap().to_input(&dex)),"visits":searches[0].visits(),"means":searches[0].means()}));
                }
            }
            rows.push(json!({"game":game["game"],"side":side,"foe_slot":slot,"swapped_slot":swap,
                "original_item":teams[1-side][slot].item,"alternate_item":alternate[1-side][slot].item,
                "public_logs_identical":true,"observations_identical":true,"projection_checks":32,
                "legal_alternate_team":true,"mask_changed":mask_changed,"seed":seed,
                "actions":searches[0].actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),
                "masks":searches.each_ref().map(|s|s.dominated()),"checkpoints":checkpoints}));
        }
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec_pretty(&json!({"openings":openings,"rows":rows})).unwrap(),
    )
    .unwrap();
}
