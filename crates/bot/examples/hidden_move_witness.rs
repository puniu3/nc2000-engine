#[allow(dead_code)]
mod certificate {
    include!("hidden_terminal.rs");
    pub fn run(dex: &Dex, b: Battle, config: Value) -> Value {
        finite::solve(dex, b, config)
    }
}
use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    state::Battle,
};
use serde_json::{json, Value};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4
            || (args.len() == 5 && ["--scheduled", "--search"].contains(&args[4].as_str())),
        "corpus census-row output [--scheduled|--search]"
    );
    let row: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let game: Value = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|g| g["game"] == row["game"])
        .unwrap();
    let dex = conformance::load_dex();
    let side = row["side"].as_u64().unwrap() as usize;
    let slot = row["foe_slot"].as_u64().unwrap() as usize;
    let original: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let mut alternate = original.clone();
    alternate[1 - side][slot].moves[row["move_slot"].as_u64().unwrap() as usize] =
        row["replacement"].as_str().unwrap().into();
    let mut battles = [&original, &alternate].map(|t| {
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &t[0], &t[1]).unwrap();
        b.set_log_enabled(true);
        b
    });
    let mut obs = battles.each_ref().map(|b| Observer::new(b, side));
    let old = SearchChoice::Move(
        dex.moves
            .id(row["original_move"].as_str().unwrap())
            .unwrap(),
    );
    let new = SearchChoice::Move(dex.moves.id(row["replacement"].as_str().unwrap()).unwrap());
    let mut checks = 0;
    for d in game["decisions"]
        .as_array()
        .unwrap()
        .iter()
        .take(row["decision"].as_u64().unwrap() as usize)
    {
        assert_eq!(
            battles[0].state_key128().to_string(),
            d["key"].as_str().unwrap()
        );
        assert_eq!(battles[0].log, battles[1].log);
        assert_eq!(
            format!("{:?}", obs[0].mons()),
            format!("{:?}", obs[1].mons())
        );
        checks += 1;
        let actions = [0, 1].map(|s| battles[0].legal_choices(&dex, s));
        let choices = [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize]));
        let mut alt_choices = choices;
        if battles[0]
            .active_id(1 - side)
            .is_some_and(|p| p.slot as usize == slot)
            && alt_choices[1 - side] == Some(old)
        {
            alt_choices[1 - side] = Some(new);
        }
        battles[0].apply_choices(&dex, choices).unwrap();
        battles[1].apply_choices(&dex, alt_choices).unwrap();
        for i in 0..2 {
            obs[i].observe(&battles[i], &dex);
        }
    }
    assert_eq!(battles[0].log, battles[1].log);
    assert_eq!(
        format!("{:?}", obs[0].mons()),
        format!("{:?}", obs[1].mons())
    );
    checks += 1;
    assert_eq!(
        battles[0].state_key128().to_string(),
        row["key"].as_str().unwrap()
    );
    assert_eq!(
        battles[1].state_key128().to_string(),
        row["alternate_key"].as_str().unwrap()
    );
    if args.get(4).is_some_and(|mode| mode == "--search") {
        let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
        let mut searches = Vec::new();
        for pinned in [false, true] {
            for seed in 0..16 {
                for variant in 0..2 {
                    let belief = if pinned {
                        Belief::pinned_from_battle(&battles[variant], &obs[variant])
                    } else {
                        Belief::new(&dex, &pool, &obs[variant])
                    };
                    let mut search = BlindSearch::new(
                        &battles[variant],
                        &dex,
                        RmConfig {
                            c: 0.4,
                            rule: SelRule::Ucb,
                            ..Default::default()
                        },
                        side,
                        0x2026110301 + seed,
                    );
                    search.step(&dex, &belief, &obs[variant], 27000);
                    searches.push(json!({"pinned":pinned,"seed":0x2026110301+seed,"variant":variant,"best":search.best().unwrap().to_input(&dex),"visits":search.visits(),"means":search.means(),"rng":search.search_rng_state(),"nodes":search.node_count()}));
                }
            }
        }
        std::fs::write(
            &args[3],
            serde_json::to_vec(&json!({"case":row,"prefix_checks":checks,"searches":searches}))
                .unwrap(),
        )
        .unwrap();
        return;
    }
    let mut ledgers = Vec::new();
    for variant in 0..2 {
        assert!(!battles[variant].quick_claw_roll_observable(&dex));
        for (target, first, next) in [
            (side, "rockslide", "rockslide"),
            (side, "swordsdance", "rockslide"),
            (1 - side, "sleeptalk", "hiddenpowerice"),
        ] {
            let mut config = json!({"game":row["game"],"decision":row["decision"],"target_side":target,"policy_side":target,"first_choice":format!("move {first}"),"priority":[format!("move {next}")],"rest_at_percent":null,"horizon":3,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true});
            if args.len() == 5 && first == "sleeptalk" {
                config["opening"] =
                    json!(["move sleeptalk", "move sleeptalk", "move hiddenpowerice"]);
            }
            let ledger = certificate::run(&dex, battles[variant].clone(), config);
            eprintln!(
                "variant={variant} first={first} runs={} bounds={}",
                ledger["runs"], ledger["bounds"]
            );
            ledgers.push(json!({"variant":variant,"first":first,"ledger":ledger}));
            std::fs::write(&args[3],serde_json::to_vec(&json!({"case":row,"prefix_checks":checks,"states":battles.each_ref().map(|b|b.essence(&dex)),"ledgers":ledgers})).unwrap()).unwrap();
        }
    }
}
