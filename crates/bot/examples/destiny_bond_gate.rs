use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::smmcts::{dominated_actions_with, MaskRules, SelRule};
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use serde_json::json;

mod combo_certify;
#[path = "../tests/support/destiny_bond_wait.rs"]
mod fixture;

fn config(enabled: bool) -> RmConfig {
    RmConfig {
        rule: SelRule::Ucb,
        c: 0.4,
        mask_rules: MaskRules {
            destiny_bond_wait: enabled,
            residual_damage_wait: false,
            ..MaskRules::default()
        },
        ..RmConfig::default()
    }
}

fn corpus() {
    use nc2000_bot::corpus::{
        corpus_files, load_battle, load_sources, reconstruct_context_with_pool,
    };
    use nc2000_bot::preview::load_meta_pool;
    let root = repo_root();
    let dex = load_dex();
    let sources = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let files = corpus_files(&root.join("tmp/corpus-spectator"));
    let mut tally = [0usize; 6];
    for (index, file) in files.iter().enumerate() {
        let battle = load_battle(file);
        tally[0] += battle.decisions.len();
        if !battle.lines.iter().any(|s| s.contains("Destiny Bond")) {
            continue;
        }
        tally[1] += 1;
        for decision in &battle.decisions {
            let Some(context) = reconstruct_context_with_pool(
                &dex,
                &sources,
                pool.clone(),
                &battle.lines,
                &battle.evidence,
                decision,
                42001,
            ) else {
                continue;
            };
            tally[2] += 1;
            let agent = context.agent;
            let b = agent.battle().unwrap();
            if !b.active_id(1 - decision.side).is_some_and(|id| {
                b.poke(id)
                    .has_volatile(dex.conds_id("destinybond").unwrap())
            }) {
                continue;
            }
            tally[3] += 1;
            let old = dominated_actions_with(b, &dex, decision.side, config(false).mask_rules);
            let new = dominated_actions_with(b, &dex, decision.side, config(true).mask_rules);
            if old == new {
                continue;
            }
            tally[4] += 1;
            let searches: Vec<_> = [false, true]
                .into_iter()
                .map(|enabled| {
                    let mut search =
                        BlindSearch::new(b, &dex, config(enabled), decision.side, 42001);
                    search.step(
                        &dex,
                        agent.belief().unwrap(),
                        agent.observer().unwrap(),
                        27000,
                    );
                    search
                })
                .collect();
            assert_eq!(searches[0].visits(), searches[1].visits());
            assert_eq!(searches[0].means(), searches[1].means());
            let changed = searches[0].best() != searches[1].best();
            tally[5] += usize::from(changed);
            println!(
                "{}",
                json!({"type":"corpus_position","battle":index,
                "turn":decision.turn,"side":decision.side,"changed":changed,
                "own_set_provenance":context.provenance,"imputed_pick":context.imputed_pick,
                "old_mask":old.iter().map(|(a,r)|json!([a.to_input(&dex),r])).collect::<Vec<_>>(),
                "new_mask":new.iter().map(|(a,r)|json!([a.to_input(&dex),r])).collect::<Vec<_>>(),
                "before":searches[0].best().map(|a|a.to_input(&dex)),
                "after":searches[1].best().map(|a|a.to_input(&dex)),
                "actions":searches[0].actions().iter().enumerate().map(|(i,a)|json!({
                    "action":a.to_input(&dex),"visits":searches[0].visits()[i],
                    "mean":searches[0].means()[i]})).collect::<Vec<_>>() })
            );
        }
    }
    println!(
        "{}",
        json!({"type":"corpus_summary","files":files.len(),"decisions":tally[0],
        "bond_files":tally[1],"reconstructed_in_bond_files":tally[2],"active_bond":tally[3],
        "mask_changed":tally[4],"choice_changed":tally[5]})
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|a| a == "--corpus") {
        return corpus();
    }
    let arg = |key: &str, default: u64| -> u64 {
        args.iter()
            .position(|a| a == key)
            .map_or(default, |i| args[i + 1].parse().unwrap())
    };
    let seed = arg("--seed", 51001);
    let seeds = arg("--seeds", 8);
    let iterations = arg("--iters", 27000) as u32;
    let dex = load_dex();
    for plain in [false, true] {
        let name = if plain {
            "bond_without_residual"
        } else {
            "bond_with_curse"
        };
        let fast = args.iter().any(|a| a == "--fast");
        let (preview, b) = fixture::position(&dex, plain, fast);
        println!(
            "{}",
            json!({"type":"fixture","case":name,"own_species":if fast {"Gengar"} else {"Misdreavus"},"log":b.log})
        );
        if args.iter().any(|a| a == "--proof") {
            println!(
                "{}",
                json!({"type":"proof","case":name,"own_species":if fast {"Gengar"} else {"Misdreavus"},
                "result":combo_certify::certify(&dex,&b,0,100000,4096)})
            );
        }
        let mut observer = Observer::new(&preview, 0);
        let blind = args.iter().any(|a| a == "--blind");
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
        for seed in seed..seed + seeds {
            let searches: Vec<_> = [false, true]
                .into_iter()
                .map(|enabled| {
                    let mut search = BlindSearch::new(&b, &dex, config(enabled), 0, seed);
                    search.step(&dex, &belief, &observer, iterations);
                    search
                })
                .collect();
            assert_eq!(searches[0].visits(), searches[1].visits());
            assert_eq!(searches[0].means(), searches[1].means());
            println!(
                "{}",
                json!({"type":"paired_search","case":name,"own_species":if fast {"Gengar"} else {"Misdreavus"},"seed":seed,"iterations":iterations,
                "information":if blind {"blind-v3"} else {"pinned"},
                "statistics_identical":true,"before":searches[0].best().unwrap().to_input(&dex),
                "after":searches[1].best().unwrap().to_input(&dex),
                "actions":searches[0].actions().iter().enumerate().map(|(i,a)|json!({
                    "action":a.to_input(&dex),"visits":searches[0].visits()[i],
                    "mean":searches[0].means()[i]})).collect::<Vec<_>>() })
            );
        }
    }
}
