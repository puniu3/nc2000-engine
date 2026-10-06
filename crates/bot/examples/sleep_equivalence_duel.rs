use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    preview::load_meta_pool, run_duel, smmcts::SelRule, BlindAgent, DuelSpec, RmConfig,
};
use serde_json::json;
use std::sync::Arc;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() == 1 || (args.len() == 2 && args[1] == "--null"));
    let is_null = args.len() == 2;
    let dex = load_dex();
    let pool = Arc::new(load_meta_pool(
        &repo_root().join("data/belief-pool-v3/belief-pool.json"),
    ));
    let catalog = load_meta_pool(&repo_root().join("data/team-pool-v2/team-pool.json"));
    let teams: Vec<_> = catalog.teams.iter().map(|t| t.sets.clone()).collect();
    let iterations = if is_null { 3000 } else { 27000 };
    let base_seed = if is_null { 2026100803 } else { 2026100804 };
    let agent = |seed, enabled| {
        Box::new(BlindAgent::new(
            RmConfig {
                iterations,
                c: 0.4,
                rule: SelRule::Ucb,
                collapse_sleep_actions: enabled,
                ..Default::default()
            },
            pool.clone(),
            None,
            seed,
        )) as Box<dyn nc2000_bot::Agent>
    };
    let result = run_duel(
        &dex,
        &teams,
        &|seed| agent(seed, !is_null),
        &|seed| agent(seed, false),
        DuelSpec {
            games: if is_null { 8 } else { 64 },
            base_seed,
            threads: 4,
            max_turns: 500,
            progress: true,
            log_on: true,
            crn_agent_seeds: true,
        },
    );
    if is_null {
        assert!(result.pair_scores.iter().all(|&score| score == 0.5));
    }
    println!(
        "{}",
        json!({"null_control":is_null,"base_seed":base_seed,"iterations":iterations,
        "games":result.games,"pairs":result.pairs,"wins":result.wins,"losses":result.losses,
        "ties":result.ties,"caps":result.turn_caps,"score":result.score,"ci95_halfwidth":result.ci95,
        "pair_scores":result.pair_scores,"game_scores":result.game_scores,"seconds":result.secs,
        "mean_think_ms":[result.a_ms_per_move,result.b_ms_per_move]})
    );
}
