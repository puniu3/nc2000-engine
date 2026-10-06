use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    preview::{load_meta_pool, MetaPool},
    run_duel,
    smmcts::{dominated_actions, SelRule},
    Agent, BlindAgent, DuelSpec, RmConfig,
};
use nc2000_engine::{battle::SearchChoice, dex::Dex, state::Battle};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Default)]
struct Counts {
    decisions: AtomicUsize,
    exposed: AtomicUsize,
}
struct Measured {
    inner: BlindAgent,
    counts: Arc<Counts>,
}
impl Agent for Measured {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() > 1 && !matches!(choices[0], SearchChoice::Team(_)) {
            self.counts.decisions.fetch_add(1, Ordering::Relaxed);
            let masked = dominated_actions(b, dex, side);
            if !masked.is_empty() && masked.len() < choices.len() {
                self.counts.exposed.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.inner.choose(b, dex, side, choices)
    }
}
fn agent(
    seed: u64,
    prune: bool,
    iterations: u32,
    pool: Arc<MetaPool>,
    counts: Arc<Counts>,
) -> Box<dyn Agent> {
    let base = BlindAgent::new(
        RmConfig {
            iterations,
            c: 0.4,
            rule: SelRule::Ucb,
            ..Default::default()
        },
        pool,
        None,
        seed,
    );
    Box::new(Measured {
        inner: if prune {
            base.with_root_pruning()
        } else {
            base
        },
        counts,
    })
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let is_null = args.iter().any(|a| a == "--null");
    let dex = load_dex();
    let pool = Arc::new(load_meta_pool(
        &repo_root().join("data/belief-pool-v3/belief-pool.json"),
    ));
    let catalog = load_meta_pool(&repo_root().join("data/team-pool-v2/team-pool.json"));
    let teams: Vec<_> = catalog.teams.iter().map(|t| t.sets.clone()).collect();
    let a = Arc::new(Counts::default());
    let b = Arc::new(Counts::default());
    let iterations = if is_null { 3000 } else { 27000 };
    let result = run_duel(
        &dex,
        &teams,
        &|seed| agent(seed, !is_null, iterations, pool.clone(), a.clone()),
        &|seed| agent(seed, false, iterations, pool.clone(), b.clone()),
        DuelSpec {
            games: if is_null { 8 } else { 64 },
            base_seed: if is_null { 2026100701 } else { 2026100702 },
            threads: 4,
            max_turns: 500,
            progress: true,
            log_on: true,
            crn_agent_seeds: true,
        },
    );
    if is_null {
        assert!(result.pair_scores.iter().all(|&x| x == 0.5));
    }
    let count = |c: &Counts| json!({"decisions":c.decisions.load(Ordering::Relaxed),"exposed":c.exposed.load(Ordering::Relaxed)});
    println!(
        "{}",
        json!({"null_control":is_null,"iterations":iterations,"games":result.games,"pairs":result.pairs,"wins":result.wins,"losses":result.losses,"ties":result.ties,"caps":result.turn_caps,
        "score":result.score,"ci95_halfwidth":result.ci95,"pair_scores":result.pair_scores,"game_scores":result.game_scores,
        "seconds":result.secs,"mean_think_ms":[result.a_ms_per_move,result.b_ms_per_move],"exposure":[count(&a),count(&b)]})
    );
}
