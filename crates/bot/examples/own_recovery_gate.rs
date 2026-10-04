use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    preview::{load_meta_pool, MetaPool},
    run_duel,
    smmcts::{dominated_actions_with, MaskRules, SelRule},
    Agent, BlindAgent, DuelSpec, RmConfig,
};
use nc2000_engine::{
    battle::{Outcome, SearchChoice},
    dex::Dex,
    prng::{BattleRng, Prng},
    state::Battle,
};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
#[path = "own_recovery_support/mod.rs"]
mod fixture;

fn config(enabled: bool, iters: u32) -> RmConfig {
    RmConfig {
        iterations: iters,
        c: 0.4,
        rule: SelRule::Ucb,
        mask_rules: MaskRules {
            own_recovery_wait: enabled,
            ..Default::default()
        },
        ..Default::default()
    }
}
#[derive(Default)]
struct Counts {
    decisions: AtomicUsize,
    exposed: AtomicUsize,
    chosen: AtomicUsize,
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
        let old = dominated_actions_with(b, dex, side, config(false, 0).mask_rules);
        let new = dominated_actions_with(b, dex, side, config(true, 0).mask_rules);
        self.counts.decisions.fetch_add(1, Ordering::Relaxed);
        if old != new {
            self.counts.exposed.fetch_add(1, Ordering::Relaxed);
        }
        let pick = self.inner.choose(b, dex, side, choices);
        if old.iter().any(|(a, _)| *a == pick) && !new.iter().any(|(a, _)| *a == pick) {
            self.counts.chosen.fetch_add(1, Ordering::Relaxed);
        }
        pick
    }
}
fn measured(
    enabled: bool,
    iters: u32,
    seed: u64,
    pool: Arc<MetaPool>,
    counts: Arc<Counts>,
) -> Box<dyn Agent> {
    Box::new(Measured {
        inner: BlindAgent::new(config(enabled, iters), pool, None, seed),
        counts,
    })
}
fn count(c: &Counts) -> serde_json::Value {
    json!({"decisions":c.decisions.load(Ordering::Relaxed),"mask_exposed":c.exposed.load(Ordering::Relaxed),"newly_allowed_chosen":c.chosen.load(Ordering::Relaxed)})
}
fn main() {
    let dex = load_dex();
    let pool = Arc::new(load_meta_pool(
        &repo_root().join("data/belief-pool-v3/belief-pool.json"),
    ));
    if std::env::args().any(|s| s == "--targeted") {
        let (_, b, _) = fixture::position(&dex);
        let rows = Mutex::new(Vec::new());
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(||loop {
                    let i=next.fetch_add(1,Ordering::Relaxed);
                    if i>=32 {break;}
                    let seed=95001+i as u64;
                    for enabled in [false,true] {
                        let mut play=b.clone();
                        play.prng=BattleRng::seeded(Prng::new(seed));
                        let counts=Arc::new(Counts::default());
                        let mut bot=measured(enabled,27000,seed,pool.clone(),counts.clone());
                        let mut trace=Vec::new();
                        for _ in 0..100 {
                            let legal=play.legal_choices(&dex,0);
                            let a=bot.choose(&play,&dex,0,&legal);
                            let legal=play.legal_choices(&dex,1);
                            let counter=SearchChoice::Move(dex.moves.id("counter").unwrap());
                            let reply=if legal.contains(&counter){counter}else{legal[0]};
                            trace.push(a.to_input(&dex));
                            play.apply_choices(&dex,[Some(a),Some(reply)]).unwrap();
                            if play.outcome().is_some(){break;}
                        }
                        let score=match play.outcome(){Some(Outcome::P1Win)=>Some(1.0),Some(Outcome::P2Win)=>Some(0.0),Some(Outcome::Tie)=>Some(0.5),None=>None};
                        rows.lock().unwrap().push(json!({"seed":seed,"enabled":enabled,"score":score,"turns":trace.len(),"choices":trace,"exposure":count(&counts)}));
                    }
                    eprintln!("targeted pair {}/32",i+1);
                });
            }
        });
        let mut rows = rows.into_inner().unwrap();
        rows.sort_by_key(|r| (r["seed"].as_u64().unwrap(), r["enabled"].as_bool().unwrap()));
        for r in rows {
            println!("{r}");
        }
        return;
    }
    let is_null = std::env::args().any(|s| s == "--null");
    let catalog = load_meta_pool(&repo_root().join("data/team-pool-v2/team-pool.json"));
    let teams: Vec<_> = catalog.teams.iter().map(|t| t.sets.clone()).collect();
    let a = Arc::new(Counts::default());
    let b = Arc::new(Counts::default());
    let iters = if is_null { 3000 } else { 27000 };
    let result = run_duel(
        &dex,
        &teams,
        &|seed| measured(!is_null, iters, seed, pool.clone(), a.clone()),
        &|seed| measured(false, iters, seed, pool.clone(), b.clone()),
        DuelSpec {
            games: if is_null { 16 } else { 64 },
            base_seed: if is_null { 96001 } else { 97001 },
            threads: 8,
            max_turns: 500,
            progress: true,
            log_on: true,
            crn_agent_seeds: true,
        },
    );
    if is_null {
        assert!(result.pair_scores.iter().all(|&s| s == 0.5));
    }
    println!(
        "{}",
        json!({"null_control":is_null,"iterations":iters,"teams":teams.len(),"games":result.games,"score":result.score,"ci95_halfwidth":result.ci95,"pair_scores":result.pair_scores,"wins":result.wins,"losses":result.losses,"ties":result.ties,"caps":result.turn_caps,"seconds":result.secs,"a":count(&a),"b":count(&b)})
    );
}
