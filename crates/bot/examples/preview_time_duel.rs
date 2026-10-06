use conformance::load_dex;
use nc2000_bot::{
    preview::{load_meta_pool, MetaPool},
    run_duel,
    smmcts::SelRule,
    Agent, Belief, BlindAgent, BlindSearch, DuelSpec, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};
use serde_json::{json, Value};
use std::{
    io::BufRead,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
enum Budget {
    Iterations(u32),
    Wall(Duration),
}

struct Session {
    side: usize,
    turn: u16,
    observer: Observer,
    belief: Belief,
}

struct BudgetAgent {
    cfg: RmConfig,
    preview_aliases: bool,
    pool: Arc<MetaPool>,
    state: Option<Session>,
    rng: u64,
    budget: Budget,
    records: Option<Arc<Mutex<Vec<Value>>>>,
}

impl BudgetAgent {
    fn new(
        pool: Arc<MetaPool>,
        seed: u64,
        enabled: bool,
        budget: Budget,
        records: Option<Arc<Mutex<Vec<Value>>>>,
    ) -> Self {
        Self {
            cfg: RmConfig {
                c: 0.4,
                rule: SelRule::Ucb,
                ..Default::default()
            },
            preview_aliases: enabled,
            pool,
            state: None,
            rng: seed,
            budget,
            records,
        }
    }
}

impl Agent for BudgetAgent {
    fn name(&self) -> String {
        format!("budget-blind:preview-aliases={}", self.preview_aliases)
    }
    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let preview = matches!(choices[0], SearchChoice::Team(_));
        if self
            .state
            .as_ref()
            .is_none_or(|g| g.side != side || preview || battle.turn < g.turn)
        {
            let observer = Observer::new(battle, side);
            let belief = Belief::new(dex, &self.pool, &observer);
            self.state = Some(Session {
                side,
                turn: battle.turn,
                observer,
                belief,
            });
        }
        let g = self.state.as_mut().unwrap();
        g.turn = battle.turn;
        g.observer.observe(battle, dex);
        g.belief.sync(dex, &g.observer);
        if choices.len() == 1 {
            return choices[0];
        }
        let started = Instant::now();
        let mut search = BlindSearch::new(battle, dex, self.cfg.clone(), side, self.rng);
        assert_eq!(search.actions(), choices);
        let removed = if self.preview_aliases {
            search.prune_preview_aliases(dex)
        } else {
            0
        };
        let mut elapsed = started.elapsed();
        let mut last_batch = elapsed;
        match self.budget {
            Budget::Iterations(n) => {
                search.step(dex, &g.belief, &g.observer, n);
                elapsed = started.elapsed();
            }
            Budget::Wall(budget) => {
                while search.iterations() == 0 || elapsed < budget {
                    let before = elapsed;
                    search.step(dex, &g.belief, &g.observer, 64);
                    elapsed = started.elapsed();
                    last_batch = elapsed - before;
                }
            }
        }
        self.rng = search.search_rng_state();
        let chosen = search.best().unwrap();
        if let Some(records) = &self.records {
            records.lock().unwrap().push(json!({"preview":preview,"removed":removed,"iterations":search.iterations(),
                "elapsed_ns":elapsed.as_nanos() as u64,"last_batch_ns":last_batch.as_nanos() as u64}));
        }
        chosen
    }
}

fn identity(corpus: &str, output: &str, dex: &Dex, pool: Arc<MetaPool>) {
    let mut checkpoints = 0;
    let mut previews = 0;
    for line in std::io::BufReader::new(std::fs::File::open(corpus).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut battle =
            Battle::from_fixture(dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        let seed = 0x2026102600 ^ game["game"].as_u64().unwrap();
        let cfg = RmConfig {
            iterations: 128,
            c: 0.4,
            rule: SelRule::Ucb,
            ..Default::default()
        };
        let mut native = [0, 1].map(|_| BlindAgent::new(cfg.clone(), pool.clone(), None, seed));
        let mut baseline = [0, 1]
            .map(|_| BudgetAgent::new(pool.clone(), seed, false, Budget::Iterations(128), None));
        let mut candidate = [0, 1]
            .map(|_| BudgetAgent::new(pool.clone(), seed, true, Budget::Iterations(128), None));
        let mut references = [0, 1].map(|side| {
            let observer = Observer::new(&battle, side);
            let belief = Belief::new(dex, &pool, &observer);
            (observer, belief, seed)
        });
        for d in game["decisions"].as_array().unwrap() {
            assert_eq!(
                battle.state_key128().to_string(),
                d["key"].as_str().unwrap()
            );
            let choices = [0, 1].map(|side| battle.legal_choices(dex, side));
            for side in 0..2 {
                if choices[side].is_empty() {
                    continue;
                }
                assert_eq!(
                    native[side].choose(&battle, dex, side, &choices[side]),
                    baseline[side].choose(&battle, dex, side, &choices[side])
                );
                assert_eq!(native[side].search_rng_state(), baseline[side].rng);
                let (observer, belief, rng) = &mut references[side];
                observer.observe(&battle, dex);
                belief.sync(dex, observer);
                let expected = if choices[side].len() == 1 {
                    choices[side][0]
                } else {
                    let mut search = BlindSearch::new(&battle, dex, cfg.clone(), side, *rng);
                    search.prune_preview_aliases(dex);
                    search.step(dex, belief, observer, 128);
                    *rng = search.search_rng_state();
                    search.best().unwrap()
                };
                assert_eq!(
                    expected,
                    candidate[side].choose(&battle, dex, side, &choices[side])
                );
                assert_eq!(*rng, candidate[side].rng);
                checkpoints += 2;
                previews += 2 * usize::from(matches!(choices[side][0], SearchChoice::Team(_)));
            }
            battle
                .apply_choices(
                    dex,
                    [0, 1].map(|s| d["choices"][s].as_u64().map(|i| choices[s][i as usize])),
                )
                .unwrap();
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&json!({"checkpoints":checkpoints,"previews":previews,"iterations":128,
        "baseline_matches_native":true,"candidate_matches_direct_search":true,"choice_and_rng_identical":true})).unwrap()).unwrap();
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        args.len() == 3 || (args.len() == 4 && args[1] == "--identity"),
        "--identity corpus output | --null output | --trial output"
    );
    let dex = load_dex();
    let pool = Arc::new(load_meta_pool(std::path::Path::new(
        "data/belief-pool-v3/belief-pool.json",
    )));
    if args[1] == "--identity" {
        identity(&args[2], &args[3], &dex, pool);
        return;
    }
    assert!(matches!(args[1].as_str(), "--null" | "--trial"));
    let is_null = args[1] == "--null";
    let catalog = load_meta_pool(std::path::Path::new("data/team-pool-v2/team-pool.json"));
    let teams = catalog
        .teams
        .iter()
        .map(|t| t.sets.clone())
        .collect::<Vec<_>>();
    let records = [0, 1].map(|_| Arc::new(Mutex::new(Vec::new())));
    let budget = if is_null {
        Budget::Iterations(3000)
    } else {
        Budget::Wall(Duration::from_millis(500))
    };
    let agent = |seed, arm: usize| {
        Box::new(BudgetAgent::new(
            pool.clone(),
            seed,
            arm == 0 && !is_null,
            budget,
            Some(records[arm].clone()),
        )) as Box<dyn Agent>
    };
    let spec = DuelSpec {
        games: if is_null { 8 } else { 64 },
        base_seed: if is_null { 2026102600 } else { 2026102601 },
        threads: 4,
        max_turns: 500,
        progress: true,
        log_on: true,
        crn_agent_seeds: true,
    };
    let r = run_duel(
        &dex,
        &teams,
        &|seed| agent(seed, 0),
        &|seed| agent(seed, 1),
        spec,
    );
    if is_null {
        assert!(r.pair_scores.iter().all(|&s| s == 0.5));
    }
    let measured = records.each_ref().map(|r| r.lock().unwrap().clone());
    std::fs::write(&args[2],serde_json::to_vec(&json!({"null_control":is_null,"base_seed":spec.base_seed,"budget_ms":if is_null {0} else {500},
        "fixed_iterations":if is_null {3000} else {0},"batch":64,"games":r.games,"pairs":r.pairs,"wins":r.wins,"losses":r.losses,
        "ties":r.ties,"caps":r.turn_caps,"score":r.score,"ci95_halfwidth":r.ci95,"pair_scores":r.pair_scores,"game_scores":r.game_scores,
        "seconds":r.secs,"mean_think_ms":[r.a_ms_per_move,r.b_ms_per_move],"moves":[r.a_moves,r.b_moves],"decisions":measured})).unwrap()).unwrap();
}
