use super::*;
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    game: usize,
    decision: usize,
    pairs: Vec<[String; 2]>,
    trials: usize,
    seed: u64,
    iterations: u32,
    max_steps: usize,
    threads: usize,
}

fn score(b: &Battle) -> Option<f64> {
    b.outcome().map(|o| match o {Outcome::P1Win => 1.0, Outcome::Tie => 0.5, Outcome::P2Win => 0.0})
}

fn named(dex: &Dex, b: &mut Battle, side: usize, name: &str) -> SearchChoice {
    b.legal_choices(dex, side).into_iter()
        .find(|&a| nc2000_bot::player::action_input(dex, a) == name).expect("illegal named action")
}

pub(super) fn execute(args: &[String]) {
    assert_eq!(args.len(), 5, "corpus root-continuations config output-directory");
    let raw = std::fs::read(&args[1]).unwrap();
    let cfg_raw = std::fs::read(&args[3]).unwrap();
    let cfg: Config = serde_json::from_slice(&cfg_raw).unwrap();
    assert!(cfg.trials > 0 && cfg.iterations > 0 && cfg.max_steps > 0 && cfg.threads > 0);
    let dex = conformance::load_dex();
    let game = std::str::from_utf8(&raw).unwrap().lines().map(|s| serde_json::from_str::<Game>(s).unwrap())
        .find(|g| g.game == cfg.game).unwrap();
    let mut root = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
    root.set_log_enabled(true);
    let preview = root.clone();
    let mut observers = [0, 1].map(|s| Observer::new(&root, s));
    let mut beliefs = [0, 1].map(|s| Belief::pinned_from_battle(&root, &observers[s]));
    for step in &game.decisions[..cfg.decision] {
        assert_eq!(root.state_key128().to_string(), step.key);
        let legal = [0, 1].map(|s| root.legal_choices(&dex, s));
        root.apply_choices(&dex, [0, 1].map(|s| step.choices[s].map(|i| legal[s][i]))).unwrap();
        for side in 0..2 {observers[side].observe(&root, &dex); beliefs[side].sync(&dex, &observers[side]);}
    }
    assert_eq!(root.state_key128().to_string(), game.decisions[cfg.decision].key);
    assert!(root.outcome().is_none());
    let pairs: Vec<_> = cfg.pairs.iter().map(|pair| [0, 1].map(|s| Some(named(&dex, &mut root, s, &pair[s])))).collect();
    let directory = std::path::Path::new(&args[4]);
    std::fs::create_dir(directory).unwrap();
    let paths = ["crates/bot/examples/fixed_reply_counterpolicy/root_continuations.rs", "crates/bot/src/blind.rs",
        "crates/bot/src/belief.rs", "crates/bot/src/observe.rs", "crates/bot/src/smmcts.rs", "crates/bot/src/mcts.rs",
        "crates/bot/src/agent.rs", "crates/bot/src/eval.rs", "crates/bot/src/rng.rs", "crates/bot/src/root_policy.rs"];
    let evidence = json!({"schema":"nc2000-root-continuations-v1", "config":cfg, "config_sha256":sha(&cfg_raw),
        "state_key":root.state_key128().to_string(), "state":root.essence(&dex), "manifest":manifest(&dex, &raw),
        "search_source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "interpretation":"Fixed joint first actions followed by current pinned-team BlindSearch for both sides. Public observations and beliefs replay from preview and synchronize after every transition. Battle and search seeds are separate and paired across root cells. Scores come only from engine terminals; censored games retain null scores. This samples concrete continuation policies, not optimal continuations, blind information-set value, or an exploitability certificate."});
    std::fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    let jobs: Vec<_> = (0..cfg.trials).flat_map(|trial| (0..pairs.len()).map(move |pair| (trial, pair))).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(trial, pair)) = jobs.get(index) else {break};
                let battle_seed = seed(cfg.seed, trial);
                let mut b = root.clone(); b.reseed(battle_seed);
                let mut obs = observers.clone();
                let mut belief = [0, 1].map(|s| Belief::pinned_from_battle(&preview, &obs[s]));
                let started = Instant::now(); let mut decisions = Vec::new();
                while b.outcome().is_none() && decisions.len() < cfg.max_steps {
                    let step = decisions.len();
                    let search_seeds = [0, 1].map(|s| seed(battle_seed ^ 0xd1b54a32d192ed03, step * 2 + s));
                    let choices = if step == 0 {pairs[pair]} else {[0, 1].map(|s| {
                        let legal = b.legal_choices(&dex, s);
                        if legal.len() < 2 {return legal.first().copied();}
                        let mut search = BlindSearch::new(&b, &dex, RmConfig {c:0.4, ..Default::default()}, s, search_seeds[s]);
                        search.step(&dex, &belief[s], &obs[s], cfg.iterations);
                        search.best()
                    })};
                    assert!(choices.iter().any(Option::is_some));
                    decisions.push(json!({"key":b.state_key128().to_string(), "turn":b.turn, "search_seeds":search_seeds,
                        "choices":choices.map(|a|a.map(|a|nc2000_bot::player::action_input(&dex,a)))}));
                    b.apply_choices(&dex, choices).unwrap();
                    for s in 0..2 {obs[s].observe(&b, &dex); belief[s].sync(&dex, &obs[s]);}
                }
                let row = json!({"trial":trial,"pair":pair,"battle_seed":battle_seed,"score":score(&b),
                    "terminal_key":b.state_key128().to_string(),"turn":b.turn,"decisions":decisions,
                    "seconds":started.elapsed().as_secs_f64()});
                std::fs::write(directory.join(format!("t{trial:04}-p{pair}.json")),serde_json::to_vec(&row).unwrap()).unwrap();
                eprintln!("{}/{} trial={trial} pair={pair} score={} steps={} seconds={:.1}", index+1,jobs.len(),row["score"],
                    row["decisions"].as_array().unwrap().len(),row["seconds"].as_f64().unwrap());
            });
        }
    });
}

pub(super) fn verify(args: &[String]) {
    assert_eq!(args.len(), 4, "corpus verify-root-continuations output-directory");
    let raw = std::fs::read(&args[1]).unwrap();
    let directory = std::path::Path::new(&args[3]);
    let evidence: Value = serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(evidence["manifest"]["corpus_sha256"], sha(&raw));
    let cfg: Config = serde_json::from_value(evidence["config"].clone()).unwrap();
    let game = std::str::from_utf8(&raw).unwrap().lines().map(|s|serde_json::from_str::<Game>(s).unwrap())
        .find(|g|g.game==cfg.game).unwrap();
    let dex = conformance::load_dex(); let root = reconstruct(&dex,&game,cfg.decision);
    assert_eq!(evidence["state_key"],root.state_key128().to_string());
    let mut count = 0;
    for trial in 0..cfg.trials {for pair in 0..cfg.pairs.len() {
        let row: Value = serde_json::from_slice(&std::fs::read(directory.join(format!("t{trial:04}-p{pair}.json"))).unwrap()).unwrap();
        assert_eq!(row["trial"],trial);assert_eq!(row["pair"],pair);
        let battle_seed = seed(cfg.seed,trial); assert_eq!(row["battle_seed"],battle_seed);
        let mut b = root.clone(); b.reseed(battle_seed);
        let decisions = row["decisions"].as_array().unwrap();
        assert_eq!(decisions[0]["choices"],json!(cfg.pairs[pair]));
        for (step, record) in decisions.iter().enumerate() {
            assert_eq!(record["key"],b.state_key128().to_string());assert_eq!(record["turn"],b.turn);
            assert_eq!(record["search_seeds"],json!([0,1].map(|s|seed(battle_seed ^ 0xd1b54a32d192ed03,step*2+s))));
            let choices = [0,1].map(|s| {
                if let Some(name) = record["choices"][s].as_str() {Some(named(&dex,&mut b,s,name))}
                else {assert!(b.legal_choices(&dex,s).is_empty());None}
            });
            b.apply_choices(&dex,choices).unwrap();count+=1;
        }
        assert_eq!(row["score"],json!(score(&b)));assert_eq!(row["terminal_key"],b.state_key128().to_string());
        assert_eq!(row["turn"],b.turn);assert!(b.outcome().is_some() || decisions.len()==cfg.max_steps);
    }}
    println!("verified {} games, {count} legal transitions and terminal scores",cfg.trials*cfg.pairs.len());
}
