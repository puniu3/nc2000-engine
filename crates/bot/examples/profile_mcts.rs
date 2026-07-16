//! Sampling profile of the M6 heavy-playout MCTS workload (the post-truncation
//! profile that decides whether any deferred M4 perf idea gets pulled in).
//! Writes target/flamegraph-mcts.svg + a top-N self-time table. Run:
//!   cargo run --release -p nc2000-bot --example profile_mcts [iters]

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::{play_game, Agent, MctsAgent, MctsConfig, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;

fn main() {
    let iters: u32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(1000);
    let dex = load_dex();
    let teams = load_team_pool();

    let guard = pprof::ProfilerGuardBuilder::default().frequency(997).build().unwrap();

    let mut rng = SplitMix64::new(0xBADC_0DE);
    for g in 0..4u64 {
        let t1 = rng.below(teams.len());
        let t2 = rng.below(teams.len());
        let seed = rng.battle_seed();
        let mut a = MctsAgent::new(MctsConfig { iterations: iters, ..Default::default() }, g);
        let mut b = MctsAgent::new(MctsConfig { iterations: iters, ..Default::default() }, !g);
        let mut battle = Battle::from_fixture(&dex, &seed, &teams[t1], &teams[t2]).unwrap();
        battle.set_log_enabled(false);
        let (pa, pb): (&mut dyn Agent, &mut dyn Agent) = (&mut a, &mut b);
        play_game(&dex, &mut battle, &mut [pa, pb], 500).unwrap();
    }

    let report = guard.report().build().unwrap();
    let path = repo_root().join("target/flamegraph-mcts.svg");
    let f = std::fs::File::create(&path).unwrap();
    report.flamegraph(f).unwrap();
    println!("wrote {}", path.display());

    let mut self_n: std::collections::HashMap<String, isize> = Default::default();
    for (frames, n) in report.data.iter() {
        if let Some(top) = frames.frames.first().and_then(|f| f.first()) {
            *self_n.entry(top.name()).or_default() += *n;
        }
    }
    let mut v: Vec<_> = self_n.into_iter().collect();
    v.sort_by_key(|(_, n)| -*n);
    let total: isize = v.iter().map(|(_, n)| n).sum();
    println!("total samples: {total}");
    for (name, n) in v.iter().take(30) {
        println!("{:6.2}%  {}", *n as f64 / total as f64 * 100.0, name);
    }
}

fn load_team_pool() -> Vec<Vec<PokemonSet>> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut teams = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            let fx = Fixture::load(&path).unwrap();
            teams.push(fx.p1team);
            teams.push(fx.p2team);
        }
    }
    teams
}
