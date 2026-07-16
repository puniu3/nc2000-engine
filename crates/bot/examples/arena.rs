//! Agent-vs-agent evaluation arena. Teams come from the golden fixture
//! corpus (120 validator-clean teams) or, with --pool meta[:LO-HI], from the
//! M8 meta pool. Games are seed-paired: each pairing is played twice with
//! sides swapped, same battle seed. Fully deterministic for a given --seed
//! regardless of --threads.
//!
//!   cargo run --release -p nc2000-bot --example arena -- \
//!       mcts:300 maxdamage --games 100 [--seed 1] [--threads N] [--max-turns 500] \
//!       [--pool fixtures|meta[:LO-HI]] [--tables data/preview-tables-v0]
//!
//! Agent specs:
//!   random | maxdamage
//!   mcts[:ITERS[:C[:EPS[:TURNS]]]]   M6 heavy playout (ε-greedy + truncated + eval)
//!   mcts5[:ITERS[:C]]                M5 baseline (uniform full rollouts, HP eval)
//!   rm[:ITERS[:PROBE[:THRESHOLD[:BUCKETS]]]]  M7 state-keyed MCTS + RM-solved mixed root
//!   skuct[:ITERS[:C[:BUCKETS]]]      M7 state-keyed argmax ablation (in-battle flagship)
//!   exploit:<inner>                  best-response probe vs a frozen <inner> policy
//!                                    (3-sample seed-marginal oracle, own budget = 3x inner's)
//!   baked:<inner> | bakedarg:<inner>       M8 baked preview (mixed sample / argmax),
//!                                          <inner> plays the battle; unknown matchup
//!                                          falls back to <inner>'s own preview search
//!   counter:<inner> | counterarg:<inner>   M8 counter-picking probe: best-responds at
//!                                          preview to the baked mixed/argmax policy

use std::sync::Arc;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{
    run_duel, Agent, BakedPreviewAgent, BrAgent, CounterPickAgent, DuelSpec, EvalWeights,
    MaxDamageAgent, MctsAgent, MctsConfig, Playout, PreviewMode, RandomAgent, RmAgent, RmConfig,
    TableSet,
};
use nc2000_engine::battle::PokemonSet;

#[derive(Clone, Debug)]
enum AgentSpec {
    Random,
    MaxDamage,
    Mcts { iterations: u32, c: f64, eps: f64, turns: u16 },
    Mcts5 { iterations: u32, c: f64 },
    Rm { iterations: u32, probe: f64, threshold: f64, buckets: i64 },
    SkUct { iterations: u32, c: f64, buckets: i64 },
    Exploit(Box<AgentSpec>),
    Baked { inner: Box<AgentSpec>, mode: PreviewMode },
    Counter { inner: Box<AgentSpec>, target: PreviewMode },
}

fn opt_num<T: std::str::FromStr>(parts: &[&str], i: usize, what: &str) -> Result<Option<T>, String> {
    parts
        .get(i)
        .map(|v| v.parse().map_err(|_| format!("bad {what}: {v}")))
        .transpose()
}

impl AgentSpec {
    fn parse(s: &str) -> Result<AgentSpec, String> {
        let parts: Vec<&str> = s.split(':').collect();
        match parts[0] {
            "random" => Ok(AgentSpec::Random),
            "maxdamage" => Ok(AgentSpec::MaxDamage),
            "mcts" => Ok(AgentSpec::Mcts {
                iterations: opt_num(&parts, 1, "iters")?.unwrap_or(1000),
                c: opt_num(&parts, 2, "c")?.unwrap_or(1.0),
                eps: opt_num(&parts, 3, "eps")?.unwrap_or(0.2),
                turns: opt_num(&parts, 4, "turns")?.unwrap_or(8),
            }),
            "mcts5" => Ok(AgentSpec::Mcts5 {
                iterations: opt_num(&parts, 1, "iters")?.unwrap_or(1000),
                c: opt_num(&parts, 2, "c")?.unwrap_or(1.0),
            }),
            "rm" => Ok(AgentSpec::Rm {
                iterations: opt_num(&parts, 1, "iters")?.unwrap_or(1000),
                probe: opt_num(&parts, 2, "probe")?.unwrap_or(0.25),
                threshold: opt_num(&parts, 3, "threshold")?.unwrap_or(0.5),
                buckets: opt_num(&parts, 4, "buckets")?.unwrap_or(16),
            }),
            "skuct" => Ok(AgentSpec::SkUct {
                iterations: opt_num(&parts, 1, "iters")?.unwrap_or(1000),
                c: opt_num(&parts, 2, "c")?.unwrap_or(1.0),
                buckets: opt_num(&parts, 3, "buckets")?.unwrap_or(16),
            }),
            "exploit" => {
                let inner = s.strip_prefix("exploit:").ok_or("exploit needs an inner spec")?;
                Ok(AgentSpec::Exploit(Box::new(AgentSpec::parse(inner)?)))
            }
            tag @ ("baked" | "bakedarg" | "counter" | "counterarg") => {
                let inner = s
                    .strip_prefix(tag)
                    .and_then(|r| r.strip_prefix(':'))
                    .ok_or_else(|| format!("{tag} needs an inner spec"))?;
                let inner = Box::new(AgentSpec::parse(inner)?);
                Ok(match tag {
                    "baked" => AgentSpec::Baked { inner, mode: PreviewMode::Mixed },
                    "bakedarg" => AgentSpec::Baked { inner, mode: PreviewMode::Argmax },
                    "counter" => AgentSpec::Counter { inner, target: PreviewMode::Mixed },
                    _ => AgentSpec::Counter { inner, target: PreviewMode::Argmax },
                })
            }
            other => Err(format!("unknown agent: {other}")),
        }
    }

    /// Search budget of this spec (exploit inherits its target's).
    fn iterations(&self) -> u32 {
        match self {
            AgentSpec::Mcts { iterations, .. }
            | AgentSpec::Mcts5 { iterations, .. }
            | AgentSpec::Rm { iterations, .. }
            | AgentSpec::SkUct { iterations, .. } => *iterations,
            AgentSpec::Exploit(inner) => inner.iterations(),
            AgentSpec::Baked { inner, .. } | AgentSpec::Counter { inner, .. } => {
                inner.iterations()
            }
            _ => 1000,
        }
    }

    fn needs_tables(&self) -> bool {
        match self {
            AgentSpec::Baked { .. } | AgentSpec::Counter { .. } => true,
            AgentSpec::Exploit(inner) => inner.needs_tables(),
            _ => false,
        }
    }

    fn build(&self, seed: u64, tables: Option<&Arc<TableSet>>) -> Box<dyn Agent> {
        match self {
            AgentSpec::Random => Box::new(RandomAgent::new(seed)),
            AgentSpec::MaxDamage => Box::new(MaxDamageAgent::new()),
            AgentSpec::Mcts { iterations, c, eps, turns } => Box::new(MctsAgent::new(
                MctsConfig {
                    iterations: *iterations,
                    c: *c,
                    playout: Playout::Heavy {
                        eps: *eps,
                        turns: *turns,
                        weights: EvalWeights::default(),
                    },
                    ..Default::default()
                },
                seed,
            )),
            AgentSpec::Mcts5 { iterations, c } => {
                Box::new(MctsAgent::new(MctsConfig::uniform(*iterations, *c), seed))
            }
            AgentSpec::Rm { iterations, probe, threshold, buckets } => Box::new(RmAgent::new(
                RmConfig {
                    iterations: *iterations,
                    probe: *probe,
                    threshold: *threshold,
                    hp_buckets: *buckets,
                    ..Default::default()
                },
                seed,
            )),
            AgentSpec::SkUct { iterations, c, buckets } => Box::new(RmAgent::new(
                RmConfig {
                    iterations: *iterations,
                    rule: SelRule::Ucb,
                    c: *c,
                    hp_buckets: *buckets,
                    ..Default::default()
                },
                seed,
            )),
            AgentSpec::Exploit(inner) => {
                let model = inner.build(seed ^ 0x517C_C1B7_2722_0A95, tables);
                let cfg =
                    MctsConfig { iterations: inner.iterations() * 3, ..Default::default() };
                Box::new(BrAgent::new(model, 3, cfg, seed))
            }
            AgentSpec::Baked { inner, mode } => Box::new(BakedPreviewAgent::new(
                tables.expect("baked agents need --tables").clone(),
                inner.build(seed ^ 0x243F_6A88_85A3_08D3, tables),
                *mode,
                seed,
            )),
            AgentSpec::Counter { inner, target } => Box::new(CounterPickAgent::new(
                tables.expect("counter agents need --tables").clone(),
                inner.build(seed ^ 0x243F_6A88_85A3_08D3, tables),
                *target,
            )),
        }
    }

    fn label(&self) -> String {
        match self {
            AgentSpec::Random => "random".into(),
            AgentSpec::MaxDamage => "maxdamage".into(),
            AgentSpec::Mcts { iterations, c, eps, turns } => {
                format!("mcts:{iterations}:{c}:{eps}:{turns}")
            }
            AgentSpec::Mcts5 { iterations, c } => format!("mcts5:{iterations}:{c}"),
            AgentSpec::Rm { iterations, probe, threshold, buckets } => {
                format!("rm:{iterations}:{probe}:{threshold}:{buckets}")
            }
            AgentSpec::SkUct { iterations, c, buckets } => {
                format!("skuct:{iterations}:{c}:{buckets}")
            }
            AgentSpec::Exploit(inner) => format!("exploit:{}", inner.label()),
            AgentSpec::Baked { inner, mode } => match mode {
                PreviewMode::Mixed => format!("baked:{}", inner.label()),
                PreviewMode::Argmax => format!("bakedarg:{}", inner.label()),
            },
            AgentSpec::Counter { inner, target } => match target {
                PreviewMode::Mixed => format!("counter:{}", inner.label()),
                PreviewMode::Argmax => format!("counterarg:{}", inner.label()),
            },
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: arena <agentA> <agentB> [--games N] [--seed S] [--threads T] [--max-turns M]");
        std::process::exit(2);
    }
    let spec_a = AgentSpec::parse(&args[0]).unwrap();
    let spec_b = AgentSpec::parse(&args[1]).unwrap();
    let games: usize = flag(&args, "--games").map(|v| v.parse().unwrap()).unwrap_or(100);
    let base_seed: u64 = flag(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(1);
    let max_turns: u16 = flag(&args, "--max-turns").map(|v| v.parse().unwrap()).unwrap_or(500);
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

    let dex = load_dex();
    let pool_spec = flag(&args, "--pool").unwrap_or_else(|| "fixtures".into());
    let teams = if let Some(range) = pool_spec.strip_prefix("meta") {
        let pool = load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"));
        let (lo, hi) = match range.strip_prefix(':') {
            None => (0, pool.teams.len() - 1),
            Some(r) => {
                let (lo, hi) = r.split_once('-').expect("--pool meta:LO-HI");
                (lo.parse().unwrap(), hi.parse::<usize>().unwrap().min(pool.teams.len() - 1))
            }
        };
        pool.teams[lo..=hi].iter().map(|t| t.sets.clone()).collect()
    } else {
        load_team_pool()
    };
    eprintln!("{} teams in pool ({pool_spec})", teams.len());

    let tables = (spec_a.needs_tables() || spec_b.needs_tables()).then(|| {
        let dir = flag(&args, "--tables")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| repo_root().join("data/preview-tables-v0"));
        let pool = load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"));
        let ts = TableSet::load(&dex, &pool, &dir);
        eprintln!("{} baked pair tables loaded from {}", ts.len(), dir.display());
        ts
    });

    let stats = run_duel(
        &dex,
        &teams,
        &|seed| spec_a.build(seed, tables.as_ref()),
        &|seed| spec_b.build(seed, tables.as_ref()),
        DuelSpec { games, base_seed, threads, max_turns, progress: true },
    );

    println!(
        "A={} vs B={}   {} games, seed {base_seed}, {threads} threads",
        spec_a.label(),
        spec_b.label(),
        stats.games
    );
    println!(
        "A: {}W {}L {}T   score {:.3} +/- {:.3} (95% CI)",
        stats.wins, stats.losses, stats.ties, stats.score, stats.ci95
    );
    println!(
        "avg turns {:.1}   {:.1}s total   {:.2} games/s   think ms/move A {:.1} B {:.1}",
        stats.avg_turns,
        stats.secs,
        stats.games as f64 / stats.secs,
        stats.a_ms_per_move,
        stats.b_ms_per_move
    );
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
