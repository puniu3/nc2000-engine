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
//!   blind[:ITERS[:C[:BUCKETS]]]      M10b imperfect-info skuct: sees only public info +
//!                                    the meta-pool prior; per-iteration belief
//!                                    determinization; baked-table preview when the
//!                                    opponent's pool identity resolves publicly
//!                                    (battles run log-ON for its observer)
//!   open[:ITERS[:C[:BUCKETS]]]       M14 open-team-sheet agent (the M12 product
//!                                    policy): the blind machinery with the opponent's
//!                                    TRUE sets pinned as a singleton belief — only
//!                                    picks stay hidden; preview by public-signature
//!                                    table lookup, else pinned live search
//!   exploit:<inner>                  best-response probe vs a frozen <inner> policy
//!                                    (3-sample seed-marginal oracle, own budget = 3x inner's)
//!   baked:<inner> | bakedarg:<inner>       M8 baked preview (mixed sample / argmax),
//!                                          <inner> plays the battle; unknown matchup
//!                                          falls back to <inner>'s own preview search
//!   counter:<inner> | counterarg:<inner>   M8 counter-picking probe: best-responds at
//!                                          preview to the baked mixed/argmax policy

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{
    run_duel, Agent, BakedPreviewAgent, BlindAgent, BrAgent, CounterPickAgent, DuelSpec,
    EvalWeights, MaxDamageAgent, MctsAgent, MctsConfig, OpenAgent, Playout, PreviewMode,
    RandomAgent, RmAgent, RmConfig, TableSet,
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
    /// skuct with the parked M16c rollout upgrades ON (A/B research arm).
    SkUctNs { iterations: u32, c: f64, buckets: i64 },
    Blind { iterations: u32, c: f64, buckets: i64 },
    Open { iterations: u32, c: f64, buckets: i64 },
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
            "skuctm16c" => Ok(AgentSpec::SkUctNs {
                iterations: opt_num(&parts, 1, "iters")?.unwrap_or(1000),
                c: opt_num(&parts, 2, "c")?.unwrap_or(1.0),
                buckets: opt_num(&parts, 3, "buckets")?.unwrap_or(16),
            }),
            "skuct" => Ok(AgentSpec::SkUct {
                iterations: opt_num(&parts, 1, "iters")?.unwrap_or(1000),
                c: opt_num(&parts, 2, "c")?.unwrap_or(1.0),
                buckets: opt_num(&parts, 3, "buckets")?.unwrap_or(16),
            }),
            "blind" => Ok(AgentSpec::Blind {
                iterations: opt_num(&parts, 1, "iters")?.unwrap_or(1000),
                c: opt_num(&parts, 2, "c")?.unwrap_or(1.0),
                buckets: opt_num(&parts, 3, "buckets")?.unwrap_or(16),
            }),
            "open" => Ok(AgentSpec::Open {
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
            | AgentSpec::SkUct { iterations, .. }
            | AgentSpec::SkUctNs { iterations, .. }
            | AgentSpec::Blind { iterations, .. }
            | AgentSpec::Open { iterations, .. } => *iterations,
            AgentSpec::Exploit(inner) => inner.iterations(),
            AgentSpec::Baked { inner, .. } | AgentSpec::Counter { inner, .. } => {
                inner.iterations()
            }
            _ => 1000,
        }
    }

    fn needs_tables(&self) -> bool {
        match self {
            AgentSpec::Baked { .. }
            | AgentSpec::Counter { .. }
            | AgentSpec::Blind { .. }
            | AgentSpec::Open { .. } => true,
            AgentSpec::Exploit(inner) => inner.needs_tables(),
            _ => false,
        }
    }

    /// Blind agents need the meta pool as their belief prior; blind AND
    /// open agents want log-on outer battles for their observer's
    /// trace-free reveal channel (product parity: the web worker's mirror
    /// battle runs log-ON).
    fn is_blind(&self) -> bool {
        match self {
            AgentSpec::Blind { .. } | AgentSpec::Open { .. } => true,
            AgentSpec::Exploit(inner)
            | AgentSpec::Baked { inner, .. }
            | AgentSpec::Counter { inner, .. } => inner.is_blind(),
            _ => false,
        }
    }

    fn build(
        &self,
        seed: u64,
        tables: Option<&Arc<TableSet>>,
        pool: Option<&Arc<MetaPool>>,
    ) -> Box<dyn Agent> {
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
            AgentSpec::SkUctNs { iterations, c, buckets } => Box::new(RmAgent::new(
                RmConfig {
                    iterations: *iterations,
                    rule: SelRule::Ucb,
                    c: *c,
                    hp_buckets: *buckets,
                    rollout_m16c: true,
                    ..Default::default()
                },
                seed,
            )),
            AgentSpec::Blind { iterations, c, buckets } => Box::new(BlindAgent::new(
                RmConfig {
                    iterations: *iterations,
                    rule: SelRule::Ucb,
                    c: *c,
                    hp_buckets: *buckets,
                    ..Default::default()
                },
                pool.expect("blind agents need the meta pool").clone(),
                tables.cloned(),
                seed,
            )),
            AgentSpec::Open { iterations, c, buckets } => Box::new(OpenAgent::new(
                RmConfig {
                    iterations: *iterations,
                    rule: SelRule::Ucb,
                    c: *c,
                    hp_buckets: *buckets,
                    ..Default::default()
                },
                tables.cloned(),
                seed,
            )),
            AgentSpec::Exploit(inner) => {
                let model = inner.build(seed ^ 0x517C_C1B7_2722_0A95, tables, pool);
                let cfg =
                    MctsConfig { iterations: inner.iterations() * 3, ..Default::default() };
                Box::new(BrAgent::new(model, 3, cfg, seed))
            }
            AgentSpec::Baked { inner, mode } => Box::new(BakedPreviewAgent::new(
                tables.expect("baked agents need --tables").clone(),
                inner.build(seed ^ 0x243F_6A88_85A3_08D3, tables, pool),
                *mode,
                seed,
            )),
            AgentSpec::Counter { inner, target } => Box::new(CounterPickAgent::new(
                tables.expect("counter agents need --tables").clone(),
                inner.build(seed ^ 0x243F_6A88_85A3_08D3, tables, pool),
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
            AgentSpec::SkUctNs { iterations, c, buckets } => {
                format!("skuctm16c:{iterations}:{c}:{buckets}")
            }
            AgentSpec::Blind { iterations, c, buckets } => {
                format!("blind:{iterations}:{c}:{buckets}")
            }
            AgentSpec::Open { iterations, c, buckets } => {
                format!("open:{iterations}:{c}:{buckets}")
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

fn write_jsonl(path: &Path, row: &serde_json::Value) {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut out = BufWriter::new(File::create(path).unwrap());
    serde_json::to_writer(&mut out, row).unwrap();
    writeln!(out).unwrap();
}

fn fnv1a64_update(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn fingerprint_part(hash: u64, bytes: &[u8]) -> u64 {
    let hash = fnv1a64_update(hash, &(bytes.len() as u64).to_le_bytes());
    fnv1a64_update(hash, bytes)
}

fn content_fingerprint<'a>(tag: &str, parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    let mut count = 0usize;
    for part in parts {
        hash = fingerprint_part(hash, part);
        count += 1;
    }
    format!("fnv1a64:{hash:016x}:{tag}:{count}parts")
}

/// Identity of the exact executable being measured, not merely a source
/// revision which could hide local edits or different compiler settings.
fn build_fingerprint() -> String {
    let executable = std::env::current_exe().expect("resolve arena executable");
    let bytes = std::fs::read(&executable)
        .unwrap_or_else(|e| panic!("read arena executable {}: {e}", executable.display()));
    content_fingerprint("arena-build-v1", [bytes.as_slice()])
}

/// Bind both the scheduled teams and (when present) the complete meta prior.
/// Blind search samples the latter even when `--pool meta:LO-HI` schedules a
/// strict subset, so both are behaviorally relevant inputs.
fn pool_fingerprint(
    pool_spec: &str,
    teams: &[Vec<PokemonSet>],
    meta_path: Option<&Path>,
) -> String {
    // PokemonSet is intentionally Deserialize-only; its derived Debug form is
    // deterministic (the only maps are BTreeMaps) and binds every field.
    let mut teams_bytes = Vec::new();
    for team in teams {
        for set in team {
            teams_bytes.extend_from_slice(format!("{set:?}\n").as_bytes());
        }
        teams_bytes.extend_from_slice(b"--team--\n");
    }
    let meta_bytes = meta_path.map(|path| {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("read meta pool {}: {e}", path.display()))
    });
    let mut parts: Vec<&[u8]> = vec![pool_spec.as_bytes(), teams_bytes.as_slice()];
    if let Some(bytes) = meta_bytes.as_ref() {
        parts.push(bytes.as_slice());
    }
    content_fingerprint("arena-pool-v1", parts)
}

/// Hash exactly the immediate JSON inputs considered by `TableSet::load`.
/// Basenames are bound but the directory path is not, so copied CX payloads
/// retain identity while additions, removals, renames, and edits do not.
fn tables_fingerprint(dir: Option<&Path>) -> String {
    let Some(dir) = dir else {
        return content_fingerprint("arena-tables-v1", std::iter::empty());
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("json"))
                .collect()
        })
        .unwrap_or_default();
    paths.sort();
    let mut owned = Vec::with_capacity(paths.len() * 2);
    for path in paths {
        owned.push(path.file_name().unwrap_or_default().to_string_lossy().as_bytes().to_vec());
        owned.push(
            std::fs::read(&path)
                .unwrap_or_else(|e| panic!("read baked table {}: {e}", path.display())),
        );
    }
    content_fingerprint("arena-tables-v1", owned.iter().map(Vec::as_slice))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: arena <agentA> <agentB> [--games N] [--seed S] [--threads T] [--max-turns M] [--jsonl FILE]");
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

    let root = repo_root();
    let dex_path = root.join("data/gen2stadium2.json");
    let dex = load_dex();
    let pool_spec = flag(&args, "--pool").unwrap_or_else(|| "fixtures".into());
    let is_blind = spec_a.is_blind() || spec_b.is_blind();
    let needs_meta =
        pool_spec.starts_with("meta") || spec_a.needs_tables() || spec_b.needs_tables();
    let meta_path = root.join("data/meta-pool-v0/meta-pool.json");
    let meta = needs_meta.then(|| Arc::new(load_meta_pool(&meta_path)));
    let teams = if let Some(range) = pool_spec.strip_prefix("meta") {
        let pool = meta.as_ref().unwrap();
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

    let tables_dir = (spec_a.needs_tables() || spec_b.needs_tables()).then(|| {
        flag(&args, "--tables")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| root.join("data/preview-tables-v0"))
    });
    let tables = tables_dir.as_ref().map(|dir| {
        let ts = TableSet::load(&dex, meta.as_ref().unwrap(), &dir);
        eprintln!("{} baked pair tables loaded from {}", ts.len(), dir.display());
        ts
    });

    let stats = run_duel(
        &dex,
        &teams,
        &|seed| spec_a.build(seed, tables.as_ref(), meta.as_ref()),
        &|seed| spec_b.build(seed, tables.as_ref(), meta.as_ref()),
        DuelSpec { games, base_seed, threads, max_turns, progress: true, log_on: is_blind },
    );

    // Hash only after all measured work. Reading the executable and large
    // data inputs immediately before the duel would warm the filesystem page
    // cache and bias timing relative to shards launched on a cold worker.
    let fingerprints = serde_json::json!({
        "build": build_fingerprint(),
        "dex": content_fingerprint(
            "arena-dex-v1",
            [std::fs::read(&dex_path)
                .unwrap_or_else(|e| panic!("read dex {}: {e}", dex_path.display()))
                .as_slice()],
        ),
        "pool": pool_fingerprint(&pool_spec, &teams, needs_meta.then_some(meta_path.as_path())),
        "tables": tables_fingerprint(tables_dir.as_deref()),
    });

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
    println!(
        "turn caps {}   think p95/p99 ms A {:.1}/{:.1} B {:.1}/{:.1}",
        stats.turn_caps, stats.a_p95_ms, stats.a_p99_ms, stats.b_p95_ms, stats.b_p99_ms
    );

    if let Some(path) = flag(&args, "--jsonl") {
        let ci95 = stats.ci95.is_finite().then_some(stats.ci95);
        let row = serde_json::json!({
            "schema": "nc2000-arena-v1",
            "agent_a": spec_a.label(),
            "agent_b": spec_b.label(),
            "config": {
                "requested_games": games,
                "base_seed": base_seed,
                "threads": threads,
                "max_turns": max_turns,
                "agent_a_iterations": spec_a.iterations(),
                "agent_b_iterations": spec_b.iterations(),
                "pool": pool_spec,
                "teams": teams.len(),
                "baked_tables": tables.as_ref().map_or(0, |t| t.len()),
                "log_on": is_blind,
            },
            "result": {
                "games": stats.games,
                "pairs": stats.pairs,
                "wins": stats.wins,
                "losses": stats.losses,
                "ties": stats.ties,
                "turn_caps": stats.turn_caps,
                "invalid_games": 0,
                "score": stats.score,
                "ci95": ci95,
                "ci_unit": "side_swap_pair",
                "pair_scores": &stats.pair_scores,
                "turns_sum": stats.turns_sum,
                "avg_turns": stats.avg_turns,
                "wall_secs": stats.secs,
            },
            "timing": {
                "unit": "choose_call",
                "a": {
                    "moves": stats.a_moves,
                    "total_ns": stats.a_total_ns,
                    "mean_ms": stats.a_ms_per_move,
                    "p95_ms": stats.a_p95_ms,
                    "p99_ms": stats.a_p99_ms,
                    "samples_ns": &stats.a_samples_ns,
                },
                "b": {
                    "moves": stats.b_moves,
                    "total_ns": stats.b_total_ns,
                    "mean_ms": stats.b_ms_per_move,
                    "p95_ms": stats.b_p95_ms,
                    "p99_ms": stats.b_p99_ms,
                    "samples_ns": &stats.b_samples_ns,
                },
            },
            "fingerprints": fingerprints,
        });
        write_jsonl(Path::new(&path), &row);
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
