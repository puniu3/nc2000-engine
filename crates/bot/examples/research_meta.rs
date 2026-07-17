//! M11a metagame-research driver: (1+λ) mutation hill-climb over the legal
//! set space, fitness = seed-paired win rate vs a gauntlet (default: the 8
//! tournament-tier teams) at a configurable skuct budget. Resumable in the
//! bake's mold: one JSON checkpoint per lineage in --out; --resume continues
//! from the stored iteration + rng state and skips completed lineages.
//!
//!   cargo run --release -p nc2000-bot --example research_meta -- \
//!       --budget-profile smoke|full \
//!       (--seed-team IDX | --seed-file FILE | --random-team) [--weaken N] \
//!       [--lineage NAME] [--iters N] [--lambda N] [--eval-games N] \
//!       [--agent-iters N] [--gauntlet LO-HI] [--max-turns M] \
//!       [--seed S] [--threads T] [--out DIR] [--resume] [--force]
//!
//! Profiles (individual flags override):
//!   smoke  iters 6,  lambda 3, eval-games 2, agent-iters 100, max-turns 300
//!          (minutes — machinery tests while the full-pool bake owns the CPU)
//!   full   iters 40, lambda 6, eval-games 8, agent-iters 300, max-turns 500
//!          (the real run's parameters: 7 evals x 64 games per iteration at
//!          skuct:300 ~ 11 s/game one-thread, ~5-6 h per lineage on 10
//!          threads — see README M11a for the launch plan)
//!
//! Loop: per iteration propose λ validator-clean mutations of the current
//! parent (crates/bot/src/teamgen) and evaluate parent + children against
//! the gauntlet under COMMON RANDOM NUMBERS — one eval seed per iteration,
//! shared by parent and children, so the same battle seeds vs the same
//! gauntlet make the comparison paired and the parent's fitness is always
//! fresh (no winner's-curse ratchet on a lucky eval: fitness at these
//! budgets is noisy, and accepting against a stale record measurably
//! selects noise — caught by this driver's own verification pass during
//! M11a smoke testing). The best child is accepted only when it strictly
//! beats the parent's same-seed fitness. After the last iteration the seed
//! team and the final parent are re-evaluated at 3x games on a held-out
//! seed (the trajectory is optimistically biased; the verification pass is
//! not).
//!
//! Equilibrium certification of a discovered team (the M11 gate's second
//! half) is the bake tooling: bake_preview --candidate <file> (this
//! checkpoint format is accepted directly) — see that example's header.

use std::path::PathBuf;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::teamgen::{gauntlet_eval, team_key, to_sets, EvalCfg, TeamGen};
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::PokemonSet;
use serde_json::{json, Value};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    flag(args, name).map(|v| v.parse().expect(name)).unwrap_or(default)
}

struct Profile {
    iters: u32,
    lambda: u32,
    eval_games: u32,
    agent_iters: u32,
    max_turns: u16,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let profile_name = flag(&args, "--budget-profile").unwrap_or_else(|| "smoke".into());
    let prof = match profile_name.as_str() {
        "smoke" => Profile { iters: 6, lambda: 3, eval_games: 2, agent_iters: 100, max_turns: 300 },
        "full" => Profile { iters: 40, lambda: 6, eval_games: 8, agent_iters: 300, max_turns: 500 },
        other => panic!("unknown --budget-profile {other} (smoke|full)"),
    };
    let iters: u32 = num(&args, "--iters", prof.iters);
    let lambda: u32 = num(&args, "--lambda", prof.lambda);
    let eval_games: u32 = num(&args, "--eval-games", prof.eval_games);
    let agent_iters: u32 = num(&args, "--agent-iters", prof.agent_iters);
    let max_turns: u16 = num(&args, "--max-turns", prof.max_turns);
    let base_seed: u64 = num(&args, "--seed", 1);
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let out_dir = flag(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("data/research-v0"));
    let resume = args.iter().any(|a| a == "--resume");
    let force = args.iter().any(|a| a == "--force");
    let weaken: usize = num(&args, "--weaken", 0);
    let gauntlet_spec = flag(&args, "--gauntlet").unwrap_or_else(|| "0-7".into());

    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text =
        std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    // ---- gauntlet: pool teams LO..=HI (default 0-7 = the tournament tier)
    let (glo, ghi) = gauntlet_spec.split_once('-').expect("--gauntlet wants LO-HI");
    let (glo, ghi): (usize, usize) =
        (glo.parse().unwrap(), ghi.parse::<usize>().unwrap().min(gen.teams().len() - 1));
    let gauntlet: Vec<Vec<PokemonSet>> = (glo..=ghi)
        .map(|i| to_sets(&gen.canonize(&dex, &gen.team_json(i)).unwrap()).unwrap())
        .collect();

    // ---- seed team + lineage name
    let mut seed_rng = SplitMix64::new(base_seed ^ 0x5EED_7EA3);
    let (mut lineage, seed_team): (String, Vec<Value>) =
        if let Some(idx) = flag(&args, "--seed-team") {
            let i: usize = idx.parse().expect("--seed-team wants a pool index");
            (format!("pool-{i:02}"), gen.canonize(&dex, &gen.team_json(i)).unwrap())
        } else if let Some(path) = flag(&args, "--seed-file") {
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let team = extract_team(&serde_json::from_str(&text).expect("seed file JSON"))
                .unwrap_or_else(|| panic!("{path}: no team found (array | sets | best.team)"));
            let team = gen
                .canonize(&dex, &team)
                .unwrap_or_else(|| panic!("{path}: team does not validate"));
            let stem = PathBuf::from(&path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "file".into());
            (format!("file-{stem}"), team)
        } else if args.iter().any(|a| a == "--random-team") {
            let team = gen
                .random_team_valid(&dex, &mut seed_rng, 200)
                .expect("failed to draw a random legal team");
            (format!("rand-{base_seed}"), team)
        } else {
            eprintln!("need --seed-team IDX | --seed-file FILE | --random-team");
            std::process::exit(2);
        };
    let seed_team = if weaken > 0 {
        lineage = format!("{lineage}-weak{weaken}");
        gen.weaken(&dex, &seed_team, &mut seed_rng, weaken).expect("weaken failed")
    } else {
        seed_team
    };
    if let Some(name) = flag(&args, "--lineage") {
        lineage = name;
    }

    std::fs::create_dir_all(&out_dir).unwrap();
    let ck_path = out_dir.join(format!("lineage-{lineage}.json"));

    // ---- checkpoint: fresh, resume, or refuse to clobber
    let mut ck: Value = if ck_path.exists() && !force {
        if !resume {
            eprintln!(
                "{} exists — use --resume to continue or --force to restart",
                ck_path.display()
            );
            std::process::exit(2);
        }
        let ck: Value =
            serde_json::from_str(&std::fs::read_to_string(&ck_path).unwrap()).unwrap();
        let stored = &ck["cfg"];
        let current = cfg_json(iters, lambda, eval_games, agent_iters, max_turns, glo, ghi, base_seed);
        if *stored != current {
            eprintln!(
                "warning: resuming with a different cfg (fitness comparability across the change is on you)\n  stored:  {stored}\n  current: {current}"
            );
        }
        eprintln!(
            "lineage {lineage}: resuming at iter {}/{iters}",
            ck["iters_done"].as_u64().unwrap_or(0)
        );
        ck
    } else {
        json!({
            "lineage": lineage,
            "profile": profile_name,
            "cfg": cfg_json(iters, lambda, eval_games, agent_iters, max_turns, glo, ghi, base_seed),
            "seed_team": seed_team,
            "seed_fitness": Value::Null,
            "parent": seed_team,
            "parent_fitness": Value::Null,
            "iters_done": 0,
            "rng": base_seed ^ 0x11EA_6E00,
            "history": [],
        })
    };

    let eval_cfg = |seed: u64, games: u32| EvalCfg {
        games_per_opponent: games,
        agent_iters,
        max_turns,
        threads,
        seed,
    };
    let evaluate = |team: &[Value], seed: u64, games: u32| -> f64 {
        let sets = to_sets(team).unwrap();
        gauntlet_eval(&dex, &sets, &gauntlet, &eval_cfg(seed, games)).score
    };

    // ---- iter 0: seed-team fitness
    if ck["parent_fitness"].is_null() {
        let t0 = std::time::Instant::now();
        let f = evaluate(ck["seed_team"].as_array().unwrap(), base_seed ^ 0x0BA5_E11E, eval_games);
        eprintln!(
            "lineage {lineage} iter 0: seed fitness {f:.3} ({} games vs {} teams, {:.0}s)",
            (eval_games + eval_games % 2) as usize * gauntlet.len(),
            gauntlet.len(),
            t0.elapsed().as_secs_f64()
        );
        ck["seed_fitness"] = json!(f);
        ck["parent_fitness"] = json!(f);
        ck["history"].as_array_mut().unwrap().push(json!({"iter": 0, "kind": "seed", "fitness": f}));
        write_checkpoint(&ck_path, &ck);
    }

    // ---- (1+λ) loop
    let mut rng = SplitMix64(ck["rng"].as_u64().unwrap());
    let mut done = ck["iters_done"].as_u64().unwrap() as u32;
    if done >= iters {
        eprintln!("lineage {lineage}: complete ({done}/{iters} iters), skipping");
    }
    while done < iters {
        let it = done + 1;
        let parent: Vec<Value> = ck["parent"].as_array().unwrap().clone();
        // Common random numbers: ONE eval seed per iteration, shared by the
        // parent re-eval and every child — paired comparison on identical
        // battle seeds vs the same gauntlet.
        let eval_seed = base_seed ^ (it as u64).wrapping_mul(0xA24B_AED4_963E_E407);
        let mut children = Vec::new();
        for k in 0..lambda as usize {
            match gen.propose_valid(&dex, &parent, &mut rng, 100) {
                Some(p) => children.push((k, p)),
                None => eprintln!("lineage {lineage} iter {it}: child {k}: no valid proposal"),
            }
        }
        let t0 = std::time::Instant::now();
        let parent_fit = evaluate(&parent, eval_seed, eval_games);
        eprintln!(
            "lineage {lineage} iter {it}: parent fitness {parent_fit:.3} ({:.0}s)",
            t0.elapsed().as_secs_f64()
        );
        ck["history"]
            .as_array_mut()
            .unwrap()
            .push(json!({"iter": it, "kind": "parent", "fitness": parent_fit}));
        let mut best: Option<(f64, usize)> = None; // (fitness, child idx)
        for (k, p) in &children {
            let t0 = std::time::Instant::now();
            let f = evaluate(&p.team, eval_seed, eval_games);
            eprintln!(
                "lineage {lineage} iter {it}: child {k} {}@{} fitness {f:.3} ({:.0}s)",
                p.op.name(),
                p.slot,
                t0.elapsed().as_secs_f64()
            );
            ck["history"].as_array_mut().unwrap().push(json!({
                "iter": it, "child": k, "op": p.op.name(), "slot": p.slot,
                "fitness": f, "accepted": false,
            }));
            if best.as_ref().is_none_or(|(bf, _)| f > *bf) {
                best = Some((f, *k));
            }
        }
        if let Some((f, k)) = best {
            if f > parent_fit {
                let team = &children.iter().find(|(ck_, _)| *ck_ == k).unwrap().1.team;
                ck["parent"] = json!(team);
                ck["parent_fitness"] = json!(f);
                let hist = ck["history"].as_array_mut().unwrap();
                let n = hist.len() - children.len();
                for e in hist[n..].iter_mut() {
                    if e["child"] == json!(k) {
                        e["accepted"] = json!(true);
                    }
                }
                eprintln!(
                    "lineage {lineage} iter {it}: ACCEPT child {k} ({f:.3} > parent {parent_fit:.3}, same seed)"
                );
            } else {
                ck["parent_fitness"] = json!(parent_fit);
                eprintln!(
                    "lineage {lineage} iter {it}: reject (best child {f:.3} <= parent {parent_fit:.3}, same seed)"
                );
            }
        }
        done = it;
        ck["iters_done"] = json!(done);
        ck["rng"] = json!(rng.0);
        write_checkpoint(&ck_path, &ck);
    }

    // ---- held-out verification at 3x games on a shifted seed
    if ck["verify"].is_null() {
        let vgames = eval_games * 3;
        let vseed = base_seed ^ 0x7E51_F1ED;
        let t0 = std::time::Instant::now();
        let seed_f = evaluate(ck["seed_team"].as_array().unwrap(), vseed, vgames);
        let best_f = if team_key(ck["parent"].as_array().unwrap())
            == team_key(ck["seed_team"].as_array().unwrap())
        {
            seed_f
        } else {
            evaluate(ck["parent"].as_array().unwrap(), vseed, vgames)
        };
        ck["verify"] = json!({
            "games_per_opponent": vgames, "seed": vseed,
            "seed_fitness": seed_f, "best_fitness": best_f,
        });
        write_checkpoint(&ck_path, &ck);
        eprintln!(
            "lineage {lineage}: verification ({} games/opponent, {:.0}s)",
            vgames,
            t0.elapsed().as_secs_f64()
        );
    }

    // ---- summary
    let traj: Vec<String> = std::iter::once(format!(
        "{:.3}",
        ck["seed_fitness"].as_f64().unwrap()
    ))
    .chain(ck["history"].as_array().unwrap().iter().filter_map(|e| {
        (e["accepted"] == json!(true))
            .then(|| format!("{:.3} (it {})", e["fitness"].as_f64().unwrap(), e["iter"]))
    }))
    .collect();
    println!("lineage {lineage}: trajectory {}", traj.join(" -> "));
    println!(
        "lineage {lineage}: verification seed {:.3} -> best {:.3} ({} games/opponent, seed-shifted)",
        ck["verify"]["seed_fitness"].as_f64().unwrap(),
        ck["verify"]["best_fitness"].as_f64().unwrap(),
        ck["verify"]["games_per_opponent"].as_u64().unwrap(),
    );
    println!("checkpoint: {}", ck_path.display());
}

fn cfg_json(
    iters: u32,
    lambda: u32,
    eval_games: u32,
    agent_iters: u32,
    max_turns: u16,
    glo: usize,
    ghi: usize,
    seed: u64,
) -> Value {
    json!({
        "iters": iters, "lambda": lambda, "eval_games": eval_games,
        "agent_iters": agent_iters, "max_turns": max_turns,
        "gauntlet": [glo, ghi], "seed": seed,
    })
}

/// Accepts a bare set array, {"sets": [...]}, or a research checkpoint
/// (whose best team is "parent"; "best.team" also accepted for
/// forward-compat with external tooling).
fn extract_team(v: &Value) -> Option<Vec<Value>> {
    if let Some(a) = v.as_array() {
        return Some(a.clone());
    }
    for path in [&v["sets"], &v["parent"], &v["best"]["team"]] {
        if let Some(a) = path.as_array() {
            return Some(a.clone());
        }
    }
    None
}

/// Atomic checkpoint write (tmp + rename), the bake's resume contract.
fn write_checkpoint(path: &std::path::Path, ck: &Value) {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(ck).unwrap()).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}
