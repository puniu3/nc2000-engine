//! EXP-PRIOR-EXPLOIT Phase 3 (docs/EXP-prior-exploit.md §7) — BLIND-BR:
//! (1+λ) exploiter hill-climb against the DEPLOYED configuration.
//!
//! The candidate (exploiter) side is always skuct on the true state — the
//! information upper bound on any real opponent. The defended side is the
//! ship-3000 mixture (weighted 3-team gauntlet from pool-artifact.json),
//! played by:
//!
//!   --defense blind   the product bot: BlindAgent, belief candidate pool
//!                     from --belief-pool (default belief-pool-v1), log-ON
//!   --defense skuct   the Gate B control: RmAgent skuct, log-off — the
//!                     same shape META-NASH's exploiter search measured
//!
//! Seed conventions mirror `teamgen::gauntlet_eval` (per-opponent rng
//! stream, battle seed shared across the side-swap pair, per-game agent
//! seeds) and `meta_nash --br` (CRN eval seed per iteration, λ proposals,
//! accept on strict improvement, holdout at 3x games on a shifted seed).
//! The defense arm never enters seed derivation, so blind and skuct
//! lineages with the same --lineage name and --seed are seed-paired.
//!
//! Checkpoints ({lineage, parent, history, holdout}) are the meta_nash
//! shape, so `prior_exploit --y lineage:FILE` can rescore a finished best
//! directly. Resumable per lineage.
//!
//!   cargo run --release -p nc2000-bot --example prior_br -- \
//!     --defense blind --lineage bbr-rand-0 --br-iters 40 --iters 300
//!   --seed-team ID | --seed-file FILE (sets JSON) | (neither = random)

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::teamgen::{to_sets, TeamGen};
use nc2000_bot::{
    play_game, Agent, BlindAgent, GameResult, RmAgent, RmConfig, SplitMix64,
};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn atomic_write(path: &Path, content: &str) {
    let tmp = path.with_extension("tmp");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&tmp, content).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

struct ShipPool {
    teams: Vec<Vec<PokemonSet>>,
    weights: Vec<f64>,
}

fn load_ship_pool(root: &Path) -> ShipPool {
    let p = root.join("data/meta-nash-v1/pool-artifact.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let teams = v["teams"].as_array().expect("pool-artifact teams");
    let w: Vec<f64> = teams.iter().map(|t| t["weight"].as_f64().unwrap()).collect();
    let total: f64 = w.iter().sum();
    ShipPool {
        teams: teams
            .iter()
            .map(|t| to_sets(t["sets"].as_array().unwrap()).unwrap())
            .collect(),
        weights: w.into_iter().map(|x| x / total).collect(),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Defense {
    Blind,
    Skuct,
}

/// Weighted fitness of `candidate` (skuct, true state) vs the ship gauntlet
/// under the chosen defense. Seed derivation is `gauntlet_eval`'s and is
/// independent of the defense arm.
#[allow(clippy::too_many_arguments)]
fn fitness(
    dex: &Dex,
    candidate: &[PokemonSet],
    ship: &ShipPool,
    defense: Defense,
    belief_pool: &Arc<MetaPool>,
    games_per_opponent: u32,
    agent_iters: u32,
    threads: usize,
    seed: u64,
) -> f64 {
    struct Job {
        opp: usize,
        battle_seed: String,
        cand_seed: u64,
        opp_seed: u64,
        cand_is_p1: bool,
    }
    let games = (games_per_opponent + games_per_opponent % 2) as usize;
    let mut jobs = Vec::with_capacity(ship.teams.len() * games);
    for opp in 0..ship.teams.len() {
        let mut rng =
            SplitMix64::new(seed ^ (opp as u64 + 1).wrapping_mul(0x9FB2_1C65_1E98_DF25));
        let mut battle_seed = String::new();
        for g in 0..games {
            if g % 2 == 0 {
                battle_seed = rng.battle_seed();
            }
            jobs.push(Job {
                opp,
                battle_seed: battle_seed.clone(),
                cand_seed: rng.next(),
                opp_seed: rng.next(),
                cand_is_p1: g % 2 == 0,
            });
        }
    }
    let cursor = AtomicUsize::new(0);
    let mut results: Vec<(usize, usize, f64)> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads.max(1) {
            let (jobs, cursor) = (&jobs, &cursor);
            let belief_pool = belief_pool.clone();
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, usize, f64)> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let job = &jobs[i];
                    let cfg = |it: u32| RmConfig {
                        iterations: it,
                        rule: SelRule::Ucb,
                        ..Default::default()
                    };
                    let mut cand_agent = RmAgent::new(cfg(agent_iters), job.cand_seed);
                    let mut def_blind: BlindAgent;
                    let mut def_skuct: RmAgent;
                    let def_agent: &mut dyn Agent = match defense {
                        Defense::Blind => {
                            def_blind = BlindAgent::new(
                                cfg(agent_iters),
                                belief_pool.clone(),
                                None,
                                job.opp_seed,
                            );
                            &mut def_blind
                        }
                        Defense::Skuct => {
                            def_skuct = RmAgent::new(cfg(agent_iters), job.opp_seed);
                            &mut def_skuct
                        }
                    };
                    let cand_ref: &mut dyn Agent = &mut cand_agent;
                    let (t1, t2): (&[PokemonSet], &[PokemonSet]) = if job.cand_is_p1 {
                        (candidate, ship.teams[job.opp].as_slice())
                    } else {
                        (ship.teams[job.opp].as_slice(), candidate)
                    };
                    let mut b = Battle::from_fixture(dex, &job.battle_seed, t1, t2).unwrap();
                    // The blind observer reads the protocol log.
                    b.set_log_enabled(defense == Defense::Blind);
                    let res = if job.cand_is_p1 {
                        play_game(dex, &mut b, &mut [cand_ref, def_agent], 500)
                    } else {
                        play_game(dex, &mut b, &mut [def_agent, cand_ref], 500)
                    }
                    .unwrap();
                    let p1s = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let score = if job.cand_is_p1 { p1s } else { 1.0 - p1s };
                    out.push((i, job.opp, score));
                }
                out
            }));
        }
        for h in handles {
            results.extend(h.join().unwrap());
        }
    });
    results.sort_by_key(|r| r.0);
    let mut per: Vec<(f64, usize)> = vec![(0.0, 0); ship.teams.len()];
    for &(_, opp, s) in &results {
        per[opp].0 += s;
        per[opp].1 += 1;
    }
    per.iter()
        .zip(&ship.weights)
        .map(|((sum, n), w)| w * sum / (*n).max(1) as f64)
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = repo_root();
    let dex = load_dex();

    let defense = match flag(&args, "--defense").as_deref() {
        Some("blind") => Defense::Blind,
        Some("skuct") => Defense::Skuct,
        other => panic!("--defense blind|skuct required, got {other:?}"),
    };
    let lineage = flag(&args, "--lineage").expect("--lineage NAME");
    let iters: usize =
        flag(&args, "--br-iters").map(|v| v.parse().unwrap()).unwrap_or(40);
    let lambda: usize = flag(&args, "--lambda").map(|v| v.parse().unwrap()).unwrap_or(4);
    let games: u32 = flag(&args, "--fit-games").map(|v| v.parse().unwrap()).unwrap_or(8);
    let agent_iters: u32 =
        flag(&args, "--iters").map(|v| v.parse().unwrap()).unwrap_or(300);
    let base_seed: u64 =
        flag(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(20260814);
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let belief_path = flag(&args, "--belief-pool")
        .unwrap_or_else(|| "data/belief-pool-v1/belief-pool.json".into());

    let ship = load_ship_pool(&root);
    let belief_pool = Arc::new(load_meta_pool(&root.join(&belief_path)));
    let defense_name = match defense {
        Defense::Blind => "blind",
        Defense::Skuct => "skuct",
    };
    eprintln!(
        "prior_br {lineage}: defense={defense_name} (belief {} teams), \
         gauntlet {} teams, {games} games/opp, skuct:{agent_iters}, λ={lambda}",
        belief_pool.teams.len(),
        ship.teams.len()
    );

    let tg = TeamGen::new(
        &dex,
        &std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap(),
        &std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap(),
    )
    .unwrap();
    let ck_path = root.join(format!("data/prior-exploit-v1/br/{lineage}.json"));

    // resume or seed. Seed derivation is defense-independent on purpose
    // (see the module docs) — the LINEAGE NAME carries the arm identity.
    let (mut parent, mut history, start_iter, seed_desc): (Vec<Value>, Vec<Value>, usize, String) =
        if let Ok(t) = std::fs::read_to_string(&ck_path) {
            let v: Value = serde_json::from_str(&t).unwrap();
            (
                v["parent"].as_array().unwrap().clone(),
                v["history"].as_array().unwrap().clone(),
                v["iters_done"].as_u64().unwrap() as usize,
                v["seed_desc"].as_str().unwrap().to_string(),
            )
        } else if let Some(f) = flag(&args, "--seed-file") {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
            (
                v["sets"].as_array().expect("seed file sets").clone(),
                Vec::new(),
                0,
                format!("file:{f}"),
            )
        } else if let Some(id) = flag(&args, "--seed-team") {
            let p = root.join("data/meta-nash-v1/candidates.json");
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            let t = v["teams"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["id"].as_str() == Some(&id))
                .unwrap_or_else(|| panic!("unknown candidate id {id}"));
            (
                t["sets"].as_array().unwrap().clone(),
                Vec::new(),
                0,
                format!("candidate:{id}"),
            )
        } else {
            let mut rng = SplitMix64::new(base_seed ^ fnv1a64(&lineage));
            let team = tg.random_team_valid(&dex, &mut rng, 500).expect("random team");
            (team, Vec::new(), 0, "random".to_string())
        };

    let write_ck = |parent: &Vec<Value>, history: &Vec<Value>, done: usize| {
        atomic_write(
            &ck_path,
            &serde_json::to_string(&json!({
                "lineage": lineage, "defense": defense_name,
                "belief_pool": belief_path,
                "target": "ship-3000 (pool-artifact.json weights)",
                "seed_desc": seed_desc,
                "iters_done": done, "iters_requested": iters,
                "lambda": lambda, "fit_games": games, "agent_iters": agent_iters,
                "base_seed": base_seed,
                "parent": parent, "history": history,
            }))
            .unwrap(),
        );
    };

    for it in start_iter..iters {
        let eval_seed = base_seed ^ fnv1a64(&format!("{lineage}|eval|{it}"));
        let mut prop_rng =
            SplitMix64::new(base_seed ^ fnv1a64(&format!("{lineage}|prop|{it}")));
        let parent_sets = to_sets(&parent).unwrap();
        let parent_fit = fitness(
            &dex, &parent_sets, &ship, defense, &belief_pool, games, agent_iters, threads,
            eval_seed,
        );
        let mut best_child: Option<(f64, Vec<Value>, String)> = None;
        for _ in 0..lambda {
            let Some(prop) = (0..40).find_map(|_| tg.propose_valid(&dex, &parent, &mut prop_rng, 200))
            else {
                continue;
            };
            let sets = to_sets(&prop.team).unwrap();
            let fit = fitness(
                &dex, &sets, &ship, defense, &belief_pool, games, agent_iters, threads,
                eval_seed,
            );
            if best_child.as_ref().is_none_or(|(bf, _, _)| fit > *bf) {
                best_child = Some((fit, prop.team, format!("{:?}", prop.op)));
            }
        }
        let (accepted, child_fit, op) = match best_child {
            Some((fit, team, op)) if fit > parent_fit => {
                parent = team;
                (true, fit, op)
            }
            Some((fit, _, op)) => (false, fit, op),
            None => (false, f64::NAN, "none".into()),
        };
        history.push(json!({
            "iter": it, "parent_fit": parent_fit, "best_child_fit": child_fit,
            "accepted": accepted, "op": op,
            "team_after": if accepted { Some(parent.clone()) } else { None },
        }));
        write_ck(&parent, &history, it + 1);
        eprintln!(
            "  [{lineage} {it}] parent {parent_fit:.3} best-child {child_fit:.3} {}",
            if accepted { "ACCEPT" } else { "keep" }
        );
    }

    // holdout: parent at 3x games on a shifted seed
    let holdout_seed = base_seed ^ fnv1a64(&format!("{lineage}|holdout"));
    let parent_sets = to_sets(&parent).unwrap();
    let holdout = fitness(
        &dex, &parent_sets, &ship, defense, &belief_pool, games * 3, agent_iters, threads,
        holdout_seed,
    );
    let t = std::fs::read_to_string(&ck_path).unwrap();
    let mut v: Value = serde_json::from_str(&t).unwrap();
    v["holdout"] =
        json!({"seed": holdout_seed, "games_per_opponent": games * 3, "fit": holdout});
    atomic_write(&ck_path, &serde_json::to_string(&v).unwrap());
    eprintln!("prior_br {lineage} done: holdout {holdout:.3} -> {}", ck_path.display());
}
