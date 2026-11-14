//! Paired Perish-trap gauntlet: two arms each play the SAME fixed opponent
//! on the same team draws, battle seeds and agent seeds, so identical arms
//! produce identical games and every difference is the arms' own.
//!
//! Self-play cannot measure this: bots rarely bring or use the trap, so a
//! bot-vs-bot arena almost never contains a Perish line. Exposure is forced by role instead:
//!
//! - `--mode defense`: the arm's team has no trapper; the opponent's has
//!   one, leads with it and plays the trap as a fixed script
//!   (`combo_common::Forced` specialist).
//! - `--mode offense`: the arm leads with its own trapper and decides for
//!   itself; the opponent is a plain agent with no trapper.
//!
//!   combo_gauntlet <armA> <armB> --vs <opp> --mode defense|offense
//!       [--games N] [--seed S] [--threads T] [--max-turns M] [--jsonl FILE]
//!       [--only G]   play game G alone (with COMBO_TRACE=1: every decision)
//!
//! Specs: (open|blind):ITERS[:C][:-perish_combo|:perish_escape] (shipped rules by default)

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{Agent, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;
use serde_json::json;

mod combo_common;
use combo_common::{is_trapper, play, Forced, Spec, Tally};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn trapper_lead(team: &[PokemonSet]) -> Option<u8> {
    team.iter().position(is_trapper).map(|i| i as u8 + 1)
}

struct Game {
    arm_team: usize,
    opp_team: usize,
    seed: String,
    arm_is_p1: bool,
    arm_seed: u64,
    opp_seed: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arms = [Spec::parse(&args[0]), Spec::parse(&args[1])];
    let opp = Spec::parse(&flag(&args, "--vs").expect("--vs OPP"));
    let defense = match flag(&args, "--mode").as_deref() {
        Some("defense") => true,
        Some("offense") => false,
        other => panic!("--mode defense|offense, got {other:?}"),
    };
    let games: usize = flag(&args, "--games").map_or(100, |v| v.parse().unwrap());
    let base_seed: u64 = flag(&args, "--seed").map_or(1, |v| v.parse().unwrap());
    let max_turns: u16 = flag(&args, "--max-turns").map_or(500, |v| v.parse().unwrap());
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let dex = load_dex();
    let pool = Arc::new(load_meta_pool(
        &repo_root().join("data/meta-pool-v0/meta-pool.json"),
    ));
    let teams: Vec<Vec<PokemonSet>> = pool.teams.iter().map(|t| t.sets.clone()).collect();
    let (with, without): (Vec<usize>, Vec<usize>) =
        (0..teams.len()).partition(|&i| trapper_lead(&teams[i]).is_some());
    let (arm_pool, opp_pool) = if defense {
        (&without, &with)
    } else {
        (&with, &without)
    };
    eprintln!(
        "{} arm teams, {} opponent teams",
        arm_pool.len(),
        opp_pool.len()
    );

    let mut sched = SplitMix64::new(base_seed ^ 0x6A09_E667_F3BC_C908);
    let schedule: Vec<Game> = (0..games)
        .map(|g| Game {
            arm_team: arm_pool[sched.below(arm_pool.len())],
            opp_team: opp_pool[sched.below(opp_pool.len())],
            seed: sched.battle_seed(),
            arm_is_p1: g % 2 == 0,
            arm_seed: base_seed ^ (g as u64).wrapping_mul(0xA24B_AED4_963E_E407),
            opp_seed: base_seed ^ (g as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25),
        })
        .collect();

    let only: Option<usize> = flag(&args, "--only").map(|v| v.parse().unwrap());
    let cursor = AtomicUsize::new(0);
    let mut rows: Vec<(usize, f64, u16, [Tally; 2])> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (schedule, cursor, dex, teams, pool, arms, opp) =
                    (&schedule, &cursor, &dex, &teams, &pool, &arms, &opp);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let job = cursor.fetch_add(1, Ordering::Relaxed);
                        if job >= 2 * schedule.len() {
                            break;
                        }
                        if only.is_some_and(|o| o != job / 2) {
                            continue;
                        }
                        let g = &schedule[job / 2];
                        let arm_spec = &arms[job % 2];
                        let arm_team = &teams[g.arm_team];
                        let opp_team = &teams[g.opp_team];
                        let mut arm: Box<dyn Agent> = Box::new(Forced {
                            inner: arm_spec.build(g.arm_seed, pool),
                            lead: if defense {
                                None
                            } else {
                                trapper_lead(arm_team)
                            },
                            specialist: false,
                        });
                        let mut foe: Box<dyn Agent> = Box::new(Forced {
                            inner: opp.build(g.opp_seed, pool),
                            lead: if defense {
                                trapper_lead(opp_team)
                            } else {
                                None
                            },
                            specialist: defense,
                        });
                        let (t1, t2) = if g.arm_is_p1 {
                            (arm_team, opp_team)
                        } else {
                            (opp_team, arm_team)
                        };
                        let mut battle = Battle::from_fixture(dex, &g.seed, t1, t2).unwrap();
                        battle.set_log_enabled(true);
                        let (p1, p2): (&mut dyn Agent, &mut dyn Agent) = if g.arm_is_p1 {
                            (&mut *arm, &mut *foe)
                        } else {
                            (&mut *foe, &mut *arm)
                        };
                        let (p1_score, turns, t) = play(dex, &mut battle, [p1, p2], max_turns);
                        let score = if g.arm_is_p1 {
                            p1_score
                        } else {
                            1.0 - p1_score
                        };
                        let by_role = if g.arm_is_p1 { t } else { [t[1], t[0]] };
                        out.push((job, score, turns, by_role));
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            rows.extend(h.join().unwrap());
        }
    });
    rows.sort_by_key(|r| r.0);

    if let Some(path) = flag(&args, "--jsonl") {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for pair in rows.chunks_exact(2) {
            let g = &schedule[pair[0].0 / 2];
            writeln!(
                f,
                "{}",
                json!({"game":pair[0].0 / 2,"arm_team":g.arm_team,"opp_team":g.opp_team,
                    "arm_is_p1":g.arm_is_p1,"seed":g.seed,
                    "a":{"score":pair[0].1,"turns":pair[0].2,"arm":pair[0].3[0].json(),"opp":pair[0].3[1].json()},
                    "b":{"score":pair[1].1,"turns":pair[1].2,"arm":pair[1].3[0].json(),"opp":pair[1].3[1].json()}})
            )
            .unwrap();
        }
    }
    let n = (rows.len() / 2) as f64;
    let a: Vec<f64> = rows.iter().step_by(2).map(|r| r.1).collect();
    let b: Vec<f64> = rows.iter().skip(1).step_by(2).map(|r| r.1).collect();
    let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let dm = mean(&d);
    let sd = (d.iter().map(|x| (x - dm).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    println!(
        "{} vs {} | {}: A {:.3}  B {:.3}  A-B {:+.3} +/- {:.3} (95%, paired) over {} games, {} differ",
        args[0],
        args[1],
        if defense { "defense" } else { "offense" },
        mean(&a),
        mean(&b),
        dm,
        1.96 * sd / n.sqrt(),
        n,
        d.iter().filter(|x| **x != 0.0).count()
    );
    for (arm, name) in [(0usize, "A"), (1, "B")] {
        let mut own = Tally::default();
        let mut foe = Tally::default();
        for r in rows.iter().skip(arm).step_by(2) {
            own.add(&r.3[0]);
            foe.add(&r.3[1]);
        }
        println!("{name} arm: {}", own.json());
        println!("{name} opp: {}", foe.json());
    }
}
