//! Paired Baton Pass / Setup gauntlet: two arms each play the SAME fixed opponent
//! on the same team draws, battle seeds, and agent seeds (CRN-paired).
//!
//! - `--mode offense`: the arm fields a team with a Baton Pass or setup sweeper
//!   lead (from `belief-pool.json` + `master-pool.json`) and plays freely after
//!   team preview; the opponent plays a standard meta team.
//! - `--mode defense`: the opponent leads with a Baton Pass / setup user; the
//!   arm plays a standard meta team and defends against it.
//! - `--mode both`: 50% offense games, 50% defense games.
//!
//! Usage:
//!   setup_gauntlet <armA> <armB> --vs <opp> [--mode offense|defense|both]
//!       [--games N] [--seed S] [--threads T] [--max-turns M]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{Agent, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;

mod combo_common;
use combo_common::{norm, play, Forced, Spec};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn is_baton_pass_user(p: &PokemonSet) -> bool {
    p.moves.iter().any(|m| norm(m) == "batonpass")
}

fn is_setup_sweeper(p: &PokemonSet) -> bool {
    p.moves.iter().any(|m| {
        matches!(
            norm(m).as_str(),
            "batonpass" | "swordsdance" | "bellydrum" | "agility" | "amnesia" | "growth"
        )
    })
}

fn setup_lead(team: &[PokemonSet]) -> Option<u8> {
    if let Some(i) = team.iter().position(is_baton_pass_user) {
        return Some(i as u8 + 1);
    }
    team.iter().position(is_setup_sweeper).map(|i| i as u8 + 1)
}

struct Game {
    arm_team: usize,
    opp_team: usize,
    seed: String,
    arm_is_p1: bool,
    defense: bool,
    arm_seed: u64,
    opp_seed: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arms = [Spec::parse(&args[0]), Spec::parse(&args[1])];
    let opp = Spec::parse(&flag(&args, "--vs").unwrap_or_else(|| "open:300".to_string()));
    let mode = flag(&args, "--mode").unwrap_or_else(|| "both".to_string());
    let games: usize = flag(&args, "--games").map_or(120, |v| v.parse().unwrap());
    let base_seed: u64 = flag(&args, "--seed").map_or(1, |v| v.parse().unwrap());
    let max_turns: u16 = flag(&args, "--max-turns").map_or(500, |v| v.parse().unwrap());
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));

    let dex = load_dex();
    let root = repo_root();
    let meta_pool = Arc::new(load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json")));
    let belief_pool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let master_pool = load_meta_pool(&root.join("data/eco-pool/master-pool.json"));

    let mut teams: Vec<Vec<PokemonSet>> = meta_pool.teams.iter().map(|t| t.sets.clone()).collect();
    for t in &belief_pool.teams {
        teams.push(t.sets.clone());
    }
    for t in &master_pool.teams {
        teams.push(t.sets.clone());
    }

    let bp_teams: Vec<usize> = (0..teams.len())
        .filter(|&i| teams[i].iter().any(is_baton_pass_user))
        .collect();
    let setup_teams: Vec<usize> = (0..teams.len())
        .filter(|&i| setup_lead(&teams[i]).is_some())
        .collect();
    let std_teams: Vec<usize> = (0..teams.len()).collect();

    eprintln!(
        "setup_gauntlet: {} total teams ({} Baton Pass, {} setup) | mode={mode} games={games}",
        teams.len(),
        bp_teams.len(),
        setup_teams.len()
    );

    let mut sched = SplitMix64::new(base_seed ^ 0x6A09_E667_F3BC_C908);
    let schedule: Vec<Game> = (0..games)
        .map(|g| {
            let defense = match mode.as_str() {
                "defense" => true,
                "offense" => false,
                "both" => (g / 2) % 2 == 1,
                other => panic!("--mode offense|defense|both, got {other}"),
            };
            // Prioritize Baton Pass teams on 60% of games, general setup on 40%
            let tac_pool = if !bp_teams.is_empty() && g % 5 < 3 {
                &bp_teams
            } else {
                &setup_teams
            };
            let (arm_team, opp_team) = if defense {
                (
                    std_teams[sched.below(std_teams.len())],
                    tac_pool[sched.below(tac_pool.len())],
                )
            } else {
                (
                    tac_pool[sched.below(tac_pool.len())],
                    std_teams[sched.below(std_teams.len())],
                )
            };
            Game {
                arm_team,
                opp_team,
                seed: sched.battle_seed(),
                arm_is_p1: g % 2 == 0,
                defense,
                arm_seed: base_seed ^ (g as u64).wrapping_mul(0xA24B_AED4_963E_E407),
                opp_seed: base_seed ^ (g as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25),
            }
        })
        .collect();

    let cursor = AtomicUsize::new(0);
    let mut rows: Vec<(usize, f64, u16)> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (schedule, cursor, dex, teams, meta_pool, arms, opp) =
                    (&schedule, &cursor, &dex, &teams, &meta_pool, &arms, &opp);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let job = cursor.fetch_add(1, Ordering::Relaxed);
                        if job >= 2 * schedule.len() {
                            break;
                        }
                        let g = &schedule[job / 2];
                        let arm_spec = &arms[job % 2];
                        let arm_team = &teams[g.arm_team];
                        let opp_team = &teams[g.opp_team];
                        let mut arm: Box<dyn Agent> = Box::new(Forced {
                            inner: arm_spec.build(g.arm_seed, meta_pool),
                            lead: if g.defense {
                                None
                            } else {
                                setup_lead(arm_team)
                            },
                            specialist: false,
                        });
                        let mut foe: Box<dyn Agent> = Box::new(Forced {
                            inner: opp.build(g.opp_seed, meta_pool),
                            lead: if g.defense {
                                setup_lead(opp_team)
                            } else {
                                None
                            },
                            specialist: false,
                        });
                        let (t1, t2) = if g.arm_is_p1 {
                            (arm_team, opp_team)
                        } else {
                            (opp_team, arm_team)
                        };
                        let mut battle = Battle::from_fixture(dex, &g.seed, t1, t2).unwrap();
                        battle.set_log_enabled(false);
                        let (p1, p2): (&mut dyn Agent, &mut dyn Agent) = if g.arm_is_p1 {
                            (&mut *arm, &mut *foe)
                        } else {
                            (&mut *foe, &mut *arm)
                        };
                        let (p1_score, turns, _) = play(dex, &mut battle, [p1, p2], max_turns);
                        let score = if g.arm_is_p1 {
                            p1_score
                        } else {
                            1.0 - p1_score
                        };
                        out.push((job, score, turns));
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

    let n = (rows.len() / 2) as f64;
    let a: Vec<f64> = rows.iter().step_by(2).map(|r| r.1).collect();
    let b: Vec<f64> = rows.iter().skip(1).step_by(2).map(|r| r.1).collect();
    let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let dm = mean(&d);
    let sd = (d.iter().map(|x| (x - dm).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    println!(
        "{} vs {} | mode={mode}: A {:.3}  B {:.3}  B-A {:+.3} +/- {:.3} (95%, paired) over {} games, {} differ",
        args[0],
        args[1],
        mean(&a),
        mean(&b),
        -dm,
        1.96 * sd / n.sqrt(),
        n as usize,
        d.iter().filter(|x| **x != 0.0).count()
    );
}
