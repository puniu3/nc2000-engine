//! PP-exhaustion census over SELF-PLAY games (read-only; no engine or eval
//! changes).
//!
//! Ladder finding under investigation: across 190 bot-vs-human games the bot
//! Struggled 40 times and the humans 0 times, concentrated in the long tail
//! (3 of the 6 games at >=80 turns ran every move of some mon to 0 PP). The
//! eval charges PP as one GLOBAL LINEAR term (`eval.rs:660-699`: `pp_num` /
//! `pp_den` summed over every living roster mon and every slot, scaled by
//! `w.pp = 0.2`), so running one mon dry is priced identically to shaving one
//! PP off six mons. This harness answers the prerequisite question for any
//! fix: does the same failure occur when the bot plays ITSELF, i.e. is there
//! a self-play population on which a tail-restricted duel could gate a fix?
//!
//! Measured per game (both sides, at every decision point with `turn >= 1`,
//! so team preview never contaminates the minima):
//!
//!   struggle_choices  times the side's chosen action was `Move(struggle)` —
//!                     the choice was forced, `search.rs:339-341` only offers
//!                     Struggle when `pokemon_choosable_moves` is empty;
//!   struggle_moves    `|move|pXa: N|Struggle` lines actually in the protocol
//!                     (a chosen Struggle can still be eaten by sleep/freeze/
//!                     flinch), which is what the ladder-log count measured;
//!   min_pp_frac       min over the game of exactly the eval's own quantity,
//!                     `pp_num/pp_den` over living roster mons;
//!   min_pp_total      min over the game of raw remaining PP, living mons;
//!   min_heal_pp       min over the game of summed PP of {rest, recover,
//!                     softboiled, milkdrink} — the same id set the KO-race
//!                     heal exemption gates on (`eval.rs:362`) — counted only
//!                     at points where a living mon still HAS such a slot, so
//!                     a fainted healer does not masquerade as an empty one;
//!   dry_mons          living mons ending the game with every slot at 0 PP.
//!
//! Games are scheduled exactly like `duel::run_duel`: seed-paired (each
//! pairing played twice with sides swapped on one battle seed), agent seeds
//! derived from the game index only, so the census is thread-count invariant
//! and its population is the one the eval gates run on.
//!
//! TRUNCATION: `--max-turns` (default 500, matching `DuelSpec::new` and
//! arena) caps a game via `runner::play_game`; capped games are counted and
//! flagged, because a cap below the tail would hide the defect by
//! construction.
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example pp_census -- \
//!     [--agent skuct:300] [--games 200] [--seed 1] [--threads N]
//!     [--max-turns 500] [--pool meta|fixtures] [--show 20] [--csv PATH]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use nc2000_bot::eval::EvalWeights;
use nc2000_bot::mcts::Playout;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::rng::SplitMix64;
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule};
use nc2000_bot::{Agent, MaxDamageAgent, MctsAgent, MctsConfig, RandomAgent};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::{Dex, MoveId};
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn opt_num<T: std::str::FromStr>(parts: &[&str], i: usize, default: T) -> T {
    parts.get(i).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Same in-battle agents `arena` exposes, minus the ones needing baked
/// tables or a belief prior (the census is about the perfect-info policy the
/// eval gates run on).
fn build_agent(spec: &str, seed: u64) -> Box<dyn Agent> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts[0] {
        "random" => Box::new(RandomAgent::new(seed)),
        "maxdamage" => Box::new(MaxDamageAgent::new()),
        "mcts" => Box::new(MctsAgent::new(
            MctsConfig {
                iterations: opt_num(&parts, 1, 1000u32),
                c: opt_num(&parts, 2, 1.0f64),
                playout: Playout::Heavy {
                    eps: opt_num(&parts, 3, 0.2f64),
                    turns: opt_num(&parts, 4, 8u16),
                    weights: EvalWeights::default(),
                },
                ..Default::default()
            },
            seed,
        )),
        "skuct" => Box::new(RmAgent::new(
            RmConfig {
                iterations: opt_num(&parts, 1, 1000u32),
                rule: SelRule::Ucb,
                c: opt_num(&parts, 2, 1.0f64),
                hp_buckets: opt_num(&parts, 3, 16i64),
                playout: Playout::Heavy {
                    eps: 0.2,
                    turns: 8,
                    weights: EvalWeights::default(),
                },
                ..Default::default()
            },
            seed,
        )),
        other => panic!("unknown agent: {other} (random|maxdamage|mcts|skuct)"),
    }
}

#[derive(Clone, Debug)]
struct GameRec {
    idx: usize,
    turns: u16,
    capped: bool,
    winner: Option<usize>,
    struggle_choices: [u32; 2],
    struggle_moves: [u32; 2],
    first_struggle_turn: [Option<u16>; 2],
    first_struggle_species: [Option<String>; 2],
    min_pp_frac: [f64; 2],
    min_pp_total: [i32; 2],
    min_heal_pp: [Option<i32>; 2],
    dry_mons: [u32; 2],
}

impl GameRec {
    fn struggled(&self) -> bool {
        self.struggle_moves[0] + self.struggle_moves[1] > 0
    }
}

/// Living-roster PP aggregates for one side: (remaining, max, heal PP,
/// whether a living mon still carries a heal slot).
fn side_pp(b: &Battle, side: usize, heal: &[MoveId]) -> (i32, i32, i32, bool) {
    let s = &b.sides[side];
    let (mut rem, mut max, mut heal_pp) = (0, 0, 0);
    let mut has_heal = false;
    for &slot in s.party.iter() {
        let p = &s.roster[slot as usize];
        if p.fainted || p.hp <= 0 {
            continue;
        }
        for ms in p.move_slots.iter() {
            rem += ms.pp;
            max += ms.maxpp;
            if heal.contains(&ms.id) {
                has_heal = true;
                heal_pp += ms.pp;
            }
        }
    }
    (rem, max, heal_pp, has_heal)
}

fn dry_mons(b: &Battle, side: usize) -> u32 {
    let s = &b.sides[side];
    s.party
        .iter()
        .filter(|&&slot| {
            let p = &s.roster[slot as usize];
            !p.fainted && p.hp > 0 && p.move_slots.iter().all(|m| m.pp <= 0)
        })
        .count() as u32
}

/// `runner::play_game` with per-decision PP sampling and Struggle accounting.
/// The battle runs log-ON; log content never affects battle state (the same
/// guarantee `DuelSpec::log_on` relies on), and draining it each step keeps
/// the transcript from growing without bound in a 500-turn game.
fn play_and_census(
    dex: &Dex,
    battle: &mut Battle,
    agents: &mut [&mut dyn Agent; 2],
    max_turns: u16,
    struggle_id: MoveId,
    heal: &[MoveId],
    idx: usize,
) -> GameRec {
    let mut rec = GameRec {
        idx,
        turns: 0,
        capped: false,
        winner: None,
        struggle_choices: [0; 2],
        struggle_moves: [0; 2],
        first_struggle_turn: [None; 2],
        first_struggle_species: [const { None }; 2],
        min_pp_frac: [1.0; 2],
        min_pp_total: [i32::MAX; 2],
        min_heal_pp: [None; 2],
        dry_mons: [0; 2],
    };
    battle.set_log_enabled(true);
    loop {
        if let Some(o) = battle.outcome() {
            rec.winner = match o {
                Outcome::P1Win => Some(0),
                Outcome::P2Win => Some(1),
                Outcome::Tie => None,
            };
            break;
        }
        if battle.turn > max_turns {
            rec.capped = true;
            break;
        }
        // Team preview (turn 0) still shows all six; sampling it would put a
        // meaningless full-roster maximum in the minima.
        if battle.turn >= 1 {
            for s in 0..2 {
                let (rem, max, heal_pp, has_heal) = side_pp(battle, s, heal);
                if max > 0 {
                    rec.min_pp_frac[s] = rec.min_pp_frac[s].min(rem as f64 / max as f64);
                }
                rec.min_pp_total[s] = rec.min_pp_total[s].min(rem);
                if has_heal {
                    rec.min_heal_pp[s] =
                        Some(rec.min_heal_pp[s].map_or(heal_pp, |m: i32| m.min(heal_pp)));
                }
            }
        }
        let mut picks = [None, None];
        for s in 0..2 {
            let cs = battle.legal_choices(dex, s);
            if cs.is_empty() {
                continue;
            }
            let pick = agents[s].choose(battle, dex, s, &cs);
            if pick == SearchChoice::Move(struggle_id) {
                rec.struggle_choices[s] += 1;
                if rec.first_struggle_turn[s].is_none() {
                    rec.first_struggle_turn[s] = Some(battle.turn);
                    rec.first_struggle_species[s] = battle
                        .active_id(s)
                        .map(|id| dex.species.key(battle.poke(id).species).to_string());
                }
            }
            picks[s] = Some(pick);
        }
        if picks == [None, None] {
            panic!("game {idx}: no side owes a choice but the battle has not ended");
        }
        battle.apply_choices(dex, picks).expect("apply_choices");
        for line in battle.log.drain(..) {
            // |move|p1a: Snorlax|Struggle|p2a: Skarmory
            if !line.starts_with("|move|") {
                continue;
            }
            let f: Vec<&str> = line.split('|').collect();
            let (Some(who), Some(name)) = (f.get(2), f.get(3)) else { continue };
            if name.eq_ignore_ascii_case("struggle") {
                let s = if who.starts_with("p2") { 1 } else { 0 };
                rec.struggle_moves[s] += 1;
            }
        }
    }
    rec.turns = battle.turn;
    for s in 0..2 {
        rec.dry_mons[s] = dry_mons(battle, s);
        if rec.min_pp_total[s] == i32::MAX {
            rec.min_pp_total[s] = 0;
        }
    }
    rec
}

fn pct<T: Copy + PartialOrd>(sorted: &[T], p: usize) -> T {
    let rank = (sorted.len() * p).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Point-biserial r between a 0/1 indicator and a continuous variable.
fn point_biserial(flag: &[bool], x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let n1 = flag.iter().filter(|&&f| f).count() as f64;
    let n0 = n - n1;
    if sd == 0.0 || n1 == 0.0 || n0 == 0.0 {
        return f64::NAN;
    }
    let m1 = flag
        .iter()
        .zip(x)
        .filter(|(f, _)| **f)
        .map(|(_, v)| *v)
        .sum::<f64>()
        / n1;
    let m0 = flag
        .iter()
        .zip(x)
        .filter(|(f, _)| !**f)
        .map(|(_, v)| *v)
        .sum::<f64>()
        / n0;
    (m1 - m0) / sd * (n1 * n0 / (n * n)).sqrt()
}

fn load_fixture_pool() -> Vec<Vec<PokemonSet>> {
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

struct GameSpec {
    team_p1: usize,
    team_p2: usize,
    battle_seed: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let agent_spec = arg_s(&args, "--agent", "skuct:300");
    let games: usize = arg_s(&args, "--games", "200").parse().unwrap();
    let base_seed: u64 = arg_s(&args, "--seed", "1").parse().unwrap();
    let max_turns: u16 = arg_s(&args, "--max-turns", "500").parse().unwrap();
    let threads: usize = arg_s(&args, "--threads", "0")
        .parse()
        .ok()
        .filter(|&t: &usize| t > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let pool_spec = arg_s(&args, "--pool", "meta");
    let show: usize = arg_s(&args, "--show", "20").parse().unwrap();
    let csv_path = arg_s(&args, "--csv", "");

    let dex = conformance::load_dex();
    let root = repo_root();
    let teams: Vec<Vec<PokemonSet>> = if pool_spec.starts_with("meta") {
        let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
        pool.teams.iter().map(|t| t.sets.clone()).collect()
    } else {
        load_fixture_pool()
    };
    let struggle_id = dex.moves.id("struggle").expect("struggle interned");
    // Same id set the KO-race heal exemption gates on (eval.rs:362).
    let heal: Vec<MoveId> = ["rest", "recover", "softboiled", "milkdrink"]
        .iter()
        .filter_map(|k| dex.moves.id(k))
        .collect();

    let games = games + games % 2;
    let mut sched = SplitMix64::new(base_seed);
    let mut specs = Vec::with_capacity(games);
    for _ in 0..games / 2 {
        let t1 = sched.below(teams.len());
        let t2 = sched.below(teams.len());
        let seed = sched.battle_seed();
        specs.push(GameSpec { team_p1: t1, team_p2: t2, battle_seed: seed.clone() });
        // Side-swapped replay of the same pairing and battle seed.
        specs.push(GameSpec { team_p1: t2, team_p2: t1, battle_seed: seed });
    }

    eprintln!(
        "{} teams ({pool_spec}), agent {agent_spec} (self-play), {} games, seed {base_seed}, \
         {threads} threads, max_turns {max_turns}",
        teams.len(),
        specs.len()
    );

    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    let mut recs: Vec<GameRec> = Vec::with_capacity(specs.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let (specs, cursor, done, dex, heal, agent_spec, teams) =
                (&specs, &cursor, &done, &dex, &heal, &agent_spec, &teams);
            handles.push(scope.spawn(move || {
                let mut out: Vec<GameRec> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    let g = &specs[i];
                    // Seeds derived from the game index only -> thread-count
                    // invariant, matching duel.rs:174-176.
                    let sa = base_seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407);
                    let sb = base_seed ^ (i as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25);
                    let mut a = build_agent(agent_spec, sa);
                    let mut b = build_agent(agent_spec, sb);
                    let mut battle = Battle::from_fixture(
                        dex,
                        &g.battle_seed,
                        &teams[g.team_p1],
                        &teams[g.team_p2],
                    )
                    .unwrap();
                    let rec = play_and_census(
                        dex,
                        &mut battle,
                        &mut [a.as_mut(), b.as_mut()],
                        max_turns,
                        struggle_id,
                        heal,
                        i,
                    );
                    out.push(rec);
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 20 == 0 || d == specs.len() {
                        eprintln!("  {d}/{} games ({:.0}s)", specs.len(), t0.elapsed().as_secs_f64());
                    }
                }
                out
            }));
        }
        for h in handles {
            recs.extend(h.join().unwrap());
        }
    });
    recs.sort_by_key(|r| r.idx);

    // ---- summary --------------------------------------------------------
    let n = recs.len();
    let mut turns: Vec<u16> = recs.iter().map(|r| r.turns).collect();
    turns.sort_unstable();
    let struggle_games = recs.iter().filter(|r| r.struggled()).count();
    let choice_games =
        recs.iter().filter(|r| r.struggle_choices[0] + r.struggle_choices[1] > 0).count();
    let total_struggles: u32 = recs.iter().map(|r| r.struggle_moves[0] + r.struggle_moves[1]).sum();
    let capped = recs.iter().filter(|r| r.capped).count();
    let dry_games = recs.iter().filter(|r| r.dry_mons[0] + r.dry_mons[1] > 0).count();

    println!("\n=== pp_census: {agent_spec} self-play, pool {pool_spec} ===");
    println!("games {n}   wall {:.0}s   TRUNCATION LIMIT --max-turns {max_turns} (capped games: {capped})", t0.elapsed().as_secs_f64());
    println!(
        "turns: median {}  p90 {}  p95 {}  p99 {}  max {}  mean {:.1}",
        pct(&turns, 50),
        pct(&turns, 90),
        pct(&turns, 95),
        pct(&turns, 99),
        turns[n - 1],
        turns.iter().map(|&t| t as f64).sum::<f64>() / n as f64
    );
    println!(
        "games with >=1 Struggle in the protocol: {struggle_games} ({:.1}%)   \
         games where Struggle was the only legal choice: {choice_games} ({:.1}%)",
        100.0 * struggle_games as f64 / n as f64,
        100.0 * choice_games as f64 / n as f64
    );
    println!(
        "total Struggle uses {total_struggles}   games with a living all-0-PP mon at the end: \
         {dry_games} ({:.1}%)",
        100.0 * dry_games as f64 / n as f64
    );

    // Struggle vs game length.
    let flags: Vec<bool> = recs.iter().map(|r| r.struggled()).collect();
    let lens: Vec<f64> = recs.iter().map(|r| r.turns as f64).collect();
    let r_pb = point_biserial(&flags, &lens);
    let mean_len = |sel: bool| {
        let v: Vec<f64> =
            recs.iter().filter(|r| r.struggled() == sel).map(|r| r.turns as f64).collect();
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    println!(
        "\nstruggle vs length: point-biserial r {r_pb:.3}   mean turns with Struggle {:.1} / \
         without {:.1}",
        mean_len(true),
        mean_len(false)
    );
    println!("  turns bucket      games   struggle games   rate    total struggles");
    let buckets: [(u16, u16); 6] =
        [(0, 19), (20, 39), (40, 59), (60, 79), (80, 119), (120, u16::MAX)];
    for (lo, hi) in buckets {
        let sel: Vec<&GameRec> = recs.iter().filter(|r| r.turns >= lo && r.turns <= hi).collect();
        if sel.is_empty() {
            continue;
        }
        let s = sel.iter().filter(|r| r.struggled()).count();
        let tot: u32 = sel.iter().map(|r| r.struggle_moves[0] + r.struggle_moves[1]).sum();
        let label = if hi == u16::MAX { format!("{lo}+") } else { format!("{lo}-{hi}") };
        println!(
            "  {label:>12}   {:6}   {:14}   {:5.1}%   {tot:6}",
            sel.len(),
            s,
            100.0 * s as f64 / sel.len() as f64
        );
    }

    // Per-side (a self-play sanity check: the two sides must be symmetric).
    for s in 0..2 {
        let g = recs.iter().filter(|r| r.struggle_moves[s] > 0).count();
        let u: u32 = recs.iter().map(|r| r.struggle_moves[s]).sum();
        println!("  side p{}: {g} games with Struggle, {u} uses", s + 1);
    }

    // min PP fraction (the eval's own quantity) and min heal PP.
    let mut fracs: Vec<f64> = recs.iter().flat_map(|r| r.min_pp_frac).collect();
    fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\nmin pp_num/pp_den per side-game (eval.rs:660-699 quantity), {} samples:",
        fracs.len()
    );
    println!(
        "  min {:.3}  p1 {:.3}  p5 {:.3}  p25 {:.3}  median {:.3}  p75 {:.3}  max {:.3}",
        fracs[0],
        pct(&fracs, 1),
        pct(&fracs, 5),
        pct(&fracs, 25),
        pct(&fracs, 50),
        pct(&fracs, 75),
        fracs[fracs.len() - 1]
    );

    let mut heals: Vec<i32> =
        recs.iter().flat_map(|r| r.min_heal_pp.into_iter().flatten()).collect();
    heals.sort_unstable();
    println!(
        "\nmin heal-move PP per side-game (rest/recover/softboiled/milkdrink, counted only while \
         a living mon still has the slot): {} of {} side-games had one",
        heals.len(),
        2 * n
    );
    if !heals.is_empty() {
        println!(
            "  min {}  p5 {}  p25 {}  median {}  p75 {}  p95 {}  max {}",
            heals[0],
            pct(&heals, 5),
            pct(&heals, 25),
            pct(&heals, 50),
            pct(&heals, 75),
            pct(&heals, 95),
            heals[heals.len() - 1]
        );
        let mut hist: std::collections::BTreeMap<i32, usize> = Default::default();
        for &h in &heals {
            let b = if h == 0 { 0 } else { ((h - 1) / 4 + 1) * 4 };
            *hist.entry(b).or_default() += 1;
        }
        println!("  histogram (bucket = upper bound, 0 is its own bucket):");
        for (b, c) in hist.iter().take(12) {
            println!(
                "    <={b:>3}  {c:5}  {:5.1}%",
                100.0 * *c as f64 / heals.len() as f64
            );
        }
        let zero = heals.iter().filter(|&&h| h == 0).count();
        println!(
            "  side-games that ran the heal move to 0 PP while its owner was alive: {zero} \
             ({:.1}%)",
            100.0 * zero as f64 / heals.len() as f64
        );
    }

    // Longest / struggling games.
    let mut worst: Vec<&GameRec> = recs.iter().collect();
    worst.sort_by_key(|r| {
        (
            std::cmp::Reverse(r.struggle_moves[0] + r.struggle_moves[1]),
            std::cmp::Reverse(r.turns),
        )
    });
    println!(
        "\ntop {show} games by Struggle count (game, turns, cap, struggle p1/p2, first p1, \
         first p2, min pp frac p1/p2, min heal pp p1/p2, dry mons p1/p2):"
    );
    for r in worst.iter().take(show) {
        let f = |o: &Option<String>, t: &Option<u16>| match (o, t) {
            (Some(sp), Some(tn)) => format!("{sp}@T{tn}"),
            _ => "-".into(),
        };
        let h = |o: Option<i32>| o.map_or("-".into(), |v| v.to_string());
        println!(
            "  g{:<5} T{:<4}{} strg {}/{}  {} {}  ppfrac {:.3}/{:.3}  heal {}/{}  dry {}/{}",
            r.idx,
            r.turns,
            if r.capped { " CAP" } else { "    " },
            r.struggle_moves[0],
            r.struggle_moves[1],
            f(&r.first_struggle_species[0], &r.first_struggle_turn[0]),
            f(&r.first_struggle_species[1], &r.first_struggle_turn[1]),
            r.min_pp_frac[0],
            r.min_pp_frac[1],
            h(r.min_heal_pp[0]),
            h(r.min_heal_pp[1]),
            r.dry_mons[0],
            r.dry_mons[1]
        );
    }

    if !csv_path.is_empty() {
        let mut s = String::from(
            "game,turns,capped,winner,strg_choice_p1,strg_choice_p2,strg_move_p1,strg_move_p2,\
             min_pp_frac_p1,min_pp_frac_p2,min_pp_total_p1,min_pp_total_p2,min_heal_pp_p1,\
             min_heal_pp_p2,dry_p1,dry_p2\n",
        );
        for r in &recs {
            s.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{}\n",
                r.idx,
                r.turns,
                r.capped as u8,
                r.winner.map_or(-1, |w| w as i32),
                r.struggle_choices[0],
                r.struggle_choices[1],
                r.struggle_moves[0],
                r.struggle_moves[1],
                r.min_pp_frac[0],
                r.min_pp_frac[1],
                r.min_pp_total[0],
                r.min_pp_total[1],
                r.min_heal_pp[0].map_or(-1, |v| v),
                r.min_heal_pp[1].map_or(-1, |v| v),
                r.dry_mons[0],
                r.dry_mons[1],
            ));
        }
        if let Some(p) = std::path::Path::new(&csv_path).parent() {
            std::fs::create_dir_all(p).ok();
        }
        std::fs::write(&csv_path, s).expect("write csv");
        println!("\ncsv: {csv_path}");
    }
}
