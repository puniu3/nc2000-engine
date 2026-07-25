//! Does the bot switch as often as equilibrium does? — the identifying test
//! for the M16b switching cluster.
//!
//! Two attempts to close that gap came back null, from opposite layers: the
//! M16c rollout switch policy and the Spikes eval term. So the question is
//! whether humans switch correctly and the bot is suboptimal, or humans
//! over-switch. Neither cheap instrument answers it:
//!
//!   * human replays — skill is a proxy and the outcome is a confound (a
//!     losing player switches BECAUSE they are losing, and a bad matchup
//!     causes both), and the corpus's rankable player pool is 20 accounts;
//!   * a perturbed-bot duel — a strength change cannot separate "switching is
//!     bad" from "this bot switches badly", in either direction.
//!
//! This one is identifying: build a position small enough to solve exactly to
//! the end, read the equilibrium switch probability off the root LP, and
//! compare the bot's root switch mass in the same position. The reference is
//! correct by construction, so a gap is the bot's and nothing else.
//!
//! The positions are "pure attack/switch rock-paper-scissors": every mon
//! carries exactly ONE move, so the action set is {attack, switch} and the
//! root is a 2x2 matrix. That removes the "maybe it switches to the wrong
//! target" and "maybe it picks the wrong move" alternatives — with one move
//! and one bench mon there is no such freedom, only the rate.
//!
//! STATUS (2026-07-26): the harness is built and identifying, but no usable row
//! has been produced yet, and the obstruction is worth recording because it is
//! structural rather than a matter of tuning.
//!
//! The first full CX sweep (16 vCPU / 32 GB, `data/switch-eq-v3/`) came back with
//! all nine rows bracketed, and reclassified the constraint: RAM held and it
//! finished in 9818 s of a 21600 s cap, so every arm stopped on `--budget` /
//! `--work`, not on the box. What it did expose is that solvability and
//! undecidedness are anti-correlated across SCENARIOS rather than along HP — the
//! position that nearly solves (`entry_gap` 0.050) is a certified 0.0 for p1,
//! and the undecided ones sit at 0.17-0.77. Hence `--hp1`/`--hp2`: skew the HP
//! so a solvable position stops being decided. See `scale_hp`.
//!
//! `--hp` sets every live mon's HP, and the two ends of that knob fail for
//! opposite reasons:
//!
//!   * At 1 HP the position solves exactly (`entry_gap` 0.0000 measured), but
//!     switching means conceding a free KO, so equilibrium never switches and
//!     the value lands on a decided 0.0 or 1.0. A decided position makes every
//!     strategy optimal, so its switch mass is LP degeneracy, not an answer.
//!   * At 25-60% HP the tension is real — a mon survives the hit it eats on the
//!     way in — but the tree is not solved to the end within budget
//!     (`entry_gap` ~0.92), and at 40% the solver OOMed this 8 GB box outright.
//!
//! There is no HP setting between them that collapses chance, and that is
//! provable rather than empirical: to make one hit never KO you need HP above
//! CRIT damage (~2x normal), and to make two hits always KO you need HP at most
//! 2x the MINIMUM roll. Since max roll > min roll, those cannot both hold. Any
//! position with a genuine stay-or-retreat trade-off therefore keeps chance
//! nodes, which is exactly what the M17e solver exists for — it just needs more
//! budget and more RAM than this box has. Next step is CX (32 GB) rather than
//! further local tuning.
//!
//! Moves are all 100% accuracy so the miss branch, at least, is gone.
//!
//! Usage: cargo run --release -p nc2000-bot --example switch_equilibrium -- \
//!          [--iters 30000] [--seed 1] [--budget 400000] [--work 20000000] \
//!          [--hp 60 | --hp1 40 --hp2 32] [--leaf-cap 20000] [--scenario SUBSTR]

use std::io::Write as _;
use std::path::Path;

use nc2000_bot::exact::{ExactConfig, ExactSolver};
use nc2000_bot::smmcts::{RmConfig, SelRule, SkuctSearch};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use conformance::load_dex;

fn set(species: &str, mv: &str, level: u8) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": species,
        "species": species,
        "item": "",
        "ability": "No Ability",
        "moves": [mv],
        "nature": "Serious",
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
        "gender": "M",
        "level": level,
        "happiness": 255
    }))
    .expect("set")
}

/// Each side picks 3 (the format's fixed size), then the third is retired so
/// the live position is 2v2. Same class of state surgery `import.rs` performs
/// when it synthesises a corpus position.
fn retire_third(b: &mut Battle) {
    for s in 0..2 {
        let slot = b.sides[s].party[2] as usize;
        let p = &mut b.sides[s].roster[slot];
        p.hp = 0;
        p.fainted = true;
        b.sides[s].pokemon_left -= 1;
    }
}

/// Scale every live mon's HP to `pct[side]`% of max.
///
/// The two sides scale independently because the uniform knob cannot reach the
/// region the experiment needs. Measured on CX (`data/switch-eq-v3/`): the one
/// position that nearly solves, `electric/ground vs water/grass`, is a certified
/// 0.0000 for p1 at every uniform setting — Thunderbolt cannot touch Quagsire at
/// all, which is also *why* it is tractable — so its mixture stays LP
/// degeneracy. Handing one side HP is what pulls such a position off the decided
/// boundary while keeping the coarse damage lattice that made it solvable.
///
/// This knob decides whether the position has any strategic content, and the
/// first attempt got it wrong in the interesting direction. At 1 HP every
/// attack is a certain KO, which does collapse chance — but it also makes
/// switching mean "concede a free kill", so equilibrium never switches and the
/// position solves to a decided 0.0 or 1.0. A decided position makes EVERY
/// strategy optimal, so the equilibrium switch rate read off it is LP
/// degeneracy, not an answer.
///
/// The tension needs a mon to SURVIVE the hit it eats on the way in: then
/// "retreat to the resist and pay one hit" trades against "stay and swing".
/// Sitting HP well above one hit's damage but under two keeps chance collapsed
/// (one hit never KOs, two always do) while leaving the trade-off real.
fn scale_hp(b: &mut Battle, pct: [u32; 2]) {
    for s in 0..2 {
        for idx in 0..2 {
            let slot = b.sides[s].party[idx] as usize;
            let p = &mut b.sides[s].roster[slot];
            if !p.fainted {
                p.hp = ((p.maxhp as u32 * pct[s]).div_ceil(100)).max(1) as i32;
            }
        }
    }
}

struct Scenario {
    name: &'static str,
    p1: Vec<PokemonSet>,
    p2: Vec<PokemonSet>,
}

fn scenarios() -> Vec<Scenario> {
    // Type cycles that make switching genuinely load-bearing: each side has
    // one mon that beats one of the foe's and loses to the other, so neither
    // "always attack" nor "always switch" can be an equilibrium.
    let filler = || set("Sunflora", "Razor Leaf", 50);
    vec![
        Scenario {
            name: "fire/water vs grass/water",
            p1: vec![set("Magmar", "Flamethrower", 50), set("Vaporeon", "Surf", 50), filler()],
            p2: vec![set("Tangela", "Giga Drain", 50), set("Golduck", "Surf", 50), filler()],
        },
        Scenario {
            name: "electric/ground vs water/grass",
            p1: vec![set("Electabuzz", "Thunderbolt", 50), set("Sandslash", "Earthquake", 50), filler()],
            p2: vec![set("Quagsire", "Surf", 50), set("Victreebel", "Giga Drain", 50), filler()],
        },
        Scenario {
            name: "ice/fighting vs dragon/psychic",
            p1: vec![set("Jynx", "Ice Punch", 50), set("Primeape", "Karate Chop", 50), filler()],
            p2: vec![set("Dragonair", "Dragon Breath", 50), set("Alakazam", "Psychic", 50), filler()],
        },
    ]
}

fn build(dex: &Dex, sc: &Scenario, hp_pct: [u32; 2]) -> Option<Battle> {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", &sc.p1, &sc.p2).ok()?;
    b.set_log_enabled(false);
    b.choose(dex, 0, "team 1,2,3").ok()?;
    b.choose(dex, 1, "team 1,2,3").ok()?;
    retire_third(&mut b);
    scale_hp(&mut b, hp_pct);
    Some(b)
}

fn bot_switch_mass(b: &Battle, dex: &Dex, iters: u32, seed: u64) -> Option<(f64, usize)> {
    let cfg = RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        c: 1.0,
        hp_buckets: 16,
        ..RmConfig::default()
    };
    let mut search = SkuctSearch::new(b, dex, cfg, seed);
    search.step(dex, iters);
    let acts = search.actions(0);
    let visits = search.visits(0);
    if acts.is_empty() {
        return None;
    }
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return None;
    }
    let sw: u32 = acts
        .iter()
        .zip(visits)
        .filter(|(a, _)| matches!(a, SearchChoice::Switch(_)))
        .map(|(_, v)| *v)
        .sum();
    Some((sw as f64 / total as f64, acts.len()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num = |k: &str, d: usize| -> usize {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let iters = num("--iters", 30_000) as u32;
    let seed = num("--seed", 1) as u64;
    let budget = num("--budget", 400_000);
    let work = num("--work", 20_000_000);
    // `--hp` sets both sides; `--hp1`/`--hp2` override one side each, which is
    // the only knob that reaches an undecided-but-tractable position (see
    // `scale_hp`).
    let hp_both = num("--hp", 60) as u32;
    let hp_pct = [
        num("--hp1", hp_both as usize) as u32,
        num("--hp2", hp_both as usize) as u32,
    ];
    // Spend a whole arm on one position. The scenarios differ by orders of
    // magnitude in how solvable they are, so a sweep over the tractable one is
    // worth more than another uniform pass over all three.
    let only: Option<String> = args
        .iter()
        .position(|a| a == "--scenario")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.to_lowercase());
    // The OOM lever. Each enumerated chance leaf holds a full Battle clone, so
    // the default 100k cap is gigabytes if any step genuinely fans out that
    // wide -- that, not the state memo, is what killed the 8 GB box at 40% HP.
    let leaf_cap = num("--leaf-cap", 20_000);

    let dex = load_dex();
    let _ = Path::new(".");
    println!(
        "pure attack/switch 2v2 positions at p1 {}% / p2 {}% HP; bot skuct:{iters} seed {seed}; \
         solver budget {budget} work {work}\n",
        hp_pct[0], hp_pct[1]
    );
    println!(
        "  {:<32} {:>9} {:>9} {:>9} {:>8} {:>8} {:>7}",
        "scenario", "value", "certwidth", "entrygap", "eq_sw", "bot_sw", "delta"
    );

    let mut rows = 0;
    let mut matched = 0;
    for sc in scenarios() {
        if let Some(pat) = &only {
            if !sc.name.to_lowercase().contains(pat.as_str()) {
                continue;
            }
        }
        matched += 1;
        eprintln!("solving: {}", sc.name);
        let Some(b) = build(&dex, &sc, hp_pct) else {
            println!("  {:<32} BUILD FAILED (illegal set or pick)", sc.name);
            continue;
        };
        let cfg = ExactConfig {
            state_budget: budget,
            work_budget: work,
            leaf_cap,
            eps: 0.0,
            stall_gain: 0.0,
            horizons: &[2, 4, 8, 12, 20, 32],
            ..ExactConfig::default()
        };
        let mut solver = ExactSolver::new(&dex, cfg);
        let Some(eq) = solver.solve_root(&b) else {
            println!("  {:<32} UNSOLVED (budget exhausted)", sc.name);
            continue;
        };
        let Some((bot_sw, n_acts)) = bot_switch_mass(&b, &dex, iters, seed) else {
            println!("  {:<32} bot produced no root policy", sc.name);
            continue;
        };
        let eq_sw = eq.switch_mass(0);
        // A decided position (value at 0 or 1) makes every strategy optimal, so
        // its equilibrium switch mass is LP degeneracy rather than an answer.
        let decided = eq.value <= 1e-9 || eq.value >= 1.0 - 1e-9;
        let flag = if eq.entry_gap > 1e-6 {
            "  UNUSABLE: payoff matrix still bracketed"
        } else if decided {
            "  UNUSABLE: position already decided, every strategy optimal"
        } else {
            ""
        };
        println!(
            "  {:<32} {:>9.4} {:>9.4} {:>9.4} {:>8.3} {:>8.3} {:>+7.3}   ({} root actions)",
            sc.name,
            eq.value,
            eq.certified.width(),
            eq.entry_gap,
            eq_sw,
            bot_sw,
            bot_sw - eq_sw,
            n_acts
        );
        if !flag.is_empty() {
            println!("  {:<32}{}", "", flag);
        }
        rows += 1;
        let _ = std::io::stdout().flush();
    }

    // A typo'd filter must fail loudly rather than read as "nothing solved":
    // the last two CX submissions were lost to silent environment errors that
    // looked like results.
    if matched == 0 {
        eprintln!(
            "--scenario {:?} matched none of {:?}",
            only.unwrap_or_default(),
            scenarios().iter().map(|s| s.name).collect::<Vec<_>>()
        );
        std::process::exit(64);
    }
    if rows == 0 {
        println!("\nno position solved — the comparison says nothing yet.");
        return;
    }
    println!(
        "\nRead only rows with certwidth and entrygap ~0: a mixture solved off a\n\
         bracketed payoff matrix is not an equilibrium of anything. delta < 0 means\n\
         the bot switches LESS than equilibrium (its disagreement with humans would\n\
         then be the bot's error); delta ~ 0 means the bot is right and the human\n\
         switch rate is the thing that needs explaining. Rows flagged UNUSABLE say\n\
         nothing either way — a decided position makes every strategy optimal, so its\n\
         equilibrium switch rate is an artifact of which vertex the LP happened to pick."
    );
}
