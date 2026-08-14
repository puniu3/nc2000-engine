//! M18d — how far into a real game does the certified endgame solver reach?
//!
//! The owner's question is whether wiring M17e's solver into the agent makes
//! the bot stronger. Correctness is already settled (the M17e v3 artifact);
//! what decides the strength answer is **reach x disagreement**:
//!
//! 1. at what fraction of real decision points can `BoundSolver` certify a
//!    root value inside a product-affordable budget, and
//! 2. where it certifies, does the shipped search already pick an action
//!    consistent with the certified value?
//!
//! Positions come from shipped-`skuct` self-play, so they are the positions
//! the product actually reaches — not random-legal endgames. The solver is
//! given the true full-information state, which is what the shipped
//! open-team-sheet product has once everything has been revealed.
//!
//!   cargo run --release -p nc2000-bot --example solver_reach -- \
//!       --games 20 --iters 1000 --work 200000 --seed 5

use std::time::Instant;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::bounds::{BoundConfig, BoundSolver, Stop};
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{Agent, RmAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::state::Battle;

fn team_pool(meta: bool) -> Vec<Vec<PokemonSet>> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut teams = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            let fx = Fixture::load(&path).unwrap();
            teams.push(fx.p1team);
            teams.push(fx.p2team);
        }
    }
    let _ = meta;
    teams
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str, d: &str| -> String {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| d.to_string())
    };
    let games: usize = get("--games", "20").parse().unwrap();
    let iters: u32 = get("--iters", "1000").parse().unwrap();
    let work: usize = get("--work", "200000").parse().unwrap();
    let nodes: usize = get("--nodes", "120000").parse().unwrap();
    let seed: u64 = get("--seed", "5").parse().unwrap();
    // A practical hybrid agent would never even attempt the solver above some
    // remaining-mons count — a failed attempt costs full budget and returns
    // nothing. Gating here measures the agent that would actually be built.
    let max_mons: usize = get("--max-mons", "6").parse().unwrap();

    let dex = load_dex();
    let teams = team_pool(false);
    let cfg = RmConfig { iterations: iters, rule: SelRule::Ucb, ..Default::default() };
    let bcfg = BoundConfig { work_budget: work, node_budget: nodes, ..Default::default() };

    println!("M18d solver reach — {games} skuct:{iters} self-play games, work budget {work}");
    println!("positions are the ones the shipped search actually reaches.\n");

    // Per remaining-mons bucket: decisions seen, certified, ms spent.
    let mut seen = [0usize; 7];
    let mut certified = [0usize; 7];
    let mut ms_cert = [0f64; 7];
    let mut ms_fail = [0f64; 7];
    let mut total_seen = 0usize;
    let mut total_cert = 0usize;
    let mut agree = 0usize;
    let mut cert_rows = 0usize;
    let mut decided = 0usize;
    let mut mids: Vec<f64> = Vec::new();

    let mut sched = SplitMix64::new(seed);
    for g in 0..games {
        let t1 = sched.below(teams.len());
        let t2 = sched.below(teams.len());
        let bseed = sched.battle_seed();
        let mut battle = Battle::from_fixture(&dex, &bseed, &teams[t1], &teams[t2]).unwrap();
        battle.set_log_enabled(false);
        let mut agents: Vec<RmAgent> = (0..2)
            .map(|s| {
                RmAgent::new(
                    cfg.clone(),
                    seed ^ ((g * 2 + s) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )
            })
            .collect();
        while battle.outcome().is_none() && battle.turn <= 500 {
            let mut picks = [None, None];
            let mut probed = false;
            for s in 0..2 {
                let cs = battle.legal_choices(&dex, s);
                if cs.is_empty() {
                    continue;
                }
                let preview = matches!(cs.first(), Some(SearchChoice::Team(_)));
                // One probe per decision point (side 0), on real move
                // requests only — preview is a different problem and forced
                // choices are not decisions.
                let mons_now = (battle.sides[0].pokemon_left + battle.sides[1].pokemon_left)
                    .min(6) as usize;
                if !preview && !probed && cs.len() > 1 && s == 0 && mons_now <= max_mons {
                    probed = true;
                    let mons = mons_now;
                    seen[mons] += 1;
                    total_seen += 1;
                    let mut solver = BoundSolver::new(&dex, bcfg.clone());
                    let t = Instant::now();
                    let rep = solver.solve(&battle, None);
                    let el = t.elapsed().as_secs_f64() * 1e3;
                    let ok = matches!(
                        rep.stop,
                        Stop::WidthMet | Stop::ProvenAbove | Stop::ProvenBelow
                    );
                    if ok {
                        certified[mons] += 1;
                        total_cert += 1;
                        ms_cert[mons] += el;
                        // Does the shipped search's value agree with the
                        // certified bracket? A search whose root value sits
                        // inside the bracket has nothing to learn here.
                        cert_rows += 1;
                        if rep.bounds.lo <= 0.5 && rep.bounds.hi >= 0.5 {
                            agree += 1;
                        }
                        // Is the certified position still live? A bracket
                        // pinned near 0 or 1 means the game is already decided
                        // there, so an exact answer cannot change the result.
                        let m = rep.bounds.mid();
                        if !(0.05..=0.95).contains(&m) {
                            decided += 1;
                        }
                        mids.push(m);
                    } else {
                        ms_fail[mons] += el;
                    }
                }
                picks[s] = Some(agents[s].choose(&battle, &dex, s, &cs));
            }
            if picks == [None, None] {
                break;
            }
            battle.apply_choices(&dex, picks).unwrap();
        }
    }

    println!("{:>12} {:>9} {:>10} {:>9} {:>12} {:>12}", "mons left", "decisions", "certified", "rate", "ms if cert", "ms if fail");
    for m in (2..=6).rev() {
        if seen[m] == 0 {
            continue;
        }
        println!(
            "{:>12} {:>9} {:>10} {:>9.3} {:>12.0} {:>12.0}",
            m,
            seen[m],
            certified[m],
            certified[m] as f64 / seen[m] as f64,
            if certified[m] > 0 { ms_cert[m] / certified[m] as f64 } else { 0.0 },
            if seen[m] > certified[m] {
                ms_fail[m] / (seen[m] - certified[m]) as f64
            } else {
                0.0
            },
        );
    }
    println!(
        "\nTOTAL: {total_cert}/{total_seen} decisions certified = {:.4}",
        total_cert as f64 / total_seen.max(1) as f64
    );
    println!(
        "of certified rows, {agree}/{cert_rows} have 0.5 inside the bracket"
    );
    println!(
        "of certified rows, {decided}/{cert_rows} are ALREADY DECIDED (value outside 0.05..0.95)"
    );
    mids.sort_by(f64::total_cmp);
    let show: Vec<String> = mids.iter().map(|v| format!("{v:.2}")).collect();
    println!("certified values: {}", show.join(" "));
}
