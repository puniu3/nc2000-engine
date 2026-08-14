//! M18a prior probe: what the PUCT prior slot *does*, beyond the duel score.
//!
//! Two instruments, one run, on self-play positions (the human spectator
//! corpus is not redistributable, so `human_agreement`'s corpus arm cannot
//! run here — see the caveat this prints):
//!
//! 1. **Footprint vs the seed floor** — the repo's standing screen
//!    (`tools/eval-candidate-screen.py`). At each captured position, ask the
//!    baseline for its top-1 under two different agent seeds (the floor: how
//!    often the search's own noise moves the answer) and the candidate under
//!    the first seed. A candidate that moves fewer decisions than the seed
//!    does cannot be resolved by any downstream instrument.
//!
//! 2. **Action-class mechanism** — the M16b cluster-2 question. Self-play
//!    class rates (damaging / status / switch) and the top status moves each
//!    configuration actually plays, so "does the prior reach the multi-turn
//!    plans" is answered by counts rather than by the duel score.
//!
//!   cargo run --release -p nc2000-bot --example prior_probe -- \
//!       --games 40 --iters 1000 --puct 2.0 --tau 0.15 --seed 5 [--positions 400]

use std::collections::HashMap;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{Agent, PriorKind, RmAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::{Category, Dex};
use nc2000_engine::state::Battle;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Class {
    Damage,
    Status,
    Switch,
    Other,
}

fn classify(dex: &Dex, c: SearchChoice) -> Class {
    match c {
        SearchChoice::Move(id) => match dex.move_static(id).category {
            Category::Status => Class::Status,
            _ => Class::Damage,
        },
        SearchChoice::Switch(_) => Class::Switch,
        _ => Class::Other,
    }
}

fn cfg_for(iters: u32, prior: PriorKind, puct: f64, tau: f64, status_bonus: f64) -> RmConfig {
    RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        prior,
        puct: if prior == PriorKind::Off { 0.0 } else { puct },
        prior_tau: tau,
        prior_status_bonus: status_bonus,
        ..Default::default()
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

struct Tally {
    class: HashMap<Class, usize>,
    status_moves: HashMap<String, usize>,
    decisions: usize,
}

impl Tally {
    fn new() -> Tally {
        Tally { class: HashMap::new(), status_moves: HashMap::new(), decisions: 0 }
    }

    fn record(&mut self, dex: &Dex, c: SearchChoice) {
        self.decisions += 1;
        let k = classify(dex, c);
        *self.class.entry(k).or_insert(0) += 1;
        if k == Class::Status {
            if let SearchChoice::Move(id) = c {
                *self.status_moves.entry(dex.moves.key(id).to_string()).or_insert(0) += 1;
            }
        }
    }

    fn rate(&self, k: Class) -> f64 {
        if self.decisions == 0 {
            return 0.0;
        }
        *self.class.get(&k).unwrap_or(&0) as f64 / self.decisions as f64
    }

    fn report(&self, label: &str) {
        println!(
            "  {label:22}  decisions {:5}   damage {:.3}  status {:.3}  switch {:.3}",
            self.decisions,
            self.rate(Class::Damage),
            self.rate(Class::Status),
            self.rate(Class::Switch),
        );
        let mut top: Vec<(&String, &usize)> = self.status_moves.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let head: Vec<String> =
            top.iter().take(8).map(|(k, v)| format!("{k}={v}")).collect();
        println!("  {:22}  top status: {}", "", head.join(" "));
    }
}

/// Self-play `games` battles with `cfg` on both sides, tallying every root
/// choice. Positions (state + side + legal set) are captured when `capture`.
#[allow(clippy::type_complexity)]
fn self_play(
    dex: &Dex,
    teams: &[Vec<PokemonSet>],
    cfg: &RmConfig,
    games: usize,
    seed: u64,
    capture: bool,
) -> (Tally, Vec<(Battle, usize, Vec<SearchChoice>)>) {
    let mut tally = Tally::new();
    let mut positions = Vec::new();
    let mut sched = SplitMix64::new(seed);
    for g in 0..games {
        let t1 = sched.below(teams.len());
        let t2 = sched.below(teams.len());
        let bseed = sched.battle_seed();
        let mut battle =
            Battle::from_fixture(dex, &bseed, &teams[t1], &teams[t2]).unwrap();
        battle.set_log_enabled(false);
        let mut agents: Vec<RmAgent> = (0..2)
            .map(|s| RmAgent::new(cfg.clone(), seed ^ ((g * 2 + s) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
            .collect();
        loop {
            if battle.outcome().is_some() || battle.turn > 500 {
                break;
            }
            let mut picks = [None, None];
            for s in 0..2 {
                let cs = battle.legal_choices(dex, s);
                if cs.is_empty() {
                    continue;
                }
                // Team preview is a different decision problem (120 ordered
                // picks, UCB1+argmax by design) — out of scope for the probe.
                if matches!(cs.first(), Some(SearchChoice::Team(_))) {
                    picks[s] = Some(agents[s].choose(&battle, dex, s, &cs));
                    continue;
                }
                if capture && cs.len() > 1 {
                    let mut snap = battle.clone();
                    snap.set_log_enabled(false);
                    positions.push((snap, s, cs.clone()));
                }
                let c = agents[s].choose(&battle, dex, s, &cs);
                if cs.len() > 1 {
                    tally.record(dex, c);
                }
                picks[s] = Some(c);
            }
            if picks == [None, None] {
                break;
            }
            battle.apply_choices(dex, picks).unwrap();
        }
    }
    (tally, positions)
}

fn top1(agent: &mut RmAgent, b: &Battle, dex: &Dex, side: usize, cs: &[SearchChoice]) -> SearchChoice {
    let p = agent.root_policy(b, dex, side, cs);
    let mut best = 0;
    for i in 1..p.len() {
        if p[i] > p[best] {
            best = i;
        }
    }
    cs[best]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str, d: &str| -> String {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| d.to_string())
    };
    let games: usize = get("--games", "40").parse().unwrap();
    let iters: u32 = get("--iters", "1000").parse().unwrap();
    let puct: f64 = get("--puct", "2.0").parse().unwrap();
    let tau: f64 = get("--tau", "0.15").parse().unwrap();
    let seed: u64 = get("--seed", "5").parse().unwrap();
    let max_positions: usize = get("--positions", "400").parse().unwrap();

    let dex = load_dex();
    let teams = load_team_pool();
    let sb: f64 = get("--status-bonus", "0.0").parse().unwrap();
    let base = cfg_for(iters, PriorKind::Off, 0.0, tau, 0.0);
    let cand = cfg_for(iters, PriorKind::Greedy, puct, tau, sb);
    let unif = cfg_for(iters, PriorKind::Uniform, puct, tau, 0.0);

    println!("M18a prior probe — iters {iters}, puct {puct}, tau {tau}, status_bonus {sb}, seed {seed}");
    println!("positions: SELF-PLAY (the 570-battle human corpus is not in-tree; this is");
    println!("           the distribution M16a warned is not the human one)\n");

    println!("[2] action-class mechanism ({games} self-play games per arm)");
    let (t_base, positions) = self_play(&dex, &teams, &base, games, seed, true);
    t_base.report("baseline skuct");
    let (t_unif, _) = self_play(&dex, &teams, &unif, games, seed, false);
    t_unif.report("puct uniform prior");
    let (t_cand, _) = self_play(&dex, &teams, &cand, games, seed, false);
    t_cand.report("puct greedy prior");

    // ---- footprint screen on the baseline-generated positions
    let mut pos = positions;
    if pos.len() > max_positions {
        let mut rng = SplitMix64::new(seed ^ 0xF00D);
        // deterministic thinning, order preserved
        let keep = max_positions as f64 / pos.len() as f64;
        pos.retain(|_| rng.next_f64() < keep);
        pos.truncate(max_positions);
    }
    println!("\n[1] footprint vs seed floor ({} positions)", pos.len());
    let (mut moved_cand, mut moved_seed) = (0usize, 0usize);
    let mut shift: HashMap<(Class, Class), usize> = HashMap::new();
    for (i, (b, side, cs)) in pos.iter().enumerate() {
        let s1 = seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407);
        let s2 = seed ^ (i as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25);
        let a = top1(&mut RmAgent::new(base.clone(), s1), b, &dex, *side, cs);
        let a2 = top1(&mut RmAgent::new(base.clone(), s2), b, &dex, *side, cs);
        let c = top1(&mut RmAgent::new(cand.clone(), s1), b, &dex, *side, cs);
        if a != a2 {
            moved_seed += 1;
        }
        if a != c {
            moved_cand += 1;
            *shift.entry((classify(&dex, a), classify(&dex, c))).or_insert(0) += 1;
        }
    }
    let n = pos.len().max(1) as f64;
    let (fc, fs) = (moved_cand as f64 / n, moved_seed as f64 / n);
    println!("  candidate top-1 change rate : {fc:.3}  ({moved_cand}/{})", pos.len());
    println!("  seed-flip floor             : {fs:.3}  ({moved_seed}/{})", pos.len());
    println!(
        "  ratio (candidate / floor)   : {:.2}x  -> {}",
        if fs > 0.0 { fc / fs } else { f64::NAN },
        if fs > 0.0 && fc / fs >= 1.5 { "resolvable by duel" } else { "below/near the floor" }
    );
    let mut sh: Vec<(&(Class, Class), &usize)> = shift.iter().collect();
    sh.sort_by(|a, b| b.1.cmp(a.1));
    let head: Vec<String> = sh
        .iter()
        .take(6)
        .map(|((f, t), v)| format!("{f:?}->{t:?}={v}"))
        .collect();
    println!("  class shifts                : {}", head.join(" "));
}
