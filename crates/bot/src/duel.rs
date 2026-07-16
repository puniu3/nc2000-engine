//! Seed-paired parallel duels — the shared evaluation harness behind the
//! arena example and the SPSA tuner.
//!
//! Each pairing is played twice with sides swapped on the same battle seed;
//! agent seeds derive from the game index only, so results are fully
//! deterministic for a given `base_seed` regardless of thread count.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::rng::SplitMix64;
use crate::runner::{play_game, GameResult};

pub type AgentBuilder<'a> = &'a (dyn Fn(u64) -> Box<dyn Agent> + Sync);

#[derive(Clone, Copy, Debug)]
pub struct DuelSpec {
    /// Rounded up to even (games are paired).
    pub games: usize,
    pub base_seed: u64,
    pub threads: usize,
    pub max_turns: u16,
    /// Progress lines on stderr every 10 games.
    pub progress: bool,
    /// Run the outer battles with the protocol log ON (M10b blind agents:
    /// the observer's trace-free reveal channel reads `battle.log`). Search
    /// clones always disable the log themselves; log content never affects
    /// battle state, so results stay comparable across this flag.
    pub log_on: bool,
}

impl DuelSpec {
    pub fn new(games: usize, base_seed: u64) -> Self {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        DuelSpec { games, base_seed, threads, max_turns: 500, progress: false, log_on: false }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DuelStats {
    pub games: usize,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    /// Agent A's mean score (win 1 / tie 0.5 / loss 0).
    pub score: f64,
    /// 95% CI half-width on `score`.
    pub ci95: f64,
    pub avg_turns: f64,
    pub secs: f64,
    /// Mean thinking time per `choose()` call — the equal-wall-clock
    /// evidence for budget-matched comparisons.
    pub a_ms_per_move: f64,
    pub b_ms_per_move: f64,
}

/// Delegating wrapper that accumulates time spent inside `choose`.
struct Timed {
    inner: Box<dyn Agent>,
    ns: u64,
    moves: u64,
}

impl Agent for Timed {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[nc2000_engine::battle::SearchChoice],
    ) -> nc2000_engine::battle::SearchChoice {
        let t = Instant::now();
        let c = self.inner.choose(battle, dex, side, choices);
        self.ns += t.elapsed().as_nanos() as u64;
        self.moves += 1;
        c
    }
}

/// One scheduled game: pool team indices, battle seed, whether agent A is p1.
struct GameSpec {
    team_p1: usize,
    team_p2: usize,
    battle_seed: String,
    a_is_p1: bool,
}

pub fn run_duel(
    dex: &Dex,
    teams: &[Vec<PokemonSet>],
    build_a: AgentBuilder,
    build_b: AgentBuilder,
    spec: DuelSpec,
) -> DuelStats {
    let games = spec.games + spec.games % 2;
    let mut sched_rng = SplitMix64::new(spec.base_seed);
    let mut specs = Vec::with_capacity(games);
    for _ in 0..games / 2 {
        let t1 = sched_rng.below(teams.len());
        let t2 = sched_rng.below(teams.len());
        let seed = sched_rng.battle_seed();
        specs.push(GameSpec { team_p1: t1, team_p2: t2, battle_seed: seed.clone(), a_is_p1: true });
        specs.push(GameSpec { team_p1: t1, team_p2: t2, battle_seed: seed, a_is_p1: false });
    }

    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();

    // per-game record: (a_score, turns, a_ns, a_moves, b_ns, b_moves)
    type Rec = (f64, u16, u64, u64, u64, u64);
    let mut results: Vec<Rec> = Vec::with_capacity(games);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..spec.threads {
            let (specs, cursor, done) = (&specs, &cursor, &done);
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, Rec)> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    let g = &specs[i];
                    // agent seeds derive from game index only -> thread-count invariant
                    let sa = spec.base_seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407);
                    let sb = spec.base_seed ^ (i as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25);
                    let mut agent_a = Timed { inner: build_a(sa), ns: 0, moves: 0 };
                    let mut agent_b = Timed { inner: build_b(sb), ns: 0, moves: 0 };
                    let mut battle = Battle::from_fixture(
                        dex,
                        &g.battle_seed,
                        &teams[g.team_p1],
                        &teams[g.team_p2],
                    )
                    .unwrap();
                    battle.set_log_enabled(spec.log_on);
                    let (p1, p2): (&mut dyn Agent, &mut dyn Agent) = if g.a_is_p1 {
                        (&mut agent_a, &mut agent_b)
                    } else {
                        (&mut agent_b, &mut agent_a)
                    };
                    let res = play_game(dex, &mut battle, &mut [p1, p2], spec.max_turns).unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let a_score = if g.a_is_p1 { p1_score } else { 1.0 - p1_score };
                    out.push((
                        i,
                        (a_score, battle.turn, agent_a.ns, agent_a.moves, agent_b.ns, agent_b.moves),
                    ));
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if spec.progress && (d % 10 == 0 || d == specs.len()) {
                        eprintln!(
                            "  {d}/{} games ({:.0}s)",
                            specs.len(),
                            t0.elapsed().as_secs_f64()
                        );
                    }
                }
                out
            }));
        }
        let mut all: Vec<(usize, Rec)> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all.into_iter().map(|(_, r)| r).collect();
    });

    let n = results.len() as f64;
    let wins = results.iter().filter(|r| r.0 == 1.0).count();
    let losses = results.iter().filter(|r| r.0 == 0.0).count();
    let ties = results.len() - wins - losses;
    let score: f64 = results.iter().map(|r| r.0).sum::<f64>() / n;
    let var: f64 = results.iter().map(|r| (r.0 - score).powi(2)).sum::<f64>() / (n - 1.0);
    let ms_per_move = |ns: u64, moves: u64| ns as f64 / 1e6 / (moves.max(1)) as f64;
    let (a_ns, a_moves) = results.iter().fold((0, 0), |(ns, m), r| (ns + r.2, m + r.3));
    let (b_ns, b_moves) = results.iter().fold((0, 0), |(ns, m), r| (ns + r.4, m + r.5));
    DuelStats {
        games: results.len(),
        wins,
        losses,
        ties,
        score,
        ci95: 1.96 * (var / n).sqrt(),
        avg_turns: results.iter().map(|r| r.1 as f64).sum::<f64>() / n,
        secs: t0.elapsed().as_secs_f64(),
        a_ms_per_move: ms_per_move(a_ns, a_moves),
        b_ms_per_move: ms_per_move(b_ns, b_moves),
    }
}
