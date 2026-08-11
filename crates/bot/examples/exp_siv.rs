//! EXP-SIV v1 — signature-scoped information-value experiment driver
//! (docs/EXP-signature-info-value.md).
//!
//! Roles: X = the pool side, holding variant v of a signature-matched pair
//! (A, B); X's own information condition is fixed in every arm (the
//! opponent's team is pinned truth — the product `OpenAgent`). Y = the
//! verification adversary on a fixed meta-pool team g; only Y's BELIEF
//! about X's sets varies between arms:
//!
//!   b=a / b=b   truth or wrong pin (`Belief::pinned`; the wrong pin is
//!               frozen — `freeze_pinned` — so contradicting reveals are
//!               held against, the stubborn envelope semantics)
//!   b=weak      C2 instrument-teeth control: a frozen pin on a gutted
//!               signature-compatible team
//!   b=open      C1 identity control: the product `OpenAgent`
//!               (`Belief::pinned_from_battle`) — statistically ≡ b=truth
//!
//! Every cell is one resumable JSON (atomic tmp+rename). The battle-seed
//! list derives from `--seed` alone and agent seeds from the game index
//! alone, so all cells share seeds (common random numbers) and results are
//! deterministic at any thread count.
//!
//! Modes:
//!   (default)    run all scheduled cells missing from --out
//!   --screen     Y's preview-policy divergence, believed-A vs believed-B
//!                (the manipulation check; writes screen-i<budget>.json)
//!   --summarize  aggregate cells in --out into envelope / control stats
//!
//! Example:
//!   cargo run --release -p nc2000-bot --example exp_siv -- \
//!     --pairs data/exp-siv-v1/pairs --gauntlet 0,4 --budget 1000 \
//!     --games 128 --seed 20260811 --out data/exp-siv-v1/cells --controls

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{
    play_game, Agent, Belief, BlindSearch, GameResult, Observer, OpenAgent, RmConfig, SplitMix64,
};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

// ---------------------------------------------------------------- pair file

#[derive(serde::Deserialize)]
struct PairFile {
    id: String,
    #[allow(dead_code)]
    axis: String,
    #[allow(dead_code)]
    base: String,
    a: Vec<PokemonSet>,
    b: Vec<PokemonSet>,
    #[serde(default)]
    weak: Option<Vec<PokemonSet>>,
}

/// Preview-public signature: (species, level, gender, item presence) per
/// display slot. Variants must match exactly or the belief's preview
/// filter would separate them (and `build_refs` would refuse the pin).
fn signature(sets: &[PokemonSet]) -> Vec<(String, u8, String, bool)> {
    sets.iter()
        .map(|s| {
            (
                s.species.clone(),
                s.level,
                s.gender.clone().unwrap_or_default(),
                !s.item.is_empty(),
            )
        })
        .collect()
}

/// Fail-closed team check through the exact machinery the arms use:
/// `Belief::pinned_checked` validates legality and aligns the sets against
/// a preview observer of themselves.
fn check_team(dex: &Dex, label: &str, sets: &[PokemonSet]) {
    let battle = Battle::from_fixture(dex, "1,2,3,4", sets, sets)
        .unwrap_or_else(|e| panic!("{label}: engine rejects team: {e:?}"));
    let obs = Observer::new(&battle, 1);
    Belief::pinned_checked(dex, label, sets, &obs)
        .unwrap_or_else(|e| panic!("{label}: pin check failed: {e}"));
}

// ----------------------------------------------------------------- Y agent

/// The wrong-pin / truth-pin adversary: the open-sheet machinery with a
/// caller-supplied candidate instead of the battle's truth. `frozen` =
/// stubborn (EXP-SIV envelope semantics — reveals never update the pin).
struct PinAgent {
    cfg: RmConfig,
    rng: SplitMix64,
    sets: Vec<PokemonSet>,
    frozen: bool,
    game: Option<PinGame>,
}

struct PinGame {
    side: usize,
    last_turn: u16,
    observer: Observer,
    belief: Belief,
}

impl PinAgent {
    fn new(cfg: RmConfig, sets: Vec<PokemonSet>, frozen: bool, seed: u64) -> Self {
        PinAgent { cfg, rng: SplitMix64::new(seed), sets, frozen, game: None }
    }
}

impl Agent for PinAgent {
    fn name(&self) -> String {
        format!("pin{}:{}", if self.frozen { "!" } else { "" }, self.cfg.iterations)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = matches!(choices[0], SearchChoice::Team(_));
        let stale = match &self.game {
            None => true,
            Some(g) => g.side != side || is_preview || battle.turn < g.last_turn,
        };
        if stale {
            // The pin aligns against preview-public facts; a mid-game
            // (re)build would see a truncated roster and must not happen.
            assert!(is_preview, "PinAgent must enter at team preview");
            let observer = Observer::new(battle, side);
            let mut belief = Belief::pinned(dex, "pin", &self.sets, &observer);
            if self.frozen {
                belief.freeze_pinned();
            }
            self.game = Some(PinGame { side, last_turn: battle.turn, observer, belief });
        }
        {
            let g = self.game.as_mut().unwrap();
            g.last_turn = battle.turn;
            g.observer.observe(battle, dex);
            g.belief.sync(dex, &g.observer);
        }
        if choices.len() == 1 {
            return choices[0];
        }
        // mirror blind::search_choose: fresh stepped search per decision
        let g = self.game.as_ref().unwrap();
        let mut bs = BlindSearch::new(battle, dex, self.cfg.clone(), side, self.rng.next());
        debug_assert_eq!(bs.actions(), choices, "root action set drifted from caller's choices");
        for _ in 0..self.cfg.iterations {
            bs.step_one(dex, &g.belief, &g.observer);
        }
        bs.best().expect("search called with a non-empty choice list")
    }
}

// ------------------------------------------------------------------- cells

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Truth,
    Wrong,
    Weak,
    Open,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Truth | Arm::Wrong => unreachable!("labeled via belief letter"),
            Arm::Weak => "weak",
            Arm::Open => "open",
        }
    }
}

#[derive(Clone)]
struct CellSpec {
    pair_idx: usize,
    g_idx: usize,
    truth_is_a: bool,
    /// "a" / "b" / "weak" / "open" — which belief Y holds.
    belief: String,
}

impl CellSpec {
    fn file_name(&self, pair_id: &str, budget: u32) -> String {
        format!(
            "cell-{}-g{}-t{}-b{}-i{}.json",
            pair_id,
            self.g_idx,
            if self.truth_is_a { "a" } else { "b" },
            self.belief,
            budget
        )
    }
}

fn agent_cfg(budget: u32) -> RmConfig {
    // exactly the arena `open`/`blind` construction (defaults c=1.0, b=16)
    RmConfig {
        iterations: budget,
        rule: SelRule::Ucb,
        c: 1.0,
        hp_buckets: 16,
        ..Default::default()
    }
}

const K_X: u64 = 0xA24B_AED4_963E_E407;
const K_Y: u64 = 0x9FB2_1C65_1E98_DF25;

#[allow(clippy::too_many_arguments)]
fn run_cell(
    dex: &Dex,
    x_sets: &[PokemonSet],
    g_sets: &[PokemonSet],
    pin_sets: Option<&[PokemonSet]>, // None = Open arm
    frozen: bool,
    budget: u32,
    games: usize,
    base_seed: u64,
    threads: usize,
    max_turns: u16,
) -> (Vec<f64>, Vec<u16>, usize, f64, f64, f64) {
    let blocks = games.div_ceil(2);
    let seeds: Vec<String> = {
        let mut r = SplitMix64::new(base_seed);
        (0..blocks).map(|_| r.battle_seed()).collect()
    };
    let cursor = AtomicUsize::new(0);
    let t0 = Instant::now();
    // per-game record: (x_score, turns, capped, x_ns, x_moves, y_ns, y_moves)
    type Rec = (f64, u16, bool, u64, u64, u64, u64);
    let mut results: Vec<Rec> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let (seeds, cursor) = (&seeds, &cursor);
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, Rec)> = Vec::new();
                loop {
                    let idx = cursor.fetch_add(1, Ordering::Relaxed);
                    if idx >= blocks * 2 {
                        break;
                    }
                    let block = idx / 2;
                    let x_is_p1 = idx % 2 == 0;
                    let sx = base_seed ^ (idx as u64).wrapping_mul(K_X);
                    let sy = base_seed ^ (idx as u64).wrapping_mul(K_Y);
                    let cfg = agent_cfg(budget);
                    let mut agent_x: Box<dyn Agent> = Box::new(OpenAgent::new(cfg.clone(), None, sx));
                    let mut agent_y: Box<dyn Agent> = match pin_sets {
                        None => Box::new(OpenAgent::new(cfg.clone(), None, sy)),
                        Some(p) => Box::new(PinAgent::new(cfg, p.to_vec(), frozen, sy)),
                    };
                    let (p1_sets, p2_sets) =
                        if x_is_p1 { (x_sets, g_sets) } else { (g_sets, x_sets) };
                    let mut battle =
                        Battle::from_fixture(dex, &seeds[block], p1_sets, p2_sets).unwrap();
                    battle.set_log_enabled(true); // observer trace channel
                    let (mut x_ns, mut y_ns, mut x_mv, mut y_mv) = (0u64, 0u64, 0u64, 0u64);
                    let res = {
                        let mut timed_x = TimedAgent {
                            inner: &mut *agent_x,
                            ns: &mut x_ns,
                            moves: &mut x_mv,
                        };
                        let mut timed_y = TimedAgent {
                            inner: &mut *agent_y,
                            ns: &mut y_ns,
                            moves: &mut y_mv,
                        };
                        let (p1, p2): (&mut dyn Agent, &mut dyn Agent) = if x_is_p1 {
                            (&mut timed_x, &mut timed_y)
                        } else {
                            (&mut timed_y, &mut timed_x)
                        };
                        play_game(dex, &mut battle, &mut [p1, p2], max_turns).unwrap()
                    };
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let x_score = if x_is_p1 { p1_score } else { 1.0 - p1_score };
                    out.push((
                        idx,
                        (
                            x_score,
                            battle.turn,
                            matches!(res, GameResult::TurnCapped),
                            x_ns,
                            x_mv,
                            y_ns,
                            y_mv,
                        ),
                    ));
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
    let scores: Vec<f64> = results.iter().map(|r| r.0).collect();
    let turns: Vec<u16> = results.iter().map(|r| r.1).collect();
    let caps = results.iter().filter(|r| r.2).count();
    let (x_ns, x_mv) = results.iter().fold((0u64, 0u64), |(n, m), r| (n + r.3, m + r.4));
    let (y_ns, y_mv) = results.iter().fold((0u64, 0u64), |(n, m), r| (n + r.5, m + r.6));
    let ms = |ns: u64, mv: u64| ns as f64 / 1e6 / mv.max(1) as f64;
    (scores, turns, caps, ms(x_ns, x_mv), ms(y_ns, y_mv), t0.elapsed().as_secs_f64())
}

/// Borrowing timing wrapper (the duel harness's `Timed`, example-local).
struct TimedAgent<'a> {
    inner: &'a mut dyn Agent,
    ns: &'a mut u64,
    moves: &'a mut u64,
}

impl Agent for TimedAgent<'_> {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let t = Instant::now();
        let c = self.inner.choose(battle, dex, side, choices);
        *self.ns += t.elapsed().as_nanos() as u64;
        *self.moves += 1;
        c
    }
}

// -------------------------------------------------------------- summarize

fn block_scores(games: &[f64]) -> Vec<f64> {
    games.chunks_exact(2).map(|p| (p[0] + p[1]) / 2.0).collect()
}

fn diff_stats(x: &[f64], y: &[f64]) -> (f64, f64, usize) {
    assert_eq!(x.len(), y.len(), "paired cells have different block counts");
    let d: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    let n = d.len();
    let mean = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean, 1.96 * (var / n as f64).sqrt(), n)
}

fn pooled_stats(diffs: &[f64]) -> (f64, f64, usize) {
    let n = diffs.len();
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let var = diffs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean, 1.96 * (var / n as f64).sqrt(), n)
}

// ------------------------------------------------------------------ screen

fn preview_policy(
    dex: &Dex,
    battle: &Battle,
    belief_sets: &[PokemonSet],
    budget: u32,
    seed: u64,
) -> (Vec<f64>, usize) {
    let obs = Observer::new(battle, 1);
    let belief = Belief::pinned(dex, "screen", belief_sets, &obs);
    let mut bs = BlindSearch::new(battle, dex, agent_cfg(budget), 1, seed);
    for _ in 0..budget {
        bs.step_one(dex, &belief, &obs);
    }
    let visits = bs.visits();
    let total: u32 = visits.iter().sum();
    let policy: Vec<f64> = visits.iter().map(|&v| v as f64 / total.max(1) as f64).collect();
    let top1 = (0..policy.len()).max_by(|&i, &j| policy[i].total_cmp(&policy[j])).unwrap();
    (policy, top1)
}

fn tvd(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

// -------------------------------------------------------------------- main

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = repo_root();
    let dex = load_dex();

    let pairs_dir = flag(&args, "--pairs")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/exp-siv-v1/pairs"));
    let out_dir = flag(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/exp-siv-v1/cells"));
    let budget: u32 = flag(&args, "--budget").map(|v| v.parse().unwrap()).unwrap_or(1000);
    let games: usize = flag(&args, "--games").map(|v| v.parse().unwrap()).unwrap_or(128);
    let base_seed: u64 = flag(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(20260811);
    let max_turns: u16 = flag(&args, "--max-turns").map(|v| v.parse().unwrap()).unwrap_or(500);
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let gauntlet: Vec<usize> = flag(&args, "--gauntlet")
        .unwrap_or_else(|| "0,4".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();

    // ---- load + fail-closed checks
    let meta = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let mut pair_paths: Vec<PathBuf> = std::fs::read_dir(&pairs_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", pairs_dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("json"))
        .collect();
    pair_paths.sort();
    let pairs: Vec<PairFile> = pair_paths
        .iter()
        .map(|p| {
            serde_json::from_str(&std::fs::read_to_string(p).unwrap())
                .unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
        })
        .collect();
    assert!(!pairs.is_empty(), "no pair files in {}", pairs_dir.display());
    for pair in &pairs {
        assert_eq!(
            signature(&pair.a),
            signature(&pair.b),
            "pair {}: A/B public signatures differ",
            pair.id
        );
        check_team(&dex, &format!("{}/a", pair.id), &pair.a);
        check_team(&dex, &format!("{}/b", pair.id), &pair.b);
        if let Some(w) = &pair.weak {
            assert_eq!(
                signature(&pair.a),
                signature(w),
                "pair {}: weak signature differs",
                pair.id
            );
            check_team(&dex, &format!("{}/weak", pair.id), w);
        }
    }
    eprintln!(
        "loaded {} pairs ({}), gauntlet {:?}, budget {budget}, games/cell {games}, seed {base_seed}",
        pairs.len(),
        pairs.iter().map(|p| p.id.as_str()).collect::<Vec<_>>().join(", "),
        gauntlet
    );

    // ---- screen mode
    if has(&args, "--screen") {
        let screen_budget: u32 =
            flag(&args, "--screen-budget").map(|v| v.parse().unwrap()).unwrap_or(2000);
        let mut report = Vec::new();
        for pair in &pairs {
            for &g in &gauntlet {
                let g_sets = &meta.teams[g].sets;
                // truth on the board is A; preview-public info is identical
                // for A and B, so Y's policy difference is belief-only.
                let mut r = SplitMix64::new(base_seed);
                let seed0 = r.battle_seed();
                let battle = Battle::from_fixture(&dex, &seed0, &pair.a, g_sets).unwrap();
                let sseeds = [1u64, 2, 3, 4];
                let mut pol_a: Vec<Vec<f64>> = Vec::new();
                let mut pol_b: Vec<Vec<f64>> = Vec::new();
                let mut top_a = Vec::new();
                let mut top_b = Vec::new();
                for &s in &sseeds {
                    let (pa, ta) = preview_policy(&dex, &battle, &pair.a, screen_budget, s);
                    let (pb, tb) = preview_policy(&dex, &battle, &pair.b, screen_budget, s);
                    pol_a.push(pa);
                    pol_b.push(pb);
                    top_a.push(ta);
                    top_b.push(tb);
                }
                // seed-noise floor: mean pairwise TVD WITHIN a belief; the
                // belief signal: mean pairwise TVD ACROSS beliefs. A screen
                // pass needs the signal clearly above the floor (the
                // eval-candidate-screen lesson: no verdict without a floor).
                let mut within = Vec::new();
                let mut across = Vec::new();
                for i in 0..sseeds.len() {
                    for j in 0..sseeds.len() {
                        if i < j {
                            within.push(tvd(&pol_a[i], &pol_a[j]));
                            within.push(tvd(&pol_b[i], &pol_b[j]));
                        }
                        across.push(tvd(&pol_a[i], &pol_b[j]));
                    }
                }
                let floor = within.iter().sum::<f64>() / within.len() as f64;
                let signal = across.iter().sum::<f64>() / across.len() as f64;
                let top1_differs = top_a.iter().zip(&top_b).filter(|(a, b)| a != b).count();
                eprintln!(
                    "screen {} vs g{}: across-belief TVD {:.3} vs within-belief floor {:.3} (x{:.2}); top1 A {:?} B {:?}",
                    pair.id, g, signal, floor, signal / floor.max(1e-9), top_a, top_b
                );
                report.push(serde_json::json!({
                    "pair": pair.id, "g": g, "g_id": meta.teams[g].id,
                    "budget": screen_budget, "tvd_across": signal, "tvd_floor": floor,
                    "ratio": signal / floor.max(1e-9),
                    "top1_a": top_a, "top1_b": top_b,
                    "top1_differs": top1_differs, "seeds": sseeds,
                }));
            }
        }
        std::fs::create_dir_all(&out_dir).unwrap();
        let path = out_dir.join(format!("screen-i{screen_budget}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        eprintln!("wrote {}", path.display());
        return;
    }

    // ---- summarize mode
    if has(&args, "--summarize") {
        summarize(&out_dir, &pairs, &gauntlet, budget);
        return;
    }

    // ---- cell schedule
    let mut cells: Vec<CellSpec> = Vec::new();
    for (pi, _) in pairs.iter().enumerate() {
        for &g in &gauntlet {
            for (truth_is_a, belief) in
                [(true, "a"), (true, "b"), (false, "b"), (false, "a")]
            {
                cells.push(CellSpec { pair_idx: pi, g_idx: g, truth_is_a, belief: belief.into() });
            }
        }
    }
    if has(&args, "--controls") {
        // C1 identity + C2 teeth: first pair, first gauntlet team, truth=A
        cells.push(CellSpec {
            pair_idx: 0,
            g_idx: gauntlet[0],
            truth_is_a: true,
            belief: Arm::Open.label().into(),
        });
        assert!(pairs[0].weak.is_some(), "--controls needs a weak team on the first pair");
        cells.push(CellSpec {
            pair_idx: 0,
            g_idx: gauntlet[0],
            truth_is_a: true,
            belief: Arm::Weak.label().into(),
        });
    }

    std::fs::create_dir_all(&out_dir).unwrap();
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    let todo: Vec<&CellSpec> = cells
        .iter()
        .filter(|c| !out_dir.join(c.file_name(&pairs[c.pair_idx].id, budget)).exists())
        .collect();
    eprintln!("{} cells scheduled, {} already done, {} to run", cells.len(), cells.len() - todo.len(), todo.len());

    for (ci, cell) in todo.iter().enumerate() {
        let pair = &pairs[cell.pair_idx];
        let x_sets = if cell.truth_is_a { &pair.a } else { &pair.b };
        let g_sets = &meta.teams[cell.g_idx].sets;
        let (pin_sets, frozen): (Option<&[PokemonSet]>, bool) = match cell.belief.as_str() {
            "a" => (Some(&pair.a), !cell.truth_is_a), // wrong iff truth is B
            "b" => (Some(&pair.b), cell.truth_is_a),
            "weak" => (Some(pair.weak.as_ref().unwrap()), true),
            "open" => (None, false),
            other => panic!("unknown belief arm {other}"),
        };
        let name = cell.file_name(&pair.id, budget);
        eprintln!("[{}/{}] {name} ...", ci + 1, todo.len());
        let (scores, turns, caps, x_ms, y_ms, secs) = run_cell(
            &dex, x_sets, g_sets, pin_sets, frozen, budget, games, base_seed, threads, max_turns,
        );
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let json = serde_json::json!({
            "pair": pair.id, "g": cell.g_idx, "g_id": meta.teams[cell.g_idx].id,
            "truth": if cell.truth_is_a { "a" } else { "b" }, "belief": cell.belief,
            "frozen": frozen, "budget": budget, "games": scores.len(),
            "base_seed": base_seed, "max_turns": max_turns,
            "engine_commit": commit,
            "x_score_mean": mean, "caps": caps,
            "avg_turns": turns.iter().map(|&t| t as f64).sum::<f64>() / turns.len() as f64,
            "x_ms_per_move": x_ms, "y_ms_per_move": y_ms, "secs": secs,
            "game_scores": scores, "turns": turns,
        });
        let path = out_dir.join(&name);
        let tmp = out_dir.join(format!("{name}.tmp"));
        std::fs::write(&tmp, serde_json::to_string(&json).unwrap()).unwrap();
        std::fs::rename(&tmp, &path).unwrap();
        eprintln!(
            "    X score {mean:.3}, {} games, caps {caps}, {:.0}s ({:.1} s/game), X {x_ms:.0} / Y {y_ms:.0} ms/move",
            scores.len(),
            secs,
            secs / scores.len() as f64
        );
    }
    eprintln!("all cells done; run with --summarize for the aggregate");
}

fn load_cell(out_dir: &Path, name: &str) -> Option<serde_json::Value> {
    let path = out_dir.join(name);
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn cell_blocks(v: &serde_json::Value) -> Vec<f64> {
    let games: Vec<f64> =
        v["game_scores"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    block_scores(&games)
}

fn summarize(out_dir: &Path, pairs: &[PairFile], gauntlet: &[usize], budget: u32) {
    let cell = |pi: usize, g: usize, t: &str, b: &str| -> Option<serde_json::Value> {
        let spec = CellSpec {
            pair_idx: pi,
            g_idx: g,
            truth_is_a: t == "a",
            belief: b.into(),
        };
        load_cell(out_dir, &spec.file_name(&pairs[pi].id, budget))
    };
    let mut grand: Vec<f64> = Vec::new();
    println!("== EXP-SIV summary (budget {budget}) ==");
    println!(
        "{:<12} {:>3} {:>18} {:>18} {:>22}",
        "pair", "g", "D_A (wrong-truth)", "D_B (wrong-truth)", "envelope mean±95CI"
    );
    for (pi, pair) in pairs.iter().enumerate() {
        for &g in gauntlet {
            let (Some(taa), Some(tab), Some(tbb), Some(tba)) = (
                cell(pi, g, "a", "a"),
                cell(pi, g, "a", "b"),
                cell(pi, g, "b", "b"),
                cell(pi, g, "b", "a"),
            ) else {
                println!("{:<12} {:>3}  (incomplete)", pair.id, g);
                continue;
            };
            let d_a: Vec<f64> = cell_blocks(&tab)
                .iter()
                .zip(cell_blocks(&taa))
                .map(|(w, t)| w - t)
                .collect();
            let d_b: Vec<f64> = cell_blocks(&tba)
                .iter()
                .zip(cell_blocks(&tbb))
                .map(|(w, t)| w - t)
                .collect();
            let (ma, cia, _) = pooled_stats(&d_a);
            let (mb, cib, _) = pooled_stats(&d_b);
            let env: Vec<f64> = d_a.iter().chain(&d_b).copied().collect();
            let (me, cie, n) = pooled_stats(&env);
            println!(
                "{:<12} {:>3} {:>11.3}±{:.3} {:>11.3}±{:.3} {:>12.3}±{:.3} (n={})",
                pair.id, g, ma, cia, mb, cib, me, cie, n
            );
            grand.extend(env);
        }
    }
    if !grand.is_empty() {
        let (m, ci, n) = pooled_stats(&grand);
        println!("---");
        println!(
            "GRAND envelope: {m:+.4} ± {ci:.4} (n={n} block diffs)  [futility bar: 95% upper < +0.05]"
        );
        println!("GRAND 95% CI: [{:+.4}, {:+.4}]", m - ci, m + ci);
    }
    // controls
    if let (Some(open), Some(truth)) = (
        cell(0, gauntlet[0], "a", "open"),
        cell(0, gauntlet[0], "a", "a"),
    ) {
        let d: Vec<f64> = cell_blocks(&open)
            .iter()
            .zip(cell_blocks(&truth))
            .map(|(o, t)| o - t)
            .collect();
        let (m, ci, n) = pooled_stats(&d);
        println!("C1 identity (open − truth-pin): {m:+.4} ± {ci:.4} (n={n})  [must straddle 0, |m| small]");
    }
    if let (Some(weak), Some(truth)) = (
        cell(0, gauntlet[0], "a", "weak"),
        cell(0, gauntlet[0], "a", "a"),
    ) {
        let d: Vec<f64> = cell_blocks(&weak)
            .iter()
            .zip(cell_blocks(&truth))
            .map(|(w, t)| w - t)
            .collect();
        let (m, ci, n) = pooled_stats(&d);
        println!("C2 teeth (weak-pin − truth-pin): {m:+.4} ± {ci:.4} (n={n})  [expect clearly > 0]");
    }
}
