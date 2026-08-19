//! Is "a greedy attacker beats the shipped search" one fiber, or a class?
//!
//! Round 2 of the 4069/4070 study found a Zapdos-alone-vs-Skarmory+Blissey
//! endgame that a pure `MaxDamageAgent` wins 0.4953 and the shipped
//! 30,000-iteration search wins 0.0857 — a −0.41 gap in a position the
//! search should find trivial. One position proves nothing about the
//! policy; this is the instrument that turns it into a distribution.
//!
//! For every endgame position reconstructed from the 570-battle corpus that
//! passes the filter, both arms are played out from the SAME state against
//! the SAME fixed opponent, trial-by-trial CRN-paired (trial `t` of both
//! arms starts from the same battle PRNG seed), and the per-position gap is
//! reported — not just its mean. A mean over positions can be carried by one
//! outlier; the whole question here is whether the −0.41 has company.
//!
//!   arm A  greedy   — a max-damage baseline, no search at all
//!   arm B  search   — `BlindAgent` at `--iters` (30000 = the ladder budget,
//!                     `tools/ps-client.js:138`)
//!
//! **The baseline trap, and why `--baseline engine` is the default.** The
//! repo's `MaxDamageAgent` scores a move by its dex `basePower`, and in
//! `data/gen2stadium2.json` the callback-powered moves carry `basePower` 0:
//! return, frustration, flail, reversal, magnitude, present, counter,
//! mirrorcoat, bide, hiddenpower, seismictoss, nightshade, psywave,
//! superfang, dragonrage, sonicboom and the OHKO moves. A Return-only
//! Snorlax therefore scores its ENTIRE kit at 0.0 and the agent's `max_by`
//! falls through to the last legal move — which is how round 2's first
//! attempt at battle 4069 measured a "max-damage tail" that never attacked.
//! An unusable baseline invalidates the comparison outright, so the default
//! baseline scores every move through the engine's own damage core
//! (`get_damage_synthetic`, which DOES run `damageCallback`/
//! `basePowerCallback` — `moveexec.rs:2673`/`:2685`), the same oracle
//! `examples/damage_conformance.rs` measures the eval against.
//! `--baseline legacy` reproduces the broken one, and every position is
//! reported with `legacy_blind` (how many of the acting mon's moves the
//! legacy scorer cannot see) and `baseline_disagree` (whether the two
//! baselines pick differently here), so the invalidated set is nameable
//! rather than guessed at.
//!
//! Usage:
//!   greedy_gap [--corpus tmp/corpus-spectator] [--battles 0-569]
//!     [--alive-me-min 1] [--alive-me-max 2] [--alive-foe-min 1] [--alive-foe-max 2]
//!     [--turn-min 0] [--turn-max 1000] [--hp-min 0.0] [--hp-max 1.0]
//!     [--side both|0|1] [--stride 1] [--max-positions 40]
//!     [--trials 100] [--iters 30000] [--max-turns 300]
//!     [--foe max-damage|search] [--baseline engine|legacy] [--allow-selfko]
//!     [--threads 12] [--seed 1] [--out FILE.jsonl] [--csv FILE.csv]

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nc2000_bot::agent::{Agent, MaxDamageAgent};
use nc2000_bot::blind::BlindAgent;
use nc2000_bot::corpus::{
    cfg, corpus_files, load_battle, load_sources, reconstruct_context_with_pool, SetSources,
};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::RmConfig;
use nc2000_engine::battle::moveexec::{get_active_move, get_damage_synthetic};
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::{Accuracy, Category, Dex, Multihit, MoveId};
use nc2000_engine::prng::{BattleRng, Prng};
use nc2000_engine::state::{Battle, PokeId};

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn arg_n<T: std::str::FromStr>(args: &[String], key: &str, default: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    arg_s(args, key, default).parse().unwrap()
}

// ---------------------------------------------------------------- baseline

/// Mean damage of one use of `id`, through the ENGINE's damage core: no crit,
/// no roll variance (the top roll scaled to the 217..255 mean, exactly the
/// convention `damage_conformance.rs` compares on), multi-hit folded in, and
/// multiplied by accuracy so a 30%-accurate OHKO does not outrank a sure hit.
///
/// Scores on a CLONE. `get_damage` runs `run_move_immunity` with messages on,
/// which appends `-immune` to the battle log, and the log is the blind
/// observer's reveal channel — scoring in place would let the greedy arm
/// write into the search arm's evidence.
fn engine_damage(scratch: &mut Battle, dex: &Dex, att: PokeId, def: PokeId, id: MoveId) -> f64 {
    let ms = dex.move_static(id);
    if ms.category == Category::Status {
        return 0.0;
    }
    let mut fake = get_active_move(dex, id);
    fake.no_damage_variance = true;
    fake.will_crit = Some(false);
    // onModifyMove is NOT run by get_damage: plant Hidden Power's real type,
    // power and gen-2 physical/special split from the attacker's DVs.
    if dex.moves.key(id) == "hiddenpower" {
        let a = scratch.poke(att);
        let (t, p) = (a.hp_type, a.hp_power);
        fake.move_type = t;
        fake.base_move_type = t;
        fake.base_power = p;
        let special = matches!(
            dex.type_name(t),
            "Fire" | "Water" | "Grass" | "Electric" | "Psychic" | "Ice" | "Dragon" | "Dark"
        );
        fake.category = if special { Category::Special } else { Category::Physical };
    }
    let Some(top) = get_damage_synthetic(scratch, dex, att, def, fake) else { return 0.0 };
    if top <= 0.0 {
        return 0.0;
    }
    let hits = match &ms.multihit {
        Some(Multihit::Fixed(n)) => *n as f64,
        Some(Multihit::Range(2, 5)) => 3.0,
        Some(Multihit::Range(lo, hi)) => (*lo + *hi) as f64 / 2.0,
        None => 1.0,
    };
    let acc = match ms.accuracy {
        Accuracy::AlwaysHits => 1.0,
        Accuracy::Pct(p) => p as f64 / 100.0,
    };
    top * (236.0 / 255.0) * hits * acc
}

/// The greedy baseline: hardest hit at the mon in front, never a voluntary
/// switch, healthiest bench on a forced one — `MaxDamageAgent`'s policy with
/// its damage model replaced by the engine's.
struct GreedyAgent {
    /// Refuse a self-KO move from the last mon. The Stadium 2 self-KO clause
    /// makes that an immediate LOSS however much damage it deals
    /// (`smmcts::certain_self_loss`), so without this the baseline throws
    /// games away for a reason that has nothing to do with greed. Reported,
    /// and switchable with `--allow-selfko`, because it is a deviation from
    /// the published baseline.
    guard_selfko: bool,
}

impl GreedyAgent {
    fn hp_frac(battle: &Battle, side: usize, display_pos: u8) -> f64 {
        let s = &battle.sides[side];
        let Some(&slot) = s.party.get((display_pos - 1) as usize) else { return -1.0 };
        let p = &s.roster[slot as usize];
        p.hp as f64 / p.maxhp as f64
    }

    /// Move scores at this exact state, aligned with `choices`.
    fn scores(&self, battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> Vec<f64> {
        let (Some(att), Some(def)) = (battle.active_id(side), battle.active_id(1 - side)) else {
            return vec![0.0; choices.len()];
        };
        let mut scratch = battle.clone();
        scratch.set_log_enabled(false);
        let last_mon = battle.sides[side].pokemon_left <= 1;
        choices
            .iter()
            .map(|c| match c {
                SearchChoice::Move(id) => {
                    if self.guard_selfko && last_mon && dex.move_static(*id).selfdestruct {
                        return -1.0;
                    }
                    engine_damage(&mut scratch, dex, att, def, *id)
                }
                _ => -1.0,
            })
            .collect()
    }
}

impl Agent for GreedyAgent {
    fn name(&self) -> String {
        "greedy-engine".into()
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if matches!(choices[0], SearchChoice::Team(_)) {
            let default = SearchChoice::Team([1, 2, 3]);
            return if choices.contains(&default) { default } else { choices[0] };
        }
        if !choices.iter().any(|c| matches!(c, SearchChoice::Move(_))) {
            // forced switch (or pass): healthiest bench mon
            return choices
                .iter()
                .copied()
                .max_by(|a, b| {
                    let f = |c: &SearchChoice| match c {
                        SearchChoice::Switch(pos) => Self::hp_frac(battle, side, *pos),
                        _ => -1.0,
                    };
                    f(a).total_cmp(&f(b))
                })
                .unwrap();
        }
        let scores = self.scores(battle, dex, side, choices);
        let best = (0..choices.len())
            .filter(|&i| matches!(choices[i], SearchChoice::Move(_)))
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap_or(0);
        choices[best]
    }
}

fn build_baseline(kind: &str, guard_selfko: bool) -> Box<dyn Agent> {
    match kind {
        "legacy" => Box::new(MaxDamageAgent::new()),
        "engine" => Box::new(GreedyAgent { guard_selfko }),
        other => panic!("unknown baseline `{other}` (engine|legacy)"),
    }
}

// ---------------------------------------------------------------- positions

struct Position {
    battle_index: usize,
    decision_index: usize,
    turn: u16,
    /// The side the two arms play. The other side gets the fixed opponent.
    side: usize,
    alive_me: usize,
    alive_foe: usize,
    hp_frac: f64,
    label: String,
    /// Moves of the acting mon whose damage the LEGACY scorer cannot see:
    /// `basePower == 0` yet the engine deals damage.
    legacy_blind: usize,
    /// The same count for the mon the FIXED OPPONENT is holding. The
    /// opponent is a baseline too, in both arms, so a blind one changes the
    /// common opponent rather than one arm — still fatal to the reading, and
    /// invisible if only the acting side is checked.
    legacy_blind_foe: usize,
    /// Do the two baselines pick different moves at this root?
    baseline_disagree: bool,
    /// …and what each of them picks, so the invalidation is exhibited rather
    /// than asserted.
    legacy_pick: String,
    engine_pick: String,
    battle: Battle,
}

fn alive(b: &Battle, side: usize) -> usize {
    let s = &b.sides[side];
    s.party.iter().filter(|&&slot| !s.roster[slot as usize].fainted && s.roster[slot as usize].hp > 0).count()
}

fn describe(b: &Battle, dex: &Dex, side: usize) -> String {
    let s = &b.sides[side];
    let parts: Vec<String> = s
        .party
        .iter()
        .map(|&slot| {
            let p = &s.roster[slot as usize];
            format!(
                "{} {:.0}%",
                dex.species.key(p.species),
                100.0 * p.hp as f64 / p.maxhp.max(1) as f64
            )
        })
        .collect();
    parts.join("/")
}

// ------------------------------------------------------------------ playout

#[derive(Default, Clone)]
struct Arm {
    wins: f64,
    per_trial: Vec<f64>,
    turns: u64,
}

/// Battle seed as a pure function of (position, trial) — identical in both
/// arms, which is what makes the pair CRN. splitmix64 rather than
/// `Prng::from_seed_str`, whose four 16-bit limbs cap the usable trial index
/// at ~65k (`prng.rs:32`).
fn battle_seed(base: u64, pos: u64, trial: u64) -> u64 {
    let mut z = base
        ^ pos.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ trial.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[allow(clippy::too_many_arguments)]
fn play_arm(
    dex: &Dex,
    pool: &Arc<MetaPool>,
    p: &Position,
    pos_id: u64,
    arm: &str,
    ctx: &Ctx,
) -> Arm {
    let mut acc = Arm::default();
    for t in 0..ctx.trials {
        let mut b = p.battle.clone();
        b.set_log_enabled(true);
        b.prng = BattleRng::seeded(Prng::new(battle_seed(ctx.seed, pos_id, t)));
        let start = b.turn;
        // Agent seeds depend on (position, trial) only, never on the arm, so
        // the fixed opponent is bit-identically the same opponent in both.
        let foe_seed = battle_seed(ctx.seed ^ 0xF0E5_EED0, pos_id, t);
        let me_seed = battle_seed(ctx.seed ^ 0x5EA2_C400, pos_id, t);
        let mut me: Box<dyn Agent> = match arm {
            "greedy" => build_baseline(&ctx.baseline, !ctx.allow_selfko),
            _ => Box::new(BlindAgent::new(
                RmConfig { iterations: ctx.iters, ..cfg() },
                pool.clone(),
                None,
                me_seed,
            )),
        };
        let mut foe: Box<dyn Agent> = match ctx.foe.as_str() {
            "search" => Box::new(BlindAgent::new(
                RmConfig { iterations: ctx.foe_iters, ..cfg() },
                pool.clone(),
                None,
                foe_seed,
            )),
            _ => build_baseline(&ctx.baseline, !ctx.allow_selfko),
        };
        let agents: [&mut dyn Agent; 2] = if p.side == 0 {
            [me.as_mut(), foe.as_mut()]
        } else {
            [foe.as_mut(), me.as_mut()]
        };
        let score = loop {
            if let Some(o) = b.outcome() {
                break match (o, p.side) {
                    (Outcome::P1Win, 0) | (Outcome::P2Win, 1) => 1.0,
                    (Outcome::Tie, _) => 0.5,
                    _ => 0.0,
                };
            }
            if b.turn > start.saturating_add(ctx.max_turns) {
                break 0.5;
            }
            let mut picks = [None, None];
            for s in 0..2 {
                let cs = b.legal_choices(dex, s);
                if !cs.is_empty() {
                    picks[s] = Some(agents[s].choose(&b, dex, s, &cs));
                }
            }
            if picks == [None, None] {
                break 0.5;
            }
            if b.apply_choices(dex, picks).is_err() {
                break 0.5;
            }
        };
        acc.wins += score;
        acc.per_trial.push(score);
        acc.turns += b.turn.saturating_sub(start) as u64;
    }
    acc
}

struct Ctx {
    trials: u64,
    iters: u32,
    foe_iters: u32,
    max_turns: u16,
    foe: String,
    baseline: String,
    allow_selfko: bool,
    seed: u64,
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-569");
    let side_sel = arg_s(&args, "--side", "both");
    let alive_me = (
        arg_n::<usize>(&args, "--alive-me-min", "1"),
        arg_n::<usize>(&args, "--alive-me-max", "2"),
    );
    let alive_foe = (
        arg_n::<usize>(&args, "--alive-foe-min", "1"),
        arg_n::<usize>(&args, "--alive-foe-max", "2"),
    );
    let turn_band =
        (arg_n::<u16>(&args, "--turn-min", "0"), arg_n::<u16>(&args, "--turn-max", "1000"));
    let hp_band =
        (arg_n::<f64>(&args, "--hp-min", "0.0"), arg_n::<f64>(&args, "--hp-max", "1.0"));
    let stride: usize = arg_n(&args, "--stride", "1");
    let max_positions: usize = arg_n(&args, "--max-positions", "40");
    let threads: usize = arg_n(&args, "--threads", "12");
    let out_path = arg_s(&args, "--out", "tmp/verdicts-4069-4070/r3/greedy-gap.jsonl");
    let csv_path = arg_s(&args, "--csv", "");
    let ctx = Ctx {
        trials: arg_n(&args, "--trials", "100"),
        iters: arg_n(&args, "--iters", "30000"),
        foe_iters: arg_n(&args, "--foe-iters", "1000"),
        max_turns: arg_n(&args, "--max-turns", "300"),
        foe: arg_s(&args, "--foe", "max-damage"),
        baseline: arg_s(&args, "--baseline", "engine"),
        allow_selfko: args.iter().any(|a| a == "--allow-selfko"),
        seed: arg_n(&args, "--seed", "1"),
    };
    assert!(ctx.baseline == "engine" || ctx.baseline == "legacy", "--baseline engine|legacy");

    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let hi: usize = it.next().unwrap_or("569").parse().unwrap_or(569);
        (lo, hi)
    };

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let apool = Arc::new(pool.clone());
    let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&root.join(&corpus))
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    eprintln!(
        "scanning {} battles (index {lo}-{hi}) for alive me {}-{} foe {}-{}, turn {}-{}, hp {:.2}-{:.2}",
        files.len(),
        alive_me.0, alive_me.1, alive_foe.0, alive_foe.1, turn_band.0, turn_band.1,
        hp_band.0, hp_band.1
    );

    // ---- phase 1: scan the corpus for positions the filter admits --------
    let found: Mutex<Vec<Position>> = Mutex::new(Vec::new());
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= files.len() {
                    return;
                }
                let (bi, path) = &files[j];
                let mut local = scan_battle(
                    &dex, &src, &pool, path, *bi, &side_sel, alive_me, alive_foe, turn_band,
                    hp_band, &ctx,
                );
                if !local.is_empty() {
                    found.lock().unwrap().append(&mut local);
                }
            });
        }
    });
    let mut positions = found.into_inner().unwrap();
    positions.sort_by_key(|p| (p.battle_index, p.decision_index, p.side));
    let total_matched = positions.len();
    let positions: Vec<Position> =
        positions.into_iter().step_by(stride.max(1)).take(max_positions).collect();
    eprintln!(
        "{total_matched} positions match; measuring {} (stride {stride}, cap {max_positions})",
        positions.len()
    );
    let blind_positions = positions.iter().filter(|p| p.legacy_blind > 0).count();
    let blind_foe = positions.iter().filter(|p| p.legacy_blind_foe > 0).count();
    let blind_either =
        positions.iter().filter(|p| p.legacy_blind > 0 || p.legacy_blind_foe > 0).count();
    let disagree = positions.iter().filter(|p| p.baseline_disagree).count();
    eprintln!(
        "legacy baseline blind to >=1 move: acting side {blind_positions}, fixed opponent \
         {blind_foe}, either {blind_either} of {}; different root pick in {disagree}",
        positions.len()
    );
    if positions.is_empty() {
        return;
    }

    // ---- phase 2: both arms, CRN-paired, per position --------------------
    eprintln!(
        "arms: greedy({}) vs search({} iters) vs a common {} opponent, {} trials each",
        ctx.baseline, ctx.iters, ctx.foe, ctx.trials
    );
    let rows: Mutex<Vec<(usize, serde_json::Value, f64)>> = Mutex::new(Vec::new());
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let k = cursor.fetch_add(1, Ordering::Relaxed);
                if k >= positions.len() {
                    return;
                }
                let p = &positions[k];
                let pos_id = (p.battle_index as u64) << 20
                    | (p.decision_index as u64) << 4
                    | p.side as u64;
                let g = play_arm(&dex, &apool, p, pos_id, "greedy", &ctx);
                let s = play_arm(&dex, &apool, p, pos_id, "search", &ctx);
                let n = ctx.trials as f64;
                let gw = g.wins / n;
                let sw = s.wins / n;
                let gap = gw - sw;
                // CRN-paired: the CI is over the per-trial DIFFERENCE, which
                // is what the shared battle seed buys.
                let diffs: Vec<f64> =
                    g.per_trial.iter().zip(&s.per_trial).map(|(a, b)| a - b).collect();
                let m = mean(&diffs);
                let var = if diffs.len() > 1 {
                    diffs.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64
                } else {
                    f64::NAN
                };
                let ci = 1.96 * (var / diffs.len() as f64).sqrt();
                let row = serde_json::json!({
                    "battle": p.battle_index,
                    "decision": p.decision_index,
                    "turn": p.turn,
                    "side": p.side,
                    "alive_me": p.alive_me,
                    "alive_foe": p.alive_foe,
                    "hp_frac": p.hp_frac,
                    "label": p.label,
                    "legacy_blind": p.legacy_blind,
                    "legacy_blind_foe": p.legacy_blind_foe,
                    "baseline_disagree": p.baseline_disagree,
                    "legacy_pick": p.legacy_pick,
                    "engine_pick": p.engine_pick,
                    "trials": ctx.trials,
                    "greedy_win": gw,
                    "search_win": sw,
                    "gap": gap,
                    "gap_ci95": ci,
                    "greedy_turns": g.turns as f64 / n,
                    "search_turns": s.turns as f64 / n,
                });
                rows.lock().unwrap().push((k, row, gap));
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("  {d}/{} positions", positions.len());
            });
        }
    });

    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|(k, _, _)| *k);
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    for (_, row, _) in &rows {
        writeln!(f, "{row}").unwrap();
    }
    drop(f);
    if !csv_path.is_empty() {
        let mut c = std::io::BufWriter::new(std::fs::File::create(&csv_path).unwrap());
        writeln!(
            c,
            "battle,decision,turn,side,alive_me,alive_foe,hp_frac,legacy_blind,\
             legacy_blind_foe,baseline_disagree,trials,greedy_win,search_win,gap,gap_ci95,label"
        )
        .unwrap();
        for (_, r, _) in &rows {
            writeln!(
                c,
                "{},{},{},{},{},{},{:.4},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{}",
                r["battle"], r["decision"], r["turn"], r["side"], r["alive_me"], r["alive_foe"],
                r["hp_frac"].as_f64().unwrap(), r["legacy_blind"], r["legacy_blind_foe"],
                r["baseline_disagree"],
                r["trials"], r["greedy_win"].as_f64().unwrap(), r["search_win"].as_f64().unwrap(),
                r["gap"].as_f64().unwrap(), r["gap_ci95"].as_f64().unwrap(),
                r["label"].as_str().unwrap()
            )
            .unwrap();
        }
    }

    // ---- the distribution, which is the whole point ----------------------
    let mut gaps: Vec<f64> = rows.iter().map(|(_, _, g)| *g).collect();
    gaps.sort_by(f64::total_cmp);
    let m = mean(&gaps);
    let sd = if gaps.len() > 1 {
        (gaps.iter().map(|g| (g - m).powi(2)).sum::<f64>() / (gaps.len() - 1) as f64).sqrt()
    } else {
        f64::NAN
    };
    println!("\npositions {}   trials/arm {}   iters {}", gaps.len(), ctx.trials, ctx.iters);
    println!(
        "gap = greedy - search   mean {m:+.4}  sd {sd:.4}  ci95 {:+.4}",
        1.96 * sd / (gaps.len() as f64).sqrt()
    );
    println!(
        "  min {:+.3}  p10 {:+.3}  p25 {:+.3}  median {:+.3}  p75 {:+.3}  p90 {:+.3}  max {:+.3}",
        quantile(&gaps, 0.0),
        quantile(&gaps, 0.10),
        quantile(&gaps, 0.25),
        quantile(&gaps, 0.50),
        quantile(&gaps, 0.75),
        quantile(&gaps, 0.90),
        quantile(&gaps, 1.0)
    );
    for thr in [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30] {
        let n = if thr < 0.0 {
            gaps.iter().filter(|g| **g <= thr).count()
        } else {
            gaps.iter().filter(|g| **g >= thr).count()
        };
        let sign = if thr < 0.0 { "<=" } else { ">=" };
        println!("  gap {sign} {thr:+.2}: {n}/{}", gaps.len());
    }
    println!("\nworst 10 positions for the search:");
    let mut by_gap: Vec<&(usize, serde_json::Value, f64)> = rows.iter().collect();
    by_gap.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (_, r, g) in by_gap.iter().take(10) {
        println!(
            "  b{:<4} d{:<4} t{:<4} side{} {:>7.3} greedy {:.3} search {:.3}  {}",
            r["battle"], r["decision"], r["turn"], r["side"], g,
            r["greedy_win"].as_f64().unwrap(), r["search_win"].as_f64().unwrap(),
            r["label"].as_str().unwrap()
        );
    }
    println!("\nrows -> {out_path}");
}

#[allow(clippy::too_many_arguments)]
fn scan_battle(
    dex: &Dex,
    src: &SetSources,
    pool: &MetaPool,
    path: &std::path::Path,
    bi: usize,
    side_sel: &str,
    alive_me: (usize, usize),
    alive_foe: (usize, usize),
    turn_band: (u16, u16),
    hp_band: (f64, f64),
    ctx: &Ctx,
) -> Vec<Position> {
    let cb = load_battle(path);
    let mut out = Vec::new();
    for (di, d) in cb.decisions.iter().enumerate() {
        if side_sel != "both" && side_sel.parse::<usize>().ok() != Some(d.side) {
            continue;
        }
        if d.turn < turn_band.0 || d.turn > turn_band.1 {
            continue;
        }
        let seed = (bi as u64).wrapping_mul(0x9E37_79B9_7F4A) ^ (di as u64) ^ ctx.seed;
        let Some(rec) =
            reconstruct_context_with_pool(dex, src, pool.clone(), &cb.lines, &cb.evidence, d, seed)
        else {
            continue;
        };
        let Some(b) = rec.agent.battle().cloned() else { continue };
        let side = d.side;
        let (am, af) = (alive(&b, side), alive(&b, 1 - side));
        if am < alive_me.0 || am > alive_me.1 || af < alive_foe.0 || af > alive_foe.1 {
            continue;
        }
        let (Some(att), Some(def)) = (b.active_id(side), b.active_id(1 - side)) else { continue };
        let hp = b.poke(att).hp as f64 / b.poke(att).maxhp.max(1) as f64;
        if hp < hp_band.0 || hp > hp_band.1 {
            continue;
        }
        // Only positions where the arms actually have something to choose.
        let choices = b.clone().legal_choices(dex, side);
        if choices.len() < 2 || !choices.iter().any(|c| matches!(c, SearchChoice::Move(_))) {
            continue;
        }
        // Baseline validity, per position: which of the acting mon's moves
        // does the LEGACY scorer score at 0 while the engine deals damage?
        let mut scratch = b.clone();
        scratch.set_log_enabled(false);
        let mut legacy_blind = 0usize;
        for c in choices.iter() {
            if let SearchChoice::Move(id) = c {
                let ms = dex.move_static(*id);
                if ms.category != Category::Status
                    && ms.base_power == 0
                    && engine_damage(&mut scratch, dex, att, def, *id) > 0.0
                {
                    legacy_blind += 1;
                }
            }
        }
        // …and the same question for the fixed opponent's kit.
        let mut legacy_blind_foe = 0usize;
        for c in b.clone().legal_choices(dex, 1 - side) {
            if let SearchChoice::Move(id) = c {
                let ms = dex.move_static(id);
                if ms.category != Category::Status
                    && ms.base_power == 0
                    && engine_damage(&mut scratch, dex, def, att, id) > 0.0
                {
                    legacy_blind_foe += 1;
                }
            }
        }
        let mut legacy = MaxDamageAgent::new();
        let mut engine = GreedyAgent { guard_selfko: !ctx.allow_selfko };
        let lp = legacy.choose(&b, dex, side, &choices);
        let ep = engine.choose(&b, dex, side, &choices);
        let baseline_disagree = lp != ep;
        let (legacy_pick, engine_pick) = (lp.to_input(dex), ep.to_input(dex));
        let label = format!(
            "[{}] {} vs {}",
            dex.species.key(b.poke(att).species),
            describe(&b, dex, side),
            describe(&b, dex, 1 - side)
        );
        out.push(Position {
            battle_index: bi,
            decision_index: di,
            turn: d.turn,
            side,
            alive_me: am,
            alive_foe: af,
            hp_frac: hp,
            label,
            legacy_blind,
            legacy_blind_foe,
            baseline_disagree,
            legacy_pick,
            engine_pick,
            battle: b,
        });
    }
    out
}
