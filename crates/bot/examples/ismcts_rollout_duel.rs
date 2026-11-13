//! Isolate visible-state ISMCTS on the pre-solver eight-turn heavy-rollout baseline.
//! Both arms use identical preview choices, seeds, evaluation, pruning and rollout rules.
//! See docs/ISMCTS-ONLY-EXPERIMENT.md for commands and interpretation.
//! Emits one JSON summary; optional per-pair traces go to --out.

#![recursion_limit = "256"]

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nc2000_bot::accounting::LeafSettlementMode;
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{compiled_preview, Agent, Belief, BlindAgent, Observer, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, RequestState};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone)]
struct TeamEntry {
    id: String,
    sets: Vec<PokemonSet>,
    weight: f64,
}

#[derive(Clone)]
struct CompiledTable {
    policies: Arc<HashMap<String, Vec<String>>>,
}

impl CompiledTable {
    fn load(path: &Path) -> Self {
        let raw: Value = serde_json::from_slice(
            &std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
        )
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        let map = raw["policies"]
            .as_object()
            .expect("compiled preview missing policies")
            .iter()
            .map(|(k, v)| {
                let choices = v
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().to_string())
                    .collect();
                (k.clone(), choices)
            })
            .collect();
        Self {
            policies: Arc::new(map),
        }
    }

    fn pick(
        &self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        seed: u64,
        legal: &[SearchChoice],
    ) -> Option<SearchChoice> {
        let key = compiled_preview::key(b, side)?;
        let choices = self.policies.get(&key)?;
        if choices.is_empty() {
            return None;
        }
        let s = choices[(seed as u32 as usize) % choices.len()].as_str();
        legal.iter().copied().find(|c| c.to_input(dex) == s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmKind {
    Ismcts,
    Legacy,
}

impl ArmKind {
    fn parse(s: &str) -> Self {
        match s {
            "ismcts" => Self::Ismcts,
            "legacy" => Self::Legacy,
            _ => panic!("unknown arm {s:?}; expected ismcts or legacy"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Ismcts => "ismcts",
            Self::Legacy => "legacy",
        }
    }

    fn config(self, iterations: u32, c: f64, hp_buckets: i64) -> RmConfig {
        RmConfig {
            iterations,
            rule: SelRule::Ucb,
            c,
            hp_buckets,
            ismcts_visible_tree: self == Self::Ismcts,
            ..RmConfig::legacy_heavy_rollout()
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PreviewMode {
    CompiledShared,
    Live,
}

impl PreviewMode {
    fn parse(s: &str) -> Self {
        match s {
            "compiled" => Self::CompiledShared,
            "live" => Self::Live,
            _ => panic!("unknown preview mode {s:?}; expected compiled or live"),
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::CompiledShared => "compiled-shared",
            Self::Live => "live",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolMode {
    Catalog,
    Offpool,
    MixedOffpool,
}

impl PoolMode {
    fn parse(s: &str) -> Self {
        match s {
            "catalog" => Self::Catalog,
            "offpool" => Self::Offpool,
            "mixed-offpool" => Self::MixedOffpool,
            _ => panic!("unknown pool mode {s:?}"),
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Catalog => "catalog",
            Self::Offpool => "offpool",
            Self::MixedOffpool => "mixed-offpool",
        }
    }
}

struct PairSchedule {
    pair_idx: usize,
    team_ids: [String; 2],
    team_sets: [Vec<PokemonSet>; 2],
    battle_seed: String,
    agent_seed: u64,
    preview_pick_seeds: [u64; 2],
}

fn load_catalog_teams(path: &Path) -> Vec<TeamEntry> {
    let raw: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    raw["teams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| TeamEntry {
            id: t["id"].as_str().unwrap().to_string(),
            sets: serde_json::from_value(t["sets"].clone()).unwrap(),
            weight: t["drawWeight"].as_f64().unwrap(),
        })
        .collect()
}

fn load_offpool_rentals(dex: &Dex, prior: &MetaPool, path: &Path) -> Vec<TeamEntry> {
    #[derive(Deserialize)]
    struct RentalTeam {
        #[serde(default)]
        archetype: Option<String>,
        sets: Vec<PokemonSet>,
    }
    #[derive(Deserialize)]
    struct RentalsFile {
        teams: Vec<RentalTeam>,
    }
    let rf: RentalsFile = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut out = Vec::new();
    for (i, t) in rf.teams.into_iter().enumerate() {
        let Ok(b) = Battle::from_fixture(dex, "1,2,3,4", &t.sets, &t.sets) else {
            continue;
        };
        let obs = Observer::new(&b, 0);
        let belief = Belief::new(dex, prior, &obs);
        if belief.is_fallback() {
            out.push(TeamEntry {
                id: format!("rental-{i:03}:{}", t.archetype.unwrap_or_default()),
                sets: t.sets,
                weight: 1.0,
            });
        }
    }
    assert!(
        !out.is_empty(),
        "expected at least one off-pool rental team"
    );
    out
}

fn draw_weighted<'a>(pool: &'a [TeamEntry], rng: &mut SplitMix64) -> &'a TeamEntry {
    let total: f64 = pool.iter().map(|t| t.weight).sum();
    let mut r = rng.next_f64() * total;
    for t in pool {
        r -= t.weight;
        if r < 0.0 {
            return t;
        }
    }
    pool.last().unwrap()
}

fn schedule_for_pair(
    pair_idx: usize,
    base_seed: u64,
    pool_mode: PoolMode,
    catalog: &[TeamEntry],
    offpool: &[TeamEntry],
) -> PairSchedule {
    let mut rng = SplitMix64::new(
        base_seed ^ (pair_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x517cc1b727220a95,
    );
    let _ = rng.next();
    let (t0, t1) = match pool_mode {
        PoolMode::Catalog => (
            draw_weighted(catalog, &mut rng),
            draw_weighted(catalog, &mut rng),
        ),
        PoolMode::Offpool => (
            draw_weighted(offpool, &mut rng),
            draw_weighted(offpool, &mut rng),
        ),
        PoolMode::MixedOffpool => {
            if pair_idx % 2 == 0 {
                (
                    draw_weighted(catalog, &mut rng),
                    draw_weighted(offpool, &mut rng),
                )
            } else {
                (
                    draw_weighted(offpool, &mut rng),
                    draw_weighted(catalog, &mut rng),
                )
            }
        }
    };
    let battle_seed = rng.battle_seed();
    let agent_seed = rng.next();
    let p0 = rng.next();
    let p1 = rng.next();
    PairSchedule {
        pair_idx,
        team_ids: [t0.id.clone(), t1.id.clone()],
        team_sets: [t0.sets.clone(), t1.sets.clone()],
        battle_seed,
        agent_seed,
        preview_pick_seeds: [p0, p1],
    }
}

struct GameExec {
    a_score: f64,
    turns: u16,
    capped: bool,
    outcome: String,
    a_battle_ns: Vec<u64>,
    b_battle_ns: Vec<u64>,
    a_preview_ns: Vec<u64>,
    b_preview_ns: Vec<u64>,
    a_compiled_hits: usize,
    b_compiled_hits: usize,
    a_fallback_used: bool,
    b_fallback_used: bool,
}

#[allow(clippy::too_many_arguments)]
fn play_single_game(
    dex: &Dex,
    prior: &Arc<MetaPool>,
    sched: &PairSchedule,
    arm_a: ArmKind,
    arm_b: ArmKind,
    a_side: usize,
    battle_iters: u32,
    preview_iters: u32,
    c: f64,
    hp_buckets: i64,
    max_turns: u16,
    preview_mode: PreviewMode,
    compiled: Option<&CompiledTable>,
) -> GameExec {
    let mut battle = Battle::from_fixture(
        dex,
        &sched.battle_seed,
        &sched.team_sets[0],
        &sched.team_sets[1],
    )
    .expect("valid scheduled teams");
    battle.set_log_enabled(true);

    let arm_for_side = |s: usize| if s == a_side { arm_a } else { arm_b };
    let mut agents = [0usize, 1].map(|s| {
        let cfg = arm_for_side(s).config(battle_iters, c, hp_buckets);
        let mut ag = BlindAgent::new(cfg, prior.clone(), None, sched.agent_seed);
        ag.set_iterations(battle_iters);
        ag.with_preview_iterations(preview_iters)
    });

    let mut a_battle_ns = Vec::new();
    let mut b_battle_ns = Vec::new();
    let mut a_preview_ns = Vec::new();
    let mut b_preview_ns = Vec::new();
    let mut a_compiled_hits = 0usize;
    let mut b_compiled_hits = 0usize;
    let mut a_fallback_used = false;
    let mut b_fallback_used = false;

    while battle.outcome().is_none() && battle.turn <= max_turns {
        let legal = [0usize, 1].map(|s| battle.legal_choices(dex, s));
        let mut picks = [None, None];
        let is_preview = battle.request_state == RequestState::TeamPreview;
        for s in 0..2 {
            if legal[s].is_empty() {
                continue;
            }
            let is_a = s == a_side;
            let mut chosen = None;
            if is_preview {
                if let Some(tbl) = compiled.filter(|_| preview_mode == PreviewMode::CompiledShared)
                {
                    let t0 = Instant::now();
                    if let Some(pick) =
                        tbl.pick(&battle, dex, s, sched.preview_pick_seeds[s], &legal[s])
                    {
                        let ns = t0.elapsed().as_nanos() as u64;
                        if is_a {
                            a_preview_ns.push(ns);
                            a_compiled_hits += 1;
                        } else {
                            b_preview_ns.push(ns);
                            b_compiled_hits += 1;
                        }
                        chosen = Some(pick);
                    }
                }
            }
            if chosen.is_none() {
                let t0 = Instant::now();
                let pick = agents[s].choose(&battle, dex, s, &legal[s]);
                let ns = t0.elapsed().as_nanos() as u64;
                if let Some(bel) = agents[s].belief() {
                    if bel.is_fallback() {
                        if is_a {
                            a_fallback_used = true;
                        } else {
                            b_fallback_used = true;
                        }
                    }
                }
                if is_preview {
                    if is_a {
                        a_preview_ns.push(ns);
                    } else {
                        b_preview_ns.push(ns);
                    }
                } else if legal[s].len() > 1 {
                    if is_a {
                        a_battle_ns.push(ns);
                    } else {
                        b_battle_ns.push(ns);
                    }
                }
                chosen = Some(pick);
            }
            picks[s] = chosen;
        }
        battle.apply_choices(dex, picks).unwrap();
    }

    let capped = battle.outcome().is_none();
    let p1_score = match battle.outcome() {
        Some(Outcome::P1Win) => 1.0,
        Some(Outcome::P2Win) => 0.0,
        Some(Outcome::Tie) | None => 0.5,
    };
    let a_score = if a_side == 0 {
        p1_score
    } else {
        1.0 - p1_score
    };
    GameExec {
        a_score,
        turns: battle.turn,
        capped,
        outcome: format!("{:?}", battle.outcome()),
        a_battle_ns,
        b_battle_ns,
        a_preview_ns,
        b_preview_ns,
        a_compiled_hits,
        b_compiled_hits,
        a_fallback_used,
        b_fallback_used,
    }
}

fn percentile_ms(sorted_ns: &[u64], q: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ns.len() - 1) as f64 * q).round() as usize;
    sorted_ns[idx] as f64 / 1e6
}

fn run_self_test(
    dex: &Dex,
    prior: &Arc<MetaPool>,
    catalog: &[TeamEntry],
    compiled: &CompiledTable,
) {
    // Guard the experimental contract and exercise real rollout advancement.
    let cfg = ArmKind::Ismcts.config(64, 0.4, 16);
    assert_eq!(cfg.settlement, LeafSettlementMode::BlindSpotLeaf);
    assert!(!cfg.semantic_hp_thresholds && !cfg.three_rank_pruning);
    assert!(!cfg.collapse_sleep_actions && !cfg.key_no_damage);
    assert!(matches!(
        cfg.playout,
        nc2000_bot::mcts::Playout::Heavy { turns: 8, .. }
    ));
    let mut without_ismcts = cfg.clone();
    without_ismcts.ismcts_visible_tree = false;
    assert_eq!(
        format!("{without_ismcts:?}"),
        format!("{:?}", ArmKind::Legacy.config(64, 0.4, 16))
    );
    let mut rollout =
        Battle::from_fixture(dex, "1,2,3,4", &catalog[0].sets, &catalog[1].sets).unwrap();
    let picks = [0, 1].map(|side| rollout.legal_choices(dex, side).into_iter().next());
    rollout.apply_choices(dex, picks).unwrap();
    let start_turn = rollout.turn;
    let value = nc2000_bot::mcts::playout_value(
        &mut rollout,
        dex,
        &cfg.playout,
        100,
        &mut SplitMix64::new(777),
        cfg.rollout_rules(),
    );
    assert!((0.0..=1.0).contains(&value));
    let end = nc2000_bot::mcts::last_sim_end();
    assert!(
        end.outcome != 0 || end.turn >= start_turn + 8,
        "heavy rollout did not advance"
    );

    // 1) Verify compiled-preview skipping TeamPreview produces identical Turn-1 Observer/Belief
    let t0 = &catalog[0].sets;
    let t1 = &catalog[1].sets;
    let mut b_with = Battle::from_fixture(dex, "1,2,3,4", t0, t1).unwrap();
    let mut b_skip = Battle::from_fixture(dex, "1,2,3,4", t0, t1).unwrap();
    b_with.set_log_enabled(true);
    b_skip.set_log_enabled(true);

    let cfg = ArmKind::Ismcts.config(16, 0.4, 16);
    let mut a_with = [0, 1].map(|_| BlindAgent::new(cfg.clone(), prior.clone(), None, 777));
    let mut a_skip = [0, 1].map(|_| BlindAgent::new(cfg.clone(), prior.clone(), None, 777));

    let legal0 = b_with.legal_choices(dex, 0);
    let legal1 = b_with.legal_choices(dex, 1);
    let p0 = compiled
        .pick(&b_with, dex, 0, 11, &legal0)
        .expect("catalog preview hit p0");
    let p1 = compiled
        .pick(&b_with, dex, 1, 22, &legal1)
        .expect("catalog preview hit p1");
    let _ = a_with[0].choose(&b_with, dex, 0, &legal0);
    let _ = a_with[1].choose(&b_with, dex, 1, &legal1);
    b_with.apply_choices(dex, [Some(p0), Some(p1)]).unwrap();
    b_skip.apply_choices(dex, [Some(p0), Some(p1)]).unwrap();

    for s in 0..2 {
        let l_w = b_with.legal_choices(dex, s);
        let l_s = b_skip.legal_choices(dex, s);
        assert_eq!(l_w, l_s);
        let _ = a_with[s].choose(&b_with, dex, s, &l_w);
        let _ = a_skip[s].choose(&b_skip, dex, s, &l_s);
        assert_eq!(
            a_with[s].belief().unwrap().alive(),
            a_skip[s].belief().unwrap().alive(),
            "Turn-1 alive belief must match regardless of skipping preview BlindAgent"
        );
    }

    // Verify CRN null controls for both arms (four pairs each).
    // Search must be invariant to a change in the actual opponent's hidden happiness.
    // Both determinizations are supplied by the same belief, never by the real hidden team.
    let mut hidden = catalog[1].sets.clone();
    for mon in &mut hidden {
        mon.happiness = Some(0);
    }
    let mut hidden_battle = Battle::from_fixture(dex, "1,2,3,4", t0, &hidden).unwrap();
    hidden_battle.set_log_enabled(true);
    hidden_battle
        .apply_choices(dex, [Some(p0), Some(p1)])
        .unwrap();
    let mut depth_stats = Vec::new();
    for side in 0..2 {
        let obs = Observer::new(&b_skip, side);
        let mut belief = Belief::new(dex, prior, &obs);
        belief.sync(dex, &obs);
        for arm in [ArmKind::Legacy, ArmKind::Ismcts] {
            let cfg = arm.config(2400, 0.4, 16);
            let mut search = nc2000_bot::BlindSearch::new(&b_skip, dex, cfg.clone(), side, 777);
            search.step(dex, &belief, &obs, 2400);
            assert!(search.mean_depth() > 0.0);
            depth_stats.push(json!({"arm": arm.as_str(), "side": side,
                "nodes": search.node_count(), "mean_depth": search.mean_depth(), "max_depth": search.max_depth()}));
            if side == 0 {
                let hidden_obs = Observer::new(&hidden_battle, side);
                let mut hidden_belief = Belief::new(dex, prior, &hidden_obs);
                hidden_belief.sync(dex, &hidden_obs);
                let mut other = nc2000_bot::BlindSearch::new(&hidden_battle, dex, cfg, side, 777);
                other.step(dex, &hidden_belief, &hidden_obs, 2400);
                assert_eq!(search.visits(), other.visits());
                assert_eq!(search.means(), other.means());
                assert_eq!(search.best(), other.best());
            }
        }
    }
    let off = Vec::new();
    for arm in [ArmKind::Legacy, ArmKind::Ismcts] {
        for pair_idx in 0..4 {
            let sched = schedule_for_pair(pair_idx, 12345, PoolMode::Catalog, catalog, &off);
            let g0 = play_single_game(
                dex,
                prior,
                &sched,
                arm,
                arm,
                0,
                64,
                64,
                0.4,
                16,
                500,
                PreviewMode::CompiledShared,
                Some(compiled),
            );
            let g1 = play_single_game(
                dex,
                prior,
                &sched,
                arm,
                arm,
                1,
                64,
                64,
                0.4,
                16,
                500,
                PreviewMode::CompiledShared,
                Some(compiled),
            );
            assert!(
                !g0.capped && !g1.capped,
                "null control must finish both games"
            );
            assert!(
                ((g0.a_score + g1.a_score) * 0.5 - 0.5).abs() < 1e-12 && (g0.turns == g1.turns),
                "CRN identical-arm null control mismatch at pair {pair_idx}"
            );
        }
    }
    println!(
        "{}",
        json!({"self_test": "ok", "null_pairs": 8, "rollout_end_turn": end.turn, "depth": depth_stats})
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help") {
        println!(
            "{}",
            json!({"usage": "ismcts_rollout_duel [--self-test] [--arm-a ismcts|legacy] [--arm-b ismcts|legacy] [--iters 27000] [--preview-iters N] [--pairs 16] [--seed 20261113] [--threads N] [--pool catalog|offpool|mixed-offpool] [--preview compiled|live] [--compiled PATH] [--c 0.4] [--hp-buckets 16] [--max-turns 1001] [--start-pair 0] [--end-pair N] [--stride 1] [--offset 0] [--out PATH]"})
        );
        return;
    }
    let value_flags = [
        "--arm-a",
        "--arm-b",
        "--iters",
        "--preview-iters",
        "--pairs",
        "--seed",
        "--threads",
        "--pool",
        "--preview",
        "--compiled",
        "--c",
        "--hp-buckets",
        "--max-turns",
        "--start-pair",
        "--end-pair",
        "--stride",
        "--offset",
        "--out",
    ];
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--self-test" {
            i += 1;
            continue;
        }
        assert!(
            value_flags.contains(&args[i].as_str()),
            "unknown flag {}",
            args[i]
        );
        assert!(
            i + 1 < args.len() && !args[i + 1].starts_with("--"),
            "missing value for {}",
            args[i]
        );
        i += 2;
    }
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has = |k: &str| args.iter().any(|a| a == k);

    let dex = conformance::load_dex();
    let prior = Arc::new(load_meta_pool(Path::new(
        "data/belief-pool-v3/belief-pool.json",
    )));
    let catalog = load_catalog_teams(Path::new("data/team-pool-v2/team-pool.json"));
    let compiled_path = get("--compiled").unwrap_or_else(|| "data/compiled-preview-v1.json".into());
    let compiled = CompiledTable::load(Path::new(&compiled_path));
    use sha2::{Digest, Sha256};
    let inputs_sha256: Value = [
        compiled_path.as_str(),
        "data/belief-pool-v3/belief-pool.json",
        "data/team-pool-v2/team-pool.json",
        "data/training-v3/rentals.json",
    ]
    .into_iter()
    .map(|path| {
        (
            path.to_owned(),
            json!(format!(
                "{:x}",
                Sha256::digest(std::fs::read(path).unwrap())
            )),
        )
    })
    .collect::<serde_json::Map<String, Value>>()
    .into();

    if has("--self-test") {
        run_self_test(&dex, &prior, &catalog, &compiled);
        return;
    }

    let arm_a = ArmKind::parse(get("--arm-a").as_deref().unwrap_or("ismcts"));
    let arm_b = ArmKind::parse(get("--arm-b").as_deref().unwrap_or("legacy"));
    let battle_iters: u32 = get("--iters").map(|s| s.parse().unwrap()).unwrap_or(27_000);
    let preview_iters: u32 = get("--preview-iters")
        .map(|s| s.parse().unwrap())
        .unwrap_or(battle_iters);
    let c: f64 = get("--c").map(|s| s.parse().unwrap()).unwrap_or(0.4);
    let hp_buckets: i64 = get("--hp-buckets")
        .map(|s| s.parse().unwrap())
        .unwrap_or(16);
    let max_turns: u16 = get("--max-turns")
        .map(|s| s.parse().unwrap())
        .unwrap_or(1001);
    let pairs: usize = get("--pairs").map(|s| s.parse().unwrap()).unwrap_or(16);
    let start_pair: usize = get("--start-pair").map(|s| s.parse().unwrap()).unwrap_or(0);
    let end_pair: usize = get("--end-pair")
        .map(|s| s.parse().unwrap())
        .unwrap_or(pairs);
    let stride: usize = get("--stride").map(|s| s.parse().unwrap()).unwrap_or(1);
    let offset: usize = get("--offset").map(|s| s.parse().unwrap()).unwrap_or(0);
    assert!(start_pair < end_pair && end_pair <= pairs && stride >= 1 && offset < stride);
    let seed: u64 = get("--seed")
        .map(|s| s.parse().unwrap())
        .unwrap_or(20261113);
    let threads: usize = get("--threads")
        .map(|s| s.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let preview_mode = PreviewMode::parse(get("--preview").as_deref().unwrap_or("compiled"));
    let pool_mode = PoolMode::parse(get("--pool").as_deref().unwrap_or("catalog"));
    let out_path = get("--out");

    assert!(battle_iters > 0 && preview_iters > 0 && threads > 0);
    assert!(c.is_finite() && c >= 0.0 && hp_buckets >= 0 && max_turns > 0);
    let offpool = if pool_mode == PoolMode::Catalog {
        Vec::new()
    } else {
        load_offpool_rentals(&dex, &prior, Path::new("data/training-v3/rentals.json"))
    };

    let schedules: Vec<PairSchedule> = (start_pair..end_pair)
        .filter(|&p| p % stride == offset)
        .map(|p| schedule_for_pair(p, seed, pool_mode, &catalog, &offpool))
        .collect();

    assert!(!schedules.is_empty(), "empty shard");
    let started = Instant::now();
    let mut game_jobs: Vec<(usize, usize)> = Vec::with_capacity(schedules.len() * 2);
    for idx in 0..schedules.len() {
        game_jobs.push((idx, 0));
        game_jobs.push((idx, 1));
    }
    let cursor = AtomicUsize::new(0);
    let slot_results: Mutex<Vec<Option<GameExec>>> =
        Mutex::new((0..game_jobs.len()).map(|_| None).collect());

    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let job_idx = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(&(sched_idx, a_side)) = game_jobs.get(job_idx) else {
                    break;
                };
                let sched = &schedules[sched_idx];
                let g = play_single_game(
                    &dex,
                    &prior,
                    sched,
                    arm_a,
                    arm_b,
                    a_side,
                    battle_iters,
                    preview_iters,
                    c,
                    hp_buckets,
                    max_turns,
                    preview_mode,
                    Some(&compiled),
                );
                slot_results.lock().unwrap()[job_idx] = Some(g);
            });
        }
    });

    let mut raw_games = slot_results.into_inner().unwrap();
    let mut done: Vec<(usize, Value, [GameExec; 2])> = Vec::with_capacity(schedules.len());
    for (sched_idx, sched) in schedules.iter().enumerate() {
        let g0 = raw_games[sched_idx * 2].take().unwrap();
        let g1 = raw_games[sched_idx * 2 + 1].take().unwrap();
        let pair_score = 0.5 * (g0.a_score + g1.a_score);
        let rec = json!({
            "pair": sched.pair_idx,
            "battle_seed": sched.battle_seed,
            "agent_seed": sched.agent_seed,
            "preview_pick_seeds": sched.preview_pick_seeds,
            "teams": sched.team_ids,
            "pair_score": pair_score,
            "g0_a_score": g0.a_score,
            "g1_a_score": g1.a_score,
            "g0_turns": g0.turns,
            "g1_turns": g1.turns,
            "g0_capped": g0.capped,
            "g1_capped": g1.capped,
            "a_battle_decisions": g0.a_battle_ns.len() + g1.a_battle_ns.len(),
            "b_battle_decisions": g0.b_battle_ns.len() + g1.b_battle_ns.len(),
            "a_battle_ms_sum": (g0.a_battle_ns.iter().sum::<u64>() + g1.a_battle_ns.iter().sum::<u64>()) as f64 / 1e6,
            "b_battle_ms_sum": (g0.b_battle_ns.iter().sum::<u64>() + g1.b_battle_ns.iter().sum::<u64>()) as f64 / 1e6,
            "a_battle_ns": [g0.a_battle_ns.clone(), g1.a_battle_ns.clone()],
            "b_battle_ns": [g0.b_battle_ns.clone(), g1.b_battle_ns.clone()],
            "a_preview_ns": [g0.a_preview_ns.clone(), g1.a_preview_ns.clone()],
            "b_preview_ns": [g0.b_preview_ns.clone(), g1.b_preview_ns.clone()],
            "a_compiled_hits": g0.a_compiled_hits + g1.a_compiled_hits,
            "b_compiled_hits": g0.b_compiled_hits + g1.b_compiled_hits,
            "a_fallback": g0.a_fallback_used || g1.a_fallback_used,
            "b_fallback": g0.b_fallback_used || g1.b_fallback_used,
            "g0_outcome": g0.outcome,
            "g1_outcome": g1.outcome,
        });
        done.push((sched.pair_idx, rec, [g0, g1]));
    }

    if let Some(path) = out_path {
        if let Some(parent) = Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut f = std::fs::File::create(&path).unwrap();
        for (_, rec, _) in &done {
            writeln!(f, "{}", rec).unwrap();
        }
    }

    let n_pairs = done.len();
    let n_games = n_pairs * 2;
    let mut pair_scores = Vec::with_capacity(n_pairs);
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut ties = 0usize;
    let mut capped = 0usize;
    let mut turn_sum = 0u64;
    let mut a_ns_all = Vec::new();
    let mut b_ns_all = Vec::new();
    let mut a_fallback_pairs = 0usize;
    let mut b_fallback_pairs = 0usize;
    let mut a_compiled_games = 0usize;
    let mut b_compiled_games = 0usize;

    for (_, rec, [g0, g1]) in done {
        pair_scores.push(0.5 * (g0.a_score + g1.a_score));
        for g in [&g0, &g1] {
            turn_sum += u64::from(g.turns);
            if g.capped {
                capped += 1;
            }
            if g.capped {
                // Operational caps are unresolved, never engine draws.
            } else if g.a_score == 1.0 {
                wins += 1;
            } else if g.a_score == 0.0 {
                losses += 1;
            } else {
                ties += 1;
            }
            if g.a_compiled_hits > 0 {
                a_compiled_games += 1;
            }
            if g.b_compiled_hits > 0 {
                b_compiled_games += 1;
            }
        }
        a_ns_all.extend(g0.a_battle_ns);
        a_ns_all.extend(g1.a_battle_ns);
        b_ns_all.extend(g0.b_battle_ns);
        b_ns_all.extend(g1.b_battle_ns);
        if rec["a_fallback"].as_bool().unwrap() {
            a_fallback_pairs += 1;
        }
        if rec["b_fallback"].as_bool().unwrap() {
            b_fallback_pairs += 1;
        }
    }

    let mean: f64 = pair_scores.iter().sum::<f64>() / n_pairs as f64;
    let var: f64 = if n_pairs > 1 {
        pair_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n_pairs - 1) as f64
    } else {
        0.0
    };
    let se = (var / n_pairs as f64).sqrt();
    let ci95 = 1.96 * se;

    a_ns_all.sort_unstable();
    b_ns_all.sort_unstable();
    let a_mean_ms = if a_ns_all.is_empty() {
        0.0
    } else {
        a_ns_all.iter().sum::<u64>() as f64 / a_ns_all.len() as f64 / 1e6
    };
    let b_mean_ms = if b_ns_all.is_empty() {
        0.0
    } else {
        b_ns_all.iter().sum::<u64>() as f64 / b_ns_all.len() as f64 / 1e6
    };
    let elapsed_s = started.elapsed().as_secs_f64();

    println!(
        "{}",
        json!({
            "arm_a": arm_a.as_str(),
            "arm_b": arm_b.as_str(),
            "battle_iters": battle_iters,
            "preview_iters": preview_iters,
            "c": c,
            "hp_buckets": hp_buckets,
            "preview_mode": preview_mode.as_str(),
            "pool_mode": pool_mode.as_str(),
            "pairs": n_pairs,
            "games": n_games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "capped": capped,
            "score": (mean * 1e6).round() / 1e6,
            "se": (se * 1e6).round() / 1e6,
            "ci95_pair_normal": [(mean-ci95).max(0.0), (mean+ci95).min(1.0)],
            "score_bounds": [mean - 0.5*capped as f64/n_games as f64, mean + 0.5*capped as f64/n_games as f64],
            "score_convention": "win=1, terminal tie=0.5, loss=0; unresolved=0.5 midpoint with bounds",
            "seed": seed,
            "start_pair": start_pair,
            "end_pair": end_pair,
            "stride": stride,
            "offset": offset,
            "threads": threads,
            "max_turns": max_turns,
            "inputs_sha256": inputs_sha256,
            "harness_sha256": format!("{:x}", Sha256::digest(include_bytes!("ismcts_rollout_duel.rs"))),
            "build_fingerprint": env!("M17E_SOLVER_BUILD_FINGERPRINT"),
            "a_config": format!("{:?}", arm_a.config(battle_iters, c, hp_buckets)),
            "b_config": format!("{:?}", arm_b.config(battle_iters, c, hp_buckets)),
            "avg_turns": (turn_sum as f64 / n_games as f64 * 100.0).round() / 100.0,
            "a_decisions": a_ns_all.len(),
            "b_decisions": b_ns_all.len(),
            "a_mean_ms": (a_mean_ms * 100.0).round() / 100.0,
            "b_mean_ms": (b_mean_ms * 100.0).round() / 100.0,
            "a_p95_ms": (percentile_ms(&a_ns_all, 0.95) * 100.0).round() / 100.0,
            "b_p95_ms": (percentile_ms(&b_ns_all, 0.95) * 100.0).round() / 100.0,
            "a_p99_ms": (percentile_ms(&a_ns_all, 0.99) * 100.0).round() / 100.0,
            "b_p99_ms": (percentile_ms(&b_ns_all, 0.99) * 100.0).round() / 100.0,
            "a_compiled_games": a_compiled_games,
            "b_compiled_games": b_compiled_games,
            "a_fallback_pairs": a_fallback_pairs,
            "b_fallback_pairs": b_fallback_pairs,
            "elapsed_s": (elapsed_s * 100.0).round() / 100.0,
            "games_per_s": ((n_games as f64 / elapsed_s) * 100.0).round() / 100.0
        })
    );
}
