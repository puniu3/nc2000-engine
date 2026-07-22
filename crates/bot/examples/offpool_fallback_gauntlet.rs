//! M17d off-pool fallback-policy A/B gauntlet.
//!
//! The evaluated side always owns a meta-pool pilot team and observes a
//! custom opponent that is certified to match no meta candidate even at
//! team preview. Both arms use the same battle/agent seeds and the same
//! blind reference opponent; the only changed variable is fallback policy:
//!
//! - legacy: revealed moves -> same-species meta set -> empty set
//!   (the engine safely supplies implicit Struggle)
//! - layered: revealed moves -> meta -> community rentals -> learnset
//!
//! Baked preview tables are structurally disabled (`None`) for every agent.
//! JSONL contains the full selected teams, source and executable/data
//! fingerprints, exact seeds, per-arm outcomes, caps, and invalids.
//!
//! Lightweight default:
//!
//!   cargo run --release -p nc2000-bot --example offpool_fallback_gauntlet -- \
//!     --profile smoke --allow-incomplete --out tmp/m17d-smoke.jsonl
//!
//! Determinism/self-contract check (no full run):
//!
//!   cargo run -p nc2000-bot --example offpool_fallback_gauntlet -- --self-test

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use conformance::fixture::{corpus_files, repo_root};
use conformance::load_dex;
use nc2000_bot::preview::{team_sig, MetaPool};
use nc2000_bot::teamgen::{to_sets, TeamGen};
use nc2000_bot::{
    play_game, Belief, BlindAgent, FallbackPolicy, FallbackSource, GameResult, Observer, RmConfig,
    SplitMix64,
};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::Battle;
use serde::Serialize;
use serde_json::{json, Value};

const SCHEMA: &str = "nc2000-m17d-offpool-gauntlet-v1";

#[derive(Clone, Debug, Serialize)]
struct Config {
    profile: String,
    community: usize,
    fixtures: usize,
    mutations: usize,
    pilots: usize,
    games_per_matchup: usize,
    agent_iters: u32,
    max_turns: u16,
    threads: usize,
    seed: u64,
    allow_incomplete: bool,
}

impl Config {
    fn smoke() -> Self {
        Self {
            profile: "smoke".into(),
            community: 1,
            fixtures: 1,
            mutations: 1,
            pilots: 1,
            games_per_matchup: 2,
            agent_iters: 5,
            max_turns: 80,
            threads: 2,
            seed: 1,
            allow_incomplete: false,
        }
    }

    fn full() -> Self {
        Self {
            profile: "full".into(),
            community: 2,
            fixtures: 8,
            mutations: 8,
            pilots: 4,
            games_per_matchup: 64,
            agent_iters: 3_000,
            max_turns: 500,
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            seed: 1,
            allow_incomplete: false,
        }
    }
}

#[derive(Clone)]
struct CustomTeam {
    id: String,
    source: Value,
    canonical: Vec<Value>,
    sets: Vec<PokemonSet>,
    fingerprint: String,
    fallback_layers: Vec<Value>,
    meta_absent_species: usize,
}

#[derive(Clone)]
struct PilotTeam {
    id: String,
    canonical: Vec<Value>,
    sets: Vec<PokemonSet>,
    fingerprint: String,
}

#[derive(Clone, Debug, Serialize)]
struct Job {
    index: usize,
    custom: usize,
    pilot: usize,
    game: usize,
    custom_is_p1: bool,
    battle_seed: String,
    evaluated_agent_seed: u64,
    reference_agent_seed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ArmResult {
    policy: &'static str,
    status: &'static str,
    outcome: Option<&'static str>,
    score: Option<f64>,
    turns: u16,
    capped: bool,
    evaluated_fallback: bool,
    reference_fallback: bool,
    error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct PairRow {
    schema: &'static str,
    kind: &'static str,
    run_fingerprint: String,
    job: usize,
    custom_id: String,
    custom_fingerprint: String,
    pilot_id: String,
    pilot_fingerprint: String,
    game: usize,
    custom_is_p1: bool,
    battle_seed: String,
    evaluated_agent_seed: u64,
    reference_agent_seed: u64,
    legacy: ArmResult,
    layered: ArmResult,
    delta_layered_minus_legacy: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Summary {
    pairs: usize,
    valid_pairs: usize,
    excluded_pairs: usize,
    both_incomplete_pairs: usize,
    legacy_only_incomplete_pairs: usize,
    layered_only_incomplete_pairs: usize,
    asymmetric_incomplete_pairs: usize,
    legacy_mean: Option<f64>,
    layered_mean: Option<f64>,
    mean_paired_delta: Option<f64>,
    delta_positive: usize,
    delta_zero: usize,
    delta_negative: usize,
    legacy_invalid: usize,
    layered_invalid: usize,
    legacy_invalid_rate: f64,
    layered_invalid_rate: f64,
    legacy_capped: usize,
    layered_capped: usize,
    legacy_cap_rate: f64,
    layered_cap_rate: f64,
    certified: bool,
    certification_failures: Vec<String>,
    result_fingerprint: String,
}

struct Prepared {
    dex: Arc<Dex>,
    pool: Arc<MetaPool>,
    custom: Vec<CustomTeam>,
    pilots: Vec<PilotTeam>,
    inputs: Value,
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn number<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    flag(args, name)
        .map(|value| value.parse().expect(name))
        .unwrap_or(default)
}

fn parse_config(args: &[String]) -> Config {
    let profile = flag(args, "--profile").unwrap_or_else(|| "smoke".into());
    let mut cfg = match profile.as_str() {
        "smoke" => Config::smoke(),
        "full" => Config::full(),
        _ => panic!("--profile must be smoke or full"),
    };
    cfg.community = number(args, "--community", cfg.community);
    cfg.fixtures = number(args, "--fixtures", cfg.fixtures);
    cfg.mutations = number(args, "--mutations", cfg.mutations);
    cfg.pilots = number(args, "--pilots", cfg.pilots);
    cfg.games_per_matchup = number(args, "--games", cfg.games_per_matchup);
    cfg.agent_iters = number(args, "--iters", cfg.agent_iters);
    cfg.max_turns = number(args, "--max-turns", cfg.max_turns);
    cfg.threads = number(args, "--threads", cfg.threads).max(1);
    cfg.seed = number(args, "--seed", cfg.seed);
    cfg.allow_incomplete = args.iter().any(|arg| arg == "--allow-incomplete");
    assert!(
        cfg.community + cfg.fixtures + cfg.mutations > 0,
        "empty custom gauntlet"
    );
    assert!(cfg.pilots > 0, "--pilots must be positive");
    assert!(cfg.games_per_matchup > 0, "--games must be positive");
    assert!(cfg.agent_iters > 0, "--iters must be positive");
    cfg
}

fn fnv_update(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn fingerprint(tag: &str, bytes: &[u8]) -> String {
    let mut hash = fnv_update(0xcbf2_9ce4_8422_2325, tag.as_bytes());
    hash = fnv_update(hash, &(bytes.len() as u64).to_le_bytes());
    hash = fnv_update(hash, bytes);
    format!("fnv1a64:{hash:016x}:{tag}")
}

fn file_fingerprint(path: &Path, tag: &str) -> String {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read {} for fingerprint: {error}", path.display()));
    fingerprint(tag, &bytes)
}

fn canonical_fingerprint(tag: &str, value: &impl Serialize) -> String {
    fingerprint(tag, &serde_json::to_vec(value).unwrap())
}

fn preview_fallback_layers(
    dex: &Dex,
    pool: &MetaPool,
    pilot: &[PokemonSet],
    custom: &[PokemonSet],
) -> Option<Vec<Value>> {
    let battle = Battle::from_fixture(dex, "1,2,3,4", pilot, custom)
        .expect("preflight battle must construct");
    let observer = Observer::new(&battle, 0);
    let legacy = Belief::with_fallback_policy(dex, pool, &observer, FallbackPolicy::LegacyMetaOnly);
    let layered = Belief::with_fallback_policy(dex, pool, &observer, FallbackPolicy::Layered);
    if !legacy.is_fallback() || !layered.is_fallback() {
        return None;
    }
    let legacy_sources = legacy.fallback_sources(dex, &observer);
    let layered_sources = layered.fallback_sources(dex, &observer);
    let legacy_det = legacy.determinize(
        dex,
        &battle,
        &observer,
        &mut SplitMix64::new(0x4c45_4741_4359),
    );
    let layered_det = layered.determinize(
        dex,
        &battle,
        &observer,
        &mut SplitMix64::new(0x4c41_5945_5245_44),
    );
    let mut layers = Vec::with_capacity(custom.len());
    for (slot, ((legacy_source, layered_source), set)) in legacy_sources
        .iter()
        .zip(&layered_sources)
        .zip(custom)
        .enumerate()
    {
        let legacy_moves = legacy_det.sides[1].roster[slot].base_move_slots.len();
        let layered_moves = layered_det.sides[1].roster[slot].base_move_slots.len();
        match legacy_source {
            FallbackSource::LegacyEmpty => {
                assert_ne!(*layered_source, FallbackSource::Meta);
                assert_eq!(legacy_moves, 0, "legacy meta-absent mon must be empty");
                assert!(
                    layered_moves > 0,
                    "layered meta-absent mon must be playable"
                );
            }
            FallbackSource::Meta => {
                assert_eq!(*layered_source, FallbackSource::Meta);
                assert!(legacy_moves > 0 && layered_moves > 0);
            }
            _ => unreachable!("legacy policy selected a non-legacy source"),
        }
        layers.push(json!({
            "slot": slot,
            "species": set.species,
            "legacy_source": legacy_source.id(),
            "layered_source": layered_source.id(),
            "legacy_move_slots": legacy_moves,
            "layered_move_slots": layered_moves,
        }));
    }
    layers
        .iter()
        .any(|layer| layer["legacy_source"] == "legacy-empty")
        .then_some(layers)
}

fn is_preview_fallback(
    dex: &Dex,
    pool: &MetaPool,
    pilot: &[PokemonSet],
    custom: &[PokemonSet],
) -> bool {
    preview_fallback_layers(dex, pool, pilot, custom).is_some()
}

fn accept_custom(
    dex: &Dex,
    pool: &MetaPool,
    gen: &TeamGen,
    raw: &[Value],
    id: String,
    source: Value,
    pilot: &[PokemonSet],
    pool_signatures: &HashSet<String>,
    seen: &mut HashSet<String>,
) -> Option<CustomTeam> {
    let canonical = gen.canonize(dex, raw)?;
    let sets = to_sets(&canonical).ok()?;
    if sets.len() != 6 {
        return None;
    }
    if sets.iter().all(|set| {
        pool.teams
            .iter()
            .flat_map(|team| &team.sets)
            .any(|meta| toid(&meta.species) == toid(&set.species))
    }) {
        return None;
    }
    let exact = format!("{:?}", team_sig(dex, &sets));
    if pool_signatures.contains(&exact) {
        return None;
    }
    let fallback_layers = preview_fallback_layers(dex, pool, pilot, &sets)?;
    let meta_absent_species = fallback_layers
        .iter()
        .filter(|layer| layer["legacy_source"] == "legacy-empty")
        .count();
    assert!(meta_absent_species > 0);
    let fingerprint = canonical_fingerprint("m17d-team-v1", &canonical);
    if !seen.insert(fingerprint.clone()) {
        return None;
    }
    Some(CustomTeam {
        id,
        source,
        canonical,
        sets,
        fingerprint,
        fallback_layers,
        meta_absent_species,
    })
}

fn fixture_paths(root: &Path) -> Vec<PathBuf> {
    let corpus = root.join("fixtures/corpus-v1");
    let mut paths = Vec::new();
    for subdir in ["directed", "directed-sleep", "full", "puredata"] {
        paths.extend(corpus_files(&corpus.join(subdir)));
    }
    paths.sort();
    paths
}

fn prepare(cfg: &Config) -> Prepared {
    let root = repo_root();
    let dex = Arc::new(load_dex());
    let dex_path = root.join("data/gen2stadium2.json");
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let rentals_path = root.join("data/community-rentals-v0/teams.json");
    let learnsets_path = root.join("data/learnsets-gen2.json");
    let pool_text = std::fs::read_to_string(&pool_path).unwrap();
    let rentals_text = std::fs::read_to_string(&rentals_path).unwrap();
    let learnsets_text = std::fs::read_to_string(&learnsets_path).unwrap();
    let pool: Arc<MetaPool> = Arc::new(serde_json::from_str(&pool_text).unwrap());
    let gen = TeamGen::new(&dex, &learnsets_text, &pool_text).unwrap();

    let mut pilots = Vec::new();
    for i in 0..cfg.pilots.min(pool.teams.len()) {
        let canonical = gen.canonize(&dex, &gen.team_json(i)).unwrap();
        let sets = to_sets(&canonical).unwrap();
        pilots.push(PilotTeam {
            id: pool.teams[i].id.clone(),
            fingerprint: canonical_fingerprint("m17d-team-v1", &canonical),
            canonical,
            sets,
        });
    }
    assert_eq!(pilots.len(), cfg.pilots, "not enough pilot teams");
    let preflight_pilot = &pilots[0].sets;

    let pool_signatures: HashSet<String> = pool
        .teams
        .iter()
        .map(|team| format!("{:?}", team_sig(&dex, &team.sets)))
        .collect();
    let mut seen = HashSet::new();
    let mut custom = Vec::new();

    let rentals: Value = serde_json::from_str(&rentals_text).unwrap();
    for team in rentals["teams"].as_array().unwrap() {
        if custom
            .iter()
            .filter(|team: &&CustomTeam| team.source["kind"] == "community")
            .count()
            >= cfg.community
        {
            break;
        }
        let cban = team["cban"].as_u64().unwrap();
        let raw = team["sets"].as_array().unwrap();
        if let Some(team) = accept_custom(
            &dex,
            &pool,
            &gen,
            raw,
            format!("community-cban-{cban:02}"),
            json!({
                "kind": "community",
                "cban": cban,
                "archetype": team["archetype"],
                "source_file": "data/community-rentals-v0/teams.json",
            }),
            preflight_pilot,
            &pool_signatures,
            &mut seen,
        ) {
            custom.push(team);
        }
    }
    assert_eq!(
        custom
            .iter()
            .filter(|team| team.source["kind"] == "community")
            .count(),
        cfg.community,
        "not enough legal preview-off-pool community rentals"
    );

    let fixtures = fixture_paths(&root);
    for path in &fixtures {
        if custom
            .iter()
            .filter(|team| team.source["kind"] == "fixture")
            .count()
            >= cfg.fixtures
        {
            break;
        }
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for (side, key) in [("p1", "p1team"), ("p2", "p2team")] {
            let raw = fixture[key].as_array().unwrap();
            let relative = path
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .to_string();
            if let Some(team) = accept_custom(
                &dex,
                &pool,
                &gen,
                raw,
                format!("fixture-{}-{side}", relative.replace('/', "-")),
                json!({
                    "kind": "fixture",
                    "path": relative,
                    "side": side,
                    "file_fingerprint": file_fingerprint(path, "m17d-fixture-file-v1"),
                    "fixture_seed": fixture["seed"],
                }),
                preflight_pilot,
                &pool_signatures,
                &mut seen,
            ) {
                custom.push(team);
            }
            if custom
                .iter()
                .filter(|team| team.source["kind"] == "fixture")
                .count()
                >= cfg.fixtures
            {
                break;
            }
        }
    }
    assert_eq!(
        custom
            .iter()
            .filter(|team| team.source["kind"] == "fixture")
            .count(),
        cfg.fixtures,
        "not enough legal preview-off-pool fixture teams"
    );

    let mut mutation_rng = SplitMix64::new(cfg.seed ^ 0x6d31_3764_6d75_7461);
    for mutation_index in 0..cfg.mutations {
        let seed_pool_index = mutation_index % pool.teams.len();
        let mut parent = gen.canonize(&dex, &gen.team_json(seed_pool_index)).unwrap();
        let mut chain = Vec::new();
        let mut accepted = None;
        for attempt in 0..10_000usize {
            let proposal = gen
                .propose_valid(&dex, &parent, &mut mutation_rng, 128)
                .expect("teamgen exhausted legal mutation proposals");
            chain
                .push(json!({"attempt": attempt, "op": proposal.op.name(), "slot": proposal.slot}));
            parent = proposal.team;
            let source = json!({
                "kind": "mutation",
                "seed_pool_index": seed_pool_index,
                "seed_pool_id": pool.teams[seed_pool_index].id,
                "generator_seed": cfg.seed ^ 0x6d31_3764_6d75_7461,
                "chain": chain,
            });
            if let Some(team) = accept_custom(
                &dex,
                &pool,
                &gen,
                &parent,
                format!("mutation-{mutation_index:02}"),
                source,
                preflight_pilot,
                &pool_signatures,
                &mut seen,
            ) {
                accepted = Some(team);
                break;
            }
        }
        custom.push(accepted.expect("failed to generate unique preview-off-pool mutation"));
    }

    for team in &custom {
        assert!(!pool_signatures.contains(&format!("{:?}", team_sig(&dex, &team.sets))));
        assert!(is_preview_fallback(
            &dex,
            &pool,
            preflight_pilot,
            &team.sets
        ));
        assert!(team.meta_absent_species > 0);
        assert!(team.fallback_layers.iter().any(|layer| {
            layer["legacy_source"] == "legacy-empty"
                && layer["layered_source"] != "meta"
                && layer["legacy_move_slots"] == 0
                && layer["layered_move_slots"]
                    .as_u64()
                    .is_some_and(|moves| moves > 0)
        }));
    }

    let fixture_manifest: Vec<_> = fixtures
        .iter()
        .map(|path| {
            json!({
                "path": path.strip_prefix(&root).unwrap().to_string_lossy(),
                "fingerprint": file_fingerprint(path, "m17d-fixture-file-v1"),
            })
        })
        .collect();
    let inputs = json!({
        "dex": file_fingerprint(&dex_path, "m17d-dex-v1"),
        "meta_pool": file_fingerprint(&pool_path, "m17d-meta-pool-v1"),
        "community_rentals": file_fingerprint(&rentals_path, "m17d-community-rentals-v1"),
        "learnsets": file_fingerprint(&learnsets_path, "m17d-learnsets-v1"),
        "fixture_manifest": canonical_fingerprint("m17d-fixture-manifest-v1", &fixture_manifest),
    });
    Prepared {
        dex,
        pool,
        custom,
        pilots,
        inputs,
    }
}

fn plan_jobs(cfg: &Config, prepared: &Prepared) -> Vec<Job> {
    let mut jobs = Vec::new();
    for custom in 0..prepared.custom.len() {
        for pilot in 0..prepared.pilots.len() {
            let mut rng = SplitMix64::new(
                cfg.seed
                    ^ ((custom as u64 + 1).wrapping_mul(0x9fb2_1c65_1e98_df25))
                    ^ ((pilot as u64 + 1).wrapping_mul(0xa24b_aed4_963e_e407)),
            );
            let mut battle_seed = String::new();
            for game in 0..cfg.games_per_matchup {
                if game % 2 == 0 {
                    battle_seed = rng.battle_seed();
                }
                jobs.push(Job {
                    index: jobs.len(),
                    custom,
                    pilot,
                    game,
                    custom_is_p1: game % 2 == 1,
                    battle_seed: battle_seed.clone(),
                    evaluated_agent_seed: rng.next(),
                    reference_agent_seed: rng.next(),
                });
            }
        }
    }
    jobs
}

fn play_arm(
    dex: &Dex,
    pool: Arc<MetaPool>,
    custom: &[PokemonSet],
    pilot: &[PokemonSet],
    job: &Job,
    policy: FallbackPolicy,
    cfg: &Config,
) -> ArmResult {
    let policy_name = policy.id();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (p1, p2) = if job.custom_is_p1 {
            (custom, pilot)
        } else {
            (pilot, custom)
        };
        let mut battle = Battle::from_fixture(dex, &job.battle_seed, p1, p2)
            .map_err(|error| format!("construct: {error:?}"))?;
        battle.set_log_enabled(true);
        let agent_cfg = RmConfig {
            iterations: cfg.agent_iters,
            ..Default::default()
        };
        let mut evaluated = BlindAgent::new_with_fallback_policy(
            agent_cfg.clone(),
            pool.clone(),
            None,
            job.evaluated_agent_seed,
            policy,
        );
        let mut reference = BlindAgent::new(agent_cfg, pool, None, job.reference_agent_seed);
        let game_result = if job.custom_is_p1 {
            play_game(
                dex,
                &mut battle,
                &mut [&mut reference, &mut evaluated],
                cfg.max_turns,
            )
        } else {
            play_game(
                dex,
                &mut battle,
                &mut [&mut evaluated, &mut reference],
                cfg.max_turns,
            )
        }
        .map_err(|error| format!("play: {error:?}"))?;
        let evaluated_fallback = evaluated.belief().is_some_and(Belief::is_fallback);
        let reference_fallback = reference.belief().is_some_and(Belief::is_fallback);
        if !evaluated_fallback {
            return Err("evaluated belief was not in fallback".to_string());
        }
        if reference_fallback {
            return Err("reference belief unexpectedly fell back on pilot team".to_string());
        }
        let (status, outcome, p1_score, capped) = match game_result {
            GameResult::Outcome(Outcome::P1Win) => ("outcome", Some("p1-win"), Some(1.0), false),
            GameResult::Outcome(Outcome::P2Win) => ("outcome", Some("p2-win"), Some(0.0), false),
            GameResult::Outcome(Outcome::Tie) => ("outcome", Some("tie"), Some(0.5), false),
            GameResult::TurnCapped => ("capped", None, None, true),
        };
        let score = p1_score.map(|p1_score| {
            if job.custom_is_p1 {
                1.0 - p1_score
            } else {
                p1_score
            }
        });
        Ok(ArmResult {
            policy: policy_name,
            status,
            outcome,
            score,
            turns: battle.turn,
            capped,
            evaluated_fallback,
            reference_fallback,
            error: None,
        })
    }));
    match result {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => ArmResult {
            policy: policy_name,
            status: "invalid",
            outcome: None,
            score: None,
            turns: 0,
            capped: false,
            evaluated_fallback: false,
            reference_fallback: false,
            error: Some(error),
        },
        Err(_) => ArmResult {
            policy: policy_name,
            status: "invalid",
            outcome: None,
            score: None,
            turns: 0,
            capped: false,
            evaluated_fallback: false,
            reference_fallback: false,
            error: Some("panic".into()),
        },
    }
}

fn run_rows(
    cfg: &Config,
    prepared: &Prepared,
    jobs: &[Job],
    run_fingerprint: &str,
    threads: usize,
) -> Vec<PairRow> {
    let cursor = AtomicUsize::new(0);
    let mut unordered = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads.max(1) {
            let cursor = &cursor;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else { break };
                    let custom = &prepared.custom[job.custom];
                    let pilot = &prepared.pilots[job.pilot];
                    let legacy = play_arm(
                        &prepared.dex,
                        prepared.pool.clone(),
                        &custom.sets,
                        &pilot.sets,
                        job,
                        FallbackPolicy::LegacyMetaOnly,
                        cfg,
                    );
                    let layered = play_arm(
                        &prepared.dex,
                        prepared.pool.clone(),
                        &custom.sets,
                        &pilot.sets,
                        job,
                        FallbackPolicy::Layered,
                        cfg,
                    );
                    let delta_layered_minus_legacy =
                        layered.score.zip(legacy.score).map(|(new, old)| new - old);
                    local.push(PairRow {
                        schema: SCHEMA,
                        kind: "pair",
                        run_fingerprint: run_fingerprint.to_string(),
                        job: job.index,
                        custom_id: custom.id.clone(),
                        custom_fingerprint: custom.fingerprint.clone(),
                        pilot_id: pilot.id.clone(),
                        pilot_fingerprint: pilot.fingerprint.clone(),
                        game: job.game,
                        custom_is_p1: job.custom_is_p1,
                        battle_seed: job.battle_seed.clone(),
                        evaluated_agent_seed: job.evaluated_agent_seed,
                        reference_agent_seed: job.reference_agent_seed,
                        legacy,
                        layered,
                        delta_layered_minus_legacy,
                    });
                }
                local
            }));
        }
        for handle in handles {
            unordered.extend(handle.join().expect("gauntlet worker panicked"));
        }
    });
    unordered.sort_by_key(|row| row.job);
    unordered
}

fn summarize(rows: &[PairRow]) -> Summary {
    let valid: Vec<_> = rows
        .iter()
        .filter_map(|row| row.layered.score.zip(row.legacy.score))
        .collect();
    let mean = |side: fn(&(f64, f64)) -> f64| {
        (!valid.is_empty()).then(|| valid.iter().map(side).sum::<f64>() / valid.len() as f64)
    };
    let deltas: Vec<_> = valid.iter().map(|&(new, old)| new - old).collect();
    let rate = |count: usize, total: usize| {
        if total == 0 {
            0.0
        } else {
            count as f64 / total as f64
        }
    };
    let legacy_capped = rows.iter().filter(|row| row.legacy.capped).count();
    let layered_capped = rows.iter().filter(|row| row.layered.capped).count();
    let legacy_invalid = rows
        .iter()
        .filter(|row| row.legacy.status == "invalid")
        .count();
    let layered_invalid = rows
        .iter()
        .filter(|row| row.layered.status == "invalid")
        .count();
    let legacy_only_incomplete_pairs = rows
        .iter()
        .filter(|row| row.legacy.status != "outcome" && row.layered.status == "outcome")
        .count();
    let layered_only_incomplete_pairs = rows
        .iter()
        .filter(|row| row.legacy.status == "outcome" && row.layered.status != "outcome")
        .count();
    let both_incomplete_pairs = rows
        .iter()
        .filter(|row| row.legacy.status != "outcome" && row.layered.status != "outcome")
        .count();
    let asymmetric_incomplete_pairs = legacy_only_incomplete_pairs + layered_only_incomplete_pairs;
    let legacy_cap_rate = rate(legacy_capped, rows.len());
    let layered_cap_rate = rate(layered_capped, rows.len());
    let mut certification_failures = Vec::new();
    if legacy_invalid + layered_invalid > 0 {
        certification_failures.push(format!(
            "invalid arms: legacy={legacy_invalid}, layered={layered_invalid}"
        ));
    }
    if legacy_cap_rate > 0.01 || layered_cap_rate > 0.01 {
        certification_failures.push(format!(
            "cap rate exceeds 1%: legacy={legacy_cap_rate:.6}, layered={layered_cap_rate:.6}"
        ));
    }
    if asymmetric_incomplete_pairs > 0 {
        certification_failures.push(format!(
            "asymmetric incomplete pairs: {asymmetric_incomplete_pairs}"
        ));
    }
    let certified = certification_failures.is_empty();
    Summary {
        pairs: rows.len(),
        valid_pairs: valid.len(),
        excluded_pairs: rows.len() - valid.len(),
        both_incomplete_pairs,
        legacy_only_incomplete_pairs,
        layered_only_incomplete_pairs,
        asymmetric_incomplete_pairs,
        legacy_mean: mean(|&(_, old)| old),
        layered_mean: mean(|&(new, _)| new),
        mean_paired_delta: (!deltas.is_empty())
            .then(|| deltas.iter().sum::<f64>() / deltas.len() as f64),
        delta_positive: deltas.iter().filter(|&&delta| delta > 0.0).count(),
        delta_zero: deltas.iter().filter(|&&delta| delta == 0.0).count(),
        delta_negative: deltas.iter().filter(|&&delta| delta < 0.0).count(),
        legacy_invalid,
        layered_invalid,
        legacy_invalid_rate: rate(legacy_invalid, rows.len()),
        layered_invalid_rate: rate(layered_invalid, rows.len()),
        legacy_capped,
        layered_capped,
        legacy_cap_rate,
        layered_cap_rate,
        certified,
        certification_failures,
        result_fingerprint: canonical_fingerprint("m17d-paired-results-v1", &rows),
    }
}

fn run_identity(cfg: &Config, prepared: &Prepared, jobs: &[Job]) -> Value {
    let executable = std::env::current_exe().unwrap();
    let semantic_cfg = json!({
        "profile": cfg.profile,
        "community": cfg.community,
        "fixtures": cfg.fixtures,
        "mutations": cfg.mutations,
        "pilots": cfg.pilots,
        "games_per_matchup": cfg.games_per_matchup,
        "agent_iters": cfg.agent_iters,
        "max_turns": cfg.max_turns,
        "seed": cfg.seed,
        "agent": {
            "kind": "blind",
            "rm_config_defaults": "nc2000-bot/RmConfig::default",
            "evaluated_policy_a": FallbackPolicy::LegacyMetaOnly.id(),
            "evaluated_policy_b": FallbackPolicy::Layered.id(),
            "reference_policy": FallbackPolicy::Layered.id(),
        },
        "baked_preview_tables": false,
        "opponent_contract": "every custom exact-off-pool and preview-fallback",
        "pairing": "same battle/evaluated-agent/reference-agent seeds per A/B arm; orientations alternate; adjacent orientations share battle seed",
        "strength_filter": "include only pairs where both arms reached a terminal outcome; exclude caps and invalids",
        "certification": {
            "invalid_arms": 0,
            "max_cap_rate_per_arm": 0.01,
            "asymmetric_incomplete_pairs": 0,
        },
    });
    let selected_custom: Vec<_> = prepared
        .custom
        .iter()
        .map(|team| {
            json!({
                "id": team.id,
                "fingerprint": team.fingerprint,
                "source": team.source,
                "fallback_layers": team.fallback_layers,
                "meta_absent_species": team.meta_absent_species,
            })
        })
        .collect();
    let selected_pilots: Vec<_> = prepared
        .pilots
        .iter()
        .map(|team| json!({"id": team.id, "fingerprint": team.fingerprint}))
        .collect();
    json!({
        "schema": SCHEMA,
        "build": {
            "package_version": env!("CARGO_PKG_VERSION"),
            "executable": file_fingerprint(&executable, "m17d-executable-v1"),
        },
        "inputs": prepared.inputs,
        "selected_custom": selected_custom,
        "selected_pilots": selected_pilots,
        "semantic_config": semantic_cfg,
        "execution": {"threads": cfg.threads, "allow_incomplete": cfg.allow_incomplete},
        "workload_fingerprint": canonical_fingerprint("m17d-workload-v1", &jobs),
    })
}

fn semantic_run_fingerprint(identity: &Value) -> String {
    canonical_fingerprint(
        "m17d-run-v1",
        &json!({
            "schema": identity["schema"],
            "build": identity["build"],
            "inputs": identity["inputs"],
            "selected_custom": identity["selected_custom"],
            "selected_pilots": identity["selected_pilots"],
            "semantic_config": identity["semantic_config"],
            "workload_fingerprint": identity["workload_fingerprint"],
        }),
    )
}

fn header(run: Value, run_fingerprint: &str, prepared: &Prepared) -> Value {
    let custom: Vec<_> = prepared
        .custom
        .iter()
        .map(|team| {
            json!({
                "id": team.id,
                "fingerprint": team.fingerprint,
                "off_pool_exact": true,
                "preview_fallback": true,
                "source": team.source,
                "fallback_layers": team.fallback_layers,
                "meta_absent_species": team.meta_absent_species,
                "canonical_team": team.canonical,
            })
        })
        .collect();
    let pilots: Vec<_> = prepared
        .pilots
        .iter()
        .map(|team| {
            json!({
                "id": team.id,
                "fingerprint": team.fingerprint,
                "canonical_team": team.canonical,
            })
        })
        .collect();
    json!({
        "schema": SCHEMA,
        "kind": "run",
        "run_fingerprint": run_fingerprint,
        "run": run,
        "custom_teams": custom,
        "pilot_teams": pilots,
    })
}

fn self_test() {
    let mut cfg = Config::smoke();
    cfg.agent_iters = 1;
    cfg.max_turns = 1;
    let prepared = prepare(&cfg);
    let jobs = plan_jobs(&cfg, &prepared);
    assert_eq!(prepared.custom.len(), 3);
    let kinds: HashSet<_> = prepared
        .custom
        .iter()
        .map(|team| team.source["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, HashSet::from(["community", "fixture", "mutation"]));
    assert!(prepared.custom.iter().all(|team| {
        team.meta_absent_species > 0
            && team.fallback_layers.iter().any(|layer| {
                layer["legacy_source"] == "legacy-empty"
                    && layer["layered_source"] != "meta"
                    && layer["legacy_move_slots"] == 0
                    && layer["layered_move_slots"]
                        .as_u64()
                        .is_some_and(|moves| moves > 0)
            })
    }));
    let identity = run_identity(&cfg, &prepared, &jobs);
    assert_eq!(identity["semantic_config"]["baked_preview_tables"], false);
    let run_fingerprint = semantic_run_fingerprint(&identity);
    let single = run_rows(&cfg, &prepared, &jobs, &run_fingerprint, 1);
    let parallel = run_rows(&cfg, &prepared, &jobs, &run_fingerprint, 3);
    assert_eq!(single, parallel, "thread count changed paired rows");
    assert_eq!(summarize(&single), summarize(&parallel));
    let summary = summarize(&single);
    assert_eq!(summary.valid_pairs, 0);
    assert_eq!(summary.excluded_pairs, single.len());
    assert!(!summary.certified);
    assert!(single.iter().all(|row| row.legacy.capped
        && row.layered.capped
        && row.legacy.score.is_none()
        && row.layered.score.is_none()));
    assert!(single.iter().all(|row| {
        row.legacy.policy == FallbackPolicy::LegacyMetaOnly.id()
            && row.layered.policy == FallbackPolicy::Layered.id()
    }));
    let mut shifted = cfg.clone();
    shifted.seed = cfg.seed.wrapping_add(1);
    assert_ne!(
        canonical_fingerprint("m17d-workload-v1", &jobs),
        canonical_fingerprint("m17d-workload-v1", &plan_jobs(&shifted, &prepared))
    );
    let mut other_threads = cfg.clone();
    other_threads.threads = cfg.threads + 7;
    assert_eq!(
        run_fingerprint,
        semantic_run_fingerprint(&run_identity(&other_threads, &prepared, &jobs)),
        "execution thread count contaminated semantic run identity"
    );
    eprintln!(
        "self-test passed: {} custom sources, {} seed-paired jobs; threads 1 == 3; invalid legacy/layered {}/{}; caps {}/{}",
        prepared.custom.len(),
        single.len(),
        summary.legacy_invalid,
        summary.layered_invalid,
        summary.legacy_capped,
        summary.layered_capped,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--self-test") {
        self_test();
        return;
    }
    let cfg = parse_config(&args);
    let prepared = prepare(&cfg);
    let jobs = plan_jobs(&cfg, &prepared);
    let identity = run_identity(&cfg, &prepared, &jobs);
    let run_fingerprint = semantic_run_fingerprint(&identity);
    let rows = run_rows(&cfg, &prepared, &jobs, &run_fingerprint, cfg.threads);
    let summary = summarize(&rows);

    let mut lines = Vec::with_capacity(rows.len() + 2);
    lines.push(serde_json::to_string(&header(identity, &run_fingerprint, &prepared)).unwrap());
    lines.extend(rows.iter().map(|row| serde_json::to_string(row).unwrap()));
    lines.push(
        serde_json::to_string(&json!({
            "schema": SCHEMA,
            "kind": "summary",
            "run_fingerprint": run_fingerprint,
            "summary": summary,
        }))
        .unwrap(),
    );
    let output = format!("{}\n", lines.join("\n"));
    if let Some(path) = flag(&args, "--out") {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let tmp = path.with_extension(format!(
            "{}.tmp",
            path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("jsonl")
        ));
        let mut file = std::fs::File::create(&tmp).unwrap();
        file.write_all(output.as_bytes()).unwrap();
        file.sync_all().unwrap();
        std::fs::rename(tmp, &path).unwrap();
        eprintln!("wrote {} paired rows to {}", rows.len(), path.display());
    } else {
        print!("{output}");
    }
    eprintln!(
        "paired={} valid={} excluded={} legacy={:?} layered={:?} delta={:?} invalid={}/{} cap={}/{} certified={}",
        summary.pairs,
        summary.valid_pairs,
        summary.excluded_pairs,
        summary.legacy_mean,
        summary.layered_mean,
        summary.mean_paired_delta,
        summary.legacy_invalid,
        summary.layered_invalid,
        summary.legacy_capped,
        summary.layered_capped,
        summary.certified,
    );
    if !summary.certified && !cfg.allow_incomplete {
        eprintln!(
            "uncertified artifact (use --allow-incomplete only for diagnostics): {}",
            summary.certification_failures.join("; ")
        );
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(policy: &'static str, status: &'static str, score: Option<f64>) -> ArmResult {
        ArmResult {
            policy,
            status,
            outcome: (status == "outcome").then_some("tie"),
            score,
            turns: 1,
            capped: status == "capped",
            evaluated_fallback: true,
            reference_fallback: false,
            error: (status == "invalid").then(|| "test".to_string()),
        }
    }

    fn row(job: usize, legacy: ArmResult, layered: ArmResult) -> PairRow {
        PairRow {
            schema: SCHEMA,
            kind: "pair",
            run_fingerprint: "test".into(),
            job,
            custom_id: "custom".into(),
            custom_fingerprint: "custom-fp".into(),
            pilot_id: "pilot".into(),
            pilot_fingerprint: "pilot-fp".into(),
            game: job,
            custom_is_p1: false,
            battle_seed: "1,2,3,4".into(),
            evaluated_agent_seed: 1,
            reference_agent_seed: 2,
            delta_layered_minus_legacy: layered.score.zip(legacy.score).map(|(n, o)| n - o),
            legacy,
            layered,
        }
    }

    #[test]
    fn fingerprint_is_stable_and_tagged() {
        assert_eq!(
            fingerprint("m17d-test", b"abc"),
            fingerprint("m17d-test", b"abc")
        );
        assert_ne!(
            fingerprint("m17d-test", b"abc"),
            fingerprint("m17d-test", b"abd")
        );
        assert!(fingerprint("m17d-test", b"abc").ends_with(":m17d-test"));
    }

    #[test]
    fn smoke_sources_are_all_preview_off_pool() {
        let cfg = Config::smoke();
        let prepared = prepare(&cfg);
        assert_eq!(prepared.custom.len(), 3);
        for custom in &prepared.custom {
            assert!(custom.meta_absent_species > 0);
            assert!(custom.fallback_layers.iter().any(|layer| {
                layer["legacy_source"] == "legacy-empty"
                    && layer["layered_source"] != "meta"
                    && layer["legacy_move_slots"] == 0
                    && layer["layered_move_slots"]
                        .as_u64()
                        .is_some_and(|moves| moves > 0)
            }));
            assert!(is_preview_fallback(
                &prepared.dex,
                &prepared.pool,
                &prepared.pilots[0].sets,
                &custom.sets,
            ));
        }
    }

    #[test]
    fn full_profile_inputs_prepare_without_running_games() {
        let cfg = Config::full();
        let prepared = prepare(&cfg);
        assert_eq!(
            prepared.custom.len(),
            cfg.community + cfg.fixtures + cfg.mutations
        );
        assert_eq!(
            prepared
                .custom
                .iter()
                .filter(|team| team.source["kind"] == "community")
                .count(),
            cfg.community
        );
        assert!(prepared
            .custom
            .iter()
            .all(|team| team.meta_absent_species > 0));
    }

    #[test]
    fn summary_excludes_incomplete_pairs_and_refuses_certification() {
        let rows = vec![
            row(
                0,
                arm(FallbackPolicy::LegacyMetaOnly.id(), "outcome", Some(0.0)),
                arm(FallbackPolicy::Layered.id(), "outcome", Some(1.0)),
            ),
            row(
                1,
                arm(FallbackPolicy::LegacyMetaOnly.id(), "capped", None),
                arm(FallbackPolicy::Layered.id(), "outcome", Some(0.0)),
            ),
            row(
                2,
                arm(FallbackPolicy::LegacyMetaOnly.id(), "invalid", None),
                arm(FallbackPolicy::Layered.id(), "invalid", None),
            ),
        ];
        let summary = summarize(&rows);
        assert_eq!(summary.valid_pairs, 1);
        assert_eq!(summary.excluded_pairs, 2);
        assert_eq!(summary.mean_paired_delta, Some(1.0));
        assert_eq!(summary.legacy_only_incomplete_pairs, 1);
        assert_eq!(summary.both_incomplete_pairs, 1);
        assert_eq!(summary.asymmetric_incomplete_pairs, 1);
        assert_eq!((summary.legacy_invalid, summary.layered_invalid), (1, 1));
        assert!(!summary.certified);
        assert_eq!(summary.certification_failures.len(), 3);
    }
}
