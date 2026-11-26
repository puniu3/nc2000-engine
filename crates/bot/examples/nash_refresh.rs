//! Native Rust drop-in runner for `web/tests/nash-refresh.mjs` schedules.
//!
//! Usage (from any directory):
//!   nash_refresh GAMES.jsonl NEW_OUTPUT_DIR [workers]
//!
//! Environment variables (matching `web/tests/nash-refresh.mjs` plus native extensions):
//!   NASH_POOLS=FILE          Optional extra `{teams, mixtures}` JSON file.
//!   NASH_RULESET=ID          Optional battle ruleset id (default: `gen2stadium2`; e.g. `castle-free`, `castle-tournament`).
//!   NASH_SAVE_LOGS=1         Save full battle protocol logs to `NEW_OUTPUT_DIR/logs/<index>.json`.
//!   NASH_PREVIEW_ITERS=N     Override live preview iterations (default: `live-search.json` `unknownPreview`).
//!   NASH_BATTLE_ITERS=N      Override battle iteration cap/budget (default: `live-search.json` `battle.adaptive`).
//!   NASH_BATTLE_MODE=MODE    `adaptive` (default) or `fixed` (fixed `agent.step(N)` per battle decision).
//!   NASH_PREVIEW_CACHE=0|1   Exact singleflight Turn-0 live preview cache (default: `1`).
//!   NASH_PREVIEW_DIST_CACHE=FILE  Optional persistent K-seed empirical Legacy preview distribution cache JSON.
//!   NASH_PREVIEW_DIST_K=N    Number of independent Legacy preview seeds per unique key (e.g. `16`).
//!   NASH_PREVIEW_DIST_ONLY=1 Precompute missing K-seed preview distributions for `GAMES.jsonl` without running battles.
//!
//! Emits a single-line JSON summary on stdout. Uses `ProtocolAgent` + `PlayerChannel`
//! + `DecisionSampler` with identical configuration to `crates/wasm/src/lib.rs`
//! (`WasmProtocolSearcher`) so `actionSha256` can be verified against WASM runs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use nc2000_bot::compiled_preview::{policy_epoch_for, POLICY_EPOCH};
use nc2000_bot::decision::{DecisionPolicy, DecisionSampler, PolicyMode};
use nc2000_bot::player::PlayerChannel;
use nc2000_bot::preview::{MetaPool, MetaTeam};
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{ProtocolAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::rules::Ruleset;
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{validate_team_with_ruleset, Learnsets};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const CANDIDATE_FILE: &str = "nash-refresh-v1/candidate";
const INCUMBENT_FILE: &str = "kodoku-r3/conclusion-equilibrium-deme";
const TEAM_FILES: [&str; 3] = [
    "kodoku-beq-v1/novel_evolved_teams",
    "team-pool-v2/team-pool",
    INCUMBENT_FILE,
];

#[derive(Clone, Deserialize, Serialize)]
struct Task {
    #[serde(default)]
    block: Value,
    #[serde(default)]
    cell: Value,
    #[serde(rename = "rowSeat")]
    row_seat: usize,
    ids: [String; 2],
    seeds: [u32; 2],
    #[serde(default, rename = "previewSeeds", skip_serializing_if = "Option::is_none")]
    preview_seeds: Option<[u32; 2]>,
    #[serde(rename = "battleSeed")]
    battle_seed: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    picks: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mixtures: Option<[String; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    policies: Option<[String; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ruleset: Option<String>,
}

#[derive(Deserialize)]
struct TeamFile {
    teams: Vec<TeamEntry>,
}

#[derive(Clone, Deserialize)]
struct TeamEntry {
    id: String,
    sets: Vec<PokemonSet>,
    #[serde(default)]
    weight: Option<f64>,
}

#[derive(Deserialize)]
struct ExtraPoolsFile {
    teams: Vec<TeamEntry>,
    mixtures: BTreeMap<String, Vec<MixtureMember>>,
}

#[derive(Deserialize)]
struct MixtureMember {
    id: String,
    weight: f64,
}

#[derive(Deserialize)]
struct LiveSearchCfg {
    #[serde(rename = "unknownPreview")]
    unknown_preview: u32,
    battle: LiveBattleCfg,
}

#[derive(Deserialize)]
struct LiveBattleCfg {
    adaptive: u32,
}

#[derive(Deserialize)]
struct SearchProfilesCfg {
    blind: BlindProfileCfg,
}

#[derive(Deserialize)]
struct BlindProfileCfg {
    c: f64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hex_encode_32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

const PREVIEW_DIST_FORMAT: &str = "nc2000-preview-dist-v1";

#[derive(Clone, Deserialize, Serialize)]
struct PreviewDistEntry {
    k: u32,
    counts: BTreeMap<String, u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    own: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    opp_mix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    side: Option<usize>,
}

#[derive(Clone, Deserialize, Serialize)]
struct PreviewDistFile {
    format: String,
    #[serde(default, rename = "policyEpoch", skip_serializing_if = "Option::is_none")]
    policy_epoch: Option<String>,
    entries: BTreeMap<String, PreviewDistEntry>,
}

fn dist_sample_seed(key_bytes: &[u8; 32], rep: u32) -> u64 {
    let mut h = Sha256::new();
    h.update(b"nc2000-preview-dist-seed-v1\0");
    h.update(key_bytes);
    h.update(rep.to_le_bytes());
    let d = h.finalize();
    u64::from_le_bytes(d[0..8].try_into().unwrap())
}

fn entry_to_policy(entry: &PreviewDistEntry) -> Option<DecisionPolicy<String>> {
    if entry.counts.is_empty() {
        return None;
    }
    DecisionPolicy::new(
        "legacy-dist",
        entry.counts.iter().map(|(a, &c)| (a.clone(), c as f64)),
    )
    .ok()
}

fn resolve_ruleset(opt: Option<&str>, fallback: Ruleset) -> Ruleset {
    match opt.map(str::trim).filter(|s| !s.is_empty()) {
        None => fallback,
        Some(id) => Ruleset::parse_id(id).unwrap_or_else(|| {
            panic!(
                "invalid ruleset {id:?} (expected e.g. gen2stadium2, castle-free, castle-tournament)"
            )
        }),
    }
}

/// Cache epoch tag that preserves bit-exact `POLICY_EPOCH` keys on PS Stadium 2
/// while isolating Castle rulesets and overlays (`castle-policy-v4:<canonical_id>`).
fn cache_epoch_tag(ruleset: Ruleset) -> String {
    let base = policy_epoch_for(ruleset);
    if ruleset.is_ps() {
        base.to_string()
    } else {
        format!("{base}:{}", ruleset.canonical_id())
    }
}

fn run_single_preview_sample(
    dex: &Dex,
    ruleset: Ruleset,
    side: usize,
    prior: MetaPool,
    cfg: RmConfig,
    own_sets: Vec<PokemonSet>,
    frame_lines: &[String],
    req_json: &str,
    preview_iters: u32,
    is_current: bool,
    sample_seed: u64,
) -> Option<String> {
    let mut p_agent = ProtocolAgent::new_with_ruleset(dex, side, prior, cfg, sample_seed, ruleset);
    p_agent.set_own_team(own_sets);
    for line in frame_lines {
        p_agent.push_line(dex, line);
    }
    let owed = p_agent
        .on_request(dex, req_json)
        .expect("preview-dist ProtocolAgent::on_request");
    assert!(owed, "expected pending preview decision on side {side}");
    if is_current {
        p_agent
            .step_adaptive_battle(dex, preview_iters)
            .expect("step_adaptive_battle");
    } else {
        p_agent.step(dex, preview_iters).expect("step");
    }
    let pol = p_agent.decision_policy(dex, PolicyMode::Legacy)?;
    let mut sampler = DecisionSampler::new(sample_seed);
    Some(sampler.commit(pol).input.clone())
}

struct PreviewDistCache {
    preloaded: HashMap<String, PreviewDistEntry>,
    computed: Mutex<HashMap<String, Arc<OnceLock<Option<PreviewDistEntry>>>>>,
    target_k: u32,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl PreviewDistCache {
    fn new(preloaded: HashMap<String, PreviewDistEntry>, target_k: u32) -> Self {
        Self {
            preloaded,
            computed: Mutex::new(HashMap::new()),
            target_k,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    fn get_or_compute(
        &self,
        key_hex: &str,
        compute: impl FnOnce(Option<&PreviewDistEntry>, u32) -> Option<PreviewDistEntry>,
    ) -> Option<DecisionPolicy<String>> {
        if let Some(entry) = self.preloaded.get(key_hex) {
            if !entry.counts.is_empty() && (self.target_k == 0 || entry.k >= self.target_k) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return entry_to_policy(entry);
            }
        }
        if self.target_k < 2 {
            return None;
        }
        let cell = {
            let mut map = self.computed.lock().unwrap();
            map.entry(key_hex.to_string())
                .or_insert_with(|| Arc::new(OnceLock::new()))
                .clone()
        };
        let mut did_compute = false;
        let opt_entry = cell.get_or_init(|| {
            did_compute = true;
            compute(self.preloaded.get(key_hex), self.target_k)
        });
        if did_compute {
            self.misses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        opt_entry.as_ref().and_then(entry_to_policy)
    }

    fn export_new_entries(&self) -> BTreeMap<String, PreviewDistEntry> {
        let map = self.computed.lock().unwrap();
        let mut out = BTreeMap::new();
        for (k, cell) in map.iter() {
            if let Some(Some(entry)) = cell.get() {
                out.insert(k.clone(), entry.clone());
            }
        }
        out
    }
}

fn preview_dist_key_with_epoch(
    policy_epoch: &str,
    side: usize,
    preview_iters: u32,
    blind_c: f64,
    policy_name: &str,
    own_sets_json: &str,
    frame_lines: &[String],
    req_json: &str,
    prior: &MetaPool,
    agent: &ProtocolAgent,
    team_json_map: &HashMap<String, String>,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"nc2000-preview-dist-v1\0");
    h.update((policy_epoch.len() as u64).to_le_bytes());
    h.update(policy_epoch.as_bytes());
    h.update([side as u8]);
    h.update(preview_iters.to_le_bytes());
    h.update(blind_c.to_bits().to_le_bytes());
    h.update((policy_name.len() as u64).to_le_bytes());
    h.update(policy_name.as_bytes());
    h.update((own_sets_json.len() as u64).to_le_bytes());
    h.update(own_sets_json.as_bytes());
    h.update((frame_lines.len() as u64).to_le_bytes());
    for line in frame_lines {
        h.update((line.len() as u64).to_le_bytes());
        h.update(line.as_bytes());
    }
    h.update((req_json.len() as u64).to_le_bytes());
    h.update(req_json.as_bytes());
    if let Some(belief) = agent.belief() {
        let alive = belief.alive();
        h.update([1u8]);
        h.update((alive.len() as u64).to_le_bytes());
        for &idx in alive {
            let cand_id = belief.candidate_id(idx);
            let cand_weight = prior
                .teams
                .iter()
                .find(|t| t.id == cand_id)
                .and_then(|t| t.weight);
            let sets_json = team_json_map
                .get(cand_id)
                .map(String::as_str)
                .unwrap_or("");
            h.update((cand_id.len() as u64).to_le_bytes());
            h.update(cand_id.as_bytes());
            h.update((sets_json.len() as u64).to_le_bytes());
            h.update(sets_json.as_bytes());
            let w_bits = cand_weight.map(f64::to_bits).unwrap_or(u64::MAX);
            h.update(w_bits.to_le_bytes());
        }
    } else {
        h.update([0u8]);
    }
    h.finalize().into()
}

fn preview_dist_key_for_ruleset(
    ruleset: Ruleset,
    side: usize,
    preview_iters: u32,
    blind_c: f64,
    policy_name: &str,
    own_sets_json: &str,
    frame_lines: &[String],
    req_json: &str,
    prior: &MetaPool,
    agent: &ProtocolAgent,
    team_json_map: &HashMap<String, String>,
) -> [u8; 32] {
    let tag = cache_epoch_tag(ruleset);
    preview_dist_key_with_epoch(
        &tag,
        side,
        preview_iters,
        blind_c,
        policy_name,
        own_sets_json,
        frame_lines,
        req_json,
        prior,
        agent,
        team_json_map,
    )
}

#[cfg(test)]
fn preview_dist_key(
    side: usize,
    preview_iters: u32,
    blind_c: f64,
    policy_name: &str,
    own_sets_json: &str,
    frame_lines: &[String],
    req_json: &str,
    prior: &MetaPool,
    agent: &ProtocolAgent,
    team_json_map: &HashMap<String, String>,
) -> [u8; 32] {
    preview_dist_key_for_ruleset(
        Ruleset::PsStadium2,
        side,
        preview_iters,
        blind_c,
        policy_name,
        own_sets_json,
        frame_lines,
        req_json,
        prior,
        agent,
        team_json_map,
    )
}

fn read_data(root: &Path, name: &str) -> String {
    let path = root.join("data").join(format!("{name}.json"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn canonical_no_spaces(s: &str) -> String {
    s.chars().filter(|&ch| ch != ' ').collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BattleMode {
    Adaptive,
    Fixed,
}

impl BattleMode {
    fn as_str(self) -> &'static str {
        match self {
            BattleMode::Adaptive => "adaptive",
            BattleMode::Fixed => "fixed",
        }
    }
}

struct CachedPreview {
    policy: Option<DecisionPolicy<String>>,
}

struct PreviewCache {
    slots: Mutex<HashMap<[u8; 32], Arc<OnceLock<CachedPreview>>>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl PreviewCache {
    fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    fn get_or_compute(
        &self,
        key: [u8; 32],
        compute: impl FnOnce() -> Option<DecisionPolicy<String>>,
    ) -> Option<DecisionPolicy<String>> {
        let cell = {
            let mut map = self.slots.lock().unwrap();
            map.entry(key)
                .or_insert_with(|| Arc::new(OnceLock::new()))
                .clone()
        };
        let mut computed = false;
        let entry = cell.get_or_init(|| {
            computed = true;
            CachedPreview { policy: compute() }
        });
        if computed {
            self.misses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        entry.policy.clone()
    }
}

fn preview_cache_key_with_epoch(
    policy_epoch: &str,
    side: usize,
    seed: u32,
    preview_iters: u32,
    blind_c: f64,
    policy_name: &str,
    own_sets_json: &str,
    frame_lines: &[String],
    req_json: &str,
    prior: &MetaPool,
    agent: &ProtocolAgent,
    team_json_map: &HashMap<String, String>,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((policy_epoch.len() as u64).to_le_bytes());
    h.update(policy_epoch.as_bytes());
    h.update([side as u8]);
    h.update(seed.to_le_bytes());
    h.update(preview_iters.to_le_bytes());
    h.update(blind_c.to_bits().to_le_bytes());
    h.update((policy_name.len() as u64).to_le_bytes());
    h.update(policy_name.as_bytes());
    h.update((own_sets_json.len() as u64).to_le_bytes());
    h.update(own_sets_json.as_bytes());
    h.update((frame_lines.len() as u64).to_le_bytes());
    for line in frame_lines {
        h.update((line.len() as u64).to_le_bytes());
        h.update(line.as_bytes());
    }
    h.update((req_json.len() as u64).to_le_bytes());
    h.update(req_json.as_bytes());
    if let Some(belief) = agent.belief() {
        let alive = belief.alive();
        h.update([1u8]);
        h.update((alive.len() as u64).to_le_bytes());
        for &idx in alive {
            let cand_id = belief.candidate_id(idx);
            let cand_weight = prior
                .teams
                .iter()
                .find(|t| t.id == cand_id)
                .and_then(|t| t.weight);
            let sets_json = team_json_map
                .get(cand_id)
                .map(String::as_str)
                .unwrap_or("");
            h.update((cand_id.len() as u64).to_le_bytes());
            h.update(cand_id.as_bytes());
            h.update((sets_json.len() as u64).to_le_bytes());
            h.update(sets_json.as_bytes());
            let w_bits = cand_weight.map(f64::to_bits).unwrap_or(u64::MAX);
            h.update(w_bits.to_le_bytes());
        }
    } else {
        h.update([0u8]);
    }
    h.finalize().into()
}

fn preview_cache_key_for_ruleset(
    ruleset: Ruleset,
    side: usize,
    seed: u32,
    preview_iters: u32,
    blind_c: f64,
    policy_name: &str,
    own_sets_json: &str,
    frame_lines: &[String],
    req_json: &str,
    prior: &MetaPool,
    agent: &ProtocolAgent,
    team_json_map: &HashMap<String, String>,
) -> [u8; 32] {
    let tag = cache_epoch_tag(ruleset);
    preview_cache_key_with_epoch(
        &tag,
        side,
        seed,
        preview_iters,
        blind_c,
        policy_name,
        own_sets_json,
        frame_lines,
        req_json,
        prior,
        agent,
        team_json_map,
    )
}

#[cfg(test)]
fn preview_cache_key(
    side: usize,
    seed: u32,
    preview_iters: u32,
    blind_c: f64,
    policy_name: &str,
    own_sets_json: &str,
    frame_lines: &[String],
    req_json: &str,
    prior: &MetaPool,
    agent: &ProtocolAgent,
    team_json_map: &HashMap<String, String>,
) -> [u8; 32] {
    preview_cache_key_for_ruleset(
        Ruleset::PsStadium2,
        side,
        seed,
        preview_iters,
        blind_c,
        policy_name,
        own_sets_json,
        frame_lines,
        req_json,
        prior,
        agent,
        team_json_map,
    )
}

struct SharedState {
    dex: Dex,
    ruleset: Ruleset,
    teams: HashMap<String, Vec<PokemonSet>>,
    team_json_map: HashMap<String, String>,
    disclosed: HashMap<String, MetaPool>,
    default_pool: Option<MetaPool>,
    preview_iters: u32,
    battle_iters: u32,
    battle_mode: BattleMode,
    blind_c: f64,
    log_dir: Option<PathBuf>,
    preview_cache: Option<PreviewCache>,
    preview_dist_cache: Option<PreviewDistCache>,
}

fn run_game(shared: &SharedState, index: usize, task: &Task) -> Value {
    let policies = task
        .policies
        .clone()
        .unwrap_or_else(|| ["current".to_string(), "current".to_string()]);
    for p in &policies {
        assert!(
            p == "current" || p == "current81",
            "unsupported policy {p:?} in native nash_refresh (only current / current81 supported)"
        );
    }
    let picks = task.picks.as_deref().unwrap_or(&[]);
    if let Some(mixtures) = &task.mixtures {
        assert!(picks.is_empty() || picks.len() == 2);
        for (side, mix) in mixtures.iter().enumerate() {
            let pool = shared
                .disclosed
                .get(mix)
                .unwrap_or_else(|| panic!("unknown disclosed mixture: {mix}"));
            assert!(
                pool.teams.iter().any(|t| t.id == task.ids[side]),
                "realized id {} missing from mixture {mix}",
                task.ids[side]
            );
        }
    }

    let sets: [Vec<PokemonSet>; 2] = [
        shared
            .teams
            .get(&task.ids[0])
            .unwrap_or_else(|| panic!("unknown team id: {}", task.ids[0]))
            .clone(),
        shared
            .teams
            .get(&task.ids[1])
            .unwrap_or_else(|| panic!("unknown team id: {}", task.ids[1]))
            .clone(),
    ];

    let ruleset = resolve_ruleset(task.ruleset.as_deref(), shared.ruleset);
    let battle_seed_str = SplitMix64::new(task.battle_seed as u64).battle_seed();
    let mut battle = Battle::from_fixture_with_ruleset(
        &shared.dex,
        &battle_seed_str,
        &sets[0],
        &sets[1],
        ruleset,
    )
    .expect("Battle::from_fixture_with_ruleset");
    let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];

    let cfg = RmConfig {
        rule: SelRule::Ucb,
        c: shared.blind_c,
        hp_buckets: 16,
        ..RmConfig::default()
    };

    let priors: [MetaPool; 2] = std::array::from_fn(|side| {
        if let Some(mixtures) = &task.mixtures {
            shared.disclosed[&mixtures[1 - side]].clone()
        } else {
            shared
                .default_pool
                .as_ref()
                .expect("default belief pool required when mixtures absent")
                .clone()
        }
    });

    let mut agents: [ProtocolAgent; 2] = std::array::from_fn(|side| {
        let mut agent = ProtocolAgent::new_with_ruleset(
            &shared.dex,
            side,
            priors[side].clone(),
            cfg.clone(),
            task.seeds[side] as u64,
            ruleset,
        );
        agent.set_own_team(sets[side].clone());
        agent
    });
    let mut decisions: [DecisionSampler<String>; 2] =
        std::array::from_fn(|side| DecisionSampler::new(task.seeds[side] as u64));

    let mut preview: [Option<Value>; 2] = [None, None];
    let mut preview_beliefs: [Option<Vec<String>>; 2] = [None, None];
    let mut transcript = Sha256::new();
    let began = Instant::now();
    let mut first = true;

    while battle.outcome().is_none() && battle.turn < 500 {
        let needs = battle.needs_choice();
        let mut inputs: [Option<String>; 2] = [None, None];
        for side in 0..2 {
            if !needs[side] {
                continue;
            }
            let frame = channels[side]
                .frame(&mut battle, &shared.dex)
                .expect("PlayerChannel::frame");
            let agent = &mut agents[side];
            for line in &frame.lines {
                agent.push_line(&shared.dex, line);
            }
            decisions[side].reset();
            let req_json = serde_json::to_string(&frame.request).unwrap();
            let owed = agent
                .on_request(&shared.dex, &req_json)
                .expect("ProtocolAgent::on_request");
            assert!(owed, "expected pending decision on side {side}");

            let mut chosen: Option<String> = None;
            let mut source = "live";

            if task.mixtures.is_some() {
                let belief: Value =
                    serde_json::from_str(&agent.belief_info()).expect("belief_info json");
                let fallback = belief["fallback"].as_bool().unwrap_or(true);
                let candidates: Vec<String> = belief["candidates"]
                    .as_array()
                    .expect("belief candidates")
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();
                assert!(
                    !fallback && candidates.iter().any(|c| c == &task.ids[1 - side]),
                    "disclosure lost the played team"
                );
                if first {
                    preview_beliefs[side] = Some(candidates);
                }
            }

            if first && task.mixtures.is_none() {
                if let Some(forced) = picks.get(side).filter(|s| !s.is_empty()) {
                    let norm = canonical_no_spaces(forced);
                    assert!(
                        frame
                            .legal_actions
                            .iter()
                            .any(|a| canonical_no_spaces(a) == norm),
                        "illegal forced selection: {forced}"
                    );
                    chosen = Some(forced.clone());
                    source = "forced";
                }
            }

            if chosen.is_none() {
                let separate_preview_seed = task.preview_seeds.map(|ps| ps[side]);
                let preview_seed = separate_preview_seed.unwrap_or(task.seeds[side]);
                let mut dist_policy: Option<DecisionPolicy<String>> = None;
                if first
                    && separate_preview_seed.is_none()
                    && agent.belief().is_some_and(|b| !b.is_fallback())
                {
                    if let Some(dcache) = &shared.preview_dist_cache {
                        let own_json = shared
                            .team_json_map
                            .get(&task.ids[side])
                            .map(String::as_str)
                            .unwrap_or("");
                        let key_bytes = preview_dist_key_for_ruleset(
                            ruleset,
                            side,
                            shared.preview_iters,
                            shared.blind_c,
                            &policies[side],
                            own_json,
                            &frame.lines,
                            &req_json,
                            &priors[side],
                            agent,
                            &shared.team_json_map,
                        );
                        let key_hex = hex_encode_32(&key_bytes);
                        let dex = &shared.dex;
                        let preview_iters = shared.preview_iters;
                        let is_current = policies[side] == "current";
                        let cfg_clone = cfg.clone();
                        let prior_clone = priors[side].clone();
                        let own_sets = sets[side].clone();
                        let own_id = task.ids[side].clone();
                        let opp_mix = task.mixtures.as_ref().map(|m| m[1 - side].clone());
                        dist_policy = dcache.get_or_compute(&key_hex, |existing, target_k| {
                            let mut counts = existing
                                .map(|e| e.counts.clone())
                                .unwrap_or_default();
                            let start_k = existing.map(|e| e.k).unwrap_or(0);
                            let mut total_k = start_k;
                            for rep in start_k..target_k {
                                let s = dist_sample_seed(&key_bytes, rep);
                                if let Some(act) = run_single_preview_sample(
                                    dex,
                                    ruleset,
                                    side,
                                    prior_clone.clone(),
                                    cfg_clone.clone(),
                                    own_sets.clone(),
                                    &frame.lines,
                                    &req_json,
                                    preview_iters,
                                    is_current,
                                    s,
                                ) {
                                    *counts.entry(act).or_insert(0) += 1;
                                    total_k += 1;
                                }
                            }
                            if counts.is_empty() {
                                None
                            } else {
                                Some(PreviewDistEntry {
                                    k: total_k,
                                    counts,
                                    own: Some(own_id),
                                    opp_mix,
                                    side: Some(side),
                                })
                            }
                        });
                        if dist_policy.is_some() {
                            source = "dist-cache";
                        }
                    }
                }

                let cached_policy = if let Some(dp) = dist_policy {
                    Some(Some(dp))
                } else if first
                    && agent.belief().is_some_and(|b| !b.is_fallback())
                {
                    if let Some(cache) = &shared.preview_cache {
                        let own_json = shared
                            .team_json_map
                            .get(&task.ids[side])
                            .map(String::as_str)
                            .unwrap_or("");
                        let key = preview_cache_key_for_ruleset(
                            ruleset,
                            side,
                            preview_seed,
                            shared.preview_iters,
                            shared.blind_c,
                            &policies[side],
                            own_json,
                            &frame.lines,
                            &req_json,
                            &priors[side],
                            agent,
                            &shared.team_json_map,
                        );
                        let dex = &shared.dex;
                        let preview_iters = shared.preview_iters;
                        let is_current = policies[side] == "current";
                        let cfg_clone = cfg.clone();
                        let prior_clone = priors[side].clone();
                        let own_sets = sets[side].clone();
                        Some(cache.get_or_compute(key, || {
                            if let Some(pseed) = separate_preview_seed {
                                // Isolated agent seeded with `previewSeeds[side]` guarantees
                                // bit-exact determinism regardless of worker scheduling or cache hit/miss,
                                // while keeping main battle `agent` RNG strictly independent per game.
                                let mut p_agent = ProtocolAgent::new_with_ruleset(
                                    dex,
                                    side,
                                    prior_clone,
                                    cfg_clone,
                                    pseed as u64,
                                    ruleset,
                                );
                                p_agent.set_own_team(own_sets);
                                for line in &frame.lines {
                                    p_agent.push_line(dex, line);
                                }
                                let owed = p_agent
                                    .on_request(dex, &req_json)
                                    .expect("preview ProtocolAgent::on_request");
                                assert!(owed, "expected pending preview decision on side {side}");
                                if is_current {
                                    p_agent
                                        .step_adaptive_battle(dex, preview_iters)
                                        .expect("step_adaptive_battle");
                                } else {
                                    p_agent.step(dex, preview_iters).expect("step");
                                }
                                p_agent.decision_policy(dex, PolicyMode::Legacy)
                            } else {
                                if is_current {
                                    agent
                                        .step_adaptive_battle(dex, preview_iters)
                                        .expect("step_adaptive_battle");
                                } else {
                                    agent.step(dex, preview_iters).expect("step");
                                }
                                agent.decision_policy(dex, PolicyMode::Legacy)
                            }
                        }))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let policy = match cached_policy {
                    Some(p) => p,
                    None => {
                        if first {
                            if let Some(pseed) = separate_preview_seed {
                                let mut p_agent = ProtocolAgent::new_with_ruleset(
                                    &shared.dex,
                                    side,
                                    priors[side].clone(),
                                    cfg.clone(),
                                    pseed as u64,
                                    ruleset,
                                );
                                p_agent.set_own_team(sets[side].clone());
                                for line in &frame.lines {
                                    p_agent.push_line(&shared.dex, line);
                                }
                                let owed = p_agent
                                    .on_request(&shared.dex, &req_json)
                                    .expect("preview ProtocolAgent::on_request");
                                assert!(owed, "expected pending preview decision on side {side}");
                                if policies[side] == "current" {
                                    p_agent
                                        .step_adaptive_battle(&shared.dex, shared.preview_iters)
                                        .expect("step_adaptive_battle");
                                } else {
                                    p_agent
                                        .step(&shared.dex, shared.preview_iters)
                                        .expect("step");
                                }
                                p_agent.decision_policy(&shared.dex, PolicyMode::Legacy)
                            } else {
                                if policies[side] == "current" {
                                    agent
                                        .step_adaptive_battle(&shared.dex, shared.preview_iters)
                                        .expect("step_adaptive_battle");
                                } else {
                                    agent.step(&shared.dex, shared.preview_iters).expect("step");
                                }
                                agent.decision_policy(&shared.dex, PolicyMode::Legacy)
                            }
                        } else {
                            if policies[side] == "current" {
                                match shared.battle_mode {
                                    BattleMode::Adaptive => {
                                        agent
                                            .step_adaptive_battle(&shared.dex, shared.battle_iters)
                                            .expect("step_adaptive_battle");
                                    }
                                    BattleMode::Fixed => {
                                        agent
                                            .step(&shared.dex, shared.battle_iters)
                                            .expect("step");
                                    }
                                }
                            } else {
                                agent.step(&shared.dex, 81_000).expect("step");
                            }
                            agent.decision_policy(&shared.dex, PolicyMode::Legacy)
                        }
                    }
                };

                if decisions[side].committed().is_none() {
                    if let Some(policy) = policy {
                        if policy.source() == "request-projected" {
                            agent.projections += 1;
                        }
                        decisions[side].commit(policy);
                    }
                }
                let committed = decisions[side]
                    .committed()
                    .expect("expected committed decision");
                chosen = Some(committed.input.clone());
            }

            if first && task.mixtures.is_some() {
                if let Some(forced) = picks.get(side).filter(|s| !s.is_empty()) {
                    let norm = canonical_no_spaces(forced);
                    assert!(
                        frame
                            .legal_actions
                            .iter()
                            .any(|a| canonical_no_spaces(a) == norm),
                        "illegal forced selection: {forced}"
                    );
                    chosen = Some(forced.clone());
                    source = "forced";
                }
            }

            let input = chosen.expect("decision input");
            if first {
                preview[side] = Some(json!({"source": source, "input": input}));
            }
            inputs[side] = Some(input);
        }

        let mut line = serde_json::to_string(&inputs).unwrap();
        line.push('\n');
        transcript.update(line.as_bytes());

        for (side, input) in inputs.iter().enumerate() {
            if let Some(inp) = input {
                battle
                    .choose(&shared.dex, side, inp)
                    .unwrap_or_else(|e| panic!("battle.choose(side={side}, {inp:?}): {e:?}"));
            }
        }
        first = false;
    }

    let elapsed_ms = began.elapsed().as_secs_f64() * 1000.0;
    let outcome = match battle.outcome() {
        Some(Outcome::P1Win) => "p1",
        Some(Outcome::P2Win) => "p2",
        Some(Outcome::Tie) => "tie",
        None => "cap",
    };

    if let Some(log_dir) = &shared.log_dir {
        let log_path = log_dir.join(format!("{index}.json"));
        fs::write(log_path, serde_json::to_string(&battle.log).unwrap()).expect("write log");
    }

    let mut obj = serde_json::to_value(task).unwrap();
    let map = obj.as_object_mut().unwrap();
    map.insert("index".into(), json!(index));
    if !ruleset.is_ps() || task.ruleset.is_some() {
        map.insert("ruleset".into(), json!(ruleset.canonical_id()));
    }
    map.insert("preview".into(), json!(preview));
    map.insert("outcome".into(), json!(outcome));
    map.insert("turns".into(), json!(battle.turn));
    map.insert("actionSha256".into(), json!(format!("{:x}", transcript.finalize())));
    map.insert("ms".into(), json!(elapsed_ms.round() as u64));
    if task.mixtures.is_some() {
        map.insert("previewBeliefs".into(), json!(preview_beliefs));
    }
    obj
}

struct PrecomputeSpec {
    ruleset: Ruleset,
    key_bytes: [u8; 32],
    key_hex: String,
    side: usize,
    prior: MetaPool,
    own_sets: Vec<PokemonSet>,
    frame_lines: Vec<String>,
    req_json: String,
    is_current: bool,
    own_id: String,
    opp_mix: Option<String>,
    existing_k: u32,
    counts: Mutex<BTreeMap<String, u32>>,
}

fn precompute_preview_dists(
    shared: &SharedState,
    tasks: &[Task],
    preloaded: &HashMap<String, PreviewDistEntry>,
    target_k: u32,
    workers: usize,
    output: &Path,
) {
    let wall_start = Instant::now();
    let cfg = RmConfig {
        rule: SelRule::Ucb,
        c: shared.blind_c,
        hp_buckets: 16,
        ..RmConfig::default()
    };

    let mut seen_keys: HashSet<String> = HashSet::new();
    let mut specs: Vec<PrecomputeSpec> = Vec::new();
    let mut cached_entries: BTreeMap<String, PreviewDistEntry> = BTreeMap::new();

    for task in tasks {
        if task.preview_seeds.is_some() {
            continue;
        }
        let ruleset = resolve_ruleset(task.ruleset.as_deref(), shared.ruleset);
        let policies = task
            .policies
            .clone()
            .unwrap_or_else(|| ["current".to_string(), "current".to_string()]);
        let sets: [Vec<PokemonSet>; 2] = [
            shared.teams[&task.ids[0]].clone(),
            shared.teams[&task.ids[1]].clone(),
        ];
        let battle_seed_str = SplitMix64::new(task.battle_seed as u64).battle_seed();
        let mut battle = Battle::from_fixture_with_ruleset(
            &shared.dex,
            &battle_seed_str,
            &sets[0],
            &sets[1],
            ruleset,
        )
        .expect("Battle::from_fixture_with_ruleset");
        let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
        let priors: [MetaPool; 2] = std::array::from_fn(|side| {
            if let Some(mixtures) = &task.mixtures {
                shared.disclosed[&mixtures[1 - side]].clone()
            } else {
                shared
                    .default_pool
                    .as_ref()
                    .expect("default belief pool required when mixtures absent")
                    .clone()
            }
        });

        for side in 0..2 {
            let frame = channels[side]
                .frame(&mut battle, &shared.dex)
                .expect("PlayerChannel::frame");
            let mut agent = ProtocolAgent::new_with_ruleset(
                &shared.dex,
                side,
                priors[side].clone(),
                cfg.clone(),
                task.seeds[side] as u64,
                ruleset,
            );
            agent.set_own_team(sets[side].clone());
            for line in &frame.lines {
                agent.push_line(&shared.dex, line);
            }
            let req_json = serde_json::to_string(&frame.request).unwrap();
            let owed = agent
                .on_request(&shared.dex, &req_json)
                .expect("ProtocolAgent::on_request");
            if !owed || !agent.belief().is_some_and(|b| !b.is_fallback()) {
                continue;
            }
            let own_json = shared
                .team_json_map
                .get(&task.ids[side])
                .map(String::as_str)
                .unwrap_or("");
            let key_bytes = preview_dist_key_for_ruleset(
                ruleset,
                side,
                shared.preview_iters,
                shared.blind_c,
                &policies[side],
                own_json,
                &frame.lines,
                &req_json,
                &priors[side],
                &agent,
                &shared.team_json_map,
            );
            let key_hex = hex_encode_32(&key_bytes);
            if !seen_keys.insert(key_hex.clone()) {
                continue;
            }
            if let Some(existing) = preloaded.get(&key_hex) {
                if !existing.counts.is_empty() && existing.k >= target_k {
                    cached_entries.insert(key_hex, existing.clone());
                    continue;
                }
                specs.push(PrecomputeSpec {
                    ruleset,
                    key_bytes,
                    key_hex,
                    side,
                    prior: priors[side].clone(),
                    own_sets: sets[side].clone(),
                    frame_lines: frame.lines,
                    req_json,
                    is_current: policies[side] == "current",
                    own_id: task.ids[side].clone(),
                    opp_mix: task.mixtures.as_ref().map(|m| m[1 - side].clone()),
                    existing_k: existing.k,
                    counts: Mutex::new(existing.counts.clone()),
                });
            } else {
                specs.push(PrecomputeSpec {
                    ruleset,
                    key_bytes,
                    key_hex,
                    side,
                    prior: priors[side].clone(),
                    own_sets: sets[side].clone(),
                    frame_lines: frame.lines,
                    req_json,
                    is_current: policies[side] == "current",
                    own_id: task.ids[side].clone(),
                    opp_mix: task.mixtures.as_ref().map(|m| m[1 - side].clone()),
                    existing_k: 0,
                    counts: Mutex::new(BTreeMap::new()),
                });
            }
        }
    }

    let mut work: Vec<(usize, u32)> = Vec::new();
    for (idx, sp) in specs.iter().enumerate() {
        for rep in sp.existing_k..target_k {
            work.push((idx, rep));
        }
    }

    let next = AtomicUsize::new(0);
    let n_workers = workers.min(work.len().max(1));
    if !work.is_empty() {
        std::thread::scope(|s| {
            for _ in 0..n_workers {
                let specs = &specs;
                let work = &work;
                let next = &next;
                let dex = &shared.dex;
                let preview_iters = shared.preview_iters;
                let cfg = &cfg;
                s.spawn(move || loop {
                    let wi = next.fetch_add(1, Ordering::Relaxed);
                    if wi >= work.len() {
                        break;
                    }
                    let (spec_idx, rep) = work[wi];
                    let sp = &specs[spec_idx];
                    let sample_seed = dist_sample_seed(&sp.key_bytes, rep);
                    if let Some(act) = run_single_preview_sample(
                        dex,
                        sp.ruleset,
                        sp.side,
                        sp.prior.clone(),
                        cfg.clone(),
                        sp.own_sets.clone(),
                        &sp.frame_lines,
                        &sp.req_json,
                        preview_iters,
                        sp.is_current,
                        sample_seed,
                    ) {
                        let mut guard = sp.counts.lock().unwrap();
                        *guard.entry(act).or_insert(0) += 1;
                    }
                });
            }
        });
    }

    let mut entries = cached_entries;
    let cached_keys = entries.len();
    for sp in specs {
        let counts = sp.counts.into_inner().unwrap();
        let total_k: u32 = counts.values().sum();
        entries.insert(
            sp.key_hex,
            PreviewDistEntry {
                k: total_k,
                counts,
                own: Some(sp.own_id),
                opp_mix: sp.opp_mix,
                side: Some(sp.side),
            },
        );
    }

    let doc = PreviewDistFile {
        format: PREVIEW_DIST_FORMAT.to_string(),
        policy_epoch: Some(policy_epoch_for(shared.ruleset).to_string()),
        entries,
    };
    fs::write(
        output.join("preview-dist.json"),
        serde_json::to_string(&doc).unwrap(),
    )
    .expect("write preview-dist.json");

    println!(
        "{}",
        json!({
            "mode": "preview-dist-only",
            "unique_keys": doc.entries.len(),
            "cached_keys": cached_keys,
            "computed_previews": work.len(),
            "target_k": target_k,
            "workers": n_workers,
            "wall_ms": (wall_start.elapsed().as_secs_f64() * 1000.0).round() as u64,
            "output": output.display().to_string(),
        })
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: {} GAMES.jsonl NEW_OUTPUT_DIR [workers]", args[0]);
        std::process::exit(2);
    }
    let task_file = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let default_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let workers: usize = args
        .get(3)
        .map(|s| s.parse().expect("invalid workers count"))
        .unwrap_or(default_workers);
    assert!(workers > 0, "workers must be > 0");

    let task_bytes = fs::read(&task_file)
        .unwrap_or_else(|e| panic!("read {}: {e}", task_file.display()));
    let source = String::from_utf8(task_bytes.clone()).expect("utf8 task file");
    let trimmed = source.trim();
    let tasks: Vec<Task> = if trimmed.starts_with('[') {
        serde_json::from_str(trimmed).expect("parse JSON array tasks")
    } else {
        trimmed
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("parse JSONL task"))
            .collect()
    };
    assert!(!tasks.is_empty(), "empty task schedule");

    // Refuse to mix builds or overwrite an earlier run (matches nash-refresh.mjs).
    fs::create_dir(&output).unwrap_or_else(|e| panic!("create {}: {e}", output.display()));
    let log_dir = if env::var("NASH_SAVE_LOGS").ok().as_deref() == Some("1") {
        let dir = output.join("logs");
        fs::create_dir(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
        Some(dir)
    } else {
        None
    };

    let root = env::var("NC2000_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest_root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .expect("workspace root")
                .to_path_buf();
            if manifest_root.join("data/live-search.json").exists() {
                manifest_root
            } else {
                env::current_dir().expect("cwd")
            }
        });
    let dex_bytes = fs::read(root.join("data/gen2stadium2.json")).expect("read data/gen2stadium2.json");
    let dex = Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).expect("parse dex");
    let learnsets_text =
        fs::read_to_string(root.join("data/learnsets-gen2.json")).expect("read learnsets");
    let learnsets = Learnsets::from_json(&learnsets_text).expect("parse learnsets");

    let live: LiveSearchCfg = serde_json::from_str(&read_data(&root, "live-search")).unwrap();
    let profiles: SearchProfilesCfg =
        serde_json::from_str(&read_data(&root, "search-profiles")).unwrap();

    let mut teams: HashMap<String, Vec<PokemonSet>> = HashMap::new();
    let mut team_json_map: HashMap<String, String> = HashMap::new();
    for name in TEAM_FILES {
        let tf: TeamFile = serde_json::from_str(&read_data(&root, name)).unwrap();
        for t in tf.teams {
            team_json_map.insert(t.id.clone(), serde_json::to_string(&t.sets).unwrap());
            teams.insert(t.id, t.sets);
        }
    }

    let has_disclosure = tasks.iter().any(|t| t.mixtures.is_some());
    let needs_default_pool = tasks.iter().any(|t| t.mixtures.is_none());

    let mut disclosed: HashMap<String, MetaPool> = HashMap::new();
    if has_disclosure {
        let cand: TeamFile = serde_json::from_str(&read_data(&root, CANDIDATE_FILE)).unwrap();
        disclosed.insert(
            "candidate".into(),
            MetaPool {
                teams: cand
                    .teams
                    .into_iter()
                    .map(|t| MetaTeam {
                        id: t.id,
                        sets: t.sets,
                        weight: t.weight,
                    })
                    .collect(),
            },
        );
        let inc: TeamFile = serde_json::from_str(&read_data(&root, INCUMBENT_FILE)).unwrap();
        disclosed.insert(
            "incumbent".into(),
            MetaPool {
                teams: inc
                    .teams
                    .into_iter()
                    .map(|t| MetaTeam {
                        id: t.id,
                        sets: t.sets,
                        weight: t.weight,
                    })
                    .collect(),
            },
        );
        let deme_sets = teams["deme-r1-00-s0030"].clone();
        disclosed.insert(
            "deme".into(),
            MetaPool {
                teams: vec![MetaTeam {
                    id: "deme-r1-00-s0030".into(),
                    sets: deme_sets,
                    weight: Some(1.0),
                }],
            },
        );
    }

    let default_ruleset = resolve_ruleset(
        env::var("NASH_RULESET").ok().as_deref(),
        Ruleset::PsStadium2,
    );
    let task_rulesets: Vec<Ruleset> = tasks
        .iter()
        .map(|t| resolve_ruleset(t.ruleset.as_deref(), default_ruleset))
        .collect();

    let pools_text = env::var("NASH_POOLS")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|p| fs::read_to_string(&p).unwrap_or_else(|e| panic!("read NASH_POOLS {p}: {e}")));

    if let Some(ref extra_text) = pools_text {
        let extra: ExtraPoolsFile = serde_json::from_str(extra_text).expect("parse NASH_POOLS");
        for t in extra.teams {
            assert!(!teams.contains_key(&t.id), "extra team ID already exists: {}", t.id);
            let sets_json = serde_json::to_string(&t.sets).unwrap();
            let mut rulesets_for_team: Vec<Ruleset> = Vec::new();
            for (task, &rs) in tasks.iter().zip(task_rulesets.iter()) {
                if (task.ids[0] == t.id || task.ids[1] == t.id) && !rulesets_for_team.contains(&rs) {
                    rulesets_for_team.push(rs);
                }
            }
            if rulesets_for_team.is_empty() {
                rulesets_for_team.push(default_ruleset);
            }
            for rs in rulesets_for_team {
                let v = validate_team_with_ruleset(&dex, &learnsets, &sets_json, rs);
                assert!(
                    v["ok"].as_bool() == Some(true),
                    "invalid team {} under {}: {v}",
                    t.id,
                    rs.canonical_id()
                );
            }
            team_json_map.insert(t.id.clone(), sets_json);
            teams.insert(t.id, t.sets);
        }
        for (name, members) in extra.mixtures {
            assert!(!disclosed.contains_key(&name), "duplicate mixture: {name}");
            let mut seen = HashSet::new();
            let mut meta_teams = Vec::with_capacity(members.len());
            for m in members {
                assert!(seen.insert(m.id.clone()), "duplicate member {} in {name}", m.id);
                let sets = teams
                    .get(&m.id)
                    .unwrap_or_else(|| panic!("unknown team {} in mixture {name}", m.id))
                    .clone();
                meta_teams.push(MetaTeam {
                    id: m.id,
                    sets,
                    weight: Some(m.weight),
                });
            }
            disclosed.insert(name, MetaPool { teams: meta_teams });
        }
    }

    for (name, pool) in &disclosed {
        let sum: f64 = pool.teams.iter().map(|t| t.weight.unwrap_or(0.0)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "mixture {name} weights sum to {sum}"
        );
        for t in &pool.teams {
            assert!(t.weight.unwrap_or(0.0) > 0.0, "non-positive weight in {name}");
        }
    }

    // Validate every referenced team once up front per active ruleset (matches nash-refresh.mjs).
    let mut validated: HashSet<(String, Ruleset)> = HashSet::new();
    for (t, &rs) in tasks.iter().zip(task_rulesets.iter()) {
        for id in &t.ids {
            if validated.insert((id.clone(), rs)) {
                let sets_json = team_json_map
                    .get(id)
                    .unwrap_or_else(|| panic!("unknown team id: {id}"));
                let v = validate_team_with_ruleset(&dex, &learnsets, sets_json, rs);
                assert!(
                    v["ok"].as_bool() == Some(true),
                    "invalid {id} under {}: {v}",
                    rs.canonical_id()
                );
            }
        }
    }

    let default_pool = if needs_default_pool {
        let pool_bytes = fs::read(root.join("data/belief-pool-v3/belief-pool.json"))
            .expect("read data/belief-pool-v3/belief-pool.json");
        let pool: MetaPool = serde_json::from_slice(&pool_bytes).expect("parse belief-pool");
        Some(pool)
    } else {
        None
    };

    let mut inputs: Vec<&str> = TEAM_FILES.to_vec();
    inputs.push("live-search");
    inputs.push("search-profiles");
    if needs_default_pool {
        inputs.push("belief-pool-v3/belief-pool");
    }
    if has_disclosure {
        inputs.push(CANDIDATE_FILE);
    }
    let mut data_hashes = serde_json::Map::new();
    for k in inputs {
        let text = read_data(&root, k);
        data_hashes.insert(k.to_string(), json!(sha256_hex(text.as_bytes())));
    }
    let exe_bytes = env::current_exe()
        .ok()
        .and_then(|p| fs::read(p).ok())
        .unwrap_or_default();
    let preview_iters: u32 = env::var("NASH_PREVIEW_ITERS")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("invalid NASH_PREVIEW_ITERS"))
        .unwrap_or(live.unknown_preview);
    let battle_iters: u32 = env::var("NASH_BATTLE_ITERS")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("invalid NASH_BATTLE_ITERS"))
        .unwrap_or(live.battle.adaptive);
    let battle_mode = match env::var("NASH_BATTLE_MODE").ok().as_deref() {
        None | Some("") | Some("adaptive") => BattleMode::Adaptive,
        Some("fixed") => BattleMode::Fixed,
        Some(other) => panic!("invalid NASH_BATTLE_MODE: {other} (expected adaptive or fixed)"),
    };
    let preview_cache_enabled = match env::var("NASH_PREVIEW_CACHE").ok().as_deref() {
        None | Some("") | Some("1") | Some("true") => true,
        Some("0") | Some("false") => false,
        Some(other) => panic!("invalid NASH_PREVIEW_CACHE: {other}"),
    };
    let preview_dist_only = match env::var("NASH_PREVIEW_DIST_ONLY").ok().as_deref() {
        None | Some("") | Some("0") | Some("false") => false,
        Some("1") | Some("true") => true,
        Some(other) => panic!("invalid NASH_PREVIEW_DIST_ONLY: {other}"),
    };
    let preview_dist_k: u32 = env::var("NASH_PREVIEW_DIST_K")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("invalid NASH_PREVIEW_DIST_K"))
        .unwrap_or(if preview_dist_only { 16 } else { 0 });
    let preview_dist_path = env::var("NASH_PREVIEW_DIST_CACHE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let mut preloaded_dist: HashMap<String, PreviewDistEntry> = HashMap::new();
    let mut dist_file_sha: Option<String> = None;
    if let Some(ref p) = preview_dist_path {
        if p.exists() {
            let raw = fs::read_to_string(p)
                .unwrap_or_else(|e| panic!("read NASH_PREVIEW_DIST_CACHE {}: {e}", p.display()));
            dist_file_sha = Some(sha256_hex(raw.as_bytes()));
            let doc: PreviewDistFile = serde_json::from_str(&raw)
                .unwrap_or_else(|e| panic!("parse NASH_PREVIEW_DIST_CACHE {}: {e}", p.display()));
            assert!(
                doc.format == PREVIEW_DIST_FORMAT,
                "unsupported preview dist cache format: {}",
                doc.format
            );
            for (k, v) in doc.entries {
                preloaded_dist.insert(k, v);
            }
        }
    }
    let use_preview_dist = preview_cache_enabled
        && (preview_dist_only || preview_dist_path.is_some() || preview_dist_k >= 2);
    let is_custom_budget = preview_iters != live.unknown_preview
        || battle_iters != live.battle.adaptive
        || battle_mode != BattleMode::Adaptive;

    let mut manifest = serde_json::Map::new();
    if let Some(ref pt) = pools_text {
        manifest.insert("poolsSha256".into(), json!(sha256_hex(pt.as_bytes())));
    }
    manifest.insert("tasks".into(), json!(sha256_hex(&task_bytes)));
    manifest.insert("runner".into(), json!("native-rust"));
    manifest.insert("ruleset".into(), json!(default_ruleset.canonical_id()));
    manifest.insert("policyEpoch".into(), json!(policy_epoch_for(default_ruleset)));
    manifest.insert("nativeBinary".into(), json!(sha256_hex(&exe_bytes)));
    if is_custom_budget {
        manifest.insert(
            "budget".into(),
            json!({
                "previewIters": preview_iters,
                "battleIters": battle_iters,
                "battleMode": battle_mode.as_str(),
            }),
        );
    }
    manifest.insert("previewCache".into(), json!(preview_cache_enabled));
    manifest.insert("previewDistCache".into(), json!(use_preview_dist));
    if preview_dist_only {
        manifest.insert("previewDistOnly".into(), json!(true));
    }
    if use_preview_dist || preview_dist_only {
        manifest.insert("previewDistK".into(), json!(preview_dist_k));
        if let Some(sha) = dist_file_sha {
            manifest.insert("previewDistSha256".into(), json!(sha));
        }
    }
    manifest.insert("data".into(), Value::Object(data_hashes));
    fs::write(
        output.join("manifest.json"),
        serde_json::to_string(&Value::Object(manifest)).unwrap(),
    )
    .expect("write manifest.json");

    let preloaded_for_only = if preview_dist_only {
        preloaded_dist.clone()
    } else {
        HashMap::new()
    };

    let shared = Arc::new(SharedState {
        dex,
        ruleset: default_ruleset,
        teams,
        team_json_map,
        disclosed,
        default_pool,
        preview_iters,
        battle_iters,
        battle_mode,
        blind_c: profiles.blind.c,
        log_dir,
        preview_cache: preview_cache_enabled.then(PreviewCache::new),
        preview_dist_cache: use_preview_dist
            .then(|| PreviewDistCache::new(preloaded_dist, preview_dist_k)),
    });

    if preview_dist_only {
        assert!(preview_dist_k >= 1, "NASH_PREVIEW_DIST_K must be >= 1 in preview-dist-only mode");
        precompute_preview_dists(&shared, &tasks, &preloaded_for_only, preview_dist_k, workers, &output);
        return;
    }

    let wall_start = Instant::now();
    let next = AtomicUsize::new(0);
    let n_workers = workers.min(tasks.len());
    std::thread::scope(|s| {
        for _ in 0..n_workers {
            let shared = &shared;
            let tasks = &tasks;
            let next = &next;
            let output = &output;
            s.spawn(move || loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= tasks.len() {
                    break;
                }
                let result = run_game(shared, idx, &tasks[idx]);
                fs::write(
                    output.join(format!("{idx}.json")),
                    serde_json::to_string(&result).unwrap(),
                )
                .expect("write game result");
            });
        }
    });

    let (cache_hits, cache_misses) = shared
        .preview_cache
        .as_ref()
        .map(|c| {
            (
                c.hits.load(Ordering::Relaxed),
                c.misses.load(Ordering::Relaxed),
            )
        })
        .unwrap_or((0, 0));
    let (dist_hits, dist_misses) = shared
        .preview_dist_cache
        .as_ref()
        .map(|c| {
            (
                c.hits.load(Ordering::Relaxed),
                c.misses.load(Ordering::Relaxed),
            )
        })
        .unwrap_or((0, 0));
    if let Some(ref dcache) = shared.preview_dist_cache {
        let new_entries = dcache.export_new_entries();
        if !new_entries.is_empty() {
            let doc = PreviewDistFile {
                format: PREVIEW_DIST_FORMAT.to_string(),
                policy_epoch: Some(policy_epoch_for(shared.ruleset).to_string()),
                entries: new_entries,
            };
            fs::write(
                output.join("preview-dist-new.json"),
                serde_json::to_string(&doc).unwrap(),
            )
            .expect("write preview-dist-new.json");
        }
    }

    println!(
        "{}",
        json!({
            "games": tasks.len(),
            "output": output.display().to_string(),
            "workers": n_workers,
            "wall_ms": (wall_start.elapsed().as_secs_f64() * 1000.0).round() as u64,
            "preview_cache_hits": cache_hits,
            "preview_cache_misses": cache_misses,
            "preview_dist_hits": dist_hits,
            "preview_dist_misses": dist_misses,
        })
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dex() -> Dex {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let bytes = fs::read(root.join("data/gen2stadium2.json")).expect("read dex");
        Dex::from_json(std::str::from_utf8(&bytes).unwrap()).expect("parse dex")
    }

    #[test]
    fn preview_keys_isolate_policy_epoch_and_policy_name() {
        let dex = test_dex();
        let prior = MetaPool { teams: vec![] };
        let cfg = RmConfig::default();
        let agent = ProtocolAgent::new(&dex, 0, prior.clone(), cfg, 42);
        let frame = vec!["|poke|p2|Snorlax, L55, M|item".to_string()];
        let team_map = HashMap::new();

        let d_v1 = preview_dist_key_with_epoch(
            "policy-v1", 0, 240_000, 0.4, "current", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        let d_cur = preview_dist_key_with_epoch(
            POLICY_EPOCH, 0, 240_000, 0.4, "current", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        let d_cur81 = preview_dist_key_with_epoch(
            POLICY_EPOCH, 0, 240_000, 0.4, "current81", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        let d_default = preview_dist_key(
            0, 240_000, 0.4, "current", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        assert_ne!(d_v1, d_cur, "preview_dist_key must differ across policy epochs");
        assert_ne!(d_cur, d_cur81, "preview_dist_key must differ across policy names");
        assert_eq!(d_cur, d_default, "default preview_dist_key must hash POLICY_EPOCH");

        let c_v1 = preview_cache_key_with_epoch(
            "policy-v1", 0, 123, 240_000, 0.4, "current", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        let c_cur = preview_cache_key_with_epoch(
            POLICY_EPOCH, 0, 123, 240_000, 0.4, "current", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        let c_cur81 = preview_cache_key_with_epoch(
            POLICY_EPOCH, 0, 123, 240_000, 0.4, "current81", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        let c_default = preview_cache_key(
            0, 123, 240_000, 0.4, "current", "[]", &frame, "{}", &prior, &agent, &team_map,
        );
        assert_ne!(c_v1, c_cur, "preview_cache_key must differ across policy epochs");
        assert_ne!(c_cur, c_cur81, "preview_cache_key must differ across policy names");
        assert_eq!(c_cur, c_default, "default preview_cache_key must hash POLICY_EPOCH");
    }

    #[test]
    fn preview_dist_file_backward_compatible_with_missing_policy_epoch() {
        let legacy = r#"{"format":"nc2000-preview-dist-v1","entries":{}}"#;
        let parsed: PreviewDistFile = serde_json::from_str(legacy).expect("parse legacy");
        assert_eq!(parsed.policy_epoch, None);
        let current = PreviewDistFile {
            format: PREVIEW_DIST_FORMAT.to_string(),
            policy_epoch: Some(POLICY_EPOCH.to_string()),
            entries: BTreeMap::new(),
        };
        let v: Value = serde_json::to_value(&current).unwrap();
        assert_eq!(v["policyEpoch"], POLICY_EPOCH);
    }

    #[test]
    fn castle_ruleset_cache_keys_isolate_overlays_and_handle_filtered_beliefs() {
        let dex = test_dex();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let tp4: TeamFile = serde_json::from_str(&read_data(root, "team-pool-v4/team-pool")).unwrap();
        let legal_a = tp4.teams[0].sets.clone();
        let legal_b = tp4.teams[1].sets.clone();
        // Put a Castle-illegal (NYPC Lovely Kiss Snorlax) entry BEFORE the legal candidate in `prior`
        // so `Belief` filters index 0 out under Castle, shifting `belief.alive()` indices relative to `prior.teams`.
        let mut illegal_nypc = legal_b.clone();
        if let Some(lax) = illegal_nypc.iter_mut().find(|s| s.species == "Snorlax") {
            lax.moves[0] = "Lovely Kiss".into();
        } else {
            illegal_nypc[0].species = "Snorlax".into();
            illegal_nypc[0].name = "Snorlax".into();
            illegal_nypc[0].moves = vec![
                "Lovely Kiss".into(),
                "Body Slam".into(),
                "Rest".into(),
                "Curse".into(),
            ];
        }
        let prior = MetaPool {
            teams: vec![
                MetaTeam { id: "illegal-nypc".into(), sets: illegal_nypc.clone(), weight: Some(0.5) },
                MetaTeam { id: "legal-b".into(), sets: legal_b.clone(), weight: Some(0.5) },
            ],
        };
        let mut team_map = HashMap::new();
        team_map.insert("illegal-nypc".into(), serde_json::to_string(&illegal_nypc).unwrap());
        team_map.insert("legal-b".into(), serde_json::to_string(&legal_b).unwrap());
        let own_json = serde_json::to_string(&legal_a).unwrap();

        assert_eq!(cache_epoch_tag(Ruleset::PsStadium2), POLICY_EPOCH);
        assert_eq!(cache_epoch_tag(Ruleset::castle_free()), "castle-policy-v4:castle-free");
        assert_eq!(
            cache_epoch_tag(Ruleset::castle_tournament()),
            "castle-policy-v4:castle-tournament"
        );

        let make_key = |rs: Ruleset| {
            let mut b = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &legal_a, &legal_b, rs)
                .unwrap();
            let mut ch = PlayerChannel::new(0);
            let frame = ch.frame(&mut b, &dex).unwrap();
            let req_json = serde_json::to_string(&frame.request).unwrap();
            let mut agent = ProtocolAgent::new_with_ruleset(
                &dex,
                0,
                prior.clone(),
                RmConfig::default(),
                42,
                rs,
            );
            agent.set_own_team(legal_a.clone());
            for line in &frame.lines {
                agent.push_line(&dex, line);
            }
            assert!(agent.on_request(&dex, &req_json).unwrap());
            let bel = agent.belief().unwrap();
            assert!(!bel.is_fallback());
            if rs.is_castle() {
                // `illegal-nypc` was filtered out, so `legal-b` sits at `belief.cands[0]`.
                assert_eq!(bel.alive(), &[0]);
                assert_eq!(bel.candidate_id(0), "legal-b");
            }
            let dk = preview_dist_key_for_ruleset(
                rs,
                0,
                720_000,
                0.4,
                "current",
                &own_json,
                &frame.lines,
                &req_json,
                &prior,
                &agent,
                &team_map,
            );
            let ck = preview_cache_key_for_ruleset(
                rs,
                0,
                99,
                720_000,
                0.4,
                "current",
                &own_json,
                &frame.lines,
                &req_json,
                &prior,
                &agent,
                &team_map,
            );
            (dk, ck)
        };

        let (dk_ps, ck_ps) = make_key(Ruleset::PsStadium2);
        let (dk_cf, ck_cf) = make_key(Ruleset::castle_free());
        let (dk_ct, ck_ct) = make_key(Ruleset::castle_tournament());
        assert_ne!(dk_ps, dk_cf);
        assert_ne!(dk_cf, dk_ct);
        assert_ne!(ck_ps, ck_cf);
        assert_ne!(ck_cf, ck_ct);
    }

    #[test]
    fn run_game_executes_cleanly_under_castle_free() {
        let dex = test_dex();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let tp4: TeamFile = serde_json::from_str(&read_data(root, "team-pool-v4/team-pool")).unwrap();
        let t0 = &tp4.teams[0];
        let t1 = &tp4.teams[1];
        let mut teams = HashMap::new();
        let mut team_json_map = HashMap::new();
        for t in [t0, t1] {
            team_json_map.insert(t.id.clone(), serde_json::to_string(&t.sets).unwrap());
            teams.insert(t.id.clone(), t.sets.clone());
        }
        let mut disclosed = HashMap::new();
        for t in [t0, t1] {
            disclosed.insert(
                t.id.clone(),
                MetaPool {
                    teams: vec![MetaTeam {
                        id: t.id.clone(),
                        sets: t.sets.clone(),
                        weight: Some(1.0),
                    }],
                },
            );
        }
        let shared = SharedState {
            dex,
            ruleset: Ruleset::castle_free(),
            teams,
            team_json_map,
            disclosed,
            default_pool: None,
            preview_iters: 256,
            battle_iters: 64,
            battle_mode: BattleMode::Fixed,
            blind_c: 0.4,
            log_dir: None,
            preview_cache: None,
            preview_dist_cache: None,
        };
        let task = Task {
            block: json!(0),
            cell: json!("castle-smoke"),
            row_seat: 0,
            ids: [t0.id.clone(), t1.id.clone()],
            seeds: [101, 202],
            preview_seeds: None,
            battle_seed: 303,
            picks: None,
            mixtures: Some([t0.id.clone(), t1.id.clone()]),
            policies: Some(["current".into(), "current".into()]),
            ruleset: None,
        };
        let res_a = run_game(&shared, 0, &task);
        let res_b = run_game(&shared, 0, &task);
        assert_eq!(res_a["ruleset"], "castle-free");
        assert_eq!(res_a["preview"][0]["source"], "live");
        assert_eq!(res_a["preview"][1]["source"], "live");
        assert!(matches!(
            res_a["outcome"].as_str(),
            Some("p1" | "p2" | "tie" | "cap")
        ));
        assert_eq!(res_a["actionSha256"], res_b["actionSha256"]);
    }
}
