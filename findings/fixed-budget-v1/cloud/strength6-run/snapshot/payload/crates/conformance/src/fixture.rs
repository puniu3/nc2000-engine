//! Schema for the golden fixtures written by `tools/gen-fixtures.js`.
//!
//! Contract with the generator (and with any future Rust replayer):
//! - `choices` are PS-canonical choice strings in inputLog order.
//! - `snapshots[0]` is the state right after both players are set (team
//!   preview pending); `snapshots[k]` (k>0) is taken after choice line
//!   `after_line` whenever processing that line grew the battle log.
//! - `snapshots[*].log` holds the log lines produced since the previous
//!   snapshot, with nondeterministic `|t:|` lines already stripped.
//! - `prng_seed` is PS `Gen5RNG.getSeed()` format: 4 decimal 16-bit limbs
//!   joined by commas. Matching it at every snapshot proves RNG-consumption
//!   parity, not just outcome parity.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub use nc2000_engine::battle::PokemonSet;

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub meta: Meta,
    pub seed: String,
    pub p1team: Vec<PokemonSet>,
    pub p2team: Vec<PokemonSet>,
    pub p1packed: String,
    pub p2packed: String,
    pub choices: Vec<ChoiceLine>,
    pub snapshots: Vec<Snapshot>,
    pub result: BattleResult,
}

#[derive(Debug, Deserialize)]
pub struct Meta {
    pub format: String,
    pub r#mod: String,
    pub pool: String,
    pub index: u32,
}

#[derive(Debug, Deserialize)]
pub struct ChoiceLine {
    pub index: u32,
    pub side: String, // "p1" | "p2"
    pub choice: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    pub after_line: i64,
    pub turn: u32,
    pub prng_seed: String,
    pub request_state: String,
    pub field: FieldEssence,
    pub sides: Vec<SideEssence>,
    pub log: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldEssence {
    pub weather: String,
    pub weather_state: BTreeMap<String, Value>,
    pub pseudo_weather: BTreeMap<String, BTreeMap<String, Value>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SideEssence {
    pub pokemon_left: u32,
    pub side_conditions: BTreeMap<String, BTreeMap<String, Value>>,
    pub active: Vec<Option<String>>,
    pub pokemon: Vec<PokemonEssence>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PokemonEssence {
    pub ident: String,
    pub species: String,
    pub hp: u32,
    pub maxhp: u32,
    pub fainted: bool,
    pub status: String,
    pub status_state: BTreeMap<String, Value>,
    pub boosts: BTreeMap<String, i8>,
    pub item: String,
    pub last_item: String,
    pub item_state: BTreeMap<String, Value>,
    pub moves: Vec<MoveEssence>,
    pub volatiles: BTreeMap<String, BTreeMap<String, Value>>,
    pub types: Vec<String>,
    pub transformed: bool,
    pub active: bool,
    pub position: u32,
}

#[derive(Debug, Deserialize)]
pub struct MoveEssence {
    pub id: String,
    pub pp: u32,
    pub disabled: bool,
}

#[derive(Debug, Deserialize)]
pub struct BattleResult {
    pub winner: String, // "P1" | "P2" | "" (tie)
    pub turns: u32,
}

impl Fixture {
    pub fn load(path: &std::path::Path) -> Result<Fixture, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path:?}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("{path:?}: {e}"))
    }
}

/// All fixture files under a corpus directory, sorted.
pub fn corpus_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// Repo root (crates/conformance/../..).
pub fn repo_root() -> std::path::PathBuf {
    if let Some(root) = std::env::var_os("NC2000_REPO_ROOT") {
        return std::path::PathBuf::from(root)
            .canonicalize()
            .expect("NC2000_REPO_ROOT must name an existing directory");
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}
