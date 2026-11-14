//! Self-contained, fail-closed artifact contract for the M17e exact-endgame
//! sweep. Shards bind their solver, reconstruction inputs, selection policy,
//! and row set; only a complete merged artifact is accepted by the gate.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

pub const SHARD_SCHEMA: &str = "nc2000-m17e-exactness-shard-v3";
pub const MERGED_SCHEMA: &str = "nc2000-m17e-exactness-merged-v3";
pub const FORMAL_PROFILE: &str = "m17e-formal-sweep-v3";
pub const CUSTOM_PROFILE: &str = "m17e-custom-sweep-v3";

fn fnv1a64_update(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn fingerprint_part(hash: u64, bytes: &[u8]) -> u64 {
    let hash = fnv1a64_update(hash, &(bytes.len() as u64).to_le_bytes());
    fnv1a64_update(hash, bytes)
}

fn tagged_fingerprint<'a>(tag: &str, parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hash = fingerprint_part(0xcbf2_9ce4_8422_2325, tag.as_bytes());
    for part in parts {
        hash = fingerprint_part(hash, part);
    }
    format!("fnv1a64:{hash:016x}:{tag}")
}

fn file_fingerprint(path: &Path, tag: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(tagged_fingerprint(tag, [bytes.as_slice()]))
}

pub fn solver_build_fingerprint() -> &'static str {
    env!("M17E_SOLVER_BUILD_FINGERPRINT")
}

pub fn generator_executable_fingerprint() -> Result<String, String> {
    let path =
        std::env::current_exe().map_err(|error| format!("resolve current executable: {error}"))?;
    file_fingerprint(&path, "m17e-generator-executable-v1")
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDataFingerprints {
    pub dex: String,
    pub meta_pool: String,
    pub community_rentals: String,
    pub learnsets: String,
}

pub fn runtime_data_fingerprints(root: &Path) -> Result<RuntimeDataFingerprints, String> {
    Ok(RuntimeDataFingerprints {
        dex: file_fingerprint(&root.join("data/gen2stadium2.json"), "m17e-dex-v1")?,
        meta_pool: file_fingerprint(
            &root.join("data/meta-pool-v0/meta-pool.json"),
            "m17e-meta-pool-v1",
        )?,
        community_rentals: file_fingerprint(
            &root.join("data/community-rentals-v0/teams.json"),
            "m17e-community-rentals-v1",
        )?,
        learnsets: file_fingerprint(&root.join("data/learnsets-gen2.json"), "m17e-learnsets-v1")?,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverConfig {
    pub work_budget: usize,
    pub node_budget: usize,
    pub cell_cap: usize,
    pub eps: f64,
    pub trial_depth: usize,
    pub descend_floor: f64,
    pub dead_damage_quotient: bool,
    pub fold_terminal_nodes: bool,
    pub fold_closed_nodes: bool,
    pub monotone_stall_scheduling: bool,
    pub two_sided_resource_scheduling: bool,
    pub certified_action_pruning: bool,
    pub support_br_scheduling: bool,
    pub threshold_radius: f64,
}

impl SolverConfig {
    pub fn formal() -> Self {
        Self {
            work_budget: 1_000_000,
            node_budget: 120_000,
            cell_cap: 4096,
            eps: 0.02,
            trial_depth: 24,
            descend_floor: 0.1,
            dead_damage_quotient: true,
            fold_terminal_nodes: true,
            fold_closed_nodes: true,
            monotone_stall_scheduling: true,
            two_sided_resource_scheduling: true,
            certified_action_pruning: true,
            support_br_scheduling: true,
            threshold_radius: 0.02,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionConfig {
    pub hp_cap: u64,
    pub max_alive_per_side: usize,
    pub per_battle: usize,
    pub side_filter: Option<usize>,
    pub turn_filter: Option<u16>,
    pub decision_order: String,
    pub reconstruction_seed: u64,
}

impl SelectionConfig {
    pub fn formal() -> Self {
        Self {
            hp_cap: 150,
            max_alive_per_side: 2,
            per_battle: 2,
            side_filter: None,
            turn_filter: None,
            decision_order: "reverse".to_string(),
            reconstruction_seed: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunIdentity {
    pub profile: String,
    pub solver_build_fingerprint: String,
    pub generator_executable_fingerprint: String,
    pub runtime_data: RuntimeDataFingerprints,
    pub corpus_fingerprint: String,
    pub corpus_count: usize,
    pub solver: SolverConfig,
    pub selection: SelectionConfig,
}

impl RunIdentity {
    pub fn is_formal(&self) -> bool {
        self.profile == FORMAL_PROFILE
            && self.solver == SolverConfig::formal()
            && self.selection == SelectionConfig::formal()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub battle: usize,
    pub decision: usize,
    pub side: usize,
    pub turn: u16,
    pub human: String,
    pub exact: f64,
    pub width: f64,
    pub stop: u16,
    pub eval: f64,
    pub alive0: usize,
    pub alive1: usize,
    pub total_hp: u64,
    pub state_key128: String,
    pub desc: String,
}

impl Row {
    pub fn coordinate(&self) -> (usize, usize) {
        (self.battle, self.decision)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowSummary {
    pub row_count: usize,
    pub coordinate_fingerprint: String,
    pub state_fingerprint: String,
    pub row_fingerprint: String,
}

fn records_fingerprint(tag: &str, records: Vec<Vec<Vec<u8>>>) -> String {
    let mut hash = fingerprint_part(0xcbf2_9ce4_8422_2325, tag.as_bytes());
    for record in records {
        hash = fingerprint_part(hash, b"record");
        for field in record {
            hash = fingerprint_part(hash, &field);
        }
    }
    format!("fnv1a64:{hash:016x}:{tag}")
}

fn row_record(row: &Row) -> Vec<Vec<u8>> {
    vec![
        row.battle.to_string().into_bytes(),
        row.decision.to_string().into_bytes(),
        row.side.to_string().into_bytes(),
        row.turn.to_string().into_bytes(),
        row.human.as_bytes().to_vec(),
        format!("{:016x}", row.exact.to_bits()).into_bytes(),
        format!("{:016x}", row.width.to_bits()).into_bytes(),
        row.stop.to_string().into_bytes(),
        format!("{:016x}", row.eval.to_bits()).into_bytes(),
        row.alive0.to_string().into_bytes(),
        row.alive1.to_string().into_bytes(),
        row.total_hp.to_string().into_bytes(),
        row.state_key128.as_bytes().to_vec(),
        row.desc.as_bytes().to_vec(),
    ]
}

pub fn summarize_rows(rows: &[Row]) -> RowSummary {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|row| row.coordinate());
    let coordinates = ordered
        .iter()
        .map(|row| {
            vec![
                row.battle.to_string().into_bytes(),
                row.decision.to_string().into_bytes(),
                row.side.to_string().into_bytes(),
                row.turn.to_string().into_bytes(),
            ]
        })
        .collect();
    let mut states = rows
        .iter()
        .map(|row| vec![row.state_key128.as_bytes().to_vec()])
        .collect::<Vec<_>>();
    states.sort();
    let row_records = ordered.into_iter().map(row_record).collect();
    RowSummary {
        row_count: rows.len(),
        coordinate_fingerprint: records_fingerprint("m17e-coordinate-v1", coordinates),
        state_fingerprint: records_fingerprint("m17e-state-v1", states),
        row_fingerprint: records_fingerprint("m17e-row-v1", row_records),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardDescriptor {
    pub battle_lo: usize,
    pub battle_hi: usize,
    pub summary: RowSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardArtifact {
    pub schema: String,
    pub run: RunIdentity,
    pub shard: ShardDescriptor,
    pub rows: Vec<Row>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeInfo {
    pub shards: Vec<ShardDescriptor>,
    pub summary: RowSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergedArtifact {
    pub schema: String,
    pub run: RunIdentity,
    pub merge: MergeInfo,
    pub rows: Vec<Row>,
}

fn validate_row(row: &Row, where_: &str) -> Result<(), String> {
    if row.side > 1 {
        return Err(format!("{where_}: side must be 0 or 1"));
    }
    if !row.exact.is_finite() || !row.width.is_finite() || row.width < 0.0 || !row.eval.is_finite()
    {
        return Err(format!("{where_}: invalid exact/width/eval"));
    }
    if row.state_key128.len() != 32
        || !row
            .state_key128
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || row
            .state_key128
            .bytes()
            .any(|byte| byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{where_}: state_key128 must be 32 lowercase hex digits"
        ));
    }
    Ok(())
}

fn validate_run_identity(run: &RunIdentity, where_: &str) -> Result<(), String> {
    if !matches!(run.profile.as_str(), FORMAL_PROFILE | CUSTOM_PROFILE) {
        return Err(format!("{where_}: unsupported profile {:?}", run.profile));
    }
    for (name, value) in [
        (
            "solver_build_fingerprint",
            run.solver_build_fingerprint.as_str(),
        ),
        (
            "generator_executable_fingerprint",
            run.generator_executable_fingerprint.as_str(),
        ),
        ("corpus_fingerprint", run.corpus_fingerprint.as_str()),
        ("runtime_data.dex", run.runtime_data.dex.as_str()),
        (
            "runtime_data.meta_pool",
            run.runtime_data.meta_pool.as_str(),
        ),
        (
            "runtime_data.community_rentals",
            run.runtime_data.community_rentals.as_str(),
        ),
        (
            "runtime_data.learnsets",
            run.runtime_data.learnsets.as_str(),
        ),
    ] {
        if value.is_empty() {
            return Err(format!("{where_}.{name} is empty"));
        }
    }
    if run.corpus_count == 0 {
        return Err(format!("{where_}: corpus_count must be positive"));
    }
    let solver = &run.solver;
    if solver.work_budget == 0
        || solver.node_budget == 0
        || solver.cell_cap == 0
        || solver.trial_depth == 0
        || !solver.eps.is_finite()
        || solver.eps <= 0.0
        || !solver.descend_floor.is_finite()
        || solver.descend_floor < 0.0
        || !solver.threshold_radius.is_finite()
        || solver.threshold_radius < 0.0
    {
        return Err(format!("{where_}: invalid solver numeric config"));
    }
    let selection = &run.selection;
    if selection.hp_cap == 0
        || selection.max_alive_per_side == 0
        || selection.per_battle == 0
        || selection.side_filter.is_some_and(|side| side > 1)
        || selection.decision_order != "reverse"
    {
        return Err(format!("{where_}: invalid selection config"));
    }
    Ok(())
}

fn validate_unique_rows(rows: &[Row], where_: &str, require_sorted: bool) -> Result<(), String> {
    let mut coordinates = HashSet::new();
    let mut states = HashSet::new();
    let mut previous = None;
    for (index, row) in rows.iter().enumerate() {
        validate_row(row, &format!("{where_}.rows[{index}]"))?;
        let coordinate = row.coordinate();
        if require_sorted && previous.is_some_and(|prior| prior >= coordinate) {
            return Err(format!("{where_}: rows are not strictly coordinate-sorted"));
        }
        previous = Some(coordinate);
        if !coordinates.insert(coordinate) {
            return Err(format!("{where_}: duplicate coordinate {coordinate:?}"));
        }
        if !states.insert(row.state_key128.as_str()) {
            return Err(format!(
                "{where_}: duplicate state_key128 {}",
                row.state_key128
            ));
        }
    }
    Ok(())
}

pub fn validate_shard(artifact: &ShardArtifact) -> Result<(), String> {
    if artifact.schema != SHARD_SCHEMA {
        return Err(format!("unsupported shard schema {:?}", artifact.schema));
    }
    validate_run_identity(&artifact.run, "shard.run")?;
    if artifact.shard.battle_lo > artifact.shard.battle_hi
        || artifact.shard.battle_hi >= artifact.run.corpus_count
    {
        return Err("shard battle range is outside the corpus".to_string());
    }
    validate_unique_rows(&artifact.rows, "shard", false)?;
    if let Some(row) = artifact
        .rows
        .iter()
        .find(|row| row.battle < artifact.shard.battle_lo || row.battle > artifact.shard.battle_hi)
    {
        return Err(format!(
            "shard row b{} is outside {}-{}",
            row.battle, artifact.shard.battle_lo, artifact.shard.battle_hi
        ));
    }
    let actual = summarize_rows(&artifact.rows);
    if actual != artifact.shard.summary {
        return Err(format!(
            "shard summary mismatch: declared {:?}, actual {:?}",
            artifact.shard.summary, actual
        ));
    }
    Ok(())
}

pub fn validate_merged(artifact: &MergedArtifact) -> Result<(), String> {
    if artifact.schema != MERGED_SCHEMA {
        return Err(format!("unsupported merged schema {:?}", artifact.schema));
    }
    validate_run_identity(&artifact.run, "merged.run")?;
    if artifact.merge.shards.is_empty() {
        return Err("merged artifact has no shards".to_string());
    }
    if artifact.rows.is_empty() {
        return Err("merged artifact has no anchor rows".to_string());
    }
    let mut next = 0usize;
    for (index, shard) in artifact.merge.shards.iter().enumerate() {
        if shard.battle_lo != next || shard.battle_lo > shard.battle_hi {
            return Err(format!(
                "merge.shards[{index}] does not continue complete range at battle {next}"
            ));
        }
        next = shard
            .battle_hi
            .checked_add(1)
            .ok_or_else(|| "shard range overflow".to_string())?;
    }
    if next != artifact.run.corpus_count {
        return Err(format!(
            "merged shard coverage ends at {next}, corpus has {} battles",
            artifact.run.corpus_count
        ));
    }

    validate_unique_rows(&artifact.rows, "merged", true)?;
    if let Some(row) = artifact
        .rows
        .iter()
        .find(|row| row.battle >= artifact.run.corpus_count)
    {
        return Err(format!(
            "merged row b{} is outside corpus_count {}",
            row.battle, artifact.run.corpus_count
        ));
    }
    let actual_global = summarize_rows(&artifact.rows);
    if actual_global != artifact.merge.summary {
        return Err(format!(
            "merged global summary mismatch: declared {:?}, actual {:?}",
            artifact.merge.summary, actual_global
        ));
    }
    for (index, shard) in artifact.merge.shards.iter().enumerate() {
        let rows = artifact
            .rows
            .iter()
            .filter(|row| row.battle >= shard.battle_lo && row.battle <= shard.battle_hi)
            .cloned()
            .collect::<Vec<_>>();
        let actual = summarize_rows(&rows);
        if actual != shard.summary {
            return Err(format!(
                "merge.shards[{index}] summary mismatch: declared {:?}, actual {:?}",
                shard.summary, actual
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(battle: usize, decision: usize, state: u128) -> Row {
        Row {
            battle,
            decision,
            side: battle % 2,
            turn: 7,
            human: "move rest".to_string(),
            exact: 0.5,
            width: 0.02,
            stop: 0,
            eval: 0.4,
            alive0: 1,
            alive1: 1,
            total_hp: 42,
            state_key128: format!("{state:032x}"),
            desc: format!("b{battle}"),
        }
    }

    fn run() -> RunIdentity {
        RunIdentity {
            profile: FORMAL_PROFILE.to_string(),
            solver_build_fingerprint: "build".to_string(),
            generator_executable_fingerprint: "exe".to_string(),
            runtime_data: RuntimeDataFingerprints {
                dex: "dex".to_string(),
                meta_pool: "meta".to_string(),
                community_rentals: "rentals".to_string(),
                learnsets: "learnsets".to_string(),
            },
            corpus_fingerprint: "corpus".to_string(),
            corpus_count: 2,
            solver: SolverConfig::formal(),
            selection: SelectionConfig::formal(),
        }
    }

    #[test]
    fn merged_requires_complete_ranges_and_summaries() {
        let rows = vec![row(0, 3, 1), row(1, 4, 2)];
        let shards = vec![
            ShardDescriptor {
                battle_lo: 0,
                battle_hi: 0,
                summary: summarize_rows(&rows[..1]),
            },
            ShardDescriptor {
                battle_lo: 1,
                battle_hi: 1,
                summary: summarize_rows(&rows[1..]),
            },
        ];
        let mut artifact = MergedArtifact {
            schema: MERGED_SCHEMA.to_string(),
            run: run(),
            merge: MergeInfo {
                shards,
                summary: summarize_rows(&rows),
            },
            rows,
        };
        validate_merged(&artifact).unwrap();
        artifact.rows.pop();
        assert!(validate_merged(&artifact)
            .unwrap_err()
            .contains("summary mismatch"));
    }

    #[test]
    fn merged_rejects_cross_shard_state_duplicate() {
        let rows = vec![row(0, 3, 1), row(1, 4, 1)];
        let artifact = MergedArtifact {
            schema: MERGED_SCHEMA.to_string(),
            run: run(),
            merge: MergeInfo {
                shards: vec![
                    ShardDescriptor {
                        battle_lo: 0,
                        battle_hi: 0,
                        summary: summarize_rows(&rows[..1]),
                    },
                    ShardDescriptor {
                        battle_lo: 1,
                        battle_hi: 1,
                        summary: summarize_rows(&rows[1..]),
                    },
                ],
                summary: summarize_rows(&rows),
            },
            rows,
        };
        assert!(validate_merged(&artifact)
            .unwrap_err()
            .contains("duplicate state_key128"));
    }
}
