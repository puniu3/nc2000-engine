//! Counterfactual 3-Stratum Maximal-Coupling Party A/B Estimator (`policy-v2`).
//!
//! Estimates the exact `policy-v2` party utility difference `Delta U = U(cand) - U(base)`
//! without altering live `SteppedCfPreview` or `nash_refresh.rs` behavior:
//!   1. Extracts exact Turn-0 `LcbGatedMixed` execution policies `(p0, q0)` and `(p1, q1)`
//!      via `ProtocolAgent::preview_execution_policy` under independent preview seeds
//!      `omega_me ⊥ omega_opp` (shared across arms `T0, T1` within each cluster).
//!   2. Decomposes `pi1 - pi0 = (p1 ⊗ q1) - (p0 ⊗ q0)` via exact 3-stratum Maximal Coupling:
//!        - Stratum 1 (`tau = TV(pi1, pi0)`): non-overlapping preview difference with
//!          secondary maximal B-coupling and tertiary lead/exact-A coupling.
//!        - Stratum 2 (`m2`): same-index overlap containing a mutated slot (`s in a`),
//!          evaluated with Turn 1+ Battle CRN.
//!        - Stratum 3a (`m3_zero`): provably bit-exact zero-Term-3 overlap (`s not in a`,
//!          under identical public frame + shared opponent belief OR exact level-cap exclusion
//!          such as Lv55-lead excluding a Lv55 mutated slot) -> skipped at 0 cost.
//!        - Stratum 3b (`m3`): remaining hidden-bench belief overlap (`s not in a`),
//!          evaluated via CRN with full (`h3 = 1.0`), constant Horvitz-Thompson (`--term3-prob`),
//!          or PPS (`--term3-pps`) unbiased sampling, or skipped in screening mode (`--skip-term3`).
//!   3. Aggregates strictly unbiased cluster-level estimates across `K` independent preview-seed
//!      clusters per opponent (`se_cluster`, `se_across_opp`, stratified bootstrap 95% CI).
//!   4. Emits a single compact structured JSON summary line on stdout (M2M telemetry).
//!      Optional per-cluster details are written to `--out FILE.jsonl`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use nc2000_bot::compiled_preview::POLICY_EPOCH;
use nc2000_bot::decision::{DecisionPolicy, DecisionSampler, PolicyMode};
use nc2000_bot::player::PlayerChannel;
use nc2000_bot::preview::{MetaPool, MetaTeam};
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{ProtocolAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{validate_team, Learnsets};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const PARTY_ORDER_MOVES: [&str; 6] = [
    "beatup",
    "metronome",
    "mimic",
    "mirrormove",
    "sketch",
    "transform",
];

const NC2000_LEVEL_CAP: u32 = 155;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BattleMode {
    Adaptive,
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AllocMode {
    /// Optimal Neyman single-draw coupling: at most 1 battle pair per cell,
    /// `p_j = w_j * (sigma_j / sigma_1)` with `(1.0, 0.80, 0.64)` for strata `(1, 2, 3b)`.
    /// Eliminates both multi-stratum over-allocation and Jensen `E[w^2]` taxes.
    Neyman,
    /// Uniform single-draw maximal coupling: `p_j = w_j` for strata `(1, 2, 3b)`, 0 battles on `m3_zero`.
    SingleDraw,
    /// Deterministic per-stratum sampling (1+ pair for every non-empty stratum + HT/PPS for 3b).
    Stratified,
}

impl AllocMode {
    fn as_str(self) -> &'static str {
        match self {
            AllocMode::Neyman => "neyman",
            AllocMode::SingleDraw => "single-draw",
            AllocMode::Stratified => "stratified",
        }
    }
}

const NEYMAN_SCALE_S1: f64 = 1.0;
const NEYMAN_SCALE_S2: f64 = 0.80;
const NEYMAN_SCALE_S3: f64 = 0.64;

#[derive(Clone, Deserialize)]
struct TeamEntry {
    id: String,
    sets: Vec<PokemonSet>,
    #[serde(default)]
    weight: Option<f64>,
}

#[derive(Deserialize)]
struct PoolOrCatalogFile {
    #[serde(default)]
    teams: Vec<TeamEntry>,
    #[serde(default)]
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

#[derive(Clone, Debug)]
struct CliConfig {
    pool_files: Vec<PathBuf>,
    base_id: String,
    cand_id: String,
    base_mix: Option<String>,
    cand_mix: Option<String>,
    shared_opp_belief: bool,
    opp_belief_mix: Option<String>,
    opps_spec: Option<String>,
    clusters: usize,
    preview_reps: usize,
    alloc_mode: AllocMode,
    diff_battles: usize,
    same_slot_battles: usize,
    term3_battles: usize,
    term3_prob: f64,
    term3_pps: Option<f64>,
    skip_term3: bool,
    decompose_adapt: bool,
    verify_zero_term3: bool,
    preview_iters: Option<u32>,
    battle_iters: Option<u32>,
    battle_mode: BattleMode,
    blind_c: Option<f64>,
    seed: u64,
    threads: usize,
    out_jsonl: Option<PathBuf>,
}

fn usage(bin: &str) -> String {
    format!(
        "usage: {bin} --pools FILE [--pools FILE...] --base ID --cand ID [options]\n\
         options:\n\
           --base-mix MIX          Disclosed mixture name for base arm (default: singleton {{base}})\n\
           --cand-mix MIX          Disclosed mixture name for cand arm (default: singleton {{cand}})\n\
           --shared-opp-belief     Use identical shared opp belief {{base:0.5, cand:0.5}} on both arms\n\
           --opp-belief-mix MIX    Use explicit shared opp belief mixture on both arms\n\
           --opps SPEC             Comma-separated opp IDs/mixtures, @file, or 'all-shells'/'all-teams'\n\
           --clusters K            Independent preview-seed clusters per (opp, seat) [default: 2]\n\
           --preview-reps R        Independent preview policies averaged per cluster [default: 1]\n\
           --alloc MODE            neyman (default) | single-draw | stratified\n\
           --diff-battles N        Battle pairs per Stratum 1 draw [default: 1]\n\
           --same-slot-battles N   Battle pairs per Stratum 2 draw [default: 1]\n\
           --term3-battles N       Battle pairs per Stratum 3b draw [default: 1]\n\
           --term3-prob P          (stratified) Constant HT inclusion prob in (0, 1] [default: 1.0]\n\
           --term3-pps SCALE       (stratified) PPS inclusion prob h3 = clamp(SCALE * m3, 1e-4, 1)\n\
           --skip-term3            Screening mode: skip Stratum 3b (sets --term3-battles 0)\n\
           --decompose-adapt       Also estimate Delta_fight vs Delta_adapt (opp preview shift)\n\
           --verify-zero-term3     Run live bit-exact actionSha256 checks on exact-zero Term-3 picks\n\
           --preview-iters N       Turn-0 preview iterations [default: live-search.json unknownPreview]\n\
           --battle-iters N        Turn 1+ battle iterations/cap [default: live-search.json battle.adaptive]\n\
           --battle-mode MODE      adaptive (default) | fixed\n\
           --c FLOAT               Blind search exploration constant [default: search-profiles.json]\n\
           --seed U64              Master RNG seed [default: 20261124]\n\
           --threads N             Worker threads [default: available_parallelism]\n\
           --out FILE.jsonl        Optional per-cluster telemetry JSONL path"
    )
}

fn parse_cli(args: &[String]) -> Result<CliConfig, String> {
    let mut cfg = CliConfig {
        pool_files: Vec::new(),
        base_id: String::new(),
        cand_id: String::new(),
        base_mix: None,
        cand_mix: None,
        shared_opp_belief: false,
        opp_belief_mix: None,
        opps_spec: None,
        clusters: 2,
        preview_reps: 1,
        alloc_mode: AllocMode::Neyman,
        diff_battles: 1,
        same_slot_battles: 1,
        term3_battles: 1,
        term3_prob: 1.0,
        term3_pps: None,
        skip_term3: false,
        decompose_adapt: false,
        verify_zero_term3: false,
        preview_iters: None,
        battle_iters: None,
        battle_mode: BattleMode::Adaptive,
        blind_c: None,
        seed: 20261124,
        threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8),
        out_jsonl: None,
    };

    let mut i = 1usize;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: &mut usize, name: &str| -> Result<String, String> {
            *idx += 1;
            args.get(*idx)
                .cloned()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match flag {
            "--help" | "-h" => return Err(usage(&args[0])),
            "--pools" | "--catalog" => cfg.pool_files.push(PathBuf::from(next_val(&mut i, flag)?)),
            "--base" => cfg.base_id = next_val(&mut i, flag)?,
            "--cand" => cfg.cand_id = next_val(&mut i, flag)?,
            "--base-mix" => cfg.base_mix = Some(next_val(&mut i, flag)?),
            "--cand-mix" => cfg.cand_mix = Some(next_val(&mut i, flag)?),
            "--shared-opp-belief" => cfg.shared_opp_belief = true,
            "--opp-belief-mix" => cfg.opp_belief_mix = Some(next_val(&mut i, flag)?),
            "--opps" => cfg.opps_spec = Some(next_val(&mut i, flag)?),
            "--clusters" => {
                cfg.clusters = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --clusters")?;
            }
            "--preview-reps" => {
                cfg.preview_reps = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --preview-reps")?;
            }
            "--alloc" => {
                let v = next_val(&mut i, flag)?;
                cfg.alloc_mode = match v.as_str() {
                    "neyman" => AllocMode::Neyman,
                    "single-draw" => AllocMode::SingleDraw,
                    "stratified" => AllocMode::Stratified,
                    _ => return Err(format!("invalid --alloc: {v}")),
                };
            }
            "--diff-battles" => {
                cfg.diff_battles = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --diff-battles")?;
            }
            "--same-slot-battles" => {
                cfg.same_slot_battles = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --same-slot-battles")?;
            }
            "--term3-battles" => {
                cfg.term3_battles = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --term3-battles")?;
            }
            "--term3-prob" => {
                cfg.term3_prob = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --term3-prob")?;
            }
            "--term3-pps" => {
                let v: f64 = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --term3-pps")?;
                cfg.term3_pps = Some(v);
            }
            "--skip-term3" => {
                cfg.skip_term3 = true;
                cfg.term3_battles = 0;
            }
            "--decompose-adapt" => cfg.decompose_adapt = true,
            "--verify-zero-term3" => cfg.verify_zero_term3 = true,
            "--preview-iters" => {
                cfg.preview_iters = Some(
                    next_val(&mut i, flag)?
                        .parse()
                        .map_err(|_| "invalid --preview-iters")?,
                );
            }
            "--battle-iters" => {
                cfg.battle_iters = Some(
                    next_val(&mut i, flag)?
                        .parse()
                        .map_err(|_| "invalid --battle-iters")?,
                );
            }
            "--battle-mode" => {
                let v = next_val(&mut i, flag)?;
                cfg.battle_mode = match v.as_str() {
                    "adaptive" => BattleMode::Adaptive,
                    "fixed" => BattleMode::Fixed,
                    _ => return Err(format!("invalid --battle-mode: {v}")),
                };
            }
            "--c" => {
                cfg.blind_c = Some(
                    next_val(&mut i, flag)?
                        .parse()
                        .map_err(|_| "invalid --c")?,
                );
            }
            "--seed" => {
                cfg.seed = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --seed")?;
            }
            "--threads" => {
                cfg.threads = next_val(&mut i, flag)?
                    .parse()
                    .map_err(|_| "invalid --threads")?;
            }
            "--out" => cfg.out_jsonl = Some(PathBuf::from(next_val(&mut i, flag)?)),
            other => return Err(format!("unknown argument: {other}\n{}", usage(&args[0]))),
        }
        i += 1;
    }

    if cfg.base_id.is_empty() || cfg.cand_id.is_empty() {
        return Err(format!("--base and --cand are required\n{}", usage(&args[0])));
    }
    if cfg.clusters == 0 {
        return Err("--clusters must be >= 1".to_string());
    }
    if cfg.preview_reps == 0 {
        return Err("--preview-reps must be >= 1".to_string());
    }
    if cfg.diff_battles == 0 || cfg.same_slot_battles == 0 {
        return Err("--diff-battles and --same-slot-battles must be >= 1".to_string());
    }
    if cfg.term3_battles == 0 {
        cfg.skip_term3 = true;
    }
    if !cfg.term3_prob.is_finite() || cfg.term3_prob <= 0.0 || cfg.term3_prob > 1.0 {
        return Err("--term3-prob must be in (0.0, 1.0]".to_string());
    }
    if let Some(pps) = cfg.term3_pps {
        if !pps.is_finite() || pps <= 0.0 {
            return Err("--term3-pps must be > 0.0".to_string());
        }
    }
    if cfg.threads == 0 {
        return Err("--threads must be >= 1".to_string());
    }
    Ok(cfg)
}

fn canonical_no_spaces(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn canonical_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn parse_team_pick(input: &str) -> [u8; 3] {
    let digits: Vec<u8> = input
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| (c as u8) - b'1')
        .collect();
    assert_eq!(
        digits.len(),
        3,
        "expected 3-slot team preview pick (e.g. 'team 123'), got {input:?}"
    );
    [digits[0], digits[1], digits[2]]
}

fn derive_seed(domain: &str, master: u64, key: &str, a: u64, b: u64, c: u64) -> u64 {
    let mut h = Sha256::new();
    h.update(b"nc2000-cf-party-ab-v1\0");
    h.update(domain.as_bytes());
    h.update([0u8]);
    h.update(master.to_le_bytes());
    h.update((key.len() as u64).to_le_bytes());
    h.update(key.as_bytes());
    h.update(a.to_le_bytes());
    h.update(b.to_le_bytes());
    h.update(c.to_le_bytes());
    let d = h.finalize();
    u64::from_le_bytes(d[0..8].try_into().unwrap())
}

fn derive_seed_u32(domain: &str, master: u64, key: &str, a: u64, b: u64, c: u64) -> u32 {
    let s = derive_seed(domain, master, key, a, b, c);
    let v = (s ^ (s >> 32)) as u32;
    if v == 0 { 1 } else { v }
}

fn team_party_order_matters(sets: &[PokemonSet]) -> bool {
    sets.iter().any(|s| {
        s.moves
            .iter()
            .any(|m| PARTY_ORDER_MOVES.contains(&canonical_id(m).as_str()))
    })
}

fn has_deterministic_gender(dex: &Dex, set: &PokemonSet) -> bool {
    if matches!(set.gender.as_deref(), Some("M" | "F" | "N")) {
        return true;
    }
    let sp_id = canonical_id(&set.species);
    if let Some(id) = dex.species.id(&sp_id) {
        let sp = dex.species.get(id);
        return matches!(sp.gender.as_deref(), Some("M" | "F" | "N"));
    }
    false
}

fn same_public_slot(a: &PokemonSet, b: &PokemonSet) -> bool {
    canonical_id(&a.species) == canonical_id(&b.species)
        && a.level == b.level
        && a.gender == b.gender
        && (!a.item.trim().is_empty()) == (!b.item.trim().is_empty())
}

fn metapool_eq(a: &MetaPool, b: &MetaPool) -> bool {
    if a.teams.len() != b.teams.len() {
        return false;
    }
    a.teams.iter().zip(&b.teams).all(|(x, y)| {
        x.id == y.id
            && x.weight.map(f64::to_bits) == y.weight.map(f64::to_bits)
            && serde_json::to_string(&x.sets).unwrap() == serde_json::to_string(&y.sets).unwrap()
    })
}

fn species_immunity_mask(dex: &Dex, set: &PokemonSet) -> u32 {
    let dt = nc2000_engine::compact::dex_tables(dex);
    let sp_id = dex
        .species
        .id(&canonical_id(&set.species))
        .unwrap_or_else(|| panic!("unknown species {:?}", set.species));
    let types = dex.species_types(sp_id);
    let has_flying = types.iter().any(|t| t == dex.known_types.flying);
    let mut mask = 0u32;
    for mt in 0..20usize {
        let immune = if mt == dt.ground_type as usize {
            has_flying
        } else {
            let row = &dt.type_chart[mt];
            types.iter().any(|t| row[(t.0 as usize).min(19)] == -9)
        };
        if immune {
            mask |= 1u32 << mt;
        }
    }
    mask
}

fn sets_json_eq(a: &[PokemonSet], b: &[PokemonSet]) -> bool {
    a.len() == b.len()
        && serde_json::to_string(a).unwrap() == serde_json::to_string(b).unwrap()
}

#[derive(Clone, Debug)]
struct ArmComparison {
    diff_slots: Vec<bool>,
    same_public_turn0_frame: bool,
    same_levels_all_slots: bool,
    same_party_order_matters: bool,
    opp_belief_identical: bool,
    opp_belief_singleton_arms: bool,
    base_levels: Vec<u32>,
    base_immune_masks: Vec<u32>,
    cand_immune_masks: Vec<u32>,
}

impl ArmComparison {
    fn build(
        dex: &Dex,
        base_sets: &[PokemonSet],
        cand_sets: &[PokemonSet],
        opp_prior_0: &MetaPool,
        opp_prior_1: &MetaPool,
    ) -> Self {
        assert_eq!(
            base_sets.len(),
            cand_sets.len(),
            "base and cand rosters must have equal length"
        );
        let diff_slots: Vec<bool> = base_sets
            .iter()
            .zip(cand_sets)
            .map(|(a, b)| {
                serde_json::to_string(a).unwrap() != serde_json::to_string(b).unwrap()
            })
            .collect();
        let same_public_turn0_frame = base_sets
            .iter()
            .zip(cand_sets)
            .all(|(a, b)| same_public_slot(a, b));
        let same_levels_all_slots = base_sets
            .iter()
            .zip(cand_sets)
            .all(|(a, b)| a.level == b.level);
        let same_party_order_matters =
            team_party_order_matters(base_sets) == team_party_order_matters(cand_sets);
        let opp_belief_identical = metapool_eq(opp_prior_0, opp_prior_1);
        // Require exact singleton belief alignment with `base_sets`/`cand_sets` so
        // opponent's `Belief` never falls back to `Bayesian` prior, agrees on all
        // unmutated slots, and matches `party_order_matters`.
        let opp_belief_singleton_arms = opp_prior_0.teams.len() == 1
            && opp_prior_1.teams.len() == 1
            && sets_json_eq(&opp_prior_0.teams[0].sets, base_sets)
            && sets_json_eq(&opp_prior_1.teams[0].sets, cand_sets);
        let base_levels = base_sets.iter().map(|s| u32::from(s.level)).collect();
        let base_immune_masks = base_sets
            .iter()
            .map(|s| species_immunity_mask(dex, s))
            .collect();
        let cand_immune_masks = cand_sets
            .iter()
            .map(|s| species_immunity_mask(dex, s))
            .collect();

        Self {
            diff_slots,
            same_public_turn0_frame,
            same_levels_all_slots,
            same_party_order_matters,
            opp_belief_identical,
            opp_belief_singleton_arms,
            base_levels,
            base_immune_masks,
            cand_immune_masks,
        }
    }

    fn contains_diff_slot(&self, pick: &[u8; 3]) -> bool {
        pick.iter()
            .any(|&slot| self.diff_slots.get(slot as usize).copied().unwrap_or(false))
    }

    /// Conservative, mathematically sound Zero-Term-3 certificate for an overlapping
    /// Turn-0 pick `a` containing NO mutated slot (`!contains_diff_slot(a)`).
    ///
    /// Returns `true` iff Turn 1+ Battle + both `ProtocolAgent`s are guaranteed bit-exact
    /// identical across arms `T0` and `T1` on the same `(battle_seed, s_me, s_opp)`:
    ///   - Condition Z1: Identical public Turn-0 frame on all 6 slots AND identical
    ///     opponent belief pool on both arms (`opp_prior_0 == opp_prior_1`), with
    ///     identical `party_order_matters` and explicit/deterministic genders.
    ///   - Condition Z2 (Exact Level-Cap Exclusion, e.g. Lv55-lead skip):
    ///     Identical levels across all 6 slots, exact disclosed singleton arm beliefs
    ///     (`{T0: 1}` == `base_sets` and `{T1: 1}` == `cand_sets`), identical
    ///     `party_order_matters`, the Turn-1 revealed lead `a[0]` strictly excludes
    ///     EVERY mutated slot `s` under the 155-level NC2000 cap
    ///     (`level[a[0]] + level[s] + min_other_level > 155`), AND `mask.rs`
    ///     `foe_switchin_candidates` cannot flip any move-type immunity check
    ///     (for every move type whose immunity differs on `s`, at least one unpicked
    ///     non-mutated slot `u` is non-immune).
    fn is_exact_zero_term3(&self, pick: &[u8; 3]) -> bool {
        if !self.same_party_order_matters || self.contains_diff_slot(pick) {
            return false;
        }
        if self.same_public_turn0_frame && self.opp_belief_identical {
            return true;
        }
        if self.same_levels_all_slots && self.opp_belief_singleton_arms {
            let lead = pick[0] as usize;
            let lead_lv = self.base_levels[lead];
            let mut has_diff = false;
            let mut flipped_immunity_mask = 0u32;
            for (s, &is_diff) in self.diff_slots.iter().enumerate() {
                if !is_diff {
                    continue;
                }
                has_diff = true;
                let min_third = self
                    .base_levels
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != lead && j != s)
                    .map(|(_, &lv)| lv)
                    .min()
                    .unwrap_or(0);
                if lead_lv + self.base_levels[s] + min_third <= NC2000_LEVEL_CAP {
                    return false;
                }
                flipped_immunity_mask |= self.base_immune_masks[s] ^ self.cand_immune_masks[s];
            }
            if !has_diff {
                return false;
            }
            if flipped_immunity_mask != 0 {
                // Guard `mask.rs::foe_switchin_candidates`, which admits every
                // never-appeared roster mon (without checking `preview_level_caps`)
                // while `!all_picks_revealed`. Every unpicked non-mutated slot `u`
                // can never appear or faint, so it is always present alongside `s`.
                // If at least one such `u` is non-immune to every flipped move type,
                // `all(|cand| type_immune_to(mt, cand))` is identically false on both arms.
                let mut unpicked_common_immune = u32::MAX;
                let mut unpicked_non_diff = 0usize;
                for u in 0..self.diff_slots.len() {
                    if !self.diff_slots[u] && !pick.contains(&(u as u8)) {
                        unpicked_common_immune &= self.base_immune_masks[u];
                        unpicked_non_diff += 1;
                    }
                }
                if unpicked_non_diff == 0 || (flipped_immunity_mask & unpicked_common_immune) != 0 {
                    return false;
                }
            }
            return true;
        }
        false
    }
}

#[derive(Clone, Debug)]
struct WeightedAction {
    input: String,
    slots: [u8; 3],
    prob: f64,
}

fn extract_weighted_actions(pol: &DecisionPolicy<String>) -> Vec<WeightedAction> {
    pol.actions()
        .iter()
        .map(|a| WeightedAction {
            input: a.input.clone(),
            slots: parse_team_pick(&a.input),
            prob: a.probability,
        })
        .collect()
}

fn marginal_tv(a0: &[WeightedAction], a1: &[WeightedAction]) -> f64 {
    let mut map: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for x in a0 {
        map.entry(x.input.as_str()).or_insert((0.0, 0.0)).0 += x.prob;
    }
    for x in a1 {
        map.entry(x.input.as_str()).or_insert((0.0, 0.0)).1 += x.prob;
    }
    0.5 * map.values().map(|(p0, p1)| (p1 - p0).abs()).sum::<f64>()
}

#[derive(Clone, Debug)]
struct WeightedPair {
    a_input: String,
    a_slots: [u8; 3],
    b_input: String,
    b_slots: [u8; 3],
    weight: f64,
}

#[derive(Clone, Debug)]
struct CouplingPlan {
    tau: f64,
    m2_same_slot: f64,
    m3_exact_zero: f64,
    m3_belief: f64,
    tv_own: f64,
    tv_opp: f64,
    nu1: Vec<WeightedPair>,
    nu0: Vec<WeightedPair>,
    stratum2: Vec<WeightedPair>,
    stratum3_zero: Vec<WeightedPair>,
    stratum3_belief: Vec<WeightedPair>,
}

fn build_coupling_plan(
    p0: &[WeightedAction],
    q0: &[WeightedAction],
    p1: &[WeightedAction],
    q1: &[WeightedAction],
    cmp: &ArmComparison,
) -> CouplingPlan {
    let tv_own = marginal_tv(p0, p1);
    let tv_opp = marginal_tv(q0, q1);

    let mut pi: BTreeMap<(String, String), ([u8; 3], [u8; 3], f64, f64)> = BTreeMap::new();
    for a in p0 {
        for b in q0 {
            let entry = pi
                .entry((a.input.clone(), b.input.clone()))
                .or_insert((a.slots, b.slots, 0.0, 0.0));
            entry.2 += a.prob * b.prob;
        }
    }
    for a in p1 {
        for b in q1 {
            let entry = pi
                .entry((a.input.clone(), b.input.clone()))
                .or_insert((a.slots, b.slots, 0.0, 0.0));
            entry.3 += a.prob * b.prob;
        }
    }

    let mut r1 = Vec::new();
    let mut r0 = Vec::new();
    let mut stratum2 = Vec::new();
    let mut stratum3_zero = Vec::new();
    let mut stratum3_belief = Vec::new();

    let mut sum_r1 = 0.0f64;
    let mut sum_r0 = 0.0f64;
    let mut m2 = 0.0f64;
    let mut m3_zero = 0.0f64;
    let mut m3_belief = 0.0f64;

    const EPS: f64 = 1e-15;
    for ((a_in, b_in), (a_sl, b_sl, w0, w1)) in pi {
        let ov = w0.min(w1);
        if ov > EPS {
            let item = WeightedPair {
                a_input: a_in.clone(),
                a_slots: a_sl,
                b_input: b_in.clone(),
                b_slots: b_sl,
                weight: ov,
            };
            if cmp.contains_diff_slot(&a_sl) {
                m2 += ov;
                stratum2.push(item);
            } else if cmp.is_exact_zero_term3(&a_sl) {
                m3_zero += ov;
                stratum3_zero.push(item);
            } else {
                m3_belief += ov;
                stratum3_belief.push(item);
            }
        }
        let d1 = (w1 - w0).max(0.0);
        if d1 > EPS {
            sum_r1 += d1;
            r1.push(WeightedPair {
                a_input: a_in.clone(),
                a_slots: a_sl,
                b_input: b_in.clone(),
                b_slots: b_sl,
                weight: d1,
            });
        }
        let d0 = (w0 - w1).max(0.0);
        if d0 > EPS {
            sum_r0 += d0;
            r0.push(WeightedPair {
                a_input: a_in,
                a_slots: a_sl,
                b_input: b_in,
                b_slots: b_sl,
                weight: d0,
            });
        }
    }

    let tau = if sum_r1 > 1e-12 && sum_r0 > 1e-12 {
        assert!(
            (sum_r1 - sum_r0).abs() < 1e-9,
            "residual mass mismatch: sum_r1={sum_r1} sum_r0={sum_r0}"
        );
        for x in &mut r1 {
            x.weight /= sum_r1;
        }
        for x in &mut r0 {
            x.weight /= sum_r0;
        }
        0.5 * (sum_r1 + sum_r0)
    } else {
        r1.clear();
        r0.clear();
        0.0
    };

    if m2 > 0.0 {
        for x in &mut stratum2 {
            x.weight /= m2;
        }
    }
    if m3_zero > 0.0 {
        for x in &mut stratum3_zero {
            x.weight /= m3_zero;
        }
    }
    if m3_belief > 0.0 {
        for x in &mut stratum3_belief {
            x.weight /= m3_belief;
        }
    }

    CouplingPlan {
        tau,
        m2_same_slot: m2,
        m3_exact_zero: m3_zero,
        m3_belief,
        tv_own,
        tv_opp,
        nu1: r1,
        nu0: r0,
        stratum2,
        stratum3_zero,
        stratum3_belief,
    }
}

fn sample_weighted<T: Clone>(items: &[(T, f64)], rng: &mut SplitMix64) -> T {
    debug_assert!(!items.is_empty());
    let total: f64 = items.iter().map(|(_, w)| *w).sum();
    let u = rng.next_f64() * total;
    let mut acc = 0.0;
    for (item, w) in items {
        acc += *w;
        if u < acc {
            return item.clone();
        }
    }
    items.last().unwrap().0.clone()
}

fn sample_pair(items: &[WeightedPair], rng: &mut SplitMix64) -> (String, String) {
    debug_assert!(!items.is_empty());
    let total: f64 = items.iter().map(|p| p.weight).sum();
    let u = rng.next_f64() * total;
    let mut acc = 0.0;
    for p in items {
        acc += p.weight;
        if u < acc {
            return (p.a_input.clone(), p.b_input.clone());
        }
    }
    let last = items.last().unwrap();
    (last.a_input.clone(), last.b_input.clone())
}

/// Exact marginal-preserving maximal coupling of two discrete action distributions
/// `(c1, c0)` over team picks:
///   1. Maximally couples exact action equality (`a1 == a0`), then
///   2. On the residual, maximally couples lead equality (`a1[0] == a0[0]`).
/// Exact marginals: `P(A1 = a) = c1(a)` and `P(A0 = a) = c0(a)`.
fn couple_picks_exact_then_lead(
    c1: &[(String, [u8; 3], f64)],
    c0: &[(String, [u8; 3], f64)],
    rng: &mut SplitMix64,
) -> (String, String) {
    let mut map: BTreeMap<String, ([u8; 3], f64, f64)> = BTreeMap::new();
    let s1: f64 = c1.iter().map(|x| x.2).sum();
    let s0: f64 = c0.iter().map(|x| x.2).sum();
    for (inp, sl, w) in c1 {
        let e = map.entry(inp.clone()).or_insert((*sl, 0.0, 0.0));
        e.1 += *w / s1;
    }
    for (inp, sl, w) in c0 {
        let e = map.entry(inp.clone()).or_insert((*sl, 0.0, 0.0));
        e.2 += *w / s0;
    }

    let mut exact_ov: Vec<(String, f64)> = Vec::new();
    let mut res1: Vec<(String, [u8; 3], f64)> = Vec::new();
    let mut res0: Vec<(String, [u8; 3], f64)> = Vec::new();
    let mut m_exact = 0.0f64;
    let mut m_res1 = 0.0f64;
    let mut m_res0 = 0.0f64;

    for (inp, (sl, w1, w0)) in map {
        let ov = w1.min(w0);
        if ov > 1e-15 {
            m_exact += ov;
            exact_ov.push((inp.clone(), ov));
        }
        let d1 = (w1 - w0).max(0.0);
        if d1 > 1e-15 {
            m_res1 += d1;
            res1.push((inp.clone(), sl, d1));
        }
        let d0 = (w0 - w1).max(0.0);
        if d0 > 1e-15 {
            m_res0 += d0;
            res0.push((inp, sl, d0));
        }
    }

    if m_exact > 1e-15 && (m_res1 <= 1e-12 || m_res0 <= 1e-12 || rng.next_f64() < m_exact) {
        let shared = sample_weighted(&exact_ov, rng);
        return (shared.clone(), shared);
    }

    // Residual lead coupling over normalized `res1` and `res0`.
    let mut lead_map: BTreeMap<u8, (f64, f64)> = BTreeMap::new();
    for (_, sl, w) in &res1 {
        lead_map.entry(sl[0]).or_insert((0.0, 0.0)).0 += *w / m_res1;
    }
    for (_, sl, w) in &res0 {
        lead_map.entry(sl[0]).or_insert((0.0, 0.0)).1 += *w / m_res0;
    }

    let mut lead_ov: Vec<(u8, f64)> = Vec::new();
    let mut lead_r1: Vec<(u8, f64)> = Vec::new();
    let mut lead_r0: Vec<(u8, f64)> = Vec::new();
    let mut m_lead = 0.0f64;
    let mut lr1_sum = 0.0f64;
    let mut lr0_sum = 0.0f64;

    for (lead, (w1, w0)) in lead_map {
        let ov = w1.min(w0);
        if ov > 1e-15 {
            m_lead += ov;
            lead_ov.push((lead, ov));
        }
        let d1 = (w1 - w0).max(0.0);
        if d1 > 1e-15 {
            lr1_sum += d1;
            lead_r1.push((lead, d1));
        }
        let d0 = (w0 - w1).max(0.0);
        if d0 > 1e-15 {
            lr0_sum += d0;
            lead_r0.push((lead, d0));
        }
    }

    let (lead1, lead0) =
        if m_lead > 1e-15 && (lr1_sum <= 1e-12 || lr0_sum <= 1e-12 || rng.next_f64() < m_lead) {
            let l = sample_weighted(&lead_ov, rng);
            (l, l)
        } else {
            (
                sample_weighted(&lead_r1, rng),
                sample_weighted(&lead_r0, rng),
            )
        };

    let c1_lead: Vec<(String, f64)> = res1
        .iter()
        .filter(|(_, sl, _)| sl[0] == lead1)
        .map(|(inp, _, w)| (inp.clone(), *w))
        .collect();
    let c0_lead: Vec<(String, f64)> = res0
        .iter()
        .filter(|(_, sl, _)| sl[0] == lead0)
        .map(|(inp, _, w)| (inp.clone(), *w))
        .collect();
    (
        sample_weighted(&c1_lead, rng),
        sample_weighted(&c0_lead, rng),
    )
}

/// Secondary maximal B-coupling + tertiary exact-A/lead coupling for Stratum 1 `(nu1, nu0)`.
/// Returns `((a1, b1), (a0, b0), b_coupled)` with EXACT marginals `(A1, B1) ~ nu1`
/// and `(A0, B0) ~ nu0`.
fn sample_stratum1_coupled(
    nu1: &[WeightedPair],
    nu0: &[WeightedPair],
    rng: &mut SplitMix64,
) -> ((String, String), (String, String), bool) {
    let mut b_map: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for p in nu1 {
        b_map.entry(p.b_input.clone()).or_insert((0.0, 0.0)).0 += p.weight;
    }
    for p in nu0 {
        b_map.entry(p.b_input.clone()).or_insert((0.0, 0.0)).1 += p.weight;
    }

    let mut b_ov: Vec<(String, f64)> = Vec::new();
    let mut b_r1: Vec<(String, f64)> = Vec::new();
    let mut b_r0: Vec<(String, f64)> = Vec::new();
    let mut gamma = 0.0f64;
    let mut br1_sum = 0.0f64;
    let mut br0_sum = 0.0f64;

    for (b_in, (g1, g0)) in b_map {
        let ov = g1.min(g0);
        if ov > 1e-15 {
            gamma += ov;
            b_ov.push((b_in.clone(), ov));
        }
        let d1 = (g1 - g0).max(0.0);
        if d1 > 1e-15 {
            br1_sum += d1;
            b_r1.push((b_in.clone(), d1));
        }
        let d0 = (g0 - g1).max(0.0);
        if d0 > 1e-15 {
            br0_sum += d0;
            b_r0.push((b_in, d0));
        }
    }

    let (b1, b0, b_coupled) =
        if gamma > 1e-15 && (br1_sum <= 1e-12 || br0_sum <= 1e-12 || rng.next_f64() < gamma) {
            let b = sample_weighted(&b_ov, rng);
            (b.clone(), b, true)
        } else {
            (
                sample_weighted(&b_r1, rng),
                sample_weighted(&b_r0, rng),
                false,
            )
        };

    let cond1: Vec<(String, [u8; 3], f64)> = nu1
        .iter()
        .filter(|p| p.b_input == b1)
        .map(|p| (p.a_input.clone(), p.a_slots, p.weight))
        .collect();
    let cond0: Vec<(String, [u8; 3], f64)> = nu0
        .iter()
        .filter(|p| p.b_input == b0)
        .map(|p| (p.a_input.clone(), p.a_slots, p.weight))
        .collect();
    let (a1, a0) = couple_picks_exact_then_lead(&cond1, &cond0, rng);
    ((a1, b1), (a0, b0), b_coupled)
}

/// Exact `tau_q` decomposition for opponent adaptation `q0 -> q1`:
/// Returns `(tau_q, eta1, eta0)` where `q1 - q0 = tau_q * (eta1 - eta0)`.
fn build_opp_shift_residuals(
    q0: &[WeightedAction],
    q1: &[WeightedAction],
) -> (f64, Vec<(String, [u8; 3], f64)>, Vec<(String, [u8; 3], f64)>) {
    let mut map: BTreeMap<String, ([u8; 3], f64, f64)> = BTreeMap::new();
    for x in q0 {
        let e = map.entry(x.input.clone()).or_insert((x.slots, 0.0, 0.0));
        e.1 += x.prob;
    }
    for x in q1 {
        let e = map.entry(x.input.clone()).or_insert((x.slots, 0.0, 0.0));
        e.2 += x.prob;
    }
    let mut eta1 = Vec::new();
    let mut eta0 = Vec::new();
    let mut s1 = 0.0f64;
    let mut s0 = 0.0f64;
    for (inp, (sl, w0, w1)) in map {
        let d1 = (w1 - w0).max(0.0);
        if d1 > 1e-15 {
            s1 += d1;
            eta1.push((inp.clone(), sl, d1));
        }
        let d0 = (w0 - w1).max(0.0);
        if d0 > 1e-15 {
            s0 += d0;
            eta0.push((inp, sl, d0));
        }
    }
    if s1 <= 1e-12 || s0 <= 1e-12 {
        return (0.0, Vec::new(), Vec::new());
    }
    for x in &mut eta1 {
        x.2 /= s1;
    }
    for x in &mut eta0 {
        x.2 /= s0;
    }
    (0.5 * (s1 + s0), eta1, eta0)
}

fn solve_preview_policy(
    dex: &Dex,
    side: usize,
    sets_p0: &[PokemonSet],
    sets_p1: &[PokemonSet],
    prior: MetaPool,
    cfg: &RmConfig,
    preview_iters: u32,
    omega_seed: u64,
) -> (DecisionPolicy<String>, f64) {
    let t0 = Instant::now();
    let mut battle =
        Battle::from_fixture(dex, "1,2,3,4", sets_p0, sets_p1).expect("Turn-0 Battle::from_fixture");
    let mut ch = PlayerChannel::new(side);
    let frame = ch
        .frame(&mut battle, dex)
        .expect("Turn-0 PlayerChannel::frame");
    let req_json = serde_json::to_string(&frame.request).unwrap();
    let mut agent = ProtocolAgent::new(dex, side, prior, cfg.clone(), omega_seed);
    let own_sets = if side == 0 {
        sets_p0.to_vec()
    } else {
        sets_p1.to_vec()
    };
    agent.set_own_team(own_sets);
    for line in &frame.lines {
        agent.push_line(dex, line);
    }
    let owed = agent
        .on_request(dex, &req_json)
        .expect("Turn-0 preview on_request");
    assert!(owed, "expected pending Turn-0 preview decision");
    agent
        .step_adaptive_battle(dex, preview_iters)
        .expect("Turn-0 preview step_adaptive_battle");
    let pol = agent
        .preview_execution_policy(dex)
        .expect("preview_execution_policy");
    (pol, t0.elapsed().as_secs_f64() * 1000.0)
}

struct BattleOutcome {
    score_for_me: f64,
    turns: u16,
    action_sha256: String,
    elapsed_ms: f64,
}

/// Run a single Turn 1+ battle with forced Turn-0 picks `(pick_0, pick_1)`,
/// faithfully feeding Turn-0 `PlayerChannel::frame` + `ProtocolAgent::on_request`
/// without stepping Turn-0 search so `Observer.revision`, Turn-0 `Belief`, and
/// `ProtocolAgent.rng` match live `nash_refresh.rs` bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn run_forced_pick_battle(
    dex: &Dex,
    sets: [&[PokemonSet]; 2],
    priors: [&MetaPool; 2],
    picks: [&str; 2],
    me_side: usize,
    cfg: &RmConfig,
    battle_iters: u32,
    battle_mode: BattleMode,
    battle_seed: u32,
    agent_seeds: [u32; 2],
) -> BattleOutcome {
    let t0 = Instant::now();
    let battle_seed_str = SplitMix64::new(battle_seed as u64).battle_seed();
    let mut battle = Battle::from_fixture(dex, &battle_seed_str, sets[0], sets[1])
        .expect("Battle::from_fixture");
    let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
    let mut agents: [ProtocolAgent; 2] = std::array::from_fn(|side| {
        let mut agent =
            ProtocolAgent::new(dex, side, priors[side].clone(), cfg.clone(), agent_seeds[side] as u64);
        agent.set_own_team(sets[side].to_vec());
        agent
    });
    let mut decisions: [DecisionSampler<String>; 2] =
        std::array::from_fn(|side| DecisionSampler::new(agent_seeds[side] as u64));

    let mut transcript = Sha256::new();
    let mut first = true;

    while battle.outcome().is_none() && battle.turn < 500 {
        let needs = battle.needs_choice();
        let mut inputs: [Option<String>; 2] = [None, None];
        for side in 0..2 {
            if !needs[side] {
                continue;
            }
            let frame = channels[side]
                .frame(&mut battle, dex)
                .expect("PlayerChannel::frame");
            let agent = &mut agents[side];
            for line in &frame.lines {
                agent.push_line(dex, line);
            }
            decisions[side].reset();
            let req_json = serde_json::to_string(&frame.request).unwrap();
            let owed = agent
                .on_request(dex, &req_json)
                .expect("ProtocolAgent::on_request");
            assert!(owed, "expected pending decision on side {side}");

            let chosen = if first {
                let forced = picks[side];
                let norm = canonical_no_spaces(forced);
                assert!(
                    frame
                        .legal_actions
                        .iter()
                        .any(|a| canonical_no_spaces(a) == norm),
                    "illegal forced pick {forced:?} on side {side}"
                );
                decisions[side]
                    .commit(DecisionPolicy::pure("forced", forced.to_string()))
                    .input
                    .clone()
            } else {
                match battle_mode {
                    BattleMode::Adaptive => {
                        agent
                            .step_adaptive_battle(dex, battle_iters)
                            .expect("step_adaptive_battle");
                    }
                    BattleMode::Fixed => {
                        agent.step(dex, battle_iters).expect("step");
                    }
                }
                let policy = agent
                    .decision_policy(dex, PolicyMode::Legacy)
                    .expect("decision_policy");
                if policy.source() == "request-projected" {
                    agent.projections += 1;
                }
                decisions[side].commit(policy).input.clone()
            };
            inputs[side] = Some(chosen);
        }

        let mut line = serde_json::to_string(&inputs).unwrap();
        line.push('\n');
        transcript.update(line.as_bytes());

        for (side, input) in inputs.iter().enumerate() {
            if let Some(inp) = input {
                battle
                    .choose(dex, side, inp)
                    .unwrap_or_else(|e| panic!("battle.choose(side={side}, {inp:?}): {e:?}"));
            }
        }
        first = false;
    }

    let score_for_me = match battle.outcome() {
        Some(Outcome::P1Win) => {
            if me_side == 0 {
                1.0
            } else {
                0.0
            }
        }
        Some(Outcome::P2Win) => {
            if me_side == 1 {
                1.0
            } else {
                0.0
            }
        }
        Some(Outcome::Tie) | None => 0.5,
    };

    BattleOutcome {
        score_for_me,
        turns: battle.turn,
        action_sha256: format!("{:x}", transcript.finalize()),
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

#[derive(Clone, Serialize)]
struct SeatClusterResult {
    opp_id: String,
    opp_realized_id: String,
    cluster: usize,
    seat: usize,
    delta_u: f64,
    term1_diff: f64,
    term2_same_slot: f64,
    term3_belief: f64,
    delta_adapt: Option<f64>,
    delta_fight: Option<f64>,
    tau: f64,
    m2_same_slot: f64,
    m3_exact_zero: f64,
    m3_belief: f64,
    tv_own: f64,
    tv_opp: f64,
    b_coupled_draws: usize,
    stratum1_draws: usize,
    term3_h3: f64,
    term3_sampled: bool,
    battles_term1: usize,
    battles_term2: usize,
    battles_term3: usize,
    battles_adapt: usize,
    previews_computed: usize,
    previews_skipped_q: usize,
    exact_zero_skipped: bool,
    verify_zero_checked: usize,
    verify_zero_mismatches: usize,
    preview_ms: f64,
    battle_ms: f64,
}

struct OppSpec {
    id: String,
    pool: MetaPool,
    weight: f64,
}

struct SharedRunner {
    dex: Dex,
    base_sets: Vec<PokemonSet>,
    cand_sets: Vec<PokemonSet>,
    opp_prior_0: MetaPool,
    opp_prior_1: MetaPool,
    arm_cmp: ArmComparison,
    opps: Vec<OppSpec>,
    rm_cfg: RmConfig,
    preview_iters: u32,
    preview_reps: usize,
    alloc_mode: AllocMode,
    battle_iters: u32,
    battle_mode: BattleMode,
    diff_battles: usize,
    same_slot_battles: usize,
    term3_battles: usize,
    term3_prob: f64,
    term3_pps: Option<f64>,
    skip_term3: bool,
    decompose_adapt: bool,
    verify_zero_term3: bool,
    master_seed: u64,
}

fn draw_realized_opp<'a>(pool: &'a MetaPool, rng: &mut SplitMix64) -> &'a MetaTeam {
    if pool.teams.len() == 1 {
        return &pool.teams[0];
    }
    let total: f64 = pool
        .teams
        .iter()
        .map(|t| t.weight.unwrap_or(1.0))
        .sum();
    let u = rng.next_f64() * total;
    let mut acc = 0.0;
    for t in &pool.teams {
        acc += t.weight.unwrap_or(1.0);
        if u < acc {
            return t;
        }
    }
    pool.teams.last().unwrap()
}

fn average_weighted_actions(reps: &[Vec<WeightedAction>]) -> Vec<WeightedAction> {
    if reps.len() == 1 {
        return reps[0].clone();
    }
    let mut map: BTreeMap<String, ([u8; 3], f64)> = BTreeMap::new();
    let mut total = 0.0;
    for r in reps {
        for a in r {
            let e = map.entry(a.input.clone()).or_insert((a.slots, 0.0));
            e.1 += a.prob;
            total += a.prob;
        }
    }
    map.into_iter()
        .map(|(input, (slots, w))| WeightedAction {
            input,
            slots,
            prob: w / total,
        })
        .collect()
}

fn run_seat_cluster(shared: &SharedRunner, opp_idx: usize, cluster: usize, seat: usize) -> SeatClusterResult {
    let opp = &shared.opps[opp_idx];
    let opp_side = 1 - seat;

    // Share realized opponent variant across seats (0, 1) within the same cluster
    // so multi-team opponent mixtures are seat-swapped on identical realized rosters.
    let mut opp_real_rng = SplitMix64::new(derive_seed(
        "opp-realized",
        shared.master_seed,
        &opp.id,
        cluster as u64,
        0,
        0,
    ));
    let opp_team = draw_realized_opp(&opp.pool, &mut opp_real_rng);
    let opp_sets = &opp_team.sets;

    let priors_for_arm = |arm: usize| -> [&MetaPool; 2] {
        let opp_prior = if arm == 0 {
            &shared.opp_prior_0
        } else {
            &shared.opp_prior_1
        };
        if seat == 0 {
            [&opp.pool, opp_prior]
        } else {
            [opp_prior, &opp.pool]
        }
    };

    let s0_pair: [&[PokemonSet]; 2] = if seat == 0 {
        [&shared.base_sets, opp_sets]
    } else {
        [opp_sets, &shared.base_sets]
    };
    let s1_pair: [&[PokemonSet]; 2] = if seat == 0 {
        [&shared.cand_sets, opp_sets]
    } else {
        [opp_sets, &shared.cand_sets]
    };

    let mut preview_ms = 0.0;
    let mut previews_computed = 0usize;
    let mut previews_skipped_q = 0usize;

    let mut p0_reps = Vec::with_capacity(shared.preview_reps);
    let mut p1_reps = Vec::with_capacity(shared.preview_reps);
    let mut q0_reps = Vec::with_capacity(shared.preview_reps);
    let mut q1_reps = Vec::with_capacity(shared.preview_reps);

    // Average `preview_reps` independent preview policies (`omega_me ⊥ omega_opp` per rep).
    // Strictly unbiased: `E[p_bar ⊗ q_bar] = E[p_bar] ⊗ E[q_bar] = E[p] ⊗ E[q]`.
    for prep in 0..shared.preview_reps {
        let omega_me = derive_seed(
            "omega-me",
            shared.master_seed,
            &opp.id,
            cluster as u64,
            seat as u64,
            prep as u64,
        );
        let omega_opp = derive_seed(
            "omega-opp",
            shared.master_seed,
            &opp.id,
            cluster as u64,
            seat as u64,
            prep as u64,
        );

        let (pol_p0, ms_p0) = solve_preview_policy(
            &shared.dex,
            seat,
            s0_pair[0],
            s0_pair[1],
            opp.pool.clone(),
            &shared.rm_cfg,
            shared.preview_iters,
            omega_me,
        );
        preview_ms += ms_p0;
        previews_computed += 1;

        let (pol_p1, ms_p1) = solve_preview_policy(
            &shared.dex,
            seat,
            s1_pair[0],
            s1_pair[1],
            opp.pool.clone(),
            &shared.rm_cfg,
            shared.preview_iters,
            omega_me,
        );
        preview_ms += ms_p1;
        previews_computed += 1;

        let (pol_q0, ms_q0) = solve_preview_policy(
            &shared.dex,
            opp_side,
            s0_pair[0],
            s0_pair[1],
            shared.opp_prior_0.clone(),
            &shared.rm_cfg,
            shared.preview_iters,
            omega_opp,
        );
        preview_ms += ms_q0;
        previews_computed += 1;

        let pol_q1 = if shared.arm_cmp.same_public_turn0_frame && shared.arm_cmp.opp_belief_identical {
            previews_skipped_q += 1;
            pol_q0.clone()
        } else {
            let (q1, ms_q1) = solve_preview_policy(
                &shared.dex,
                opp_side,
                s1_pair[0],
                s1_pair[1],
                shared.opp_prior_1.clone(),
                &shared.rm_cfg,
                shared.preview_iters,
                omega_opp,
            );
            preview_ms += ms_q1;
            previews_computed += 1;
            q1
        };

        p0_reps.push(extract_weighted_actions(&pol_p0));
        p1_reps.push(extract_weighted_actions(&pol_p1));
        q0_reps.push(extract_weighted_actions(&pol_q0));
        q1_reps.push(extract_weighted_actions(&pol_q1));
    }

    let p0 = average_weighted_actions(&p0_reps);
    let p1 = average_weighted_actions(&p1_reps);
    let q0 = average_weighted_actions(&q0_reps);
    let q1 = average_weighted_actions(&q1_reps);

    let plan = build_coupling_plan(&p0, &q0, &p1, &q1, &shared.arm_cmp);

    let mut battle_ms = 0.0;
    let mut battles_term1 = 0usize;
    let mut battles_term2 = 0usize;
    let mut battles_term3 = 0usize;
    let mut battles_adapt = 0usize;
    let mut b_coupled_draws = 0usize;
    let mut stratum1_draws = 0usize;

    let run_crn_pair = |stratum_tag: &str,
                        rep: u64,
                        pick_arm1: (&str, &str),
                        pick_arm0: (&str, &str)|
     -> (BattleOutcome, BattleOutcome) {
        let bseed = derive_seed_u32(
            &format!("battle-{stratum_tag}"),
            shared.master_seed,
            &opp.id,
            cluster as u64,
            seat as u64,
            rep,
        );
        let sme = derive_seed_u32(
            &format!("agent-me-{stratum_tag}"),
            shared.master_seed,
            &opp.id,
            cluster as u64,
            seat as u64,
            rep,
        );
        let sopp = derive_seed_u32(
            &format!("agent-opp-{stratum_tag}"),
            shared.master_seed,
            &opp.id,
            cluster as u64,
            seat as u64,
            rep,
        );
        let aseeds = if seat == 0 { [sme, sopp] } else { [sopp, sme] };
        let picks1 = if seat == 0 {
            [pick_arm1.0, pick_arm1.1]
        } else {
            [pick_arm1.1, pick_arm1.0]
        };
        let picks0 = if seat == 0 {
            [pick_arm0.0, pick_arm0.1]
        } else {
            [pick_arm0.1, pick_arm0.0]
        };
        let o1 = run_forced_pick_battle(
            &shared.dex,
            s1_pair,
            priors_for_arm(1),
            picks1,
            seat,
            &shared.rm_cfg,
            shared.battle_iters,
            shared.battle_mode,
            bseed,
            aseeds,
        );
        let o0 = run_forced_pick_battle(
            &shared.dex,
            s0_pair,
            priors_for_arm(0),
            picks0,
            seat,
            &shared.rm_cfg,
            shared.battle_iters,
            shared.battle_mode,
            bseed,
            aseeds,
        );
        (o1, o0)
    };

    // Stratum 3a: exact-zero Term-3 (`m3_exact_zero`). Skipped at 0 cost unless `--verify-zero-term3`.
    let exact_zero_skipped = plan.m3_exact_zero > 0.0;
    let mut verify_zero_checked = 0usize;
    let mut verify_zero_mismatches = 0usize;
    if shared.verify_zero_term3 && plan.m3_exact_zero > 0.0 {
        let mut rng_z = SplitMix64::new(derive_seed(
            "verify-s3z",
            shared.master_seed,
            &opp.id,
            cluster as u64,
            seat as u64,
            0,
        ));
        let (a, b) = sample_pair(&plan.stratum3_zero, &mut rng_z);
        let (o1, o0) = run_crn_pair("s3z-verify", 0, (&a, &b), (&a, &b));
        battle_ms += o1.elapsed_ms + o0.elapsed_ms;
        verify_zero_checked += 1;
        if o1.action_sha256 != o0.action_sha256 || (o1.score_for_me - o0.score_for_me).abs() > 1e-12 {
            verify_zero_mismatches += 1;
        }
    }

    let mut term1_diff = 0.0;
    let mut term2_same_slot = 0.0;
    let mut term3_belief = 0.0;
    let mut term3_sampled = false;
    let term3_h3;

    match shared.alloc_mode {
        AllocMode::Neyman | AllocMode::SingleDraw => {
            let (s1, s2, s3) = if shared.alloc_mode == AllocMode::Neyman {
                (NEYMAN_SCALE_S1, NEYMAN_SCALE_S2, NEYMAN_SCALE_S3)
            } else {
                (1.0, 1.0, 1.0)
            };
            let p1_inc = plan.tau * s1;
            let p2_inc = plan.m2_same_slot * s2;
            let p3_inc = if shared.skip_term3 || shared.term3_battles == 0 {
                0.0
            } else {
                plan.m3_belief * s3
            };
            term3_h3 = p3_inc;
            let mut rng_alloc = SplitMix64::new(derive_seed(
                "alloc-single-draw",
                shared.master_seed,
                &opp.id,
                cluster as u64,
                seat as u64,
                0,
            ));
            let u = rng_alloc.next_f64();
            if p1_inc > 0.0 && u < p1_inc {
                let mut rng1 = SplitMix64::new(derive_seed(
                    "couple-s1",
                    shared.master_seed,
                    &opp.id,
                    cluster as u64,
                    seat as u64,
                    0,
                ));
                let mut sum_d = 0.0;
                for rep in 0..shared.diff_battles {
                    let ((a1, b1), (a0, b0), b_coup) =
                        sample_stratum1_coupled(&plan.nu1, &plan.nu0, &mut rng1);
                    stratum1_draws += 1;
                    if b_coup {
                        b_coupled_draws += 1;
                    }
                    let (o1, o0) = run_crn_pair("s1", rep as u64, (&a1, &b1), (&a0, &b0));
                    battle_ms += o1.elapsed_ms + o0.elapsed_ms;
                    battles_term1 += 2;
                    sum_d += o1.score_for_me - o0.score_for_me;
                }
                term1_diff = (plan.tau / p1_inc) * (sum_d / shared.diff_battles as f64);
            } else if p2_inc > 0.0 && u < p1_inc + p2_inc {
                let mut rng2 = SplitMix64::new(derive_seed(
                    "couple-s2",
                    shared.master_seed,
                    &opp.id,
                    cluster as u64,
                    seat as u64,
                    0,
                ));
                let mut sum_d = 0.0;
                for rep in 0..shared.same_slot_battles {
                    let (a, b) = sample_pair(&plan.stratum2, &mut rng2);
                    let (o1, o0) = run_crn_pair("s2", rep as u64, (&a, &b), (&a, &b));
                    battle_ms += o1.elapsed_ms + o0.elapsed_ms;
                    battles_term2 += 2;
                    sum_d += o1.score_for_me - o0.score_for_me;
                }
                term2_same_slot = (plan.m2_same_slot / p2_inc) * (sum_d / shared.same_slot_battles as f64);
            } else if p3_inc > 0.0 && u < p1_inc + p2_inc + p3_inc {
                term3_sampled = true;
                let mut rng3 = SplitMix64::new(derive_seed(
                    "couple-s3",
                    shared.master_seed,
                    &opp.id,
                    cluster as u64,
                    seat as u64,
                    0,
                ));
                let mut sum_d = 0.0;
                for rep in 0..shared.term3_battles {
                    let (a, b) = sample_pair(&plan.stratum3_belief, &mut rng3);
                    let (o1, o0) = run_crn_pair("s3", rep as u64, (&a, &b), (&a, &b));
                    battle_ms += o1.elapsed_ms + o0.elapsed_ms;
                    battles_term3 += 2;
                    sum_d += o1.score_for_me - o0.score_for_me;
                }
                term3_belief = (plan.m3_belief / p3_inc) * (sum_d / shared.term3_battles as f64);
            }
        }
        AllocMode::Stratified => {
            // Stratum 1: non-overlapping residual difference (`tau`).
            if plan.tau > 0.0 {
                let mut rng1 = SplitMix64::new(derive_seed(
                    "couple-s1",
                    shared.master_seed,
                    &opp.id,
                    cluster as u64,
                    seat as u64,
                    0,
                ));
                let mut sum_d = 0.0;
                for rep in 0..shared.diff_battles {
                    let ((a1, b1), (a0, b0), b_coup) =
                        sample_stratum1_coupled(&plan.nu1, &plan.nu0, &mut rng1);
                    stratum1_draws += 1;
                    if b_coup {
                        b_coupled_draws += 1;
                    }
                    let (o1, o0) = run_crn_pair("s1", rep as u64, (&a1, &b1), (&a0, &b0));
                    battle_ms += o1.elapsed_ms + o0.elapsed_ms;
                    battles_term1 += 2;
                    sum_d += o1.score_for_me - o0.score_for_me;
                }
                term1_diff = plan.tau * (sum_d / shared.diff_battles as f64);
            }

            // Stratum 2: same-index overlap with mutated slot in `a` (`m2_same_slot`).
            if plan.m2_same_slot > 0.0 {
                let mut rng2 = SplitMix64::new(derive_seed(
                    "couple-s2",
                    shared.master_seed,
                    &opp.id,
                    cluster as u64,
                    seat as u64,
                    0,
                ));
                let mut sum_d = 0.0;
                for rep in 0..shared.same_slot_battles {
                    let (a, b) = sample_pair(&plan.stratum2, &mut rng2);
                    let (o1, o0) = run_crn_pair("s2", rep as u64, (&a, &b), (&a, &b));
                    battle_ms += o1.elapsed_ms + o0.elapsed_ms;
                    battles_term2 += 2;
                    sum_d += o1.score_for_me - o0.score_for_me;
                }
                term2_same_slot = plan.m2_same_slot * (sum_d / shared.same_slot_battles as f64);
            }

            // Stratum 3b: non-exact-zero hidden-bench belief stratum (`m3_belief`).
            term3_h3 = if shared.skip_term3 || plan.m3_belief <= 0.0 {
                0.0
            } else if let Some(pps) = shared.term3_pps {
                // Clamp away from `SplitMix64::next_f64()` 2^-53 resolution floor and
                // extreme HT inverse-probability variance spikes.
                (pps * plan.m3_belief).clamp(1e-4, 1.0)
            } else {
                shared.term3_prob
            };

            if !shared.skip_term3 && plan.m3_belief > 0.0 && shared.term3_battles > 0 {
                let mut rng3 = SplitMix64::new(derive_seed(
                    "couple-s3",
                    shared.master_seed,
                    &opp.id,
                    cluster as u64,
                    seat as u64,
                    0,
                ));
                let u3 = rng3.next_f64();
                if u3 < term3_h3 {
                    term3_sampled = true;
                    let mut sum_d = 0.0;
                    for rep in 0..shared.term3_battles {
                        let (a, b) = sample_pair(&plan.stratum3_belief, &mut rng3);
                        let (o1, o0) = run_crn_pair("s3", rep as u64, (&a, &b), (&a, &b));
                        battle_ms += o1.elapsed_ms + o0.elapsed_ms;
                        battles_term3 += 2;
                        sum_d += o1.score_for_me - o0.score_for_me;
                    }
                    term3_belief = (plan.m3_belief / term3_h3) * (sum_d / shared.term3_battles as f64);
                }
            }
        }
    }

    let delta_u = term1_diff + term2_same_slot + term3_belief;

    // Optional opponent adaptation decomposition:
    //   Delta_adapt = tau_q * E_{a ~ p1, (b1, b0) ~ couple(eta1, eta0)}[V1(a, b1) - V1(a, b0)]
    //   Delta_fight = Delta_u - Delta_adapt
    let (delta_adapt, delta_fight) = if shared.decompose_adapt {
        let (tau_q, eta1, eta0) = build_opp_shift_residuals(&q0, &q1);
        let d_adapt = if tau_q > 0.0 {
            let mut rng_a = SplitMix64::new(derive_seed(
                "couple-adapt",
                shared.master_seed,
                &opp.id,
                cluster as u64,
                seat as u64,
                0,
            ));
            let (run_adapt, adapt_weight) = match shared.alloc_mode {
                AllocMode::Neyman | AllocMode::SingleDraw => {
                    if rng_a.next_f64() < tau_q {
                        (true, 1.0)
                    } else {
                        (false, 0.0)
                    }
                }
                AllocMode::Stratified => (true, tau_q),
            };
            if run_adapt {
                let p1_items: Vec<(String, f64)> =
                    p1.iter().map(|x| (x.input.clone(), x.prob)).collect();
                let mut sum_ad = 0.0;
                for rep in 0..shared.diff_battles {
                    let a = sample_weighted(&p1_items, &mut rng_a);
                    let (b1, b0) = couple_picks_exact_then_lead(&eta1, &eta0, &mut rng_a);
                    let bseed = derive_seed_u32(
                        "battle-adapt",
                        shared.master_seed,
                        &opp.id,
                        cluster as u64,
                        seat as u64,
                        rep as u64,
                    );
                    let sme = derive_seed_u32(
                        "agent-me-adapt",
                        shared.master_seed,
                        &opp.id,
                        cluster as u64,
                        seat as u64,
                        rep as u64,
                    );
                    let sopp = derive_seed_u32(
                        "agent-opp-adapt",
                        shared.master_seed,
                        &opp.id,
                        cluster as u64,
                        seat as u64,
                        rep as u64,
                    );
                    let aseeds = if seat == 0 { [sme, sopp] } else { [sopp, sme] };
                    let p_hi = if seat == 0 {
                        [a.as_str(), b1.as_str()]
                    } else {
                        [b1.as_str(), a.as_str()]
                    };
                    let p_lo = if seat == 0 {
                        [a.as_str(), b0.as_str()]
                    } else {
                        [b0.as_str(), a.as_str()]
                    };
                    let o_q1 = run_forced_pick_battle(
                        &shared.dex,
                        s1_pair,
                        priors_for_arm(1),
                        p_hi,
                        seat,
                        &shared.rm_cfg,
                        shared.battle_iters,
                        shared.battle_mode,
                        bseed,
                        aseeds,
                    );
                    let o_q0 = run_forced_pick_battle(
                        &shared.dex,
                        s1_pair,
                        priors_for_arm(1),
                        p_lo,
                        seat,
                        &shared.rm_cfg,
                        shared.battle_iters,
                        shared.battle_mode,
                        bseed,
                        aseeds,
                    );
                    battle_ms += o_q1.elapsed_ms + o_q0.elapsed_ms;
                    battles_adapt += 2;
                    sum_ad += o_q1.score_for_me - o_q0.score_for_me;
                }
                adapt_weight * (sum_ad / shared.diff_battles as f64)
            } else {
                0.0
            }
        } else {
            0.0
        };
        (Some(d_adapt), Some(delta_u - d_adapt))
    } else {
        (None, None)
    };

    SeatClusterResult {
        opp_id: opp.id.clone(),
        opp_realized_id: opp_team.id.clone(),
        cluster,
        seat,
        delta_u,
        term1_diff,
        term2_same_slot,
        term3_belief,
        delta_adapt,
        delta_fight,
        tau: plan.tau,
        m2_same_slot: plan.m2_same_slot,
        m3_exact_zero: plan.m3_exact_zero,
        m3_belief: plan.m3_belief,
        tv_own: plan.tv_own,
        tv_opp: plan.tv_opp,
        b_coupled_draws,
        stratum1_draws,
        term3_h3,
        term3_sampled,
        battles_term1,
        battles_term2,
        battles_term3,
        battles_adapt,
        previews_computed,
        previews_skipped_q,
        exact_zero_skipped,
        verify_zero_checked,
        verify_zero_mismatches,
        preview_ms,
        battle_ms,
    }
}

fn find_workspace_root() -> PathBuf {
    env::var("NC2000_ROOT")
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
        })
}

fn resolve_team_or_singleton<'a>(
    id: &str,
    teams: &'a HashMap<String, Vec<PokemonSet>>,
    mixtures: &'a BTreeMap<String, MetaPool>,
) -> Result<(String, &'a Vec<PokemonSet>), String> {
    if let Some(sets) = teams.get(id) {
        return Ok((id.to_string(), sets));
    }
    if let Some(pool) = mixtures.get(id) {
        if pool.teams.len() == 1 {
            let tid = &pool.teams[0].id;
            if let Some(sets) = teams.get(tid) {
                return Ok((tid.clone(), sets));
            }
        }
        return Err(format!(
            "id {id:?} is a multi-team mixture ({} teams); specify a concrete team ID",
            pool.teams.len()
        ));
    }
    Err(format!("unknown team or mixture ID: {id:?}"))
}

/// Rao-Wu rescaled stratified bootstrap 95% CI.
/// When `K >= 2`, rescales centered stratum observations by `sqrt(k / (k - 1))`
/// before drawing `k` samples with replacement per opponent stratum so bootstrap
/// variance matches the exact unbiased stratified variance `sum_o w_o^2 * s_o^2 / k`
/// without `(k - 1) / k` small-cluster shrinkage (50% variance loss at `k = 2`).
fn stratified_bootstrap_ci(
    per_opp_clusters: &[Vec<f64>],
    opp_weights: &[f64],
    seed: u64,
    reps: usize,
) -> (f64, f64) {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(reps);
    let k_min = per_opp_clusters.iter().map(|v| v.len()).min().unwrap_or(0);
    let rescaled: Vec<Vec<f64>> = if k_min >= 2 {
        per_opp_clusters
            .iter()
            .map(|vals| {
                let k = vals.len() as f64;
                let mean = vals.iter().sum::<f64>() / k;
                let scale = (k / (k - 1.0)).sqrt();
                vals.iter().map(|&v| mean + scale * (v - mean)).collect()
            })
            .collect()
    } else {
        Vec::new()
    };
    let w_sum: f64 = opp_weights.iter().sum();
    let opp_weighted_items: Vec<(usize, f64)> = opp_weights
        .iter()
        .enumerate()
        .map(|(o, &w)| (o, if w_sum > 0.0 { w / w_sum } else { 1.0 / per_opp_clusters.len() as f64 }))
        .collect();
    for _ in 0..reps {
        let mut est = 0.0;
        if k_min >= 2 {
            for (o, vals) in rescaled.iter().enumerate() {
                let k = vals.len();
                let mut sum = 0.0;
                for _ in 0..k {
                    sum += vals[rng.below(k)];
                }
                est += opp_weights[o] * (sum / k as f64);
            }
        } else {
            // Resample `n_eff = round(1 / sum w_o^2)` weighted draws so variance matches `se_across_opp`.
            let w_sq_sum: f64 = opp_weights.iter().map(|w| w * w).sum();
            let n_draws = if w_sq_sum > 0.0 {
                (1.0 / w_sq_sum).round().max(1.0) as usize
            } else {
                per_opp_clusters.len()
            };
            for _ in 0..n_draws {
                let o = sample_weighted(&opp_weighted_items, &mut rng);
                est += per_opp_clusters[o][0] / n_draws as f64;
            }
        }
        samples.push(est);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = ((0.025 * reps as f64).floor() as usize).min(reps - 1);
    let hi_idx = ((0.975 * reps as f64).ceil() as usize).min(reps - 1);
    (samples[lo_idx], samples[hi_idx])
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = match parse_cli(&args) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let wall_start = Instant::now();
    let root = find_workspace_root();
    let dex_bytes =
        fs::read(root.join("data/gen2stadium2.json")).expect("read data/gen2stadium2.json");
    let dex = Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).expect("parse dex");
    let learnsets_text =
        fs::read_to_string(root.join("data/learnsets-gen2.json")).expect("read learnsets");
    let learnsets = Learnsets::from_json(&learnsets_text).expect("parse learnsets");

    let live: LiveSearchCfg = serde_json::from_str(
        &fs::read_to_string(root.join("data/live-search.json")).expect("read live-search.json"),
    )
    .expect("parse live-search.json");
    let profiles: SearchProfilesCfg = serde_json::from_str(
        &fs::read_to_string(root.join("data/search-profiles.json"))
            .expect("read search-profiles.json"),
    )
    .expect("parse search-profiles.json");

    let preview_iters = cfg.preview_iters.unwrap_or(live.unknown_preview);
    let battle_iters = cfg.battle_iters.unwrap_or(live.battle.adaptive);
    let blind_c = cfg.blind_c.unwrap_or(profiles.blind.c);

    let mut teams: HashMap<String, Vec<PokemonSet>> = HashMap::new();
    let mut team_order: Vec<String> = Vec::new();
    let mut raw_mixtures: BTreeMap<String, Vec<MixtureMember>> = BTreeMap::new();

    let pool_files = if cfg.pool_files.is_empty() {
        vec![root.join("data/team-pool-v4/team-pool.json")]
    } else {
        cfg.pool_files.clone()
    };

    for path in &pool_files {
        let text =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let doc: PoolOrCatalogFile = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        for t in doc.teams {
            if !teams.contains_key(&t.id) {
                team_order.push(t.id.clone());
            }
            teams.insert(t.id, t.sets);
        }
        for (name, members) in doc.mixtures {
            raw_mixtures.insert(name, members);
        }
    }

    let mut mixtures: BTreeMap<String, MetaPool> = BTreeMap::new();
    for (name, members) in raw_mixtures {
        let mut seen = HashSet::new();
        let mut meta_teams = Vec::with_capacity(members.len());
        let mut sum_w = 0.0;
        for m in members {
            assert!(
                m.weight.is_finite() && m.weight > 0.0,
                "invalid non-positive weight {} for member {} in mixture {name}",
                m.weight,
                m.id
            );
            assert!(
                seen.insert(m.id.clone()),
                "duplicate member {} in mixture {name}",
                m.id
            );
            let sets = teams
                .get(&m.id)
                .unwrap_or_else(|| panic!("unknown team {} in mixture {name}", m.id))
                .clone();
            sum_w += m.weight;
            meta_teams.push(MetaTeam {
                id: m.id,
                sets,
                weight: Some(m.weight),
            });
        }
        assert!(
            sum_w.is_finite() && sum_w > 0.0,
            "mixture {name} has non-positive weight sum {sum_w}"
        );
        mixtures.insert(name, MetaPool { teams: meta_teams });
    }

    let (base_real_id, base_sets_ref) = resolve_team_or_singleton(&cfg.base_id, &teams, &mixtures)
        .unwrap_or_else(|e| panic!("--base: {e}"));
    let (cand_real_id, cand_sets_ref) = resolve_team_or_singleton(&cfg.cand_id, &teams, &mixtures)
        .unwrap_or_else(|e| panic!("--cand: {e}"));
    let base_sets = base_sets_ref.clone();
    let cand_sets = cand_sets_ref.clone();

    let (opp_prior_0, opp_prior_1, opp_belief_mode) = if let Some(ref mix_name) = cfg.opp_belief_mix
    {
        let pool = mixtures
            .get(mix_name)
            .unwrap_or_else(|| panic!("unknown --opp-belief-mix {mix_name:?}"))
            .clone();
        (pool.clone(), pool, format!("shared-mix:{mix_name}"))
    } else if cfg.shared_opp_belief {
        let shared = if base_real_id == cand_real_id && sets_json_eq(&base_sets, &cand_sets) {
            MetaPool {
                teams: vec![MetaTeam {
                    id: base_real_id.clone(),
                    sets: base_sets.clone(),
                    weight: Some(1.0),
                }],
            }
        } else {
            MetaPool {
                teams: vec![
                    MetaTeam {
                        id: base_real_id.clone(),
                        sets: base_sets.clone(),
                        weight: Some(0.5),
                    },
                    MetaTeam {
                        id: cand_real_id.clone(),
                        sets: cand_sets.clone(),
                        weight: Some(0.5),
                    },
                ],
            }
        };
        (shared.clone(), shared, "shared-50-50".to_string())
    } else {
        let p0 = match &cfg.base_mix {
            Some(name) => mixtures
                .get(name)
                .unwrap_or_else(|| panic!("unknown --base-mix {name:?}"))
                .clone(),
            None => MetaPool {
                teams: vec![MetaTeam {
                    id: base_real_id.clone(),
                    sets: base_sets.clone(),
                    weight: Some(1.0),
                }],
            },
        };
        let p1 = match &cfg.cand_mix {
            Some(name) => mixtures
                .get(name)
                .unwrap_or_else(|| panic!("unknown --cand-mix {name:?}"))
                .clone(),
            None => MetaPool {
                teams: vec![MetaTeam {
                    id: cand_real_id.clone(),
                    sets: cand_sets.clone(),
                    weight: Some(1.0),
                }],
            },
        };
        (p0, p1, "disclosed".to_string())
    };

    // Resolve opponent list.
    let opp_ids: Vec<String> = match cfg.opps_spec.as_deref() {
        Some("all-shells") => mixtures
            .keys()
            .filter(|k| k.starts_with("shell:"))
            .cloned()
            .collect(),
        Some("all-teams") => team_order
            .iter()
            .filter(|id| *id != &base_real_id && *id != &cand_real_id)
            .cloned()
            .collect(),
        Some(spec) if spec.starts_with('@') => {
            let p = Path::new(&spec[1..]);
            fs::read_to_string(p)
                .unwrap_or_else(|e| panic!("read opps file {}: {e}", p.display()))
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_string)
                .collect()
        }
        Some(spec) => spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        None => {
            let shells: Vec<String> = mixtures
                .keys()
                .filter(|k| k.starts_with("shell:"))
                .cloned()
                .collect();
            if !shells.is_empty() {
                shells
            } else {
                team_order
                    .iter()
                    .filter(|id| *id != &base_real_id && *id != &cand_real_id)
                    .cloned()
                    .collect()
            }
        }
    };
    assert!(!opp_ids.is_empty(), "empty opponent set");

    let n_opps = opp_ids.len();
    let uniform_w = 1.0 / (n_opps as f64);
    let opps: Vec<OppSpec> = opp_ids
        .into_iter()
        .map(|oid| {
            let pool = if let Some(m) = mixtures.get(&oid) {
                m.clone()
            } else if let Some(sets) = teams.get(&oid) {
                MetaPool {
                    teams: vec![MetaTeam {
                        id: oid.clone(),
                        sets: sets.clone(),
                        weight: Some(1.0),
                    }],
                }
            } else {
                panic!("unknown opponent team or mixture: {oid:?}");
            };
            OppSpec {
                id: oid,
                pool,
                weight: uniform_w,
            }
        })
        .collect();

    // Pre-flight guards: validate every participating team and verify deterministic genders.
    let mut validated = HashSet::new();
    let mut check_team = |tid: &str, sets: &[PokemonSet]| {
        if !validated.insert(tid.to_string()) {
            return;
        }
        let sets_json = serde_json::to_string(sets).unwrap();
        let v = validate_team(&dex, &learnsets, &sets_json);
        assert!(
            v["ok"].as_bool() == Some(true),
            "pre-flight validate_team failed for {tid}: {v}"
        );
        for (slot, s) in sets.iter().enumerate() {
            assert!(
                has_deterministic_gender(&dex, s),
                "pre-flight guard failed: team {tid} slot {} ({}) lacks explicit/fixed gender (would advance Battle PRNG)",
                slot + 1,
                s.species
            );
        }
    };
    check_team(&base_real_id, &base_sets);
    check_team(&cand_real_id, &cand_sets);
    for t in &opp_prior_0.teams {
        check_team(&t.id, &t.sets);
    }
    for t in &opp_prior_1.teams {
        check_team(&t.id, &t.sets);
    }
    for opp in &opps {
        for t in &opp.pool.teams {
            check_team(&t.id, &t.sets);
        }
    }

    let arm_cmp = ArmComparison::build(&dex, &base_sets, &cand_sets, &opp_prior_0, &opp_prior_1);
    let rm_cfg = RmConfig {
        rule: SelRule::Ucb,
        c: blind_c,
        hp_buckets: 16,
        ..RmConfig::default()
    };

    let shared = Arc::new(SharedRunner {
        dex,
        base_sets,
        cand_sets,
        opp_prior_0,
        opp_prior_1,
        arm_cmp,
        opps,
        rm_cfg,
        preview_iters,
        preview_reps: cfg.preview_reps,
        alloc_mode: cfg.alloc_mode,
        battle_iters,
        battle_mode: cfg.battle_mode,
        diff_battles: cfg.diff_battles,
        same_slot_battles: cfg.same_slot_battles,
        term3_battles: cfg.term3_battles,
        term3_prob: cfg.term3_prob,
        term3_pps: cfg.term3_pps,
        skip_term3: cfg.skip_term3,
        decompose_adapt: cfg.decompose_adapt,
        verify_zero_term3: cfg.verify_zero_term3,
        master_seed: cfg.seed,
    });

    let mut work: Vec<(usize, usize, usize)> = Vec::with_capacity(n_opps * cfg.clusters * 2);
    for opp_idx in 0..n_opps {
        for cluster in 0..cfg.clusters {
            for seat in 0..2 {
                work.push((opp_idx, cluster, seat));
            }
        }
    }

    let total_tasks = work.len();
    let results: Vec<std::sync::Mutex<Option<SeatClusterResult>>> =
        (0..total_tasks).map(|_| std::sync::Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let n_workers = cfg.threads.min(total_tasks.max(1));

    std::thread::scope(|s| {
        for _ in 0..n_workers {
            let shared = &shared;
            let work = &work;
            let results = &results;
            let next = &next;
            s.spawn(move || loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= work.len() {
                    break;
                }
                let (opp_idx, cluster, seat) = work[idx];
                let res = run_seat_cluster(shared, opp_idx, cluster, seat);
                *results[idx].lock().unwrap() = Some(res);
            });
        }
    });

    let flat_results: Vec<SeatClusterResult> = results
        .into_iter()
        .map(|m| m.into_inner().unwrap().unwrap())
        .collect();

    if let Some(ref out_path) = cfg.out_jsonl {
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).expect("create parent dir for --out");
            }
        }
        let mut f = fs::File::create(out_path)
            .unwrap_or_else(|e| panic!("create {}: {e}", out_path.display()));
        for row in &flat_results {
            writeln!(f, "{}", serde_json::to_string(row).unwrap()).expect("write jsonl");
        }
    }

    // Aggregate seat-balanced cluster estimates `Delta_{o, r} = 0.5 * (seat0 + seat1)`.
    let mut per_opp_delta: Vec<Vec<f64>> = vec![vec![0.0; cfg.clusters]; n_opps];
    let mut per_opp_t1: Vec<Vec<f64>> = vec![vec![0.0; cfg.clusters]; n_opps];
    let mut per_opp_t2: Vec<Vec<f64>> = vec![vec![0.0; cfg.clusters]; n_opps];
    let mut per_opp_t3: Vec<Vec<f64>> = vec![vec![0.0; cfg.clusters]; n_opps];
    let mut per_opp_adapt: Vec<Vec<f64>> = vec![vec![0.0; cfg.clusters]; n_opps];
    let mut per_opp_fight: Vec<Vec<f64>> = vec![vec![0.0; cfg.clusters]; n_opps];

    let mut sum_tau = 0.0;
    let mut sum_m2 = 0.0;
    let mut sum_m3z = 0.0;
    let mut sum_m3 = 0.0;
    let mut sum_tv_own = 0.0;
    let mut sum_tv_opp = 0.0;
    let mut b_coupled_total = 0usize;
    let mut s1_draws_total = 0usize;
    let mut battles_t1 = 0usize;
    let mut battles_t2 = 0usize;
    let mut battles_t3 = 0usize;
    let mut battles_adapt = 0usize;
    let mut previews_computed = 0usize;
    let mut previews_skipped_q = 0usize;
    let mut exact_zero_skips = 0usize;
    let mut verify_checked = 0usize;
    let mut verify_mismatches = 0usize;
    let mut preview_cpu_ms = 0.0;
    let mut battle_cpu_ms = 0.0;

    for (idx, row) in flat_results.iter().enumerate() {
        let (opp_idx, cluster, _seat) = work[idx];
        per_opp_delta[opp_idx][cluster] += 0.5 * row.delta_u;
        per_opp_t1[opp_idx][cluster] += 0.5 * row.term1_diff;
        per_opp_t2[opp_idx][cluster] += 0.5 * row.term2_same_slot;
        per_opp_t3[opp_idx][cluster] += 0.5 * row.term3_belief;
        if let (Some(da), Some(df)) = (row.delta_adapt, row.delta_fight) {
            per_opp_adapt[opp_idx][cluster] += 0.5 * da;
            per_opp_fight[opp_idx][cluster] += 0.5 * df;
        }
        sum_tau += row.tau;
        sum_m2 += row.m2_same_slot;
        sum_m3z += row.m3_exact_zero;
        sum_m3 += row.m3_belief;
        sum_tv_own += row.tv_own;
        sum_tv_opp += row.tv_opp;
        b_coupled_total += row.b_coupled_draws;
        s1_draws_total += row.stratum1_draws;
        battles_t1 += row.battles_term1;
        battles_t2 += row.battles_term2;
        battles_t3 += row.battles_term3;
        battles_adapt += row.battles_adapt;
        previews_computed += row.previews_computed;
        previews_skipped_q += row.previews_skipped_q;
        if row.exact_zero_skipped {
            exact_zero_skips += 1;
        }
        verify_checked += row.verify_zero_checked;
        verify_mismatches += row.verify_zero_mismatches;
        preview_cpu_ms += row.preview_ms;
        battle_cpu_ms += row.battle_ms;
    }

    if cfg.verify_zero_term3 {
        assert_eq!(
            verify_mismatches, 0,
            "Zero-Term-3 verification FAILED: {verify_mismatches}/{verify_checked} mismatches"
        );
    }

    let opp_weights: Vec<f64> = shared.opps.iter().map(|o| o.weight).collect();
    let agg_mean_and_se = |mat: &[Vec<f64>]| -> (f64, Option<f64>, Option<f64>) {
        let mut total_mean = 0.0;
        let mut opp_means = Vec::with_capacity(n_opps);
        let mut var_fixed_opp = 0.0;
        for (o, vals) in mat.iter().enumerate() {
            let k = vals.len();
            let mean_o = vals.iter().sum::<f64>() / k as f64;
            opp_means.push(mean_o);
            total_mean += opp_weights[o] * mean_o;
            if k >= 2 {
                let s2 = vals.iter().map(|&v| (v - mean_o).powi(2)).sum::<f64>() / (k - 1) as f64;
                var_fixed_opp += opp_weights[o].powi(2) * (s2 / k as f64);
            }
        }
        let se_across_opp = if n_opps >= 2 {
            let w_sq_sum: f64 = opp_weights.iter().map(|w| w * w).sum();
            let denom = 1.0 - w_sq_sum;
            let s2_w = if denom > 1e-12 {
                opp_means
                    .iter()
                    .enumerate()
                    .map(|(o, &v)| opp_weights[o] * (v - total_mean).powi(2))
                    .sum::<f64>()
                    / denom
            } else {
                0.0
            };
            Some((s2_w * w_sq_sum).sqrt())
        } else {
            None
        };
        let se_cluster = if cfg.clusters >= 2 {
            Some(var_fixed_opp.sqrt())
        } else {
            None
        };
        (total_mean, se_cluster, se_across_opp)
    };

    let (delta_u, se_cluster, se_across_opp) = agg_mean_and_se(&per_opp_delta);
    let (term1_diff, _, _) = agg_mean_and_se(&per_opp_t1);
    let (term2_same_slot, _, _) = agg_mean_and_se(&per_opp_t2);
    let (term3_belief, _, _) = agg_mean_and_se(&per_opp_t3);
    let ci95_boot = if cfg.clusters >= 2 || n_opps >= 2 {
        let (ci_lo, ci_hi) = stratified_bootstrap_ci(
            &per_opp_delta,
            &opp_weights,
            derive_seed(
                "bootstrap-ci",
                cfg.seed,
                &base_real_id,
                n_opps as u64,
                cfg.clusters as u64,
                0,
            ),
            4000,
        );
        json!([(ci_lo * 1e6).round() / 1e6, (ci_hi * 1e6).round() / 1e6])
    } else {
        Value::Null
    };

    let n_sc = total_tasks as f64;
    let battles_core = battles_t1 + battles_t2 + battles_t3;
    let battles_total = battles_core + battles_adapt;
    // Theoretical independent A/B SE at same core battles (assuming p(1-p) = 0.25):
    //   N_per_arm = battles_core / 2 => Var(p1_hat - p0_hat) = 0.25/N + 0.25/N = 1 / battles_core.
    let se_indep_equiv = if battles_core > 0 {
        (1.0 / battles_core as f64).sqrt()
    } else {
        f64::INFINITY
    };
    let vr_vs_indep_per_battle: Option<f64> = se_cluster
        .filter(|&se| se > 1e-12 && battles_core > 0)
        .map(|se| ((se_indep_equiv / se).powi(2) * 1e2).round() / 1e2);
    let z_score: Option<f64> = se_cluster
        .filter(|&se| se > 1e-12)
        .map(|se| ((delta_u / se) * 1e3).round() / 1e3);
    let round6 = |v: Option<f64>| v.map(|x| (x * 1e6).round() / 1e6);

    let estimator_mode = if cfg.skip_term3 {
        match cfg.alloc_mode {
            AllocMode::Neyman => "screening_neyman_skip_term3",
            AllocMode::SingleDraw => "screening_single_draw_skip_term3",
            AllocMode::Stratified => "screening_stratified_skip_term3",
        }
    } else {
        match cfg.alloc_mode {
            AllocMode::Neyman => "unbiased_3stratum_neyman_single_draw",
            AllocMode::SingleDraw => "unbiased_3stratum_single_draw",
            AllocMode::Stratified => {
                if cfg.term3_pps.is_some() {
                    "unbiased_3stratum_stratified_pps"
                } else if cfg.term3_prob < 1.0 {
                    "unbiased_3stratum_stratified_ht"
                } else {
                    "unbiased_3stratum_stratified_exact"
                }
            }
        }
    };

    let mut summary = json!({
        "policy_epoch": POLICY_EPOCH,
        "mode": estimator_mode,
        "alloc_mode": cfg.alloc_mode.as_str(),
        "preview_reps": cfg.preview_reps,
        "opp_belief_mode": opp_belief_mode,
        "base": base_real_id,
        "cand": cand_real_id,
        "diff_slots_1based": shared.arm_cmp.diff_slots.iter().enumerate().filter(|(_, &d)| d).map(|(i, _)| i + 1).collect::<Vec<_>>(),
        "same_public_frame": shared.arm_cmp.same_public_turn0_frame,
        "n_opps": n_opps,
        "clusters": cfg.clusters,
        "delta_u": (delta_u * 1e6).round() / 1e6,
        "se_cluster": round6(se_cluster),
        "se_across_opp": round6(se_across_opp),
        "se_mode": if cfg.clusters >= 2 { "fixed_opp_cluster" } else { "across_opp_only_k1" },
        "z_cluster": z_score,
        "ci95_boot": ci95_boot,
        "term1_diff": (term1_diff * 1e6).round() / 1e6,
        "term2_same_slot": (term2_same_slot * 1e6).round() / 1e6,
        "term3_belief": (term3_belief * 1e6).round() / 1e6,
        "mean_tau": ((sum_tau / n_sc) * 1e5).round() / 1e5,
        "mean_m2_same_slot": ((sum_m2 / n_sc) * 1e5).round() / 1e5,
        "mean_m3_exact_zero": ((sum_m3z / n_sc) * 1e5).round() / 1e5,
        "mean_m3_belief": ((sum_m3 / n_sc) * 1e5).round() / 1e5,
        "mean_tv_own": ((sum_tv_own / n_sc) * 1e5).round() / 1e5,
        "mean_tv_opp": ((sum_tv_opp / n_sc) * 1e5).round() / 1e5,
        "b_coupling_rate": if s1_draws_total > 0 {
            ((b_coupled_total as f64 / s1_draws_total as f64) * 1e4).round() / 1e4
        } else {
            1.0
        },
        "battles_total": battles_total,
        "battles_core": battles_core,
        "battles_term1": battles_t1,
        "battles_term2": battles_t2,
        "battles_term3": battles_t3,
        "battles_adapt": battles_adapt,
        "previews_computed": previews_computed,
        "previews_skipped_q": previews_skipped_q,
        "exact_zero_skips": exact_zero_skips,
        "vr_vs_indep_per_battle": vr_vs_indep_per_battle,
        "preview_cpu_ms": preview_cpu_ms.round() as u64,
        "battle_cpu_ms": battle_cpu_ms.round() as u64,
        "wall_ms": (wall_start.elapsed().as_secs_f64() * 1000.0).round() as u64,
        "threads": n_workers,
    });

    if cfg.decompose_adapt {
        let (d_adapt, se_adapt, _) = agg_mean_and_se(&per_opp_adapt);
        let (d_fight, se_fight, _) = agg_mean_and_se(&per_opp_fight);
        let obj = summary.as_object_mut().unwrap();
        obj.insert("delta_fight".into(), json!((d_fight * 1e6).round() / 1e6));
        obj.insert("se_fight".into(), json!(round6(se_fight)));
        obj.insert("delta_adapt".into(), json!((d_adapt * 1e6).round() / 1e6));
        obj.insert("se_adapt".into(), json!(round6(se_adapt)));
    }
    if cfg.verify_zero_term3 {
        let obj = summary.as_object_mut().unwrap();
        obj.insert("verify_zero_checked".into(), json!(verify_checked));
        obj.insert("verify_zero_mismatches".into(), json!(verify_mismatches));
    }
    if let Some(ref p) = cfg.out_jsonl {
        summary
            .as_object_mut()
            .unwrap()
            .insert("out_jsonl".into(), json!(p.display().to_string()));
    }

    println!("{}", serde_json::to_string(&summary).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dex() -> Dex {
        let root = find_workspace_root();
        let bytes = fs::read(root.join("data/gen2stadium2.json")).expect("read dex");
        Dex::from_json(std::str::from_utf8(&bytes).unwrap()).expect("parse dex")
    }

    fn mk_set(species: &str, level: u8, item: &str, moves: &[&str], gender: &str) -> PokemonSet {
        serde_json::from_value(json!({
            "name": species,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "level": level,
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255,
            "gender": gender
        }))
        .unwrap()
    }

    #[test]
    fn three_stratum_coupling_and_ht_and_adapt_are_strictly_unbiased() {
        // Synthetic 4x3 asymmetric payoff game with slot 1 (0-based) mutated (Lv55) and slot 0 Lv55.
        // Pick "team 134" => lead=0 (Lv55), no slot 1 => exact-zero Term 3!
        // Pick "team 314" => lead=2 (Lv50), no slot 1 => non-exact Term 3 (m3_belief)!
        // Pick "team 234" and "team 123" => contain slot 1 => Stratum 2 (m2_same_slot)!
        let a_keys = ["team 134", "team 314", "team 234", "team 123"];
        let b_keys = ["team 123", "team 145", "team 236"];

        let p0 = vec![
            WeightedAction { input: a_keys[0].into(), slots: parse_team_pick(a_keys[0]), prob: 0.35 },
            WeightedAction { input: a_keys[1].into(), slots: parse_team_pick(a_keys[1]), prob: 0.30 },
            WeightedAction { input: a_keys[2].into(), slots: parse_team_pick(a_keys[2]), prob: 0.25 },
            WeightedAction { input: a_keys[3].into(), slots: parse_team_pick(a_keys[3]), prob: 0.10 },
        ];
        let p1 = vec![
            WeightedAction { input: a_keys[0].into(), slots: parse_team_pick(a_keys[0]), prob: 0.20 },
            WeightedAction { input: a_keys[1].into(), slots: parse_team_pick(a_keys[1]), prob: 0.25 },
            WeightedAction { input: a_keys[2].into(), slots: parse_team_pick(a_keys[2]), prob: 0.40 },
            WeightedAction { input: a_keys[3].into(), slots: parse_team_pick(a_keys[3]), prob: 0.15 },
        ];
        let q0 = vec![
            WeightedAction { input: b_keys[0].into(), slots: parse_team_pick(b_keys[0]), prob: 0.50 },
            WeightedAction { input: b_keys[1].into(), slots: parse_team_pick(b_keys[1]), prob: 0.30 },
            WeightedAction { input: b_keys[2].into(), slots: parse_team_pick(b_keys[2]), prob: 0.20 },
        ];
        let q1 = vec![
            WeightedAction { input: b_keys[0].into(), slots: parse_team_pick(b_keys[0]), prob: 0.30 },
            WeightedAction { input: b_keys[1].into(), slots: parse_team_pick(b_keys[1]), prob: 0.45 },
            WeightedAction { input: b_keys[2].into(), slots: parse_team_pick(b_keys[2]), prob: 0.25 },
        ];

        let cmp = ArmComparison {
            diff_slots: vec![false, true, false, false, false, false],
            same_public_turn0_frame: false,
            same_levels_all_slots: true,
            same_party_order_matters: true,
            opp_belief_identical: false,
            opp_belief_singleton_arms: true,
            base_levels: vec![55, 55, 50, 50, 50, 50],
            base_immune_masks: vec![0, 0, 0, 0, 0, 0],
            cand_immune_masks: vec![0, 0, 0, 0, 0, 0],
        };
        assert!(cmp.is_exact_zero_term3(&parse_team_pick("team 134")));
        assert!(!cmp.is_exact_zero_term3(&parse_team_pick("team 314")));
        assert!(!cmp.is_exact_zero_term3(&parse_team_pick("team 234")));

        let plan = build_coupling_plan(&p0, &q0, &p1, &q1, &cmp);
        let mass_sum = plan.tau + plan.m2_same_slot + plan.m3_exact_zero + plan.m3_belief;
        assert!((mass_sum - 1.0).abs() < 1e-12, "mass_sum={mass_sum}");
        assert!(plan.tau > 0.1 && plan.m2_same_slot > 0.1 && plan.m3_exact_zero > 0.05 && plan.m3_belief > 0.1);

        // Payoff matrices V0, V1 (identical on row 0 because row 0 is exact-zero Term 3).
        let v0 = [
            [0.61, 0.42, 0.77],
            [0.53, 0.29, 0.81],
            [0.44, 0.68, 0.35],
            [0.72, 0.51, 0.26],
        ];
        let v1 = [
            [0.61, 0.42, 0.77], // exact zero diff on row 0
            [0.58, 0.25, 0.84], // non-zero Term 3 belief effect on row 1
            [0.63, 0.74, 0.49], // Stratum 2 mutated slot effect on row 2
            [0.80, 0.55, 0.39], // Stratum 2 mutated slot effect on row 3
        ];
        let lookup = |m: &[[f64; 3]; 4], a: &str, b: &str| -> f64 {
            let ai = a_keys.iter().position(|&k| k == a).unwrap();
            let bi = b_keys.iter().position(|&k| k == b).unwrap();
            m[ai][bi]
        };

        let mut true_u0 = 0.0;
        let mut true_u1 = 0.0;
        let mut true_p1_q0 = 0.0;
        for (ai, pa0) in p0.iter().enumerate() {
            for (bi, qb0) in q0.iter().enumerate() {
                true_u0 += pa0.prob * qb0.prob * v0[ai][bi];
            }
        }
        for (ai, pa1) in p1.iter().enumerate() {
            for (bi, qb1) in q1.iter().enumerate() {
                true_u1 += pa1.prob * qb1.prob * v1[ai][bi];
            }
            for (bi, qb0) in q0.iter().enumerate() {
                true_p1_q0 += pa1.prob * qb0.prob * v1[ai][bi];
            }
        }
        let true_delta = true_u1 - true_u0;
        let true_adapt = true_u1 - true_p1_q0;
        let true_fight = true_p1_q0 - true_u0;

        let (tau_q, eta1, eta0) = build_opp_shift_residuals(&q0, &q1);
        let p1_items: Vec<(String, f64)> = p1.iter().map(|x| (x.input.clone(), x.prob)).collect();

        let n_mc = 60_000usize;
        let mut rng = SplitMix64::new(1234567);
        let h3_const = 0.35f64;
        let mut sum_est = 0.0;
        let mut sum_adapt = 0.0;
        let mut b_coupled_count = 0usize;

        for _ in 0..n_mc {
            let ((a1, b1), (a0, b0), b_coup) =
                sample_stratum1_coupled(&plan.nu1, &plan.nu0, &mut rng);
            if b_coup {
                b_coupled_count += 1;
            }
            let t1 = plan.tau * (lookup(&v1, &a1, &b1) - lookup(&v0, &a0, &b0));
            let (a2, b2) = sample_pair(&plan.stratum2, &mut rng);
            let t2 = plan.m2_same_slot * (lookup(&v1, &a2, &b2) - lookup(&v0, &a2, &b2));
            let t3 = if rng.next_f64() < h3_const {
                let (a3, b3) = sample_pair(&plan.stratum3_belief, &mut rng);
                (plan.m3_belief / h3_const) * (lookup(&v1, &a3, &b3) - lookup(&v0, &a3, &b3))
            } else {
                0.0
            };
            sum_est += t1 + t2 + t3;

            let a_ad = sample_weighted(&p1_items, &mut rng);
            let (bq1, bq0) = couple_picks_exact_then_lead(&eta1, &eta0, &mut rng);
            sum_adapt += tau_q * (lookup(&v1, &a_ad, &bq1) - lookup(&v1, &a_ad, &bq0));
        }

        let est_delta = sum_est / n_mc as f64;
        let est_adapt = sum_adapt / n_mc as f64;
        let est_fight = est_delta - est_adapt;

        assert!(
            (est_delta - true_delta).abs() < 0.0025,
            "delta_u biased: est={est_delta:.6} true={true_delta:.6}"
        );
        assert!(
            (est_adapt - true_adapt).abs() < 0.0025,
            "delta_adapt biased: est={est_adapt:.6} true={true_adapt:.6}"
        );
        assert!(
            (est_fight - true_fight).abs() < 0.003,
            "delta_fight biased: est={est_fight:.6} true={true_fight:.6}"
        );
        assert!(
            b_coupled_count > 0,
            "secondary B-coupling should couple overlapping B values in Stratum 1"
        );
    }

    #[test]
    fn zero_term3_conditions_z1_and_z2_are_bit_exact_on_live_battles() {
        let dex = test_dex();
        let opp_sets = vec![
            mk_set("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"], "N"),
            mk_set("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"], "N"),
            mk_set("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"], "M"),
            mk_set("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"], "M"),
            mk_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"], "M"),
            mk_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"], "M"),
        ];
        let base = vec![
            mk_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"], "M"),
            mk_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"], "N"),
            mk_set("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"], "M"),
            mk_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Hidden Power Bug"], "M"),
            mk_set("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"], "M"),
            mk_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"], "M"),
        ];
        // Z1 candidate: same species/level/gender/has-item on slot 4 (Marowak moveset tweak).
        let mut cand_z1 = base.clone();
        cand_z1[3] = mk_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"], "M");
        // Z2 candidate: Lv55 species swap on slot 2 (Zapdos55 -> Gengar55).
        let mut cand_z2 = base.clone();
        cand_z2[1] = mk_set("Gengar", 55, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"], "M");

        let opp_pool = MetaPool {
            teams: vec![MetaTeam { id: "opp".into(), sets: opp_sets.clone(), weight: Some(1.0) }],
        };
        let cfg = RmConfig { rule: SelRule::Ucb, c: 0.4, hp_buckets: 16, ..RmConfig::default() };

        // Condition Z1: shared opponent belief + unpicked mutated slot 4 ("team 135").
        let shared_z1 = MetaPool {
            teams: vec![
                MetaTeam { id: "base".into(), sets: base.clone(), weight: Some(0.5) },
                MetaTeam { id: "z1".into(), sets: cand_z1.clone(), weight: Some(0.5) },
            ],
        };
        let cmp_z1 = ArmComparison::build(&dex, &base, &cand_z1, &shared_z1, &shared_z1);
        assert!(cmp_z1.is_exact_zero_term3(&parse_team_pick("team 1, 3, 5")));
        let b0 = run_forced_pick_battle(
            &dex, [&base, &opp_sets], [&opp_pool, &shared_z1], ["team 1, 3, 5", "team 5, 1, 6"],
            0, &cfg, 96, BattleMode::Fixed, 1001, [2001, 3001],
        );
        let b1 = run_forced_pick_battle(
            &dex, [&cand_z1, &opp_sets], [&opp_pool, &shared_z1], ["team 1, 3, 5", "team 5, 1, 6"],
            0, &cfg, 96, BattleMode::Fixed, 1001, [2001, 3001],
        );
        assert_eq!(b0.action_sha256, b1.action_sha256, "Condition Z1 must be bit-exact");
        assert_eq!(b0.score_for_me, b1.score_for_me);

        // Condition Z2: disclosed singleton beliefs + Lv55 lead ("team 1, 3, 6") excludes Lv55 slot 2.
        let p_base = MetaPool {
            teams: vec![MetaTeam { id: "base".into(), sets: base.clone(), weight: Some(1.0) }],
        };
        let p_z2 = MetaPool {
            teams: vec![MetaTeam { id: "z2".into(), sets: cand_z2.clone(), weight: Some(1.0) }],
        };
        let cmp_z2 = ArmComparison::build(&dex, &base, &cand_z2, &p_base, &p_z2);
        assert!(cmp_z2.is_exact_zero_term3(&parse_team_pick("team 1, 3, 6")));
        assert!(!cmp_z2.is_exact_zero_term3(&parse_team_pick("team 3, 1, 6")));
        let z0 = run_forced_pick_battle(
            &dex, [&base, &opp_sets], [&opp_pool, &p_base], ["team 1, 3, 6", "team 5, 2, 6"],
            0, &cfg, 96, BattleMode::Fixed, 4001, [5001, 6001],
        );
        let z1 = run_forced_pick_battle(
            &dex, [&cand_z2, &opp_sets], [&opp_pool, &p_z2], ["team 1, 3, 6", "team 5, 2, 6"],
            0, &cfg, 96, BattleMode::Fixed, 4001, [5001, 6001],
        );
        assert_eq!(z0.action_sha256, z1.action_sha256, "Condition Z2 (Lv55-lead skip) must be bit-exact");
        assert_eq!(z0.score_for_me, z1.score_for_me);

        // Audit regression #1: `mask.rs::foe_switchin_candidates` ignores the 155 level cap.
        // If all unpicked non-mutated mons share an immunity that flips on mutated slot `s`
        // (e.g. Zapdos55 + 4 Lv50 Flying mons, Snorlax55 vs Moltres55 flipping Ground immunity),
        // Z2 MUST conservatively return `false`.
        let fly_base = vec![
            mk_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"], "N"),
            mk_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"], "M"),
            mk_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"], "M"),
            mk_set("Dragonite", 50, "Leftovers", &["Dynamic Punch", "Ice Beam", "Thunderbolt", "Rest"], "M"),
            mk_set("Aerodactyl", 50, "Leftovers", &["Rock Slide", "Earthquake", "Wing Attack", "Fire Blast"], "M"),
            mk_set("Gligar", 50, "Leftovers", &["Earthquake", "Wing Attack", "Screech", "Counter"], "M"),
        ];
        let mut fly_cand = fly_base.clone();
        fly_cand[1] = mk_set("Moltres", 55, "Leftovers", &["Fire Blast", "Hidden Power Grass", "Rest", "Sleep Talk"], "N");
        let p_fb = MetaPool { teams: vec![MetaTeam { id: "fb".into(), sets: fly_base.clone(), weight: Some(1.0) }] };
        let p_fc = MetaPool { teams: vec![MetaTeam { id: "fc".into(), sets: fly_cand.clone(), weight: Some(1.0) }] };
        let cmp_fly = ArmComparison::build(&dex, &fly_base, &fly_cand, &p_fb, &p_fc);
        assert!(
            !cmp_fly.is_exact_zero_term3(&parse_team_pick("team 1, 3, 4")),
            "Z2 must reject Ground-immunity flip when all unpicked non-diff mons are Flying"
        );

        // Audit regression #2: mismatched singleton belief (`opp_prior_0 != base_sets`) MUST reject Z2.
        let p_wrong = MetaPool { teams: vec![MetaTeam { id: "wrong".into(), sets: opp_sets, weight: Some(1.0) }] };
        let cmp_wrong = ArmComparison::build(&dex, &base, &cand_z2, &p_wrong, &p_z2);
        assert!(
            !cmp_wrong.is_exact_zero_term3(&parse_team_pick("team 1, 3, 6")),
            "Z2 must reject misaligned singleton opponent beliefs"
        );
    }

    #[test]
    fn rao_wu_rescaled_bootstrap_matches_exact_cluster_se_at_k2() {
        // 32 opponents with K=2 clusters each: naive bootstrap would shrink SE by sqrt(1/2) = 0.707x.
        // Rao-Wu rescaled bootstrap must match analytical `se_cluster` closely.
        let m = 32usize;
        let w = vec![1.0 / m as f64; m];
        let mut rng = SplitMix64::new(987654);
        let mut mat: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut var_exact = 0.0;
        for o in 0..m {
            let v0 = (rng.next_f64() - 0.5) * 0.8 + (o as f64) * 0.01;
            let v1 = (rng.next_f64() - 0.5) * 0.8 + (o as f64) * 0.01;
            let mean = 0.5 * (v0 + v1);
            let s2 = (v0 - mean).powi(2) + (v1 - mean).powi(2); // (k - 1) = 1
            var_exact += w[o].powi(2) * (s2 / 2.0);
            mat.push(vec![v0, v1]);
        }
        let se_exact = var_exact.sqrt();
        let (lo, hi) = stratified_bootstrap_ci(&mat, &w, 424242, 16_000);
        let se_from_ci = (hi - lo) / (2.0 * 1.959964);
        let ratio = se_from_ci / se_exact;
        assert!(
            (ratio - 1.0).abs() < 0.08,
            "Rao-Wu bootstrap SE ratio {ratio:.4} drifted from 1.0 (ci_se={se_from_ci:.6}, exact={se_exact:.6})"
        );
    }

    #[test]
    fn forced_pick_turn0_matches_live_turn1_bit_exact() {
        let dex = test_dex();
        let t0 = vec![
            mk_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"], "M"),
            mk_set("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"], "N"),
            mk_set("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"], "M"),
            mk_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Hidden Power Bug"], "M"),
            mk_set("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"], "M"),
            mk_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"], "M"),
        ];
        let t1 = vec![
            mk_set("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"], "N"),
            mk_set("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"], "N"),
            mk_set("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"], "M"),
            mk_set("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"], "M"),
            mk_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"], "M"),
            mk_set("Umbreon", 50, "Leftovers", &["Pursuit", "Charm", "Rest", "Baton Pass"], "M"),
        ];
        let p0 = MetaPool { teams: vec![MetaTeam { id: "t1".into(), sets: t1.clone(), weight: Some(1.0) }] };
        let p1 = MetaPool { teams: vec![MetaTeam { id: "t0".into(), sets: t0.clone(), weight: Some(1.0) }] };
        let cfg = RmConfig { rule: SelRule::Ucb, c: 0.4, hp_buckets: 16, ..RmConfig::default() };

        // Run full live game (stepping Turn 0 preview at 256 iters, then Turn 1+ at 96 iters).
        let bseed = 7777u32;
        let aseeds = [8888u32, 9999u32];
        let bseed_str = SplitMix64::new(bseed as u64).battle_seed();
        let mut battle = Battle::from_fixture(&dex, &bseed_str, &t0, &t1).unwrap();
        let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
        let mut agents = [
            {
                let mut a = ProtocolAgent::new(&dex, 0, p0.clone(), cfg.clone(), aseeds[0] as u64);
                a.set_own_team(t0.clone());
                a
            },
            {
                let mut a = ProtocolAgent::new(&dex, 1, p1.clone(), cfg.clone(), aseeds[1] as u64);
                a.set_own_team(t1.clone());
                a
            },
        ];
        let mut decisions = [
            DecisionSampler::new(aseeds[0] as u64),
            DecisionSampler::new(aseeds[1] as u64),
        ];
        let mut transcript = Sha256::new();
        let mut live_picks = [String::new(), String::new()];
        let mut first = true;
        while battle.outcome().is_none() && battle.turn < 500 {
            let needs = battle.needs_choice();
            let mut inputs: [Option<String>; 2] = [None, None];
            for side in 0..2 {
                if !needs[side] {
                    continue;
                }
                let frame = channels[side].frame(&mut battle, &dex).unwrap();
                for line in &frame.lines {
                    agents[side].push_line(&dex, line);
                }
                decisions[side].reset();
                let req_json = serde_json::to_string(&frame.request).unwrap();
                assert!(agents[side].on_request(&dex, &req_json).unwrap());
                if first {
                    agents[side].step_adaptive_battle(&dex, 256).unwrap();
                } else {
                    agents[side].step(&dex, 96).unwrap();
                }
                let pol = agents[side].decision_policy(&dex, PolicyMode::Legacy).unwrap();
                let chosen = decisions[side].commit(pol).input.clone();
                if first {
                    live_picks[side] = chosen.clone();
                }
                inputs[side] = Some(chosen);
            }
            let mut line = serde_json::to_string(&inputs).unwrap();
            line.push('\n');
            transcript.update(line.as_bytes());
            for (side, inp) in inputs.iter().enumerate() {
                if let Some(s) = inp {
                    battle.choose(&dex, side, s).unwrap();
                }
            }
            first = false;
        }
        let live_sha = format!("{:x}", transcript.finalize());

        // Now run `run_forced_pick_battle` with the exact Turn-0 picks from the live run.
        let forced = run_forced_pick_battle(
            &dex,
            [&t0, &t1],
            [&p0, &p1],
            [&live_picks[0], &live_picks[1]],
            0,
            &cfg,
            96,
            BattleMode::Fixed,
            bseed,
            aseeds,
        );
        assert_eq!(
            forced.action_sha256, live_sha,
            "forced-pick Turn-0 runner diverged from live Turn 1+ battle"
        );
    }

    #[test]
    fn neyman_single_draw_and_preview_averaging_are_strictly_unbiased() {
        // Test both `AllocMode::Neyman` and `average_weighted_actions` (`--preview-reps R`)
        // on a stochastic mixture of preview policies (simulating LcbGatedMixed seed jitter).
        let a_keys = ["team 1, 3, 4", "team 3, 1, 4", "team 2, 3, 4", "team 1, 2, 3"];
        let b_keys = ["team 1, 2, 3", "team 1, 4, 5", "team 2, 3, 6"];
        let mk_wa = |probs: [f64; 4], keys: &[&str]| -> Vec<WeightedAction> {
            keys.iter()
                .zip(probs)
                .filter(|(_, p)| *p > 0.0)
                .map(|(&k, p)| WeightedAction {
                    input: k.to_string(),
                    slots: parse_team_pick(k),
                    prob: p,
                })
                .collect()
        };
        let mk_wb = |probs: [f64; 3], keys: &[&str]| -> Vec<WeightedAction> {
            keys.iter()
                .zip(probs)
                .filter(|(_, p)| *p > 0.0)
                .map(|(&k, p)| WeightedAction {
                    input: k.to_string(),
                    slots: parse_team_pick(k),
                    prob: p,
                })
                .collect()
        };

        // Two latent preview states for omega_me (prob 0.5/0.5) and omega_opp (prob 0.5/0.5).
        let p0_states = [mk_wa([0.6, 0.2, 0.1, 0.1], &a_keys), mk_wa([0.2, 0.4, 0.3, 0.1], &a_keys)];
        let p1_states = [mk_wa([0.3, 0.2, 0.4, 0.1], &a_keys), mk_wa([0.1, 0.3, 0.4, 0.2], &a_keys)];
        let q0_states = [mk_wb([0.7, 0.2, 0.1], &b_keys), mk_wb([0.3, 0.5, 0.2], &b_keys)];
        let q1_states = [mk_wb([0.4, 0.4, 0.2], &b_keys), mk_wb([0.2, 0.5, 0.3], &b_keys)];

        let p0_bar = average_weighted_actions(&p0_states);
        let p1_bar = average_weighted_actions(&p1_states);
        let q0_bar = average_weighted_actions(&q0_states);
        let q1_bar = average_weighted_actions(&q1_states);

        let v0 = [
            [0.61, 0.42, 0.77],
            [0.53, 0.29, 0.81],
            [0.44, 0.68, 0.35],
            [0.72, 0.51, 0.26],
        ];
        let v1 = [
            [0.61, 0.42, 0.77],
            [0.58, 0.25, 0.84],
            [0.63, 0.74, 0.49],
            [0.80, 0.55, 0.39],
        ];
        let lookup = |m: &[[f64; 3]; 4], a: &str, b: &str| -> f64 {
            let ai = a_keys.iter().position(|&k| k == a).unwrap();
            let bi = b_keys.iter().position(|&k| k == b).unwrap();
            m[ai][bi]
        };

        let mut true_u0 = 0.0;
        let mut true_u1 = 0.0;
        let mut true_p1_q0 = 0.0;
        for a in &p0_bar {
            for b in &q0_bar {
                true_u0 += a.prob * b.prob * lookup(&v0, &a.input, &b.input);
            }
        }
        for a in &p1_bar {
            for b in &q1_bar {
                true_u1 += a.prob * b.prob * lookup(&v1, &a.input, &b.input);
            }
            for b in &q0_bar {
                true_p1_q0 += a.prob * b.prob * lookup(&v1, &a.input, &b.input);
            }
        }
        let true_delta = true_u1 - true_u0;
        let true_adapt = true_u1 - true_p1_q0;

        let cmp = ArmComparison {
            diff_slots: vec![false, true, false, false, false, false],
            same_public_turn0_frame: false,
            same_levels_all_slots: true,
            same_party_order_matters: true,
            opp_belief_identical: false,
            opp_belief_singleton_arms: true,
            base_levels: vec![55, 55, 50, 50, 50, 50],
            base_immune_masks: vec![0; 6],
            cand_immune_masks: vec![0; 6],
        };

        let mut rng = SplitMix64::new(314159265);
        let n_mc = 80_000usize;
        let mut sum_neyman = 0.0;
        let mut sum_adapt = 0.0;
        let mut mean_tau_r1 = 0.0;
        let mut mean_tau_r4 = 0.0;

        for _ in 0..n_mc {
            // Draw `R = 4` independent preview states with omega_me ⊥ omega_opp.
            let mut r_p0 = Vec::with_capacity(4);
            let mut r_p1 = Vec::with_capacity(4);
            let mut r_q0 = Vec::with_capacity(4);
            let mut r_q1 = Vec::with_capacity(4);
            for _ in 0..4 {
                let s_me = rng.below(2);
                let s_opp = rng.below(2);
                r_p0.push(p0_states[s_me].clone());
                r_p1.push(p1_states[s_me].clone());
                r_q0.push(q0_states[s_opp].clone());
                r_q1.push(q1_states[s_opp].clone());
            }
            let plan_r1 = build_coupling_plan(&r_p0[0], &r_q0[0], &r_p1[0], &r_q1[0], &cmp);
            mean_tau_r1 += plan_r1.tau;

            let ap0 = average_weighted_actions(&r_p0);
            let ap1 = average_weighted_actions(&r_p1);
            let aq0 = average_weighted_actions(&r_q0);
            let aq1 = average_weighted_actions(&r_q1);
            let plan = build_coupling_plan(&ap0, &aq0, &ap1, &aq1, &cmp);
            mean_tau_r4 += plan.tau;

            let p1_inc = plan.tau * NEYMAN_SCALE_S1;
            let p2_inc = plan.m2_same_slot * NEYMAN_SCALE_S2;
            let p3_inc = plan.m3_belief * NEYMAN_SCALE_S3;
            assert!(p1_inc + p2_inc + p3_inc <= 1.0 + 1e-12);
            let u = rng.next_f64();
            let est = if p1_inc > 0.0 && u < p1_inc {
                let ((a1, b1), (a0, b0), _) =
                    sample_stratum1_coupled(&plan.nu1, &plan.nu0, &mut rng);
                (plan.tau / p1_inc) * (lookup(&v1, &a1, &b1) - lookup(&v0, &a0, &b0))
            } else if p2_inc > 0.0 && u < p1_inc + p2_inc {
                let (a2, b2) = sample_pair(&plan.stratum2, &mut rng);
                (plan.m2_same_slot / p2_inc) * (lookup(&v1, &a2, &b2) - lookup(&v0, &a2, &b2))
            } else if p3_inc > 0.0 && u < p1_inc + p2_inc + p3_inc {
                let (a3, b3) = sample_pair(&plan.stratum3_belief, &mut rng);
                (plan.m3_belief / p3_inc) * (lookup(&v1, &a3, &b3) - lookup(&v0, &a3, &b3))
            } else {
                0.0
            };
            sum_neyman += est;

            let (tau_q, eta1, eta0) = build_opp_shift_residuals(&aq0, &aq1);
            if tau_q > 0.0 && rng.next_f64() < tau_q {
                let p1_items: Vec<(String, f64)> =
                    ap1.iter().map(|x| (x.input.clone(), x.prob)).collect();
                let a = sample_weighted(&p1_items, &mut rng);
                let (b1, b0) = couple_picks_exact_then_lead(&eta1, &eta0, &mut rng);
                sum_adapt += lookup(&v1, &a, &b1) - lookup(&v1, &a, &b0);
            }
        }

        let est_delta = sum_neyman / n_mc as f64;
        let est_adapt = sum_adapt / n_mc as f64;
        assert!(
            (est_delta - true_delta).abs() < 0.0025,
            "Neyman + R=4 delta_u biased: est={est_delta:.6} true={true_delta:.6}"
        );
        assert!(
            (est_adapt - true_adapt).abs() < 0.0025,
            "single-draw + R=4 delta_adapt biased: est={est_adapt:.6} true={true_adapt:.6}"
        );
        assert!(
            mean_tau_r4 <= mean_tau_r1,
            "preview averaging R=4 should not inflate TV: r4={} r1={}",
            mean_tau_r4 / n_mc as f64,
            mean_tau_r1 / n_mc as f64
        );
    }
}
