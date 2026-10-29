//! Continuous-Abundance Evolutionary Ecosystem (`meta_eco`) & Decoupled Extractors.
//!
//! Replaces binary `(1+λ)` threshold replacement with Replicator-Mutator dynamics
//! over a living population where each genotype `i` carries a continuous abundance
//! `w_i > 0` (`sum_i w_i = 1`, 分布数) updated smoothly by empirical win rate:
//!
//!     w_i^{(t+1)} = w_i^{(t)} * exp(eta * (u_i - u_bar)) / Z
//!
//! Parents are NEVER overwritten by a single noisy child evaluation: budded mutants
//! enter the ecosystem alongside their parent with juvenile abundance `beta * w_parent`
//! while the parent retains `(1 - beta) * w_parent`. Cumulative game counts `n_i` and
//! wins `s_i` accumulate across waves, shrinking standard error `1 / sqrt(n_i)` for
//! persistent strains and letting neutral pre-adapted variants (`Rest` on 4th slot)
//! persist at small positive abundance until a 2nd mutation (`Sleep Talk`) lands.
//!
//! Extraction for actual battle deployment is completely decoupled from the
//! population dynamics via dedicated subcommands:
//!   - `evolve-island`: run continuous Replicator-Mutator waves on an island ledger
//!   - `extract-pool` : extract a diverse, high-LCB `team-pool` (Jaccard QD filter)
//!   - `extract-nash` : build cross-payoff matrix & solve exact Nash equilibrium
//!   - `extract-belief`: export abundance-weighted opponent sets

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::MetaPool;
use nc2000_bot::teamgen::{
    gauntlet_eval_weighted, team_key, to_sets, NicheConstraint, TeamGen, WeightedEvalCfg,
};
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::{toid, Dex};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NicheSpec {
    name: String,
    anchors: String,
    max_staples: String,
}

fn default_one() -> f64 {
    1.0
}

type PreviewSig = Vec<(String, i64, String, bool)>;

fn preview_sig(team: &[Value]) -> PreviewSig {
    let mut v: PreviewSig = team
        .iter()
        .map(|s| {
            let sp = toid(s["species"].as_str().unwrap_or(""));
            let lv = s["level"].as_i64().unwrap_or(50);
            let g = s["gender"].as_str().unwrap_or("").to_uppercase();
            let item = !toid(s["item"].as_str().unwrap_or("")).is_empty();
            (sp, lv, g, item)
        })
        .collect();
    v.sort();
    v
}

fn preview_sig_sets(sets: &[PokemonSet]) -> PreviewSig {
    let mut v: PreviewSig = sets
        .iter()
        .map(|s| {
            let sp = toid(&s.species);
            let lv = s.level as i64;
            let g = s.gender.as_deref().unwrap_or("").to_uppercase();
            let item = !toid(&s.item).is_empty();
            (sp, lv, g, item)
        })
        .collect();
    v.sort();
    v
}

/// Normalize a `MetaPool` in-memory so that any catalog team whose special-only Pokemon
/// had their Attack DV / gender optimized (e.g. `M -> F`) or had a dominated item upgraded
/// is ALSO present in its canonicalized form inside `mp.teams`. This guarantees 100%
/// Team Preview recognition by `Belief::build_refs` (`roster[i].gender == mo.gender`)
/// without altering the frozen `data/belief-pool-v3/belief-pool.json` SHA-256 on disk.
fn normalize_belief_pool(dex: &Dex, gen: &TeamGen, mp: &MetaPool) -> MetaPool {
    let mut out = mp.clone();
    let mut rng = SplitMix64::new(0xC0DE_BA5E_2026_1029);
    let mut extras = Vec::new();
    for t in &mp.teams {
        let Ok(v) = serde_json::to_value(&t.sets) else {
            continue;
        };
        let Some(arr) = v.as_array() else {
            continue;
        };
        let mut upgraded = arr.clone();
        gen.optimize_team_dominance(dex, &mut upgraded, &mut rng);
        let Some(canon) = gen.canonize(dex, &upgraded) else {
            continue;
        };
        let Ok(c_sets) = to_sets(&canon) else {
            continue;
        };
        let diff_gender_or_item = t
            .sets
            .iter()
            .zip(c_sets.iter())
            .any(|(a, b)| a.gender != b.gender || a.item.is_empty() != b.item.is_empty());
        if diff_gender_or_item {
            extras.push(nc2000_bot::preview::MetaTeam {
                id: format!("{}-opt", t.id),
                sets: c_sets,
                weight: t.weight,
            });
        }
    }
    out.teams.extend(extras);
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Strain {
    id: String,
    parent_id: String,
    niche: String,
    anchors: String,
    max_staples: String,
    born_wave: u32,
    last_op: String,
    abundance: f64,
    cum_games: f64,
    cum_wins: f64,
    bench_games: f64,
    bench_wins: f64,
    eco_games: f64,
    eco_wins: f64,
    last_wave_fit: f64,
    parent_prior: f64,
    bayes_fit: f64,
    lcb95: f64,
    #[serde(default)]
    slot_picks: [f64; 6],
    #[serde(default)]
    slot_wins: [f64; 6],
    #[serde(default)]
    slot_marginals: [f64; 6],
    #[serde(default)]
    slot_wars: [f64; 6],
    #[serde(default)]
    move_dqs: [[f64; 4]; 6],
    #[serde(default)]
    last_wave_slot_picks: [f64; 6],
    #[serde(default)]
    last_wave_slot_marginals: [f64; 6],
    #[serde(default)]
    effective_cores: f64,
    #[serde(default)]
    symbiont_of: Option<String>,
    #[serde(default = "default_one")]
    mix_alpha: f64,
    #[serde(default)]
    deme_bonus: f64,
    team: Vec<Value>,
}

impl Strain {
    fn recompute_stats(&mut self, prior_games: f64) {
        let raw = if self.cum_games > 0.0 {
            self.cum_wins / self.cum_games
        } else {
            self.parent_prior
        };
        self.bayes_fit =
            (self.cum_wins + prior_games * self.parent_prior) / (self.cum_games + prior_games);
        let eff_n = self.cum_games.max(1.0);
        let p = self.bayes_fit.clamp(0.02, 0.98);
        let se = (p * (1.0 - p) / eff_n).sqrt();
        self.lcb95 = (raw - 1.96 * se).max(0.0);
    }

    fn species_set(&self) -> BTreeSet<String> {
        self.team
            .iter()
            .filter_map(|s| s["species"].as_str().map(toid))
            .collect()
    }
}

/// Endogenously calibrate the Replacement Level `\hat{\mu}_{pop}(L)` as a function of
/// slot level `L \in [50, 55]` via OLS regression over all evaluated slots in `strains`
/// (zero hand-coded level prior). Updates `s.slot_wars[i] = \Delta V_{-i} - \hat{\mu}_{pop}(L_i)`
/// and returns `(mu_50, mu_55, beta_l)`.
fn recompute_population_wars(strains: &mut [Strain]) -> (f64, f64, f64) {
    let mut sum_l = 0.0f64;
    let mut sum_m = 0.0f64;
    let mut n = 0usize;
    for s in strains.iter().filter(|s| s.cum_games > 0.0) {
        let g = s.cum_games.max(1.0);
        for i in 0..s.team.len().min(6) {
            let lv = s.team[i]["level"].as_u64().unwrap_or(50) as f64;
            let m = s.slot_marginals[i] / g;
            sum_l += lv;
            sum_m += m;
            n += 1;
        }
    }
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean_l = sum_l / (n as f64);
    let mean_m = sum_m / (n as f64);
    let mut cov_lm = 0.0f64;
    let mut var_l = 0.0f64;
    for s in strains.iter().filter(|s| s.cum_games > 0.0) {
        let g = s.cum_games.max(1.0);
        for i in 0..s.team.len().min(6) {
            let dl = (s.team[i]["level"].as_u64().unwrap_or(50) as f64) - mean_l;
            let dm = (s.slot_marginals[i] / g) - mean_m;
            cov_lm += dl * dm;
            var_l += dl * dl;
        }
    }
    let beta_l = if var_l > 1e-6 {
        cov_lm / var_l
    } else {
        0.0
    };
    for s in strains.iter_mut().filter(|s| s.cum_games > 0.0) {
        let g = s.cum_games.max(1.0);
        for i in 0..s.team.len().min(6) {
            let lv = s.team[i]["level"].as_u64().unwrap_or(50) as f64;
            let m = s.slot_marginals[i] / g;
            let rep_lv = mean_m + beta_l * (lv - mean_l);
            s.slot_wars[i] = m - rep_lv;
        }
    }
    let mu_50 = mean_m + beta_l * (50.0 - mean_l);
    let mu_55 = mean_m + beta_l * (55.0 - mean_l);
    (mu_50, mu_55, beta_l)
}

fn build_endogenous_pool(strains: &[Strain]) -> nc2000_bot::teamgen::EndogenousPool {
    let mut pool = nc2000_bot::teamgen::EndogenousPool::default();
    for s in strains.iter().filter(|s| s.cum_games > 0.0) {
        let g = s.cum_games.max(1.0);
        for i in 0..s.team.len().min(6) {
            let pick_rate = (s.slot_picks[i] / g).clamp(0.0, 1.0);
            pool.record_slot(&s.team[i], s.slot_wars[i], pick_rate, &s.move_dqs[i]);
        }
    }
    pool
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WaveSummary {
    wave: u32,
    mean_payoff: f64,
    top_strain_id: String,
    top_abundance: f64,
    top_bayes_fit: f64,
    top_lcb95: f64,
    active_count: usize,
    archived_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct IslandLedger {
    island: String,
    wave: u32,
    next_strain_seq: u64,
    rng: u64,
    niches: Vec<NicheSpec>,
    active_strains: Vec<Strain>,
    archived_strains: Vec<Strain>,
    wave_history: Vec<WaveSummary>,
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    flag(args, name)
        .map(|v| v.parse().expect(name))
        .unwrap_or(default)
}

fn build_constraint(dex: &Dex, anchors_str: &str, max_staples_str: &str) -> (Vec<String>, NicheConstraint) {
    let mut constraint = NicheConstraint::default();
    let mut anchor_list: Vec<String> = Vec::new();
    if !anchors_str.is_empty() {
        for sp in anchors_str.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let sid = toid(sp);
            assert!(dex.species.id(&sid).is_some(), "unknown anchor species: {sp}");
            anchor_list.push(sid);
        }
        constraint.anchor_slots = anchor_list.len();
    }
    if !max_staples_str.is_empty() {
        let (sp_str, cap_str) = max_staples_str
            .split_once(':')
            .expect("max_staples wants SPECIES_LIST:MAX");
        let set: BTreeSet<String> = sp_str
            .split(',')
            .map(toid)
            .filter(|s| !s.is_empty())
            .collect();
        let cap: usize = cap_str.parse().expect("max_staples cap must be integer");
        constraint.staple_cap = Some((set, cap));
    }
    (anchor_list, constraint)
}

fn parse_agent_spec(
    args: &[String],
    root: &Path,
    dex: &Dex,
    gen: &TeamGen,
) -> (u32, f64, Option<Arc<MetaPool>>, bool) {
    if let Some(spec) = flag(args, "--agent") {
        let parts: Vec<&str> = spec.split(':').collect();
        assert!(
            parts.len() >= 2,
            "--agent expects KIND:ITERS[:C], got {spec}"
        );
        let kind = parts[0];
        let it: u32 = parts[1].parse().expect("agent iters");
        let c: f64 = parts.get(2).map(|s| s.parse().expect("agent c")).unwrap_or(0.4);
        match kind {
            "open" => (it, c, None, true),
            "blind" => {
                let bpath = flag(args, "--belief-pool")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| root.join("data/belief-pool-v3/belief-pool.json"));
                let btext = std::fs::read_to_string(&bpath)
                    .unwrap_or_else(|e| panic!("read belief-pool {}: {e}", bpath.display()));
                let mp: MetaPool = serde_json::from_str(&btext).expect("parse belief-pool");
                let norm_mp = normalize_belief_pool(dex, gen, &mp);
                (it, c, Some(Arc::new(norm_mp)), false)
            }
            "skuct" => (it, c, None, false),
            other => panic!("unsupported --agent kind {other} (expected open, blind, or skuct)"),
        }
    } else {
        (1000, 0.4, None, true)
    }
}

fn load_gauntlet_file(dex: &Dex, gen: &TeamGen, gfile: &str) -> (Vec<Vec<PokemonSet>>, Vec<f64>) {
    let gtext = std::fs::read_to_string(gfile).unwrap_or_else(|e| panic!("{gfile}: {e}"));
    let gv: Value = serde_json::from_str(&gtext).unwrap();
    let mut teams = Vec::new();
    let mut weights = Vec::new();
    for t in gv["teams"].as_array().expect("gauntlet-file: missing teams") {
        let raw = t["sets"].as_array().expect("gauntlet-file team missing sets");
        let canon = gen.canonize(dex, raw).expect("gauntlet-file team invalid");
        teams.push(to_sets(&canon).unwrap());
        let w = t["weight"]
            .as_f64()
            .or_else(|| t["drawWeight"].as_f64())
            .unwrap_or(1.0);
        weights.push(w);
    }
    let wsum: f64 = weights.iter().sum();
    if wsum > 0.0 {
        for w in &mut weights {
            *w /= wsum;
        }
    }
    (teams, weights)
}

fn write_json_atomic<T: Serialize>(path: &Path, val: &T) {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(val).unwrap()).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

fn parse_niches(spec: &str) -> Vec<NicheSpec> {
    // Format: name|anchors|max_staples;name2|anchors2|max_staples2
    let mut out = Vec::new();
    for part in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let fields: Vec<&str> = part.split('|').collect();
        let name = fields.first().copied().unwrap_or("rand").to_string();
        let anchors = fields.get(1).copied().unwrap_or("").to_string();
        let max_staples = fields.get(2).copied().unwrap_or("").to_string();
        out.push(NicheSpec {
            name,
            anchors,
            max_staples,
        });
    }
    if out.is_empty() {
        out.push(NicheSpec {
            name: "rand".into(),
            anchors: String::new(),
            max_staples: String::new(),
        });
    }
    out
}

/// Try to load matching seed teams from `--seed-dir` (`lineage-*.json` or `island-*.json`)
/// if available and upgrade any dominated moves/items via `optimize_team_dominance`.
fn import_seed_teams_from_dir(
    dex: &Dex,
    gen: &TeamGen,
    seed_dir: &Path,
    niche: &NicheSpec,
    rng: &mut SplitMix64,
) -> Vec<Vec<Value>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(seed_dir) else {
        return out;
    };
    let (anchors, constraint) = build_constraint(dex, &niche.anchors, &niche.max_staples);
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|x| x.path())).collect();
    paths.sort();

    let mut raw_candidates: Vec<(f64, Vec<Value>)> = Vec::new();
    for p in paths {
        if p.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(txt) = std::fs::read_to_string(&p) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(&txt) else {
            continue;
        };
        // Support both IslandLedger (`active_strains` / `archived_strains`) and R1 checkpoints (`parent` / `sets`)
        if let Some(act) = v["active_strains"].as_array() {
            let empty = Vec::new();
            let arch = v["archived_strains"].as_array().unwrap_or(&empty);
            for s in act.iter().chain(arch.iter()) {
                if s["cum_games"].as_f64().unwrap_or(0.0) < 40.0 {
                    continue;
                }
                if let Some(t) = s["team"].as_array() {
                    let bf = s["bayes_fit"].as_f64().unwrap_or(0.35);
                    raw_candidates.push((bf, t.clone()));
                }
            }
        } else if let Some(raw_arr) = v["parent"].as_array().or_else(|| v["sets"].as_array()) {
            let bf = v["best_fit"].as_f64().unwrap_or(0.40);
            raw_candidates.push((bf, raw_arr.clone()));
        }
    }
    // Sort highest fitness first
    raw_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen_keys: BTreeSet<String> = BTreeSet::new();
    for (_bf, mut raw) in raw_candidates {
        if !anchors.is_empty() {
            // Require that all anchor species already appear somewhere in the team
            let team_sp: BTreeSet<String> = raw
                .iter()
                .filter_map(|s| s["species"].as_str().map(toid))
                .collect();
            if !anchors.iter().all(|a| team_sp.contains(a)) {
                continue;
            }
            // Place anchor species at leading slots 0..anchors.len()
            for (i, a) in anchors.iter().enumerate() {
                if let Some(pos) = raw
                    .iter()
                    .position(|s| s["species"].as_str().map(toid).as_deref() == Some(a.as_str()))
                {
                    raw.swap(i, pos);
                }
            }
        }
        gen.optimize_team_dominance(dex, &mut raw, rng);
        if !constraint.allows_team(&raw, None) {
            continue;
        }
        if let Some(canon) = gen.canonize(dex, &raw) {
            let k = team_key(&canon);
            if seen_keys.insert(k) {
                out.push(canon);
            }
        }
    }
    out
}

fn cmd_evolve_island(args: &[String]) {
    let t_run_start = std::time::Instant::now();
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let island = flag(args, "--island").unwrap_or_else(|| "island-00".into());
    let out_dir = flag(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/eco-v1"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let ledger_path = out_dir.join(format!("{island}.json"));

    let waves: u32 = num(args, "--waves", 9999);
    let max_seconds: u64 = num(args, "--max-seconds", 0);
    let carry: usize = num(args, "--carry", 8);
    let buds: usize = num(args, "--buds", 4);
    let eval_games: u32 = num(args, "--eval-games", 8);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let eta: f64 = num(args, "--eta", 5.0);
    let prior_games: f64 = num(args, "--prior-games", 32.0);
    let _bud_share: f64 = num(args, "--bud-share", 0.35);
    let env_self_weight: f64 = num(args, "--env-self-weight", 0.70);
    let base_seed: u64 = num(args, "--seed", 20261028);
    let resume = args.iter().any(|a| a == "--resume");
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let gfile = flag(args, "--gauntlet-file")
        .unwrap_or_else(|| root.join("data/meta-nash-v3/pool-artifact.json").to_string_lossy().into_owned());
    let (bench_teams, bench_weights) = load_gauntlet_file(&dex, &gen, &gfile);
    let (agent_iters, agent_c, blind_pool, open_sheet) = parse_agent_spec(args, &root, &dex, &gen);
    let max_per_archetype: usize = num(args, "--max-per-archetype", 2);

    let niches_spec = flag(args, "--niches").unwrap_or_else(|| "rand||".into());
    let niches = parse_niches(&niches_spec);

    let seed_dir = flag(args, "--seed-dir")
        .filter(|s| !s.is_empty() && s != "none")
        .map(PathBuf::from);
    let reset_bench = args.iter().any(|a| a == "--reset-bench");

    // Check if a matching same-island ledger exists in `--seed-dir` (e.g. `island-r4-07.json`
    // for `island-r5-07`) to continue the isolated island ecosystem across epochs.
    let prev_island_ledger: Option<IslandLedger> = if !(resume && ledger_path.exists()) {
        seed_dir.as_ref().and_then(|sdir| {
            let suffix = island.rsplit('-').next().unwrap_or("");
            let entries = std::fs::read_dir(sdir).ok()?;
            for e in entries.flatten() {
                let p = e.path();
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if !suffix.is_empty() && stem.ends_with(&format!("-{suffix}")) {
                    if let Ok(txt) = std::fs::read_to_string(&p) {
                        if let Ok(mut prev) = serde_json::from_str::<IslandLedger>(&txt) {
                            prev.island = island.clone();
                            if reset_bench {
                                prev.archived_strains.clear();
                            }
                            let mut upgrade_rng = SplitMix64::new(prev.rng ^ 0xD0AD_CAFE_u64);
                            for s in prev
                                .active_strains
                                .iter_mut()
                                .chain(prev.archived_strains.iter_mut())
                            {
                                s.anchors.clear();
                                s.max_staples.clear();
                                let mut upgraded = s.team.clone();
                                gen.optimize_team_dominance(&dex, &mut upgraded, &mut upgrade_rng);
                                if let Some(canon) = gen.canonize(&dex, &upgraded) {
                                    s.team = canon;
                                }
                                if reset_bench {
                                    s.bench_games = 0.0;
                                    s.bench_wins = 0.0;
                                    s.eco_games *= 0.5;
                                    s.eco_wins *= 0.5;
                                    s.cum_games = s.eco_games;
                                    s.cum_wins = s.eco_wins;
                                } else {
                                    // Anchor cross-epoch continuation to decayed Master Pool benchmark
                                    // performance (`bench_games`, `bench_wins`), purging historical
                                    // `eco_wins` inflation from extinct local punching bags while
                                    // allowing `env_self_weight` co-evolution within the new epoch.
                                    s.bench_games *= 0.5;
                                    s.bench_wins *= 0.5;
                                    s.eco_games = 0.0;
                                    s.eco_wins = 0.0;
                                    s.cum_games = s.bench_games;
                                    s.cum_wins = s.bench_wins;
                                    if s.bench_games > 0.0 {
                                        s.parent_prior =
                                            (s.bench_wins / s.bench_games).clamp(0.15, 0.85);
                                    }
                                }
                                s.recompute_stats(prior_games);
                            }
                            eprintln!(
                                "[{island}] continuing isolated ecosystem from {} (wave={}, active={}, reset_bench={})",
                                p.display(),
                                prev.wave,
                                prev.active_strains.len(),
                                reset_bench
                            );
                            return Some(prev);
                        }
                    }
                }
            }
            None
        })
    } else {
        None
    };

    let mut ledger: IslandLedger = if resume && ledger_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&ledger_path).unwrap()).unwrap()
    } else if let Some(prev) = prev_island_ledger {
        write_json_atomic(&ledger_path, &prev);
        prev
    } else {
        let mut rng = SplitMix64::new(base_seed ^ 0xEC05_757E_u64);
        let mut active: Vec<Strain> = Vec::new();
        let mut seq = 1u64;

        let target_init = carry + buds;
        for idx in 0..target_init {
            let niche = &niches[idx % niches.len()];
            let slot_in_niche = idx / niches.len();
            let (anchors, constraint) = build_constraint(&dex, &niche.anchors, &niche.max_staples);
            let mut team_opt: Option<Vec<Value>> = None;
            if let Some(ref sdir) = seed_dir {
                let imported = import_seed_teams_from_dir(&dex, &gen, sdir, niche, &mut rng);
                if !imported.is_empty() {
                    let pick_idx = (slot_in_niche + ((rng.next_f64() * (imported.len() as f64)) as usize)) % imported.len();
                    let chosen = &imported[pick_idx];
                    if slot_in_niche == 0 {
                        team_opt = Some(chosen.clone());
                    } else if let Some(prop) =
                        gen.propose_constrained(&dex, chosen, &mut rng, &constraint, 150)
                    {
                        team_opt = Some(prop.team);
                    }
                }
            }
            let team = team_opt.unwrap_or_else(|| {
                gen.random_team_constrained(&dex, &mut rng, &anchors, &constraint, 300)
                    .expect("failed to generate initial constrained team")
            });
            let id = format!("{island}-s{seq:04}");
            seq += 1;
            active.push(Strain {
                id,
                parent_id: "seed".into(),
                niche: niche.name.clone(),
                anchors: niche.anchors.clone(),
                max_staples: niche.max_staples.clone(),
                born_wave: 0,
                last_op: "init".into(),
                abundance: 1.0 / (target_init as f64),
                cum_games: 0.0,
                cum_wins: 0.0,
                bench_games: 0.0,
                bench_wins: 0.0,
                eco_games: 0.0,
                eco_wins: 0.0,
                last_wave_fit: 0.35,
                parent_prior: 0.35,
                bayes_fit: 0.35,
                lcb95: 0.0,
                slot_picks: [0.0; 6],
                slot_wins: [0.0; 6],
                slot_marginals: [0.0; 6],
                slot_wars: [0.0; 6],
                move_dqs: [[0.0; 4]; 6],
                last_wave_slot_picks: [0.0; 6],
                last_wave_slot_marginals: [0.0; 6],
                effective_cores: 1.0,
                symbiont_of: None,
                mix_alpha: 1.0,
                deme_bonus: 0.0,
                team,
            });
        }
        let l = IslandLedger {
            island: island.clone(),
            wave: 0,
            next_strain_seq: seq,
            rng: rng.0,
            niches,
            active_strains: active,
            archived_strains: Vec::new(),
            wave_history: Vec::new(),
        };
        write_json_atomic(&ledger_path, &l);
        l
    };

    let mut rng = SplitMix64(ledger.rng);
    let mut waves_this_run = 0u32;
    let mut avg_wave_secs = 120.0f64;

    while ledger.wave < waves {
        if max_seconds > 0 && waves_this_run > 0 {
            let elapsed = t_run_start.elapsed().as_secs_f64();
            if elapsed + avg_wave_secs * 0.65 >= max_seconds as f64 {
                eprintln!(
                    "[{island}] wall-clock budget reached ({elapsed:.0}s / {max_seconds}s after {waves_this_run} waves, total wave={}); stopping cleanly.",
                    ledger.wave
                );
                break;
            }
        }

        let w = ledger.wave + 1;
        let t_wave = std::time::Instant::now();
        let wave_seed = base_seed ^ (w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

        // If all active strains were already evaluated in a previous wave (e.g. when resuming
        // after a post-eval checkpoint), run per-niche survivor selection + budding first.
        if w > 1 && ledger.active_strains.iter().all(|s| s.cum_games > 0.0) {
            bud_next_wave_by_niche(
                &dex,
                &gen,
                &island,
                w - 1,
                carry,
                buds,
                max_per_archetype,
                &mut ledger,
                &mut rng,
            );
        }

        // Build co-evolutionary opponent pool from the top 4 structurally distinct archetypes
        // (< 4 shared species) on the island.
        let mut vet_indices: Vec<usize> = (0..ledger.active_strains.len())
            .filter(|&i| ledger.active_strains[i].cum_games > 0.0)
            .collect();
        if vet_indices.is_empty() {
            vet_indices = (0..ledger.active_strains.len()).collect();
        }
        vet_indices.sort_by(|&a, &b| {
            ledger.active_strains[b]
                .bayes_fit
                .partial_cmp(&ledger.active_strains[a].bayes_fit)
                .unwrap()
        });
        let mut chosen_vets: Vec<usize> = Vec::new();
        for &vi in &vet_indices {
            if chosen_vets.len() >= 4 {
                break;
            }
            let sp_i = ledger.active_strains[vi].species_set();
            let distinct = chosen_vets.iter().all(|&vj| {
                let sp_j = ledger.active_strains[vj].species_set();
                sp_i.intersection(&sp_j).count() < 4
            });
            if distinct {
                chosen_vets.push(vi);
            }
        }
        for vi in vet_indices {
            if chosen_vets.len() >= 4 {
                break;
            }
            if !chosen_vets.contains(&vi) {
                chosen_vets.push(vi);
            }
        }

        let mut eco_opps: Vec<Vec<PokemonSet>> = Vec::new();
        let mut eco_weights: Vec<f64> = Vec::new();
        for &vi in &chosen_vets {
            if let Ok(sets) = to_sets(&ledger.active_strains[vi].team) {
                eco_opps.push(sets);
                eco_weights.push(ledger.active_strains[vi].abundance.max(0.05));
            }
        }
        let eco_wsum: f64 = eco_weights.iter().sum();
        if eco_wsum > 0.0 {
            for ew in &mut eco_weights {
                *ew /= eco_wsum;
            }
        }

        // Rotate an 8-team slice of the Master Pool each wave so every 3 waves a surviving
        // strain faces all 24 teams of the Master Pool without bloating per-wave wall-clock.
        let slice_len = bench_teams.len().min(8);
        let start_idx = ((w as usize).saturating_sub(1) * slice_len) % bench_teams.len().max(1);
        let mut wave_bench_teams = Vec::with_capacity(slice_len);
        let mut wave_bench_weights = Vec::with_capacity(slice_len);
        for k in 0..slice_len {
            let idx = (start_idx + k) % bench_teams.len();
            wave_bench_teams.push(bench_teams[idx].clone());
            wave_bench_weights.push(bench_weights[idx]);
        }
        let wb_sum: f64 = wave_bench_weights.iter().sum();
        if wb_sum > 0.0 {
            for wv in &mut wave_bench_weights {
                *wv /= wb_sum;
            }
        }

        let bench_games_per_opp =
            ((eval_games as usize * 3) / slice_len.max(1)).clamp(2, 6) as u32;
        let bench_n_games = (bench_games_per_opp as usize * slice_len) as f64;
        let eco_n_games = (eval_games as usize * eco_opps.len()) as f64;
        let total_w_games =
            bench_n_games + if eco_opps.is_empty() { 0.0 } else { eco_n_games };

        let pre_wave_bayes: BTreeMap<String, f64> = ledger
            .active_strains
            .iter()
            .filter(|s| s.cum_games > 0.0)
            .map(|s| (s.id.clone(), s.bayes_fit))
            .collect();

        // Evaluate all active strains concurrently across a single wave barrier so a single
        // long-turn stall game on one strain never idles the other 15 CPU cores across 24 barriers.
        let inner_threads = (threads / 4).max(2);
        let dex_ref = &dex;
        let wb_teams_ref = &wave_bench_teams;
        let wb_weights_ref = &wave_bench_weights;
        let eco_opps_ref = &eco_opps;
        let eco_weights_ref = &eco_weights;
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for s in &mut ledger.active_strains {
                let bp = blind_pool.clone();
                handles.push(scope.spawn(move || {
                    let cand = to_sets(&s.team).expect("valid strain team");
                    let bench_res = gauntlet_eval_weighted(
                        dex_ref,
                        &cand,
                        wb_teams_ref,
                        wb_weights_ref,
                        &WeightedEvalCfg {
                            games_per_opponent: bench_games_per_opp,
                            agent_iters,
                            c: agent_c,
                            open_sheet,
                            max_turns,
                            seed: wave_seed,
                            threads: inner_threads,
                            blind_pool: bp.clone(),
                        },
                    );
                    let bench_fit = bench_res.score;
                    s.bench_games += bench_n_games;
                    s.bench_wins += bench_fit * bench_n_games;

                    let mut wave_picks = [0.0f64; 6];
                    let mut wave_wins = [0.0f64; 6];
                    for k in 0..6 {
                        wave_picks[k] += bench_res.slot_picks[k] * bench_n_games;
                        wave_wins[k] += bench_res.slot_wins[k] * bench_n_games;
                    }
                    s.effective_cores = bench_res.effective_cores();

                    let wave_fit = if !eco_opps_ref.is_empty() && env_self_weight > 0.0 {
                        let eco_res = gauntlet_eval_weighted(
                            dex_ref,
                            &cand,
                            eco_opps_ref,
                            eco_weights_ref,
                            &WeightedEvalCfg {
                                games_per_opponent: eval_games,
                                agent_iters,
                                c: agent_c,
                                open_sheet,
                                max_turns,
                                seed: wave_seed ^ 0x5555_AAAA_5555_AAAA,
                                threads: inner_threads,
                                blind_pool: bp,
                            },
                        );
                        let eco_fit = eco_res.score;
                        s.eco_games += eco_n_games;
                        s.eco_wins += eco_fit * eco_n_games;
                        for k in 0..6 {
                            wave_picks[k] += eco_res.slot_picks[k] * eco_n_games;
                            wave_wins[k] += eco_res.slot_wins[k] * eco_n_games;
                        }
                        s.effective_cores = (1.0 - env_self_weight) * bench_res.effective_cores()
                            + env_self_weight * eco_res.effective_cores();
                        (1.0 - env_self_weight) * bench_fit + env_self_weight * eco_fit
                    } else {
                        bench_fit
                    };

                    for k in 0..6 {
                        s.slot_picks[k] += wave_picks[k];
                        s.slot_wins[k] += wave_wins[k];
                        s.last_wave_slot_picks[k] = wave_picks[k] / total_w_games.max(1.0);
                    }
                    s.last_wave_fit = wave_fit;
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });

        // Common-Random-Numbers (CRN) Paired Slice-Deconfounding + Active-Pick Deconfounding:
        // 1. When a young bud (age <= 3 waves) and its parent are evaluated in the same wave on
        //    the exact same 8-team benchmark slice, eco opponents, and `wave_seed`, use the parent's
        //    wave score as a control variate so rotating slice difficulty never masks true improvements.
        // 2. Active-Pick Deconfounding: if a 1-wave-old bud mutated a single slot `k` (`op@k` other
        //    than `level-tweak`) and slot `k` was picked in ZERO games this wave (`last_wave_slot_picks[k] == 0`),
        //    the mutated Pokemon never stepped onto the field, so any positive `s.last_wave_fit > p_wave`
        //    is pure MCTS noise on the unchanged 5 slots and is clamped to `0.0` gain.
        let wave_fit_by_id: BTreeMap<String, f64> = ledger
            .active_strains
            .iter()
            .map(|s| (s.id.clone(), s.last_wave_fit))
            .collect();
        for s in &mut ledger.active_strains {
            let eff_wave_fit = if s.born_wave > 0 && w.saturating_sub(s.born_wave) <= 3 {
                if let (Some(&p_bayes), Some(&p_wave)) = (
                    pre_wave_bayes.get(&s.parent_id),
                    wave_fit_by_id.get(&s.parent_id),
                ) {
                    s.parent_prior = p_bayes;
                    let mut delta = s.last_wave_fit - p_wave;
                    if w.saturating_sub(s.born_wave) == 1 && !s.last_op.starts_with("level-tweak") {
                        if let Some((_, slot_str)) = s.last_op.split_once('@') {
                            if let Ok(slot_idx) = slot_str.parse::<usize>() {
                                if slot_idx < 6 && s.last_wave_slot_picks[slot_idx] <= 1e-6 && delta > 0.0 {
                                    delta = 0.0;
                                }
                            }
                        }
                    }
                    (p_bayes + delta).clamp(0.0, 1.0)
                } else {
                    s.last_wave_fit
                }
            } else {
                s.last_wave_fit
            };
            s.cum_games += total_w_games;
            s.cum_wins += eff_wave_fit * total_w_games;
            s.last_wave_fit = eff_wave_fit;
            s.recompute_stats(prior_games);
        }

        // Island-Wide Damped Boltzmann-Replicator Update on continuous abundance w_i:
        // Damped replicator dynamics w_i <- w_i^0.5 * exp(eta * (u_i - u_bar))
        // lets stronger archetypes and offspring grow smoothly across the whole island
        // while structural diversity is guaranteed by the <= 2 per-archetype survivor cap.
        let mut w_sum = 0.0;
        let mut u_bar = 0.0;
        for s in &ledger.active_strains {
            let u = 0.35 * s.last_wave_fit + 0.65 * s.bayes_fit;
            u_bar += s.abundance * u;
            w_sum += s.abundance;
        }
        if w_sum > 0.0 {
            u_bar /= w_sum;
        }
        let mut new_w_sum = 0.0;
        for s in &mut ledger.active_strains {
            let u = 0.35 * s.last_wave_fit + 0.65 * s.bayes_fit;
            s.abundance = (s.abundance.max(1e-4).sqrt() * (eta * (u - u_bar)).exp()).max(1e-6);
            new_w_sum += s.abundance;
        }
        for s in &mut ledger.active_strains {
            s.abundance /= new_w_sum.max(1e-9);
        }

        let u_bar: f64 = ledger
            .active_strains
            .iter()
            .map(|s| 0.35 * s.last_wave_fit + 0.65 * s.bayes_fit)
            .sum::<f64>()
            / (ledger.active_strains.len().max(1) as f64);

        // Sort active strains by Empirical-Bayes fitness (`bayes_fit`) then abundance
        ledger.active_strains.sort_by(|a, b| {
            b.bayes_fit
                .partial_cmp(&a.bayes_fit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let wave_secs = t_wave.elapsed().as_secs_f64();
        waves_this_run += 1;
        avg_wave_secs = if waves_this_run == 1 {
            wave_secs
        } else {
            0.7 * avg_wave_secs + 0.3 * wave_secs
        };

        let top = &ledger.active_strains[0];
        eprintln!(
            "[{island} wave {w}] ({wave_secs:.0}s, elapsed={:.0}s) u_bar={u_bar:.3} | top={} (niche={}, w={:.3}, wave={:.3}, bayes={:.3}, lcb95={:.3}, games={:.0})",
            t_run_start.elapsed().as_secs_f64(),
            top.id,
            top.niche,
            top.abundance,
            top.last_wave_fit,
            top.bayes_fit,
            top.lcb95,
            top.cum_games,
        );

        ledger.wave_history.push(WaveSummary {
            wave: w,
            mean_payoff: u_bar,
            top_strain_id: top.id.clone(),
            top_abundance: top.abundance,
            top_bayes_fit: top.bayes_fit,
            top_lcb95: top.lcb95,
            active_count: ledger.active_strains.len(),
            archived_count: ledger.archived_strains.len(),
        });

        ledger.wave = w;
        ledger.rng = rng.0;
        write_json_atomic(&ledger_path, &ledger);
    }
}

fn count_same_archetype(sp: &BTreeSet<String>, group: &[Strain]) -> usize {
    group
        .iter()
        .filter(|other| sp.intersection(&other.species_set()).count() >= 4)
        .count()
}

fn bud_next_wave_by_niche(
    dex: &Dex,
    gen: &TeamGen,
    island: &str,
    born_wave: u32,
    carry: usize,
    buds: usize,
    max_per_archetype: usize,
    ledger: &mut IslandLedger,
    rng: &mut SplitMix64,
) {
    recompute_population_wars(&mut ledger.active_strains);
    let mut seen_keys: BTreeSet<String> = BTreeSet::new();

    // Island-wide selection: separate active strains into 1-wave-old grace-period buds
    // and veterans, clearing any legacy founder niche restrictions (`anchors`, `max_staples`).
    let mut grace_buds: Vec<Strain> = Vec::new();
    let mut veterans: Vec<Strain> = Vec::new();
    for mut s in ledger.active_strains.drain(..) {
        s.anchors.clear();
        s.max_staples.clear();
        if s.born_wave > 0 && s.born_wave + 1 >= born_wave {
            grace_buds.push(s);
        } else {
            veterans.push(s);
        }
    }

    veterans.sort_by(|a, b| {
        let qa = a.bayes_fit + a.deme_bonus;
        let qb = b.bayes_fit + b.deme_bonus;
        qb.partial_cmp(&qa)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    grace_buds.sort_by(|a, b| {
        let qa = a.bayes_fit + a.deme_bonus;
        let qb = b.bayes_fit + b.deme_bonus;
        qb.partial_cmp(&qa)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_vet_fit = veterans
        .first()
        .map(|s| s.bayes_fit + s.deme_bonus)
        .unwrap_or(0.35);

    // Separate certified Symbionts (`symbiont_of.is_some()`) from Primary Deme candidates
    // so Symbionts ride along with their Primary Deme and never consume a `carry` slot
    // or evict their parent Primary.
    let mut symbiont_pool: BTreeMap<String, Strain> = BTreeMap::new();
    let mut primary_veterans: Vec<Strain> = Vec::new();
    for s in veterans {
        if let Some(ref pid) = s.symbiont_of {
            symbiont_pool.entry(pid.clone()).or_insert(s);
        } else {
            primary_veterans.push(s);
        }
    }
    let mut primary_grace: Vec<Strain> = Vec::new();
    for s in grace_buds {
        if let Some(ref pid) = s.symbiont_of {
            symbiont_pool.entry(pid.clone()).or_insert(s);
        } else {
            primary_grace.push(s);
        }
    }

    // 1. Select up to `carry` Primary Demes island-wide subject to:
    //    - Same-Signature Deme Uniqueness: at most 1 Primary per `preview_sig` (`seen_sigs`),
    //      so a MoveSwap/ItemSwap child with the same preview signature competes for Primary
    //      of its Deme rather than duplicating or evicting another Deme.
    //    - Viability floor: veterans with >= 20 bench games that have < 26% Master Pool
    //      winrate or trail `top_vet_fit` by > 0.14 (dead founder lineages) are evicted.
    //    - Structural Archetype Cap: at most `max_per_archetype` (2) survivors may share >= 4 species.
    let mut seen_sigs: BTreeSet<PreviewSig> = BTreeSet::new();
    let mut survivors: Vec<Strain> = Vec::new();
    for s in primary_veterans {
        let k = team_key(&s.team);
        let sig = preview_sig(&s.team);
        let b_wr = if s.bench_games > 0.0 {
            s.bench_wins / s.bench_games
        } else {
            s.bayes_fit
        };
        let viable = if max_per_archetype == usize::MAX {
            b_wr >= 0.32 || survivors.len() < carry
        } else {
            s.bench_games < 20.0 || (b_wr >= 0.26 && s.bayes_fit + s.deme_bonus + 0.14 >= top_vet_fit)
        };
        let sp = s.species_set();
        if survivors.len() < carry
            && viable
            && count_same_archetype(&sp, &survivors) < max_per_archetype
            && seen_sigs.insert(sig)
            && seen_keys.insert(k)
        {
            survivors.push(s);
        } else {
            ledger.archived_strains.push(s);
        }
    }

    // 2. If dead founder niches were evicted and `survivors.len() < carry`, first check if any
    //    viable distinct-archetype strain in `archived_strains` can fill a slot.
    if survivors.len() < carry && !ledger.archived_strains.is_empty() {
        ledger.archived_strains.sort_by(|a, b| {
            let qa = a.bayes_fit + a.deme_bonus;
            let qb = b.bayes_fit + b.deme_bonus;
            qb.partial_cmp(&qa)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut remaining_archive = Vec::with_capacity(ledger.archived_strains.len());
        for mut s in ledger.archived_strains.drain(..) {
            s.anchors.clear();
            s.max_staples.clear();
            let k = team_key(&s.team);
            let sig = preview_sig(&s.team);
            let b_wr = if s.bench_games > 0.0 {
                s.bench_wins / s.bench_games
            } else {
                s.bayes_fit
            };
            let viable = if max_per_archetype == usize::MAX {
                b_wr >= 0.35
            } else {
                s.bench_games >= 20.0 && b_wr >= 0.28 && s.bayes_fit + s.deme_bonus + 0.12 >= top_vet_fit
            };
            let sp = s.species_set();
            if s.symbiont_of.is_none()
                && survivors.len() < carry
                && viable
                && count_same_archetype(&sp, &survivors) < max_per_archetype
                && seen_sigs.insert(sig)
                && seen_keys.insert(k)
            {
                survivors.push(s);
            } else {
                remaining_archive.push(s);
            }
        }
        ledger.archived_strains = remaining_archive;
    }

    // 3. Fill any still-open survivor slots up to `carry` by alternating between multi-step
    //    speciation branches from top survivors (mutated until they form a new archetype with
    //    < 4 shared species) and fresh de-novo Pareto-level teams (skipped in pure Kodoku mode).
    let free_constraint = NicheConstraint::default();
    let base_survivors = survivors.len();
    let mut fill_attempt = 0usize;
    while max_per_archetype != usize::MAX && survivors.len() < carry && fill_attempt < carry * 6 {
        fill_attempt += 1;
        let mut candidate_opt: Option<(Vec<Value>, String, String, f64)> = None;
        if base_survivors > 0 && fill_attempt % 2 == 1 {
            let p_idx = (fill_attempt / 2) % base_survivors;
            let parent = &survivors[p_idx];
            let mut cur_team = parent.team.clone();
            for _ in 0..3 {
                if let Some(prop) = gen.propose_targeted(
                    dex,
                    &cur_team,
                    &parent.slot_picks,
                    &parent.slot_wins,
                    rng,
                    &free_constraint,
                    150,
                ) {
                    cur_team = prop.team;
                    let sp = team_species_set(&cur_team);
                    if count_same_archetype(&sp, &survivors) < max_per_archetype {
                        break;
                    }
                }
            }
            let sp = team_species_set(&cur_team);
            if count_same_archetype(&sp, &survivors) < max_per_archetype {
                candidate_opt = Some((
                    cur_team,
                    parent.id.clone(),
                    "speciate".into(),
                    (parent.bayes_fit - 0.03).max(0.35),
                ));
            }
        }
        if candidate_opt.is_none() {
            if let Some(team) = gen.random_team_constrained(dex, rng, &[], &free_constraint, 300) {
                let sp = team_species_set(&team);
                if count_same_archetype(&sp, &survivors) < max_per_archetype {
                    candidate_opt = Some((team, "reseed".into(), "reseed".into(), 0.35));
                }
            }
        }
        if let Some((team, parent_id, last_op, prior)) = candidate_opt {
            let k = team_key(&team);
            if seen_keys.insert(k) {
                let id = format!("{island}-s{:04}", ledger.next_strain_seq);
                ledger.next_strain_seq += 1;
                survivors.push(Strain {
                    id,
                    parent_id,
                    niche: "eco".into(),
                    anchors: String::new(),
                    max_staples: String::new(),
                    born_wave,
                    last_op,
                    abundance: 1.0 / ((carry + buds).max(1) as f64),
                    cum_games: 0.0,
                    cum_wins: 0.0,
                    bench_games: 0.0,
                    bench_wins: 0.0,
                    eco_games: 0.0,
                    eco_wins: 0.0,
                    last_wave_fit: prior,
                    parent_prior: prior,
                    bayes_fit: prior,
                    lcb95: 0.0,
                    slot_picks: [0.0; 6],
                    slot_wins: [0.0; 6],
                    slot_marginals: [0.0; 6],
                    slot_wars: [0.0; 6],
                    move_dqs: [[0.0; 4]; 6],
                    last_wave_slot_picks: [0.0; 6],
                    last_wave_slot_marginals: [0.0; 6],
                    effective_cores: 1.0,
                    symbiont_of: None,
                    mix_alpha: 1.0,
                    deme_bonus: 0.0,
                    team,
                });
            }
        }
    }

    // 4. Keep up to `buds / 2` promising 1-wave-old grace buds (early-stopping buds that
    //    lost sharply to their parent on the CRN-paired wave slice) so neutral/beneficial
    //    buds get a 2nd wave of evaluation while every wave also spawns fresh mutations.
    let max_grace = (buds / 2).max(1);
    let mut kept_grace: Vec<Strain> = Vec::new();
    for s in primary_grace {
        let k = team_key(&s.team);
        let b_wr = if s.bench_games > 0.0 {
            s.bench_wins / s.bench_games
        } else {
            s.bayes_fit
        };
        let mut active_graft = true;
        if let Some((op_name, slot_str)) = s.last_op.split_once('@') {
            if let Ok(slot_idx) = slot_str.parse::<usize>() {
                if slot_idx < 6 && !op_name.starts_with("level-tweak") && s.last_wave_slot_picks[slot_idx] < 0.08 {
                    active_graft = false;
                }
            }
        }
        let promising = active_graft
            && s.last_wave_fit + 0.03 >= s.parent_prior
            && (s.bench_games < 16.0 || b_wr >= 0.28)
            && s.bayes_fit + s.deme_bonus + 0.14 >= top_vet_fit;
        if kept_grace.len() < max_grace && promising && seen_keys.insert(k) {
            kept_grace.push(s);
        } else {
            ledger.archived_strains.push(s);
        }
    }

    // 5. Spawn fresh buds from the surviving veterans for remaining bud slots,
    //    using within-party vitality telemetry (`propose_targeted`) to target hitchhikers.
    let mut endo_pool = build_endogenous_pool(&survivors);
    for s in kept_grace.iter().chain(ledger.archived_strains.iter()) {
        if s.cum_games <= 0.0 {
            continue;
        }
        let g = s.cum_games.max(1.0);
        for i in 0..s.team.len().min(6) {
            if s.slot_wars[i] >= 0.0 {
                let pick_rate = (s.slot_picks[i] / g).clamp(0.0, 1.0);
                endo_pool.record_slot(&s.team[i], s.slot_wars[i], pick_rate, &s.move_dqs[i]);
            }
        }
    }
    let num_parents = survivors.len().max(1);
    let new_buds_needed = buds.saturating_sub(kept_grace.len());
    let mut island_buds: Vec<Strain> = kept_grace;
    for b_idx in 0..new_buds_needed {
        let p_idx = (born_wave as usize + b_idx) % num_parents;
        let parent = &survivors[p_idx];
        if let Some(prop) = gen.propose_marginal_targeted(
            dex,
            &parent.team,
            &parent.slot_picks,
            &parent.slot_wins,
            &parent.slot_wars,
            Some(&parent.move_dqs),
            Some(&endo_pool),
            rng,
            &free_constraint,
            150,
        ) {
            let k = team_key(&prop.team);
            if seen_keys.insert(k) {
                let id = format!("{island}-s{:04}", ledger.next_strain_seq);
                ledger.next_strain_seq += 1;
                island_buds.push(Strain {
                    id,
                    parent_id: parent.id.clone(),
                    niche: parent.niche.clone(),
                    anchors: String::new(),
                    max_staples: String::new(),
                    born_wave,
                    last_op: format!("{}@{}", prop.op.name(), prop.slot),
                    abundance: parent.abundance,
                    cum_games: 0.0,
                    cum_wins: 0.0,
                    bench_games: 0.0,
                    bench_wins: 0.0,
                    eco_games: 0.0,
                    eco_wins: 0.0,
                    last_wave_fit: parent.bayes_fit,
                    parent_prior: parent.bayes_fit,
                    bayes_fit: parent.bayes_fit,
                    lcb95: 0.0,
                    slot_picks: [0.0; 6],
                    slot_wins: [0.0; 6],
                    slot_marginals: [0.0; 6],
                    slot_wars: [0.0; 6],
                    move_dqs: [[0.0; 4]; 6],
                    last_wave_slot_picks: [0.0; 6],
                    last_wave_slot_marginals: [0.0; 6],
                    effective_cores: parent.effective_cores,
                    symbiont_of: None,
                    mix_alpha: 1.0,
                    deme_bonus: 0.0,
                    team: prop.team,
                });
            }
        }
    }

    // 6. Attach any certified Symbiont (`symbiont_pool`) whose Primary survived in `survivors`.
    let mut attached_symbionts: Vec<Strain> = Vec::new();
    for p in &mut survivors {
        if let Some(sym) = symbiont_pool.remove(&p.id) {
            let k = team_key(&sym.team);
            if seen_keys.insert(k) {
                p.mix_alpha = (1.0 - sym.mix_alpha).clamp(0.50, 0.95);
                attached_symbionts.push(sym);
            } else {
                ledger.archived_strains.push(sym);
            }
        } else {
            p.mix_alpha = 1.0;
            p.deme_bonus = 0.0;
        }
    }
    for (_, orphan_sym) in symbiont_pool {
        ledger.archived_strains.push(orphan_sym);
    }

    survivors.extend(attached_symbionts);
    survivors.extend(island_buds);
    let total_w: f64 = survivors.iter().map(|s| s.abundance.max(1e-4)).sum();
    for s in &mut survivors {
        s.abundance = s.abundance.max(1e-4) / total_w.max(1e-9);
    }
    ledger.active_strains = survivors;
}

fn collect_all_strains(in_dir: &Path, min_games: f64) -> Vec<Strain> {
    let mut by_key: BTreeMap<String, Strain> = BTreeMap::new();
    let entries = std::fs::read_dir(in_dir).unwrap_or_else(|e| panic!("{}: {e}", in_dir.display()));
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|x| x.path())).collect();
    paths.sort();
    for p in paths {
        if p.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(txt) = std::fs::read_to_string(&p) else {
            continue;
        };
        let Ok(ledger) = serde_json::from_str::<IslandLedger>(&txt) else {
            continue;
        };
        for s in ledger
            .active_strains
            .into_iter()
            .chain(ledger.archived_strains.into_iter())
        {
            if s.cum_games < min_games || s.bench_games < 48.0 {
                continue;
            }
            let k = team_key(&s.team);
            match by_key.get_mut(&k) {
                Some(existing) => {
                    let q_new = 0.65 * s.bayes_fit + 0.35 * s.lcb95;
                    let q_old = 0.65 * existing.bayes_fit + 0.35 * existing.lcb95;
                    if q_new > q_old {
                        *existing = s;
                    }
                }
                None => {
                    by_key.insert(k, s);
                }
            }
        }
    }
    let mut out: Vec<Strain> = by_key.into_values().collect();
    // Sort by Empirical-Bayes Master Pool benchmark winrate (75%) + island Bayes fitness (25%)
    // so newly emerged 2-to-4 wave island leaders are not buried behind 800-game legacy strains.
    let strain_quality = |s: &Strain| -> f64 {
        let bench_eb = if s.bench_games > 0.0 {
            (s.bench_wins + 24.0 * 0.40) / (s.bench_games + 24.0)
        } else {
            s.bayes_fit
        };
        0.75 * bench_eb + 0.25 * s.bayes_fit
    };
    out.sort_by(|a, b| {
        strain_quality(b)
            .partial_cmp(&strain_quality(a))
            .unwrap()
    });
    out
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn strain_island_key(id: &str) -> String {
    // Extract island index (e.g. "09" from "island-r14-09-s0696")
    let parts: Vec<&str> = id.split('-').collect();
    if parts.len() >= 3 {
        parts[2].to_string()
    } else {
        id.to_string()
    }
}

fn cmd_extract_pool(args: &[String]) {
    let in_dir = PathBuf::from(flag(args, "--in").expect("--in DIR"));
    let out_file = PathBuf::from(flag(args, "--out").expect("--out FILE"));
    let min_games: f64 = num(args, "--min-games", 40.0);
    let min_bayes: f64 = num(args, "--min-bayes", 0.40);
    let top_k: usize = num(args, "--top", 24);
    let max_jaccard: f64 = num(args, "--max-jaccard", 0.50);

    let all = collect_all_strains(&in_dir, min_games);
    let root = repo_root();
    let master_path = flag(args, "--master")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/eco-pool/master-pool.json"));
    let mut existing_master_keys: BTreeSet<String> = BTreeSet::new();
    if let Ok(mtxt) = std::fs::read_to_string(&master_path) {
        if let Ok(mv) = serde_json::from_str::<Value>(&mtxt) {
            if let Some(marr) = mv["teams"].as_array() {
                for mt in marr {
                    if let Some(sets) = mt["sets"].as_array() {
                        existing_master_keys.insert(team_key(sets));
                    }
                }
            }
        }
    }

    let mut selected: Vec<Strain> = Vec::new();
    let mut selected_species: Vec<BTreeSet<String>> = Vec::new();
    let mut seen_islands: BTreeSet<String> = BTreeSet::new();

    // Pass 1: take top validated representative of each distinct island that meets `min_bayes`
    for s in &all {
        if selected.len() >= top_k {
            break;
        }
        if s.bayes_fit < min_bayes || existing_master_keys.contains(&team_key(&s.team)) {
            continue;
        }
        let sp = s.species_set();
        let ikey = strain_island_key(&s.id);
        if !seen_islands.contains(&ikey)
            && selected_species
                .iter()
                .all(|prev| jaccard(&sp, prev) <= max_jaccard)
        {
            seen_islands.insert(ikey);
            selected_species.push(sp);
            selected.push(s.clone());
        }
    }
    // Pass 2: fill remaining slots with highest quality strains honoring max_jaccard
    for s in &all {
        if selected.len() >= top_k {
            break;
        }
        if selected.iter().any(|x| x.id == s.id)
            || existing_master_keys.contains(&team_key(&s.team))
        {
            continue;
        }
        let sp = s.species_set();
        if selected_species
            .iter()
            .all(|prev| jaccard(&sp, prev) <= max_jaccard)
        {
            selected_species.push(sp);
            selected.push(s.clone());
        }
    }

    // Re-sort `selected` by quality descending so `--top-candidates` in `update-pool`
    // always evaluates the highest-ranked strains across both Pass 1 and Pass 2.
    selected.sort_by(|a, b| {
        let qa = 0.75 * ((a.bench_wins + 24.0 * 0.40) / (a.bench_games + 24.0)) + 0.25 * a.bayes_fit;
        let qb = 0.75 * ((b.bench_wins + 24.0 * 0.40) / (b.bench_games + 24.0)) + 0.25 * b.bayes_fit;
        qb.partial_cmp(&qa).unwrap()
    });

    let w_sum: f64 = selected.iter().map(|s| s.abundance.max(1e-4)).sum();
    let teams_json: Vec<Value> = selected
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "parent_id": s.parent_id,
                "niche": s.niche,
                "born_wave": s.born_wave,
                "abundance": s.abundance,
                "drawWeight": s.abundance.max(1e-4) / w_sum.max(1e-9),
                "cum_games": s.cum_games,
                "raw_winrate": s.cum_wins / s.cum_games.max(1.0),
                "bayes_winrate": s.bayes_fit,
                "lcb95": s.lcb95,
                "bench_games": s.bench_games,
                "bench_wins": s.bench_wins,
                "bench_winrate": if s.bench_games > 0.0 { s.bench_wins / s.bench_games } else { s.bayes_fit },
                "sets": s.team,
            })
        })
        .collect();

    let out = json!({
        "format": "nc2000-eco-extracted-pool-v1",
        "total_measured_strains": all.len(),
        "selected_count": selected.len(),
        "min_games": min_games,
        "max_jaccard": max_jaccard,
        "teams": teams_json,
    });
    if let Some(parent) = out_file.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    write_json_atomic(&out_file, &out);
    println!(
        "Extracted {} diverse strains (from {} measured) -> {}",
        selected.len(),
        all.len(),
        out_file.display()
    );
    for (i, s) in selected.iter().enumerate() {
        let mons: Vec<String> = s
            .team
            .iter()
            .map(|m| format!("{}(L{})", m["species"].as_str().unwrap_or("?"), m["level"]))
            .collect();
        println!(
            "  {:2}. {:16} [niche={:12}] lcb95={:.3} bayes={:.3} raw={:.3} (n={:4.0}, w={:.3}) | {}",
            i + 1,
            s.id,
            s.niche,
            s.lcb95,
            s.bayes_fit,
            s.cum_wins / s.cum_games.max(1.0),
            s.cum_games,
            s.abundance,
            mons.join(", ")
        );
    }
}

fn solve_symmetric_nash(payoff: &[Vec<f64>], iters: usize) -> Vec<f64> {
    let n = payoff.len();
    let mut cum_regret = vec![0.0f64; n];
    let mut cum_strat = vec![0.0f64; n];
    let mut strat = vec![1.0 / (n as f64); n];

    for _ in 0..iters {
        for i in 0..n {
            cum_strat[i] += strat[i];
        }
        // Expected payoff of each pure action i against `strat`
        let mut u = vec![0.0f64; n];
        let mut u_mix = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += payoff[i][j] * strat[j];
            }
            u[i] = s;
            u_mix += strat[i] * s;
        }
        for i in 0..n {
            cum_regret[i] = (cum_regret[i] + (u[i] - u_mix)).max(0.0);
        }
        let pos_sum: f64 = cum_regret.iter().sum();
        if pos_sum > 1e-12 {
            for i in 0..n {
                strat[i] = cum_regret[i] / pos_sum;
            }
        } else {
            strat.fill(1.0 / (n as f64));
        }
    }
    let sum: f64 = cum_strat.iter().sum();
    for x in &mut cum_strat {
        *x /= sum.max(1e-12);
    }
    cum_strat
}

fn cmd_extract_nash(args: &[String]) {
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let pool_file = PathBuf::from(flag(args, "--pool").expect("--pool EXTRACTED_POOL.json"));
    let out_file = PathBuf::from(flag(args, "--out").expect("--out NASH.json"));
    let top_evolved: usize = num(args, "--top-evolved", 10);
    let eval_games: u32 = num(args, "--eval-games", 16);
    let gate_games: u32 = num(args, "--gate-games", 16);
    let min_gate_wr: f64 = num(args, "--min-gate-wr", 0.505);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let seed: u64 = num(args, "--seed", 20261201);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    let (agent_iters, agent_c, blind_pool, open_sheet) = parse_agent_spec(args, &root, &dex, &gen);

    // 1. Load existing Old-Nash benchmark teams and weights
    let gfile = flag(args, "--gauntlet-file").unwrap_or_else(|| {
        root.join("data/meta-nash-v3/pool-artifact.json")
            .to_string_lossy()
            .into_owned()
    });
    let (old_teams, old_weights) = load_gauntlet_file(&dex, &gen, &gfile);
    let gv: Value = serde_json::from_str(&std::fs::read_to_string(&gfile).unwrap()).unwrap();

    // Load known reference pairwise matrix (e.g. data/meta-nash-v3/solution-blind27000.json)
    // so small-sample noise NEVER scrambles the known equilibrium among existing benchmark teams!
    let ref_matrix_file = flag(args, "--ref-matrix").unwrap_or_else(|| {
        root.join("data/meta-nash-v3/solution-blind27000.json")
            .to_string_lossy()
            .into_owned()
    });
    let mut ref_payoff_map: BTreeMap<(String, String), f64> = BTreeMap::new();
    if let Ok(rtxt) = std::fs::read_to_string(&ref_matrix_file) {
        if let Ok(rv) = serde_json::from_str::<Value>(&rtxt) {
            if let (Some(rids), Some(rmat)) = (rv["ids"].as_array(), rv["matrix"].as_array()) {
                for (i, ri) in rids.iter().enumerate() {
                    for (j, rj) in rids.iter().enumerate() {
                        if let (Some(si), Some(sj), Some(p)) = (
                            ri.as_str(),
                            rj.as_str(),
                            rmat.get(i).and_then(|row| row.get(j)).and_then(Value::as_f64),
                        ) {
                            ref_payoff_map.insert((si.to_string(), sj.to_string()), p);
                        }
                    }
                }
            }
        }
    }

    let mut ids: Vec<String> = Vec::new();
    let mut raw_teams: Vec<Vec<Value>> = Vec::new();
    let mut sets_list: Vec<Vec<PokemonSet>> = Vec::new();

    let mut existing_old_keys: BTreeSet<String> = BTreeSet::new();
    for t in gv["teams"].as_array().unwrap() {
        let raw = t["sets"].as_array().unwrap().clone();
        if let Some(canon) = gen.canonize(&dex, &raw) {
            existing_old_keys.insert(team_key(&canon));
        }
    }

    // 2. Out-of-Sample Entry Gate: only admit evolved challengers (`born_wave > 0`, distinct from
    //    `old_teams`) that strictly beat Old-Nash (`score_vs_old >= min_gate_wr` and `fresh >= 0.495`)
    //    on independent verification seeds!
    let pv: Value = serde_json::from_str(&std::fs::read_to_string(&pool_file).unwrap()).unwrap();
    eprintln!(
        "=== Step 1: Out-of-Sample Challenger Entry Gate vs Old-Nash (min_gate_wr={min_gate_wr:.3}, {gate_games} games/opp) ==="
    );
    for (c_idx, t) in pv["teams"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| {
            if t["born_wave"].as_u64().unwrap_or(1) == 0 {
                return false;
            }
            // Reject "Addition by Subtraction" dead-slot mutations (`pick_rate < 0.10`)
            if let Some(op) = t["last_op"].as_str() {
                if !op.starts_with("level-tweak") {
                    if let Some((_kind, slot_str)) = op.split_once('@') {
                        if let Ok(mut_slot) = slot_str.parse::<usize>() {
                            if let Some(tel) = t["slot_telemetry"].as_array().and_then(|a| a.get(mut_slot)) {
                                let pr = tel["pick_rate"].as_f64().unwrap_or(0.0);
                                if pr < 0.10 {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
            true
        })
        .take(top_evolved)
        .enumerate()
    {
        let id = t["id"].as_str().unwrap().to_string();
        let raw = t["sets"].as_array().unwrap().clone();
        let canon = gen.canonize(&dex, &raw).unwrap();
        if existing_old_keys.contains(&team_key(&canon)) {
            continue;
        }
        let cand_sets = to_sets(&canon).unwrap();
        let vs_old_fresh = gauntlet_eval_weighted(
            &dex,
            &cand_sets,
            &old_teams,
            &old_weights,
            &WeightedEvalCfg {
                games_per_opponent: gate_games,
                agent_iters,
                c: agent_c,
                max_turns,
                seed: seed ^ 0x7777_1111_u64.wrapping_add(c_idx as u64 * 9973),
                threads,
                blind_pool: blind_pool.clone(),
                open_sheet,
            },
        )
        .score;
        let fresh_n = (gate_games as usize * old_teams.len()) as f64;
        let prev_n = t["bench_games"].as_f64().unwrap_or(0.0);
        let prev_w = t["bench_wins"].as_f64().unwrap_or(0.0);
        let total_n = prev_n + fresh_n;
        let vs_old = (prev_w + vs_old_fresh * fresh_n) / total_n.max(1.0);
        let passed = vs_old >= min_gate_wr && vs_old_fresh >= 0.495 && total_n >= 160.0;
        eprintln!(
            "  challenger {:20} vs Old-Nash = {:.4} ({:.1}% over {:.0} games, fresh={:.1}%) -> {}",
            id,
            vs_old,
            vs_old * 100.0,
            total_n,
            vs_old_fresh * 100.0,
            if passed { "ADMITTED" } else { "REJECTED (below gate)" }
        );
        if passed {
            ids.push(id);
            sets_list.push(cand_sets);
            raw_teams.push(canon);
        }
    }

    let num_admitted = ids.len();
    eprintln!("Admitted {num_admitted} verified challengers above {min_gate_wr:.3} gate.");

    for (idx, t) in gv["teams"].as_array().unwrap().iter().enumerate() {
        let id = t["id"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| format!("bench-{idx}"));
        let raw = t["sets"].as_array().unwrap().clone();
        let canon = gen.canonize(&dex, &raw).unwrap();
        ids.push(id);
        sets_list.push(to_sets(&canon).unwrap());
        raw_teams.push(canon);
    }

    let n = sets_list.len();
    let prior_pair_games: f64 = num(args, "--prior-pair-games", 6.0);
    let mut raw_payoff = vec![vec![0.5f64; n]; n];
    let mut shrunk_payoff = vec![vec![0.5f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if let Some(&ref_p) = ref_payoff_map.get(&(ids[i].clone(), ids[j].clone())) {
                raw_payoff[i][j] = ref_p;
                raw_payoff[j][i] = 1.0 - ref_p;
                shrunk_payoff[i][j] = ref_p;
                shrunk_payoff[j][i] = 1.0 - ref_p;
                eprintln!(
                    "  pair ({:2}/{n}) {} vs {}: ref={:.3} (preserved)",
                    i + 1,
                    ids[i],
                    ids[j],
                    ref_p
                );
                continue;
            }
            let p_ij = gauntlet_eval_weighted(
                &dex,
                &sets_list[i],
                std::slice::from_ref(&sets_list[j]),
                &[1.0],
                &WeightedEvalCfg {
                    games_per_opponent: eval_games,
                    agent_iters,
                    c: agent_c,
                    max_turns,
                    seed: seed ^ ((i * 131 + j) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    threads,
                    blind_pool: blind_pool.clone(),
                    open_sheet,
                },
            )
            .score;
            raw_payoff[i][j] = p_ij;
            raw_payoff[j][i] = 1.0 - p_ij;
            let g = eval_games as f64;
            let s_ij = (p_ij * g + 0.5 * prior_pair_games) / (g + prior_pair_games);
            shrunk_payoff[i][j] = s_ij;
            shrunk_payoff[j][i] = 1.0 - s_ij;
            eprintln!(
                "  pair ({:2}/{n}) {} vs {}: raw={:.3} shrunk={:.3}",
                i + 1,
                ids[i],
                ids[j],
                p_ij,
                s_ij
            );
        }
    }

    let weights = solve_symmetric_nash(&shrunk_payoff, 200_000);
    let mut support: Vec<Value> = Vec::new();
    for i in 0..n {
        if weights[i] >= 0.005 {
            support.push(json!({
                "id": ids[i],
                "weight": weights[i],
                "sets": raw_teams[i],
            }));
        }
    }
    support.sort_by(|a, b| {
        b["weight"]
            .as_f64()
            .unwrap()
            .partial_cmp(&a["weight"].as_f64().unwrap())
            .unwrap()
    });

    let out = json!({
        "format": "nc2000-eco-extracted-nash-v1",
        "candidates": ids,
        "weights": weights,
        "payoff": raw_payoff,
        "shrunk_payoff": shrunk_payoff,
        "teams": support,
    });
    if let Some(parent) = out_file.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    write_json_atomic(&out_file, &out);
    println!("Solved Nash equilibrium over {n} teams -> {}", out_file.display());
    for s in &support {
        println!(
            "  {:20} weight={:.4}",
            s["id"].as_str().unwrap(),
            s["weight"].as_f64().unwrap()
        );
    }
}

fn cmd_verify_nash(args: &[String]) {
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let new_file = flag(args, "--new").expect("--new NEW_NASH.json");
    let old_file = flag(args, "--old").unwrap_or_else(|| {
        root.join("data/meta-nash-v3/pool-artifact.json")
            .to_string_lossy()
            .into_owned()
    });
    let eval_games: u32 = num(args, "--eval-games", 16);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let seed: u64 = num(args, "--seed", 20269999);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    let (agent_iters, agent_c, blind_pool, open_sheet) = parse_agent_spec(args, &root, &dex, &gen);

    let (new_teams, new_weights) = load_gauntlet_file(&dex, &gen, &new_file);
    let (old_teams, old_weights) = load_gauntlet_file(&dex, &gen, &old_file);

    let nv: Value = serde_json::from_str(&std::fs::read_to_string(&new_file).unwrap()).unwrap();
    let new_ids: Vec<String> = nv["teams"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, t)| t["id"].as_str().map(String::from).unwrap_or_else(|| format!("new-{i}")))
        .collect();
    let ov: Value = serde_json::from_str(&std::fs::read_to_string(&old_file).unwrap()).unwrap();
    let old_ids: Vec<String> = ov["teams"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, t)| t["id"].as_str().map(String::from).unwrap_or_else(|| format!("old-{i}")))
        .collect();

    let ref_matrix_file = flag(args, "--ref-matrix").unwrap_or_else(|| {
        root.join("data/meta-nash-v3/solution-blind27000.json")
            .to_string_lossy()
            .into_owned()
    });
    let mut ref_payoff_map: BTreeMap<(String, String), f64> = BTreeMap::new();
    if let Ok(rtxt) = std::fs::read_to_string(&ref_matrix_file) {
        if let Ok(rv) = serde_json::from_str::<Value>(&rtxt) {
            if let (Some(rids), Some(rmat)) = (rv["ids"].as_array(), rv["matrix"].as_array()) {
                for (i, ri) in rids.iter().enumerate() {
                    for (j, rj) in rids.iter().enumerate() {
                        if let (Some(si), Some(sj), Some(p)) = (
                            ri.as_str(),
                            rj.as_str(),
                            rmat.get(i).and_then(|row| row.get(j)).and_then(Value::as_f64),
                        ) {
                            ref_payoff_map.insert((si.to_string(), sj.to_string()), p);
                        }
                    }
                }
            }
        }
    }

    println!(
        "=== Head-to-Head Verification ({} new vs {} old, {} games/pair) ===",
        new_teams.len(),
        old_teams.len(),
        eval_games
    );
    let mut total_ev = 0.0;
    for (i, (cand, &w_i)) in new_teams.iter().zip(new_weights.iter()).enumerate() {
        let nid = new_ids.get(i).map(String::as_str).unwrap_or("?");
        // Check if all old_ids have known exact reference payoffs for `nid`
        let all_ref = old_ids
            .iter()
            .all(|oid| nid == oid || ref_payoff_map.contains_key(&(nid.to_string(), oid.clone())));
        let score_i = if all_ref {
            let mut s = 0.0;
            for (j, oid) in old_ids.iter().enumerate() {
                let p = if nid == oid {
                    0.5
                } else {
                    *ref_payoff_map.get(&(nid.to_string(), oid.clone())).unwrap()
                };
                s += old_weights[j] * p;
            }
            s
        } else {
            gauntlet_eval_weighted(
                &dex,
                cand,
                &old_teams,
                &old_weights,
                &WeightedEvalCfg {
                    games_per_opponent: eval_games,
                    agent_iters,
                    c: agent_c,
                    max_turns,
                    seed: seed ^ ((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    threads,
                    blind_pool: blind_pool.clone(),
                    open_sheet,
                },
            )
            .score
        };
        total_ev += w_i * score_i;
        println!(
            "  {:22} (w_new={:.4}) vs Old-Nash: {:.4} ({:.2}%){}",
            nid,
            w_i,
            score_i,
            score_i * 100.0,
            if all_ref { " [exact ref]" } else { "" }
        );
    }
    println!(
        "==> E[New-Nash vs Old-Nash] = {:.4} ({:.2}%)",
        total_ev,
        total_ev * 100.0
    );
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RingGenEntry {
    gen: String,
    teams: Vec<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RingBufferFile {
    capacity: usize,
    generations: Vec<RingGenEntry>,
}

/// Maintain a ring buffer of recent generation Nash/elite distributions (`--ring`)
/// and export their time-averaged mixture (`--out`) as the evolutionary gauntlet
/// (Fictitious Play / Historical Averaging to eliminate Red-Queen oscillation and
/// absorb single-generation Nash estimation noise).
fn cmd_update_ring(args: &[String]) {
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let ring_path = flag(args, "--ring")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/eco-ring/ring-buffer.json"));
    let out_path = flag(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/eco-ring/ring-gauntlet.json"));
    let capacity: usize = num(args, "--capacity", 5);
    let max_per_gen: usize = num(args, "--max-per-gen", 4);

    let mut ring: RingBufferFile = if ring_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&ring_path).unwrap()).unwrap()
    } else {
        // Initialize slot 0 from baseline human Nash equilibrium (meta-nash-v3)
        let base_file = root.join("data/meta-nash-v3/pool-artifact.json");
        let bv: Value = serde_json::from_str(&std::fs::read_to_string(&base_file).unwrap()).unwrap();
        let mut base_teams: Vec<Value> = Vec::new();
        let mut w_sum = 0.0;
        for t in bv["teams"].as_array().unwrap() {
            let w = t["weight"]
                .as_f64()
                .or_else(|| t["drawWeight"].as_f64())
                .unwrap_or(1.0);
            w_sum += w;
            base_teams.push(json!({
                "id": t["id"],
                "weight": w,
                "sets": t["sets"],
            }));
        }
        for t in &mut base_teams {
            let w = t["weight"].as_f64().unwrap() / w_sum.max(1e-9);
            t["weight"] = json!(w);
        }
        RingBufferFile {
            capacity,
            generations: vec![RingGenEntry {
                gen: "g0-human-nash".into(),
                teams: base_teams,
            }],
        }
    };
    ring.capacity = capacity;

    if let Some(add_file) = flag(args, "--add-file") {
        let gen_name = flag(args, "--gen-name").unwrap_or_else(|| format!("g{}", ring.generations.len()));
        let av: Value = serde_json::from_str(&std::fs::read_to_string(&add_file).unwrap()).unwrap();
        let mut gen_teams: Vec<Value> = Vec::new();
        let mut w_sum = 0.0;
        if let Some(arr) = av["teams"].as_array() {
            for t in arr.iter().take(max_per_gen) {
                let raw = t["sets"].as_array().unwrap().clone();
                let Some(canon) = gen.canonize(&dex, &raw) else {
                    continue;
                };
                let w = t["weight"]
                    .as_f64()
                    .or_else(|| t["lcb95"].as_f64())
                    .or_else(|| t["drawWeight"].as_f64())
                    .unwrap_or(1.0)
                    .max(1e-3);
                w_sum += w;
                gen_teams.push(json!({
                    "id": t["id"].as_str().unwrap_or("evolved"),
                    "weight": w,
                    "sets": canon,
                }));
            }
        }
        if !gen_teams.is_empty() {
            for t in &mut gen_teams {
                let w = t["weight"].as_f64().unwrap() / w_sum.max(1e-9);
                t["weight"] = json!(w);
            }
            if let Some(pos) = ring.generations.iter().position(|g| g.gen == gen_name) {
                ring.generations[pos].teams = gen_teams;
            } else {
                ring.generations.push(RingGenEntry {
                    gen: gen_name,
                    teams: gen_teams,
                });
            }
            // Keep baseline g0-human-nash at index 0 + most recent (capacity - 1) evolved generations
            while ring.generations.len() > ring.capacity && ring.generations.len() >= 2 {
                ring.generations.remove(1);
            }
        }
    }

    // Export combined uniform-generation-averaged gauntlet
    let num_gens = ring.generations.len().max(1) as f64;
    let mut combined: Vec<Value> = Vec::new();
    let mut key_to_idx: BTreeMap<String, usize> = BTreeMap::new();
    for g in &ring.generations {
        let g_share = 1.0 / num_gens;
        for t in &g.teams {
            let sets = t["sets"].as_array().unwrap();
            let k = team_key(sets);
            let w = t["weight"].as_f64().unwrap_or(0.0) * g_share;
            if let Some(&idx) = key_to_idx.get(&k) {
                let prev_w = combined[idx]["weight"].as_f64().unwrap_or(0.0);
                combined[idx]["weight"] = json!(prev_w + w);
                combined[idx]["drawWeight"] = json!(prev_w + w);
            } else {
                let idx = combined.len();
                key_to_idx.insert(k, idx);
                combined.push(json!({
                    "id": t["id"],
                    "gen": g.gen,
                    "weight": w,
                    "drawWeight": w,
                    "sets": t["sets"],
                }));
            }
        }
    }

    if let Some(p) = ring_path.parent() {
        std::fs::create_dir_all(p).unwrap();
    }
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).unwrap();
    }
    write_json_atomic(&ring_path, &ring);
    let out_json = json!({
        "format": "nc2000-eco-ring-gauntlet-v1",
        "capacity": ring.capacity,
        "active_generations": ring.generations.iter().map(|g| &g.gen).collect::<Vec<_>>(),
        "teams": combined,
    });
    write_json_atomic(&out_path, &out_json);
    println!(
        "Updated ring buffer ({}/{} generations, {} unique teams) -> {}",
        ring.generations.len(),
        ring.capacity,
        combined.len(),
        out_path.display()
    );
    for t in &combined {
        println!(
            "  [{:14}] {:20} weight={:.4}",
            t["gen"].as_str().unwrap_or("?"),
            t["id"].as_str().unwrap_or("?"),
            t["weight"].as_f64().unwrap_or(0.0)
        );
    }
}

fn team_species_set(sets: &[Value]) -> BTreeSet<String> {
    sets.iter()
        .filter_map(|s| s["species"].as_str().map(toid))
        .collect()
}

/// Maintain the single Master Strong Pool (`data/eco-pool/master-pool.json`):
/// 1. Starts from the 24 strong teams of `data/team-pool-v2/team-pool.json` (with their
///    measured Stage A 3000-iter scores against the 24-team Master Pool).
/// 2. Evaluates candidate veterans against ALL teams in the Master Pool (`--games-per-opp`).
/// 3. Enforces Monotonic Diverse Pool Invariants:
///    - Same Archetype (>= 4 shared species with closest team `u` in pool): replaces `u`
///      iff `cand_score >= u.score + 0.015 && cand_score >= min_admit_score`.
///    - New Archetype (<= 3 shared species with all teams in pool): added to pool iff
///      `cand_score >= min_admit_score`; if pool exceeds `--max-pool` (default 30),
///      evicts the lowest-scoring team in the pool.
fn cmd_update_pool(args: &[String]) {
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let master_path = flag(args, "--master")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/eco-pool/master-pool.json"));
    let max_pool: usize = num(args, "--max-pool", 30);
    let top_cands: usize = num(args, "--top-candidates", 6);
    let games_per_opp: u32 = num(args, "--games-per-opp", 6);
    let min_admit_score: f64 = num(args, "--min-admit-score", 0.520);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let threads: usize = num(args, "--threads", 8);
    let seed: u64 = num(args, "--seed", 2026102850);
    let (agent_iters, agent_c, blind_pool, open_sheet) = parse_agent_spec(args, &root, &dex, &gen);

    let mut master_teams: Vec<Value> = if master_path.exists() {
        let mv: Value = serde_json::from_str(&std::fs::read_to_string(&master_path).unwrap()).unwrap();
        mv["teams"].as_array().unwrap().clone()
    } else {
        // Initialize from the 24 teams of `data/team-pool-v2/team-pool.json`
        let tp_path = root.join("data/team-pool-v2/team-pool.json");
        let tp: Value = serde_json::from_str(&std::fs::read_to_string(&tp_path).unwrap()).unwrap();
        let raw_teams = tp["teams"].as_array().unwrap();
        let ids24: Vec<String> = raw_teams
            .iter()
            .map(|t| t["id"].as_str().unwrap().to_string())
            .collect();

        // Load exact 24x24 head-to-head scores from `data/team-selection-v2/stage-a-3000/cells/` if available
        let cells_dir = root.join("data/team-selection-v2/stage-a-3000/cells");
        let mut pair_scores: BTreeMap<(String, String), f64> = BTreeMap::new();
        for (i, a) in ids24.iter().enumerate() {
            for b in ids24.iter().skip(i + 1) {
                for (fname, flip) in [
                    (format!("cell-{a}__{b}.jsonl"), false),
                    (format!("cell-{b}__{a}.jsonl"), true),
                ] {
                    let p = cells_dir.join(fname);
                    if let Ok(txt) = std::fs::read_to_string(&p) {
                        let mut sum = 0.0;
                        let mut cnt = 0usize;
                        for line in txt.lines() {
                            if let Ok(v) = serde_json::from_str::<Value>(line) {
                                if let Some(sc) = v["score"].as_f64() {
                                    sum += sc;
                                    cnt += 1;
                                }
                            }
                        }
                        if cnt > 0 {
                            let sc = sum / (cnt as f64);
                            let ab = if flip { 1.0 - sc } else { sc };
                            pair_scores.insert((a.clone(), b.clone()), ab);
                            pair_scores.insert((b.clone(), a.clone()), 1.0 - ab);
                            break;
                        }
                    }
                }
            }
        }

        let mut init_teams = Vec::new();
        for t in raw_teams {
            let tid = t["id"].as_str().unwrap();
            let sets = t["sets"].as_array().unwrap();
            let canon = gen.canonize(&dex, sets).expect("canonize team-pool-v2 team");
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for oid in &ids24 {
                if oid != tid {
                    sum += pair_scores
                        .get(&(tid.to_string(), oid.clone()))
                        .copied()
                        .unwrap_or(0.5);
                    cnt += 1;
                }
            }
            let pool24_wr = if cnt > 0 { sum / (cnt as f64) } else { 0.5 };
            let panel_wr = t["score"].as_f64().unwrap_or(0.58);
            // Blended baseline score against the 24-team Master Pool + panel
            let score = 0.65 * pool24_wr + 0.35 * panel_wr;
            init_teams.push(json!({
                "id": tid,
                "origin": "team-pool-v2",
                "score": score,
                "pool24_wr": pool24_wr,
                "panel_wr": panel_wr,
                "weight": 1.0 / (raw_teams.len() as f64),
                "drawWeight": 1.0 / (raw_teams.len() as f64),
                "species": t["species"],
                "sets": canon,
            }));
        }
        init_teams
    };

    if let Some(cand_file) = flag(args, "--candidates") {
        let origin_tag = flag(args, "--origin").unwrap_or_else(|| "evolved".into());
        let cv: Value = serde_json::from_str(&std::fs::read_to_string(&cand_file).unwrap()).unwrap();
        let cand_arr = cv["teams"].as_array().expect("--candidates missing teams array");

        // Snapshot current Master Pool teams as benchmark opponents
        let opp_teams: Vec<Vec<PokemonSet>> = master_teams
            .iter()
            .map(|t| to_sets(t["sets"].as_array().unwrap()).unwrap())
            .collect();
        let opp_weights = vec![1.0 / (opp_teams.len() as f64); opp_teams.len()];

        println!(
            "=== Master Pool Admission Test ({} candidates vs {} Master Pool teams, {} games/opp = {} games/cand, min_admit={:.3}) ===",
            cand_arr.len().min(top_cands),
            opp_teams.len(),
            games_per_opp,
            games_per_opp as usize * opp_teams.len(),
            min_admit_score
        );

        let mut dom_rng = SplitMix64::new(seed ^ 0xCAFE_BEEF);
        let mut admitted = 0usize;
        for (ci, ct) in cand_arr.iter().take(top_cands).enumerate() {
            let cid = ct["id"].as_str().unwrap_or("cand");
            let mut raw_sets = ct["sets"].as_array().unwrap().clone();
            gen.optimize_team_dominance(&dex, &mut raw_sets, &mut dom_rng);
            let Some(canon) = gen.canonize(&dex, &raw_sets) else {
                continue;
            };
            let cand_key = team_key(&canon);
            if master_teams
                .iter()
                .any(|m| team_key(m["sets"].as_array().unwrap()) == cand_key)
            {
                continue;
            }
            let cand_sets = to_sets(&canon).unwrap();
            let eval_res = gauntlet_eval_weighted(
                &dex,
                &cand_sets,
                &opp_teams,
                &opp_weights,
                &WeightedEvalCfg {
                    games_per_opponent: games_per_opp,
                    agent_iters,
                    c: agent_c,
                    max_turns,
                    seed: seed ^ ((ci as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    threads,
                    blind_pool: blind_pool.clone(),
                    open_sheet,
                },
            );
            let cand_score = eval_res.score;
            let cand_sp = team_species_set(&canon);

            // Find closest archetype in Master Pool by shared species count
            let mut best_overlap = 0usize;
            let mut best_idx = 0usize;
            for (mi, mt) in master_teams.iter().enumerate() {
                let m_sp = team_species_set(mt["sets"].as_array().unwrap());
                let ov = cand_sp.intersection(&m_sp).count();
                if ov > best_overlap {
                    best_overlap = ov;
                    best_idx = mi;
                }
            }

            let species_names: Vec<String> = canon
                .iter()
                .filter_map(|s| s["species"].as_str().map(String::from))
                .collect();

            if best_overlap >= 4 {
                let inc_id = master_teams[best_idx]["id"].as_str().unwrap_or("?").to_string();
                let inc_score = master_teams[best_idx]["score"].as_f64().unwrap_or(0.5);
                if cand_score >= inc_score + 0.015 && cand_score >= min_admit_score {
                    println!(
                        "  {:22} vs MasterPool = {:.4} ({:.1}%) [overlap={} w/ {} ({:.4})] -> REPLACED INCUMBENT",
                        cid,
                        cand_score,
                        cand_score * 100.0,
                        best_overlap,
                        inc_id,
                        inc_score
                    );
                    master_teams[best_idx] = json!({
                        "id": cid,
                        "origin": origin_tag,
                        "score": cand_score,
                        "pool24_wr": cand_score,
                        "weight": 1.0,
                        "drawWeight": 1.0,
                        "species": species_names,
                        "sets": canon,
                    });
                    admitted += 1;
                } else {
                    println!(
                        "  {:22} vs MasterPool = {:.4} ({:.1}%) [overlap={} w/ {} ({:.4})] -> REJECTED",
                        cid,
                        cand_score,
                        cand_score * 100.0,
                        best_overlap,
                        inc_id,
                        inc_score
                    );
                }
            } else if cand_score >= min_admit_score {
                println!(
                    "  {:22} vs MasterPool = {:.4} ({:.1}%) [new archetype, max_overlap={}] -> ADMITTED TO POOL",
                    cid,
                    cand_score,
                    cand_score * 100.0,
                    best_overlap
                );
                master_teams.push(json!({
                    "id": cid,
                    "origin": origin_tag,
                    "score": cand_score,
                    "pool24_wr": cand_score,
                    "weight": 1.0,
                    "drawWeight": 1.0,
                    "species": species_names,
                    "sets": canon,
                }));
                admitted += 1;
                if master_teams.len() > max_pool {
                    // Evict weakest team in Master Pool
                    let mut min_i = 0usize;
                    let mut min_sc = f64::INFINITY;
                    for (mi, mt) in master_teams.iter().enumerate() {
                        let sc = mt["score"].as_f64().unwrap_or(0.5);
                        if sc < min_sc {
                            min_sc = sc;
                            min_i = mi;
                        }
                    }
                    let evicted_id = master_teams[min_i]["id"].as_str().unwrap_or("?").to_string();
                    println!(
                        "    [Pool > {}] Evicted weakest team {} (score={:.4})",
                        max_pool, evicted_id, min_sc
                    );
                    master_teams.remove(min_i);
                }
            } else {
                println!(
                    "  {:22} vs MasterPool = {:.4} ({:.1}%) [new archetype, max_overlap={}] -> REJECTED (below {:.3})",
                    cid,
                    cand_score,
                    cand_score * 100.0,
                    best_overlap,
                    min_admit_score
                );
            }
        }
        println!("Master Pool admission summary: {} admitted.", admitted);
    }

    // Sort Master Pool by descending score and assign uniform draw weights
    master_teams.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
            .unwrap()
    });
    let n_teams = master_teams.len().max(1) as f64;
    let mut mean_score = 0.0;
    for t in &mut master_teams {
        t["weight"] = json!(1.0 / n_teams);
        t["drawWeight"] = json!(1.0 / n_teams);
        mean_score += t["score"].as_f64().unwrap_or(0.5) / n_teams;
    }

    if let Some(p) = master_path.parent() {
        std::fs::create_dir_all(p).unwrap();
    }
    let out_json = json!({
        "format": "nc2000-eco-master-pool-v1",
        "max_pool": max_pool,
        "mean_score": mean_score,
        "teams": master_teams,
    });
    write_json_atomic(&master_path, &out_json);
    println!(
        "Saved Master Pool ({} teams, mean_score={:.4}) -> {}",
        master_teams.len(),
        mean_score,
        master_path.display()
    );
}

fn count_legal_155_cores(sets: &[Value]) -> usize {
    let n = sets.len().min(6);
    let mut cnt = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let li = sets[i]["level"].as_u64().unwrap_or(50);
                let lj = sets[j]["level"].as_u64().unwrap_or(50);
                let lk = sets[k]["level"].as_u64().unwrap_or(50);
                if li + lj + lk <= 155 {
                    cnt += 1;
                }
            }
        }
    }
    cnt
}

fn format_core_mask(mask: u8, sets: &[Value]) -> String {
    let mut names = Vec::new();
    for slot in 0..sets.len().min(6) {
        if (mask & (1u8 << slot)) != 0 {
            let sp = sets[slot]["species"].as_str().unwrap_or("?");
            let lv = sets[slot]["level"].as_u64().unwrap_or(50);
            names.push(format!("{sp}(L{lv})"));
        }
    }
    names.join("+")
}

/// Open-Sheet Core Tomography (`meta_eco tomography`):
/// Measures exact per-slot pick frequency `p_i`, conditional winrate `w_i`,
/// vitality `v_i = p_i * w_i`, effective core count `K_eff = exp(H(pi))`,
/// and identifies vestigial hitchhikers vs true 4-Ace / flex-core structures.
fn cmd_tomography(args: &[String]) {
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let pool_file = flag(args, "--pool").expect("--pool TEAMS.json");
    let gfile = flag(args, "--gauntlet-file").unwrap_or_else(|| {
        root.join("data/team-pool-v2/team-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let eval_games: u32 = num(args, "--eval-games", 2);
    let top_n: usize = num(args, "--top", 50);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let seed: u64 = num(args, "--seed", 2026102901);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    let (agent_iters, agent_c, blind_pool, open_sheet) = parse_agent_spec(args, &root, &dex, &gen);

    let (bench_teams, bench_weights) = load_gauntlet_file(&dex, &gen, &gfile);
    let pv: Value = serde_json::from_str(&std::fs::read_to_string(&pool_file).unwrap()).unwrap();
    let teams_arr = pv["teams"]
        .as_array()
        .expect("--pool must contain a 'teams' array");

    println!(
        "=== Open-Sheet Core Tomography ({} teams vs {} benchmark teams, {} games/opp = {} games/team, open_sheet={}) ===",
        teams_arr.len().min(top_n),
        bench_teams.len(),
        eval_games,
        eval_games as usize * bench_teams.len(),
        open_sheet
    );

    let mut report_teams: Vec<Value> = Vec::new();
    for (idx, t) in teams_arr.iter().take(top_n).enumerate() {
        let tid = t["id"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| format!("team-{idx}"));
        let raw_sets = t["sets"].as_array().expect("team missing sets");
        let canon = gen.canonize(&dex, raw_sets).expect("canonize team");
        let cand_sets = to_sets(&canon).expect("to_sets");

        let res = gauntlet_eval_weighted(
            &dex,
            &cand_sets,
            &bench_teams,
            &bench_weights,
            &WeightedEvalCfg {
                games_per_opponent: eval_games,
                agent_iters,
                c: agent_c,
                max_turns,
                seed,
                threads,
                blind_pool: blind_pool.clone(),
                open_sheet,
            },
        );

        let k_eff = res.effective_cores();
        let legal_cores = count_legal_155_cores(&canon);

        let mut sorted_cores: Vec<(u8, f64, f64)> = res
            .core_stats
            .iter()
            .map(|(&mask, &(picks, wins))| (mask, picks, wins))
            .collect();
        sorted_cores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut hitchhikers = Vec::new();
        let mut slot_rows = Vec::new();
        for slot in 0..canon.len().min(6) {
            let sp = canon[slot]["species"].as_str().unwrap_or("?");
            let lv = canon[slot]["level"].as_u64().unwrap_or(50);
            let item = canon[slot]["item"].as_str().unwrap_or("(none)");
            let p_i = res.slot_picks[slot];
            let w_i = if p_i > 1e-9 {
                res.slot_wins[slot] / p_i
            } else {
                0.0
            };
            let v_i = res.slot_wins[slot];
            let role = if p_i < 0.08 {
                hitchhikers.push(format!("{sp}(L{lv})"));
                "HITCHHIKER"
            } else if p_i >= 0.65 {
                "PRIMARY_ACE"
            } else if p_i >= 0.30 {
                "FLEX_CORE"
            } else {
                "COUNTER_PICK"
            };
            slot_rows.push(json!({
                "slot": slot,
                "species": sp,
                "level": lv,
                "item": item,
                "pick_rate": p_i,
                "cond_winrate": w_i,
                "vitality": v_i,
                "role": role,
            }));
        }

        println!(
            "\n[{:2}] {:22} | Open-WR: {:5.1}% ({:.4}) | K_eff: {:4.2} / {:2} legal cores | Hitchhikers: {}",
            idx + 1,
            tid,
            res.score * 100.0,
            res.score,
            k_eff,
            legal_cores,
            if hitchhikers.is_empty() {
                "None (all 6 active)".to_string()
            } else {
                hitchhikers.join(", ")
            }
        );
        for srow in &slot_rows {
            println!(
                "     slot {}: {:12} L{:2} @ {:14} | pick={:5.1}% | cond_wr={:5.1}% | vitality={:.3} [{}]",
                srow["slot"].as_u64().unwrap(),
                srow["species"].as_str().unwrap(),
                srow["level"].as_u64().unwrap(),
                srow["item"].as_str().unwrap(),
                srow["pick_rate"].as_f64().unwrap() * 100.0,
                srow["cond_winrate"].as_f64().unwrap() * 100.0,
                srow["vitality"].as_f64().unwrap(),
                srow["role"].as_str().unwrap(),
            );
        }
        let top_cores_str: Vec<String> = sorted_cores
            .iter()
            .take(3)
            .map(|&(mask, picks, wins)| {
                let share = picks * 100.0;
                let cwr = wins / picks.max(1e-9) * 100.0;
                format!("{} ({:.0}% pick, {:.0}% wr)", format_core_mask(mask, &canon), share, cwr)
            })
            .collect();
        println!("     top cores: {}", top_cores_str.join("  ||  "));

        let top_cores_json: Vec<Value> = sorted_cores
            .iter()
            .take(6)
            .map(|&(mask, picks, wins)| {
                json!({
                    "mask": mask,
                    "core": format_core_mask(mask, &canon),
                    "pick_rate": picks,
                    "cond_winrate": wins / picks.max(1e-9),
                })
            })
            .collect();

        report_teams.push(json!({
            "id": tid,
            "open_score": res.score,
            "effective_cores": k_eff,
            "legal_155_cores": legal_cores,
            "hitchhiker_count": hitchhikers.len(),
            "hitchhikers": hitchhikers,
            "slots": slot_rows,
            "top_cores": top_cores_json,
            "sets": canon,
        }));
    }

    report_teams.sort_by(|a, b| {
        b["open_score"]
            .as_f64()
            .unwrap()
            .partial_cmp(&a["open_score"].as_f64().unwrap())
            .unwrap()
    });

    if let Some(out_path) = flag(args, "--out") {
        let out_p = PathBuf::from(out_path);
        if let Some(parent) = out_p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let out_json = json!({
            "format": "nc2000-core-tomography-v1",
            "gauntlet_file": gfile,
            "eval_games_per_opp": eval_games,
            "open_sheet": open_sheet,
            "teams": report_teams,
        });
        write_json_atomic(&out_p, &out_json);
        println!("\nSaved tomography report -> {}", out_p.display());
    }
}

fn export_kodoku_elites(
    ledger: &IslandLedger,
    carry: usize,
    open_sheet: bool,
    elites_path: &Path,
) -> usize {
    // Export top Kodoku elites with full per-slot vitality + marginal value telemetry.
    // Require evolved strains (`born_wave > 0`) to have survived at least 2 waves (`cum_games >= 400.0`
    // when multi-wave) and sort conservatively by `0.5 * bayes_fit + 0.5 * lcb95` to eliminate 1-wave
    // Winner's Curse spikes.
    let min_evolved_games = if ledger.wave >= 2 { 400.0 } else { 24.0 };
    let mut all_measured: Vec<Strain> = ledger
        .active_strains
        .iter()
        .chain(ledger.archived_strains.iter())
        .filter(|s| {
            if s.cum_games < 24.0 {
                return false;
            }
            if s.born_wave > 0 {
                if s.cum_games < min_evolved_games {
                    return false;
                }
                if !s.last_op.starts_with("level-tweak") {
                    if let Some((_op, slot_str)) = s.last_op.split_once('@') {
                        if let Ok(mut_slot) = slot_str.parse::<usize>() {
                            let g = s.cum_games.max(1.0);
                            if mut_slot < 6 && s.slot_picks[mut_slot] / g < 0.10 {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        })
        .cloned()
        .collect();
    all_measured.sort_by(|a, b| {
        let qa = 0.5 * (a.bayes_fit + a.deme_bonus) + 0.5 * a.lcb95;
        let qb = 0.5 * (b.bayes_fit + b.deme_bonus) + 0.5 * b.lcb95;
        qb.partial_cmp(&qa).unwrap()
    });

    let mut elite_json = Vec::new();
    let mut seen_k = BTreeSet::new();
    for s in all_measured.into_iter().take(carry * 2) {
        let k = team_key(&s.team);
        if !seen_k.insert(k) {
            continue;
        }
        let g = s.cum_games.max(1.0);
        let slot_telemetry: Vec<Value> = (0..s.team.len().min(6))
            .map(|i| {
                let sp = s.team[i]["species"].as_str().unwrap_or("?");
                let lv = s.team[i]["level"].as_u64().unwrap_or(50);
                let p_i = s.slot_picks[i] / g;
                let w_i = if s.slot_picks[i] > 0.0 {
                    s.slot_wins[i] / s.slot_picks[i]
                } else {
                    0.0
                };
                let dv_i = s.slot_marginals[i] / g;
                let war_i = s.slot_wars[i];
                let moves_arr = s.team[i]["moves"].as_array();
                let move_telemetry: Vec<Value> = (0..4)
                    .filter_map(|mk| {
                        let mname = moves_arr.and_then(|a| a.get(mk)).and_then(Value::as_str)?;
                        Some(json!({
                            "move": mname,
                            "marginal_dq": s.move_dqs[i][mk],
                        }))
                    })
                    .collect();
                json!({
                    "slot": i,
                    "species": sp,
                    "level": lv,
                    "pick_rate": p_i,
                    "cond_winrate": w_i,
                    "vitality": s.slot_wins[i] / g,
                    "marginal_value": dv_i,
                    "war": war_i,
                    "move_telemetry": move_telemetry,
                })
            })
            .collect();
        elite_json.push(json!({
            "id": s.id,
            "island": ledger.island,
            "parent_id": s.parent_id,
            "born_wave": s.born_wave,
            "last_op": s.last_op,
            "symbiont_of": s.symbiont_of,
            "mix_alpha": s.mix_alpha,
            "deme_bonus": s.deme_bonus,
            "cum_games": s.cum_games,
            "raw_winrate": s.cum_wins / g,
            "bayes_winrate": s.bayes_fit,
            "lcb95": s.lcb95,
            "bench_games": s.bench_games,
            "bench_wins": s.bench_wins,
            "bench_winrate": s.bench_wins / s.bench_games.max(1.0),
            "effective_cores": s.effective_cores,
            "slot_telemetry": slot_telemetry,
            "weight": s.mix_alpha,
            "drawWeight": s.mix_alpha,
            "sets": s.team,
        }));
    }
    let count = elite_json.len();
    let out_val = json!({
        "format": "nc2000-kodoku-elites-v1",
        "island": ledger.island,
        "waves": ledger.wave,
        "open_sheet": open_sheet,
        "teams": elite_json,
    });
    write_json_atomic(elites_path, &out_val);
    count
}

/// Pure-Strength Open-Sheet Kodoku (`meta_eco kodoku`):
/// Evolves a population seeded from diverse human + evolved teams under Open Team Sheet
/// (`OpenAgent`), with:
/// 1. ZERO diversity quotas or archetype caps (`max_per_archetype = usize::MAX`),
/// 2. Within-party vitality-targeted mutation (`TeamGen::propose_targeted`) that surgical-strikes
///    low-vitality / hitchhiker slots while preserving high-vitality 3-mon cores,
/// 3. Active-Pick Deconfounding (clamping positive wave deltas of unpicked `@slot` mutations to 0).
fn cmd_kodoku(args: &[String]) {
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let island = flag(args, "--island").unwrap_or_else(|| "kodoku".into());
    let out_dir = PathBuf::from(flag(args, "--out-dir").expect("--out-dir DIR"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let default_ledger_name = if island == "kodoku" {
        "kodoku-ledger.json".to_string()
    } else {
        format!("{island}-ledger.json")
    };
    let default_elites_name = if island == "kodoku" {
        "kodoku-elites.json".to_string()
    } else {
        format!("{island}-elites.json")
    };
    let ledger_path = flag(args, "--ledger")
        .map(PathBuf::from)
        .unwrap_or_else(|| out_dir.join(default_ledger_name));
    let elites_path = flag(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| out_dir.join(default_elites_name));

    let waves: u32 = num(args, "--waves", 6);
    let carry: usize = num(args, "--carry", 10);
    let buds: usize = num(args, "--buds", 10);
    let eval_games: u32 = num(args, "--eval-games", 2);
    let prior_games: f64 = num(args, "--prior-games", 16.0);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let max_seconds: u64 = num(args, "--max-seconds", 0);
    let base_seed: u64 = num(args, "--seed", 2026102999);
    let shuffle_seeds = args.iter().any(|a| a == "--shuffle-seeds");
    let random_founders: usize = num(args, "--random-founders", 0);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    let (agent_iters, agent_c, blind_pool, open_sheet) = parse_agent_spec(args, &root, &dex, &gen);

    let gfile = flag(args, "--gauntlet-file").unwrap_or_else(|| {
        root.join("data/team-pool-v2/team-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let (bench_teams, bench_weights) = load_gauntlet_file(&dex, &gen, &gfile);
    let catalog_path = root
        .join("data/team-pool-v2/team-pool.json")
        .to_string_lossy()
        .into_owned();
    let (catalog_teams, _) = load_gauntlet_file(&dex, &gen, &catalog_path);
    let belief_path = flag(args, "--belief-pool")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data/belief-pool-v3/belief-pool.json"));
    let bp_text = std::fs::read_to_string(&belief_path).unwrap();
    let raw_mp: MetaPool = serde_json::from_str(&bp_text).expect("parse belief-pool");
    let deme_base_mp = normalize_belief_pool(&dex, &gen, &raw_mp);
    let deme_cand_pool = Arc::new(deme_base_mp.clone());

    let t_run_start = std::time::Instant::now();
    let mut ledger: IslandLedger = if ledger_path.exists() {
        let txt = std::fs::read_to_string(&ledger_path).unwrap();
        let mut l: IslandLedger = serde_json::from_str(&txt).expect("valid kodoku-ledger.json");
        l.island = island.clone();
        l
    } else {
        let seed_files_str = flag(args, "--seeds").unwrap_or_else(|| {
            root.join("data/team-pool-v2/team-pool.json")
                .to_string_lossy()
                .into_owned()
        });
        let mut rng = SplitMix64::new(base_seed);
        let mut active: Vec<Strain> = Vec::new();
        let mut seen_keys: BTreeSet<String> = BTreeSet::new();
        let mut seq = 0u64;
        let pop_target = carry + buds;
        let seed_quota = pop_target.saturating_sub(random_founders);

        let mut raw_candidates: Vec<(String, Vec<Value>)> = Vec::new();
        for sf in seed_files_str.split(',') {
            let sf = sf.trim();
            if sf.is_empty() {
                continue;
            }
            let Ok(txt) = std::fs::read_to_string(sf) else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<Value>(&txt) else {
                continue;
            };
            let Some(arr) = v["teams"].as_array() else {
                continue;
            };
            for t in arr {
                let Some(sets) = t["sets"].as_array() else {
                    continue;
                };
                let origin_id = t["id"].as_str().unwrap_or("seed").to_string();
                raw_candidates.push((origin_id, sets.clone()));
            }
        }
        if shuffle_seeds && raw_candidates.len() > 1 {
            for i in (1..raw_candidates.len()).rev() {
                let j = (rng.next() as usize) % (i + 1);
                raw_candidates.swap(i, j);
            }
        }

        for (origin_id, mut raw_sets) in raw_candidates {
            if active.len() >= seed_quota {
                break;
            }
            gen.optimize_team_dominance(&dex, &mut raw_sets, &mut rng);
            let Some(canon) = gen.canonize(&dex, &raw_sets) else {
                continue;
            };
            let k = team_key(&canon);
            if !seen_keys.insert(k) {
                continue;
            }
            let id = format!("{island}-s{seq:04}");
            seq += 1;
            active.push(Strain {
                id,
                parent_id: origin_id,
                niche: "kodoku".into(),
                anchors: String::new(),
                max_staples: String::new(),
                born_wave: 0,
                last_op: "founder".into(),
                abundance: 1.0 / (pop_target.max(1) as f64),
                cum_games: 0.0,
                cum_wins: 0.0,
                bench_games: 0.0,
                bench_wins: 0.0,
                eco_games: 0.0,
                eco_wins: 0.0,
                last_wave_fit: 0.50,
                parent_prior: 0.50,
                bayes_fit: 0.50,
                lcb95: 0.0,
                slot_picks: [0.0; 6],
                slot_wins: [0.0; 6],
                slot_marginals: [0.0; 6],
                slot_wars: [0.0; 6],
                move_dqs: [[0.0; 4]; 6],
                last_wave_slot_picks: [0.0; 6],
                last_wave_slot_marginals: [0.0; 6],
                effective_cores: 1.0,
                symbiont_of: None,
                mix_alpha: 1.0,
                deme_bonus: 0.0,
                team: canon,
            });
        }

        let free_constraint = NicheConstraint::default();
        let mut rand_tries = 0usize;
        while active.len() < pop_target && rand_tries < pop_target * 8 {
            rand_tries += 1;
            let Some(canon) = gen.random_team_constrained(&dex, &mut rng, &[], &free_constraint, 300) else {
                continue;
            };
            let k = team_key(&canon);
            if !seen_keys.insert(k) {
                continue;
            }
            let id = format!("{island}-s{seq:04}");
            seq += 1;
            active.push(Strain {
                id,
                parent_id: "random-founder".into(),
                niche: "kodoku".into(),
                anchors: String::new(),
                max_staples: String::new(),
                born_wave: 0,
                last_op: "founder".into(),
                abundance: 1.0 / (pop_target.max(1) as f64),
                cum_games: 0.0,
                cum_wins: 0.0,
                bench_games: 0.0,
                bench_wins: 0.0,
                eco_games: 0.0,
                eco_wins: 0.0,
                last_wave_fit: 0.45,
                parent_prior: 0.45,
                bayes_fit: 0.45,
                lcb95: 0.0,
                slot_picks: [0.0; 6],
                slot_wins: [0.0; 6],
                slot_marginals: [0.0; 6],
                slot_wars: [0.0; 6],
                move_dqs: [[0.0; 4]; 6],
                last_wave_slot_picks: [0.0; 6],
                last_wave_slot_marginals: [0.0; 6],
                effective_cores: 1.0,
                symbiont_of: None,
                mix_alpha: 1.0,
                deme_bonus: 0.0,
                team: canon,
            });
        }

        let l = IslandLedger {
            island: island.clone(),
            wave: 0,
            next_strain_seq: seq,
            rng: rng.0,
            niches: Vec::new(),
            active_strains: active,
            archived_strains: Vec::new(),
            wave_history: Vec::new(),
        };
        write_json_atomic(&ledger_path, &l);
        l
    };

    let mut rng = SplitMix64(ledger.rng);
    let mut waves_this_run = 0u32;
    let mut avg_wave_secs = 90.0f64;

    while ledger.wave < waves {
        if max_seconds > 0 && waves_this_run > 0 {
            let elapsed = t_run_start.elapsed().as_secs_f64();
            if elapsed + avg_wave_secs * 0.65 >= max_seconds as f64 {
                eprintln!(
                    "[{island}] wall-clock budget reached ({elapsed:.0}s / {max_seconds}s after {waves_this_run} waves, total wave={}); stopping cleanly.",
                    ledger.wave
                );
                break;
            }
        }

        let w = ledger.wave + 1;
        let t_wave = std::time::Instant::now();
        let wave_seed = base_seed ^ (w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

        // Pure-strength selection + vitality-targeted budding with ZERO archetype caps (`usize::MAX`)
        if w > 1 && ledger.active_strains.iter().all(|s| s.cum_games > 0.0) {
            bud_next_wave_by_niche(
                &dex,
                &gen,
                &island,
                w - 1,
                carry,
                buds,
                usize::MAX,
                &mut ledger,
                &mut rng,
            );
        }

        // Co-evolutionary apex predator panel (top 2 active leaders) + Rotating Full-Catalog Reservoir (2 teams from `team-pool-v2`)
        let mut vet_indices: Vec<usize> = (0..ledger.active_strains.len())
            .filter(|&i| ledger.active_strains[i].cum_games > 0.0)
            .collect();
        if vet_indices.is_empty() {
            vet_indices = (0..ledger.active_strains.len()).collect();
        }
        vet_indices.sort_by(|&a, &b| {
            ledger.active_strains[b]
                .bayes_fit
                .partial_cmp(&ledger.active_strains[a].bayes_fit)
                .unwrap()
        });
        let mut eco_opps: Vec<Vec<PokemonSet>> = Vec::new();
        let mut eco_weights: Vec<f64> = Vec::new();
        for &vi in vet_indices.iter().take(2) {
            if let Ok(sets) = to_sets(&ledger.active_strains[vi].team) {
                eco_opps.push(sets);
                eco_weights.push(1.0);
            }
        }
        if !catalog_teams.is_empty() {
            let cat_start = ((w as usize).saturating_sub(1) * 2) % catalog_teams.len();
            for k in 0..2.min(catalog_teams.len()) {
                let c_idx = (cat_start + k) % catalog_teams.len();
                eco_opps.push(catalog_teams[c_idx].clone());
                eco_weights.push(1.0);
            }
        }
        let eco_wsum: f64 = eco_weights.iter().sum();
        if eco_wsum > 0.0 {
            for ew in &mut eco_weights {
                *ew /= eco_wsum;
            }
        }

        // Rotate a 12-team slice of the benchmark each wave (staggered per island when `island != "kodoku"`)
        let slice_len = bench_teams.len().min(12);
        let island_offset = if island == "kodoku" {
            0usize
        } else {
            (base_seed as usize) % bench_teams.len().max(1)
        };
        let start_idx =
            (island_offset + (w as usize).saturating_sub(1) * slice_len) % bench_teams.len().max(1);
        let mut wave_bench_teams = Vec::with_capacity(slice_len);
        let mut wave_bench_weights = Vec::with_capacity(slice_len);
        for k in 0..slice_len {
            let idx = (start_idx + k) % bench_teams.len();
            wave_bench_teams.push(bench_teams[idx].clone());
            wave_bench_weights.push(bench_weights[idx]);
        }
        let wb_sum: f64 = wave_bench_weights.iter().sum();
        if wb_sum > 0.0 {
            for wv in &mut wave_bench_weights {
                *wv /= wb_sum;
            }
        }

        let screen_games: u32 = num(args, "--screen-games", 2);
        let refine_games: u32 = num(args, "--refine-games", eval_games.max(2));
        let support_k: usize = num(args, "--support", 3);
        let use_core_nash = !args.iter().any(|a| a == "--legacy-eval");

        let bench_n_games = if use_core_nash {
            (refine_games as usize * support_k * support_k * slice_len) as f64
        } else {
            (eval_games as usize * slice_len) as f64
        };
        let eco_n_games = if use_core_nash {
            (refine_games as usize * support_k * support_k * eco_opps.len()) as f64
        } else {
            (eval_games as usize * eco_opps.len()) as f64
        };
        let total_w_games = bench_n_games + eco_n_games;

        let inner_threads = (threads / 4).max(2);
        let dex_ref = &dex;
        let wb_teams_ref = &wave_bench_teams;
        let wb_weights_ref = &wave_bench_weights;
        let eco_opps_ref = &eco_opps;
        let eco_weights_ref = &eco_weights;

        // Partition active strains into Pass-1 (founders/parents) and Pass-2 (single-slot children
        // whose parent is evaluated in Pass 1) so Pass-2 children inherit `R_{-ms}(P)` verbatim.
        let active_id_set: BTreeSet<String> = ledger
            .active_strains
            .iter()
            .map(|s| s.id.clone())
            .collect();
        let single_slot_mut = |s: &Strain| -> Option<usize> {
            if !use_core_nash || s.born_wave == 0 || s.last_op.starts_with("level-tweak") {
                return None;
            }
            let (_op, slot_str) = s.last_op.split_once('@')?;
            let ms = slot_str.parse::<usize>().ok()?;
            if ms < 6 { Some(ms) } else { None }
        };
        let pass1_id_set: BTreeSet<String> = ledger
            .active_strains
            .iter()
            .filter(|s| single_slot_mut(s).is_none() || !active_id_set.contains(&s.parent_id))
            .map(|s| s.id.clone())
            .collect();

        let mut parent_subgames: BTreeMap<String, (Vec<OpponentStage2Subgame>, Vec<OpponentStage2Subgame>, f64)> =
            BTreeMap::new();
        let mut eval_outputs: BTreeMap<
            String,
            (
                nc2000_bot::teamgen::EvalResult,
                [f64; 6],
                [[f64; 4]; 6],
                nc2000_bot::teamgen::EvalResult,
                [f64; 6],
                [[f64; 4]; 6],
            ),
        > = BTreeMap::new();

        // Pass 1: Evaluate all parents / veterans / multi-slot strains
        if use_core_nash {
            let pass1_cands: Vec<FlatCoreNashCand<'_>> = ledger
                .active_strains
                .iter()
                .filter(|s| pass1_id_set.contains(&s.id))
                .map(|s| FlatCoreNashCand {
                    sid: s.id.clone(),
                    sets: to_sets(&s.team).expect("valid strain team"),
                    inh_bench: None,
                    inh_eco: None,
                })
                .collect();
            let p1_out = core_nash_eval_flat_pass(
                dex_ref,
                &pass1_cands,
                wb_teams_ref,
                wb_weights_ref,
                wave_seed ^ 0xA5A5_5A5A_u64,
                eco_opps_ref,
                eco_weights_ref,
                wave_seed ^ 0x5A5A_A5A5_u64,
                screen_games,
                refine_games,
                support_k,
                agent_iters,
                agent_c,
                max_turns,
                threads,
            );
            for (
                sid,
                bench_res,
                bench_marg,
                bench_mdq,
                bench_sub,
                eco_res,
                eco_marg,
                eco_mdq,
                eco_sub,
            ) in p1_out
            {
                let w_sc = 0.70 * bench_res.score + 0.30 * eco_res.score;
                parent_subgames.insert(sid.clone(), (bench_sub, eco_sub, w_sc));
                eval_outputs.insert(
                    sid,
                    (bench_res, bench_marg, bench_mdq, eco_res, eco_marg, eco_mdq),
                );
            }
        } else {
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for s in ledger
                    .active_strains
                    .iter()
                    .filter(|s| pass1_id_set.contains(&s.id))
                {
                    let sid = s.id.clone();
                    let team_c = s.team.clone();
                    let bp = blind_pool.clone();
                    handles.push(scope.spawn(move || {
                        let cand = to_sets(&team_c).expect("valid strain team");
                        let bench_res = gauntlet_eval_weighted(
                            dex_ref,
                            &cand,
                            wb_teams_ref,
                            wb_weights_ref,
                            &WeightedEvalCfg {
                                games_per_opponent: eval_games,
                                agent_iters,
                                c: agent_c,
                                max_turns,
                                seed: wave_seed ^ 0xA5A5_5A5A_u64,
                                threads: inner_threads,
                                blind_pool: bp.clone(),
                                open_sheet,
                            },
                        );
                        let eco_res = if eco_opps_ref.is_empty() {
                            nc2000_bot::teamgen::EvalResult {
                                score: bench_res.score,
                                per_opponent: Vec::new(),
                                games: 0,
                                slot_picks: [0.0; 6],
                                slot_wins: [0.0; 6],
                                core_stats: BTreeMap::new(),
                            }
                        } else {
                            gauntlet_eval_weighted(
                                dex_ref,
                                &cand,
                                eco_opps_ref,
                                eco_weights_ref,
                                &WeightedEvalCfg {
                                    games_per_opponent: eval_games,
                                    agent_iters,
                                    c: agent_c,
                                    max_turns,
                                    seed: wave_seed ^ 0x5A5A_A5A5_u64,
                                    threads: inner_threads,
                                    blind_pool: bp,
                                    open_sheet,
                                },
                            )
                        };
                        (sid, bench_res, eco_res)
                    }));
                }
                for h in handles {
                    let (sid, bench_res, eco_res) =
                        h.join().expect("kodoku pass1 eval thread panicked");
                    let w_sc = 0.70 * bench_res.score + 0.30 * eco_res.score;
                    parent_subgames.insert(sid.clone(), (Vec::new(), Vec::new(), w_sc));
                    eval_outputs.insert(
                        sid,
                        (bench_res, [0.0; 6], [[0.0; 4]; 6], eco_res, [0.0; 6], [[0.0; 4]; 6]),
                    );
                }
            });
        }

        // Pass 2: Evaluate single-slot mutated children with Monotonic Subgame Inheritance `R_{-ms}(P)`
        let p_sub_ref = &parent_subgames;
        let pass2_cands: Vec<FlatCoreNashCand<'_>> = ledger
            .active_strains
            .iter()
            .filter(|s| !pass1_id_set.contains(&s.id))
            .map(|s| {
                let ms_opt = single_slot_mut(s);
                let p_entry = p_sub_ref.get(&s.parent_id);
                FlatCoreNashCand {
                    sid: s.id.clone(),
                    sets: to_sets(&s.team).expect("valid strain team"),
                    inh_bench: ms_opt.zip(p_entry.map(|(bs, _, _)| bs.as_slice())),
                    inh_eco: ms_opt.zip(p_entry.map(|(_, es, _)| es.as_slice())),
                }
            })
            .collect();
        let p2_out = core_nash_eval_flat_pass(
            dex_ref,
            &pass2_cands,
            wb_teams_ref,
            wb_weights_ref,
            wave_seed ^ 0xA5A5_5A5A_u64,
            eco_opps_ref,
            eco_weights_ref,
            wave_seed ^ 0x5A5A_A5A5_u64,
            screen_games,
            refine_games,
            support_k,
            agent_iters,
            agent_c,
            max_turns,
            threads,
        );
        for (sid, bench_res, bench_marg, bench_mdq, _, eco_res, eco_marg, eco_mdq, _) in p2_out {
            eval_outputs.insert(
                sid,
                (bench_res, bench_marg, bench_mdq, eco_res, eco_marg, eco_mdq),
            );
        }

        // Update Pass-1 parents first so their updated `bayes_fit` is available to anchor Pass-2 children
        let mut parent_bayes_by_id: BTreeMap<String, f64> = BTreeMap::new();
        for pass in 0..2 {
            for s in &mut ledger.active_strains {
                let is_p1 = pass1_id_set.contains(&s.id);
                if (pass == 0) != is_p1 {
                    continue;
                }
                let Some((bench_res, bench_marg, bench_mdq, eco_res, eco_marg, eco_mdq)) =
                    eval_outputs.remove(&s.id)
                else {
                    continue;
                };
                let mut bench_score = bench_res.score;
                let mut eco_score = eco_res.score;
                let mut wave_score = 0.70 * bench_score + 0.30 * eco_score;
                let prev_g = s.cum_games;
                let new_g = prev_g + total_w_games;
                for slot in 0..6 {
                    let wp = bench_res.slot_picks[slot] * bench_n_games
                        + eco_res.slot_picks[slot] * eco_n_games;
                    let ww = bench_res.slot_wins[slot] * bench_n_games
                        + eco_res.slot_wins[slot] * eco_n_games;
                    let wm = bench_marg[slot] * bench_n_games
                        + eco_marg[slot] * eco_n_games;
                    s.last_wave_slot_picks[slot] = wp / total_w_games.max(1.0);
                    s.last_wave_slot_marginals[slot] = wm / total_w_games.max(1.0);
                    s.slot_picks[slot] += wp;
                    s.slot_wins[slot] += ww;
                    s.slot_marginals[slot] += wm;
                    for mk in 0..4 {
                        let w_mdq = 0.70 * bench_mdq[slot][mk] + 0.30 * eco_mdq[slot][mk];
                        s.move_dqs[slot][mk] =
                            (s.move_dqs[slot][mk] * prev_g + w_mdq * total_w_games) / new_g.max(1.0);
                    }
                }
                // Paired Subgame Delta Anchoring for Pass-2 children:
                // Because Child `C` and Parent `P` share the exact same Stage-2 submatrix `R_{-ms}(P)`
                // on this wave, `wave_score - p_wave_score` is the exact counterfactual marginal delta
                // `\Delta V_{-ms}(C) - \Delta V_{-ms}(P)` (<= 0 whenever `p_{ms}^*(C) == 0`).
                if !is_p1 {
                    if let (Some(&p_bayes), Some(&(_, _, p_wave))) = (
                        parent_bayes_by_id.get(&s.parent_id),
                        parent_subgames.get(&s.parent_id),
                    ) {
                        let paired_dv = wave_score - p_wave;
                        s.parent_prior = p_bayes;
                        wave_score = (p_bayes + paired_dv).clamp(0.0, 1.0);
                        bench_score = (p_bayes + paired_dv).clamp(0.0, 1.0);
                        eco_score = (p_bayes + paired_dv).clamp(0.0, 1.0);
                    }
                }
                s.bench_games += bench_n_games;
                s.bench_wins += bench_score * bench_n_games;
                s.eco_games += eco_n_games;
                s.eco_wins += eco_score * eco_n_games;
                s.effective_cores = 0.70 * bench_res.effective_cores()
                    + 0.30 * eco_res.effective_cores();
                s.last_wave_fit = wave_score;
                s.cum_games += total_w_games;
                s.cum_wins += wave_score * total_w_games;
                s.recompute_stats(prior_games);
                if is_p1 {
                    parent_bayes_by_id.insert(s.id.clone(), s.bayes_fit);
                }
            }
        }

        let (mu_50, mu_55, beta_l) = recompute_population_wars(&mut ledger.active_strains);

        // Pass 3 (Same-Signature Deme Synergy Certification):
        // When a single-slot MoveSwap or ItemSwap child `C` shares the exact same Team Preview
        // signature `preview_sig(&C.team) == preview_sig(&P.team)` as its parent `P`:
        // - `P` and `C` belong to the same Preview Deme `\Sigma`, where the Primary is the higher
        //   `bayes_fit` strain (`P`) and the Secondary is the lower `bayes_fit` strain (`C`).
        // - If `0.0 < P.bayes_fit - C.bayes_fit <= 0.06` and the mutated slot was actively picked
        //   (`>= 12%`), test whether mixing `C` at equilibrium weight `\alpha^* = 0.20` (4:1)
        //   raises the Deme's blind expected winrate (`V_deme(0.20) > V_blind(P|0) + 0.002`)
        //   and boosts `P` (`V_blind(P|0.20) >= V_blind(P|0)`). If so, certify `C` as `P`'s Symbiont!
        let mut p_blind0_cache: BTreeMap<String, f64> = BTreeMap::new();
        let n_active = ledger.active_strains.len();
        for c_idx in 0..n_active {
            if ledger.active_strains[c_idx].born_wave == 0
                || ledger.active_strains[c_idx].symbiont_of.is_some()
            {
                continue;
            }
            let op = ledger.active_strains[c_idx].last_op.clone();
            let Some((op_kind, slot_str)) = op.split_once('@') else {
                continue;
            };
            if !matches!(op_kind, "move-swap" | "item-swap") {
                continue;
            }
            let Ok(ms) = slot_str.parse::<usize>() else {
                continue;
            };
            if ms >= 6 {
                continue;
            }
            let pid = ledger.active_strains[c_idx].parent_id.clone();
            let Some(p_idx) = ledger.active_strains.iter().position(|x| x.id == pid) else {
                continue;
            };
            if preview_sig(&ledger.active_strains[c_idx].team)
                != preview_sig(&ledger.active_strains[p_idx].team)
            {
                continue;
            }
            // Identify Primary (higher open `bayes_fit`) and Secondary (lower open `bayes_fit`)
            let (prim_idx, sec_idx) = if ledger.active_strains[c_idx].bayes_fit
                > ledger.active_strains[p_idx].bayes_fit
            {
                (c_idx, p_idx)
            } else {
                (p_idx, c_idx)
            };
            let prim_fit = ledger.active_strains[prim_idx].bayes_fit;
            let sec_fit = ledger.active_strains[sec_idx].bayes_fit;
            let sec_pick = ledger.active_strains[sec_idx].last_wave_slot_picks[ms];
            if prim_fit - sec_fit > 0.06 || sec_pick < 0.12 {
                continue;
            }
            let prim_id = ledger.active_strains[prim_idx].id.clone();
            let sec_id = ledger.active_strains[sec_idx].id.clone();
            let prim_sets = to_sets(&ledger.active_strains[prim_idx].team).unwrap();
            let sec_sets = to_sets(&ledger.active_strains[sec_idx].team).unwrap();
            let deme_sig = preview_sig_sets(&prim_sets);
            let deme_seed = wave_seed ^ 0xDE3E_51B1_u64;
            let v_p0 = *p_blind0_cache.entry(prim_id.clone()).or_insert_with(|| {
                let mut mp0 = deme_base_mp.clone();
                mp0.teams.retain(|t| preview_sig_sets(&t.sets) != deme_sig);
                mp0.teams.push(nc2000_bot::preview::MetaTeam {
                    id: format!("{prim_id}-pure"),
                    sets: prim_sets.clone(),
                    weight: Some(0.5),
                });
                eval_blind_asymmetric_prior(
                    &dex,
                    &prim_sets,
                    &wave_bench_teams,
                    &wave_bench_weights,
                    deme_cand_pool.clone(),
                    Arc::new(mp0),
                    2,
                    agent_iters.max(400),
                    agent_c,
                    max_turns,
                    threads,
                    deme_seed,
                )
                .score
            });
            let alpha_star = 0.20;
            let mut mp_mix = deme_base_mp.clone();
            mp_mix.teams.retain(|t| preview_sig_sets(&t.sets) != deme_sig);
            mp_mix.teams.push(nc2000_bot::preview::MetaTeam {
                id: format!("{prim_id}-prim"),
                sets: prim_sets.clone(),
                weight: Some(0.5 * (1.0 - alpha_star)),
            });
            mp_mix.teams.push(nc2000_bot::preview::MetaTeam {
                id: format!("{sec_id}-sec"),
                sets: sec_sets.clone(),
                weight: Some(0.5 * alpha_star),
            });
            let opp_mix = Arc::new(mp_mix);
            let v_p_mix = eval_blind_asymmetric_prior(
                &dex,
                &prim_sets,
                &wave_bench_teams,
                &wave_bench_weights,
                deme_cand_pool.clone(),
                opp_mix.clone(),
                2,
                agent_iters.max(400),
                agent_c,
                max_turns,
                threads,
                deme_seed,
            )
            .score;
            let v_s_mix = eval_blind_asymmetric_prior(
                &dex,
                &sec_sets,
                &wave_bench_teams,
                &wave_bench_weights,
                deme_cand_pool.clone(),
                opp_mix,
                2,
                agent_iters.max(400),
                agent_c,
                max_turns,
                threads,
                deme_seed,
            )
            .score;
            let v_deme = (1.0 - alpha_star) * v_p_mix + alpha_star * v_s_mix;
            let d_deme = v_deme - v_p0;
            eprintln!(
                "  [deme-check wave {w}] Primary={prim_id} (open={prim_fit:.3}) + Symbiont={sec_id} ({op}, open={sec_fit:.3}): V_blind(P|0)={v_p0:.3} -> V_blind(P|0.20)={v_p_mix:.3}, V_blind(S|0.20)={v_s_mix:.3} | V_deme={v_deme:.3} (dDeme={:+.2}pt)",
                d_deme * 100.0
            );
            if d_deme >= 0.002 && v_p_mix >= v_p0 {
                if d_deme > ledger.active_strains[prim_idx].deme_bonus {
                    ledger.active_strains[prim_idx].deme_bonus = d_deme;
                    ledger.active_strains[prim_idx].mix_alpha = 1.0 - alpha_star;
                    ledger.active_strains[sec_idx].symbiont_of = Some(prim_id);
                    ledger.active_strains[sec_idx].mix_alpha = alpha_star;
                    ledger.active_strains[sec_idx].deme_bonus = d_deme;
                }
            }
        }

        ledger.active_strains.sort_by(|a, b| {
            let qa = a.bayes_fit + a.deme_bonus;
            let qb = b.bayes_fit + b.deme_bonus;
            qb.partial_cmp(&qa)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let wave_secs = t_wave.elapsed().as_secs_f64();
        waves_this_run += 1;
        avg_wave_secs = if waves_this_run == 1 {
            wave_secs
        } else {
            0.7 * avg_wave_secs + 0.3 * wave_secs
        };

        let u_bar: f64 = ledger
            .active_strains
            .iter()
            .map(|s| s.bayes_fit)
            .sum::<f64>()
            / (ledger.active_strains.len().max(1) as f64);
        let top = &ledger.active_strains[0];
        let top_g = top.cum_games.max(1.0);
        let hitchhikers = (0..6)
            .filter(|&i| top.slot_picks[i] / top_g < 0.08 && top.slot_wars[i] < -0.015)
            .count();
        let slot_summary: Vec<String> = (0..top.team.len().min(6))
            .map(|i| {
                let sp = top.team[i]["species"].as_str().unwrap_or("?");
                let lv = top.team[i]["level"].as_u64().unwrap_or(50);
                let p_i = top.slot_picks[i] / top_g * 100.0;
                let dv_i = top.slot_marginals[i] / top_g * 100.0;
                let war_i = top.slot_wars[i] * 100.0;
                format!("{sp}(L{lv},p={p_i:.0}%,dV={dv_i:+.1}pt,WAR={war_i:+.1}pt)")
            })
            .collect();
        eprintln!(
            "[{island} wave {w}] ({wave_secs:.0}s) mean_fit={u_bar:.3} | repl_level: mu50={:+.1}pt, mu55={:+.1}pt (beta_L={:+.2}pt/Lv) | #1={} (parent={}, op={}, bayes={:.3}, raw={:.3}, K_eff={:.2}, hitchhikers={}, games={:.0})\n  #1 slots: {}",
            mu_50 * 100.0,
            mu_55 * 100.0,
            beta_l * 100.0,
            top.id,
            top.parent_id,
            top.last_op,
            top.bayes_fit,
            top.cum_wins / top_g,
            top.effective_cores,
            hitchhikers,
            top.cum_games,
            slot_summary.join(" | "),
        );

        ledger.wave_history.push(WaveSummary {
            wave: w,
            mean_payoff: u_bar,
            top_strain_id: top.id.clone(),
            top_abundance: top.abundance,
            top_bayes_fit: top.bayes_fit,
            top_lcb95: top.lcb95,
            active_count: ledger.active_strains.len(),
            archived_count: ledger.archived_strains.len(),
        });
        ledger.wave = w;
        ledger.rng = rng.0;
        write_json_atomic(&ledger_path, &ledger);
        export_kodoku_elites(&ledger, carry, open_sheet, &elites_path);
    }

    let n_saved = export_kodoku_elites(&ledger, carry, open_sheet, &elites_path);
    println!(
        "Saved {} Kodoku elites ({}) -> {}",
        n_saved,
        island,
        elites_path.display()
    );
}

struct CoreJob {
    cell: usize,
    ta: [u8; 3],
    tb: [u8; 3],
    swap: bool,
    battle_seed: String,
    agent_seed: u64,
}

fn push_core_jobs(
    jobs: &mut Vec<CoreJob>,
    cell: usize,
    ta: [u8; 3],
    tb: [u8; 3],
    games: u32,
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed ^ (cell as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25));
    for g in 0..games {
        jobs.push(CoreJob {
            cell,
            ta,
            tb,
            swap: g % 2 == 1,
            battle_seed: rng.battle_seed(),
            agent_seed: rng.next(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn run_forced_preview_games(
    dex: &Dex,
    row_sets: &[PokemonSet],
    col_sets: &[PokemonSet],
    jobs: &[CoreJob],
    threads: usize,
    max_turns: u16,
    build_row_col: &(dyn Fn(&nc2000_engine::state::Battle, usize, usize, u64, u64) -> (Box<dyn nc2000_bot::Agent>, Box<dyn nc2000_bot::Agent>) + Sync),
    log_enabled: bool,
) -> Vec<(usize, f64)> {
    use nc2000_bot::{play_game, GameResult};
    use nc2000_engine::battle::{Outcome, SearchChoice};
    use nc2000_engine::state::Battle;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let run_one = |job: &CoreJob| -> (usize, f64) {
        let seed_a = job.agent_seed ^ 0xA24B_AED4_963E_E407;
        let seed_b = job.agent_seed ^ 0x9FB2_1C65_1E98_DF25;
        let (t1, t2, team1, team2, side_a, side_b) = if job.swap {
            (job.tb, job.ta, col_sets, row_sets, 1usize, 0usize)
        } else {
            (job.ta, job.tb, row_sets, col_sets, 0usize, 1usize)
        };
        let mut b = Battle::from_fixture(dex, &job.battle_seed, team1, team2).unwrap();
        b.set_log_enabled(log_enabled);
        let (mut agent_a, mut agent_b) = build_row_col(&b, side_a, side_b, seed_a, seed_b);
        b.apply_choices(
            dex,
            [Some(SearchChoice::Team(t1)), Some(SearchChoice::Team(t2))],
        )
        .unwrap();
        let res = if job.swap {
            play_game(dex, &mut b, &mut [agent_b.as_mut(), agent_a.as_mut()], max_turns)
        } else {
            play_game(dex, &mut b, &mut [agent_a.as_mut(), agent_b.as_mut()], max_turns)
        }
        .unwrap();
        let p1_score = match res {
            GameResult::Outcome(Outcome::P1Win) => 1.0,
            GameResult::Outcome(Outcome::P2Win) => 0.0,
            GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
        };
        let score = if job.swap { 1.0 - p1_score } else { p1_score };
        (job.cell, score)
    };

    if threads <= 1 {
        return jobs.iter().map(run_one).collect();
    }

    let cursor = AtomicUsize::new(0);
    let mut results: Vec<(usize, (usize, f64))> = Vec::with_capacity(jobs.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads.max(1) {
            let (cursor, run_one) = (&cursor, &run_one);
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let n = cursor.fetch_add(1, Ordering::Relaxed);
                    if n >= jobs.len() {
                        break;
                    }
                    out.push((n, run_one(&jobs[n])));
                }
                out
            }));
        }
        let mut all = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all;
    });
    results.into_iter().map(|(_, r)| r).collect()
}

fn subset_3_sets(sets: &[PokemonSet], triple: [u8; 3]) -> Vec<PokemonSet> {
    vec![
        sets[(triple[0] - 1) as usize].clone(),
        sets[(triple[1] - 1) as usize].clone(),
        sets[(triple[2] - 1) as usize].clone(),
    ]
}

#[derive(Clone, Debug)]
struct OpponentStage2Subgame {
    sup_a: Vec<usize>,
    sup_b: Vec<usize>,
    p_a_s1: Vec<f64>,
    p_b_s1: Vec<f64>,
    mat_v: Vec<f64>,
}

struct CoreNashGauntletCfg<'a> {
    screen_games: u32,
    refine_games: u32,
    support_k: usize,
    agent_iters: u32,
    c: f64,
    smoothing: f64,
    max_turns: u16,
    seed: u64,
    threads: usize,
    inherited: Option<(usize, &'a [OpponentStage2Subgame])>,
}

#[derive(Clone, Debug)]
struct CoreNashEvalResult {
    eval: nc2000_bot::teamgen::EvalResult,
    slot_marginals: [f64; 6],
    move_dqs: [[f64; 4]; 6],
    move_raw_sum: [[f64; 4]; 6],
    move_raw_turns: [u32; 6],
    opp_move_dqs: Vec<[[f64; 4]; 6]>,
    subgames: Vec<OpponentStage2Subgame>,
}

struct TelemetryOpenAgent {
    inner: nc2000_bot::OpenAgent,
    sink: Arc<std::sync::Mutex<([[f64; 4]; 6], [u32; 6])>>,
}

impl nc2000_bot::Agent for TelemetryOpenAgent {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn choose(
        &mut self,
        battle: &nc2000_engine::state::Battle,
        dex: &Dex,
        side: usize,
        choices: &[nc2000_engine::battle::SearchChoice],
    ) -> nc2000_engine::battle::SearchChoice {
        self.inner.choose(battle, dex, side, choices)
    }
}

impl Drop for TelemetryOpenAgent {
    fn drop(&mut self) {
        let (m, t) = self.inner.move_telemetry();
        let mut g = self.sink.lock().unwrap();
        for s in 0..6 {
            g.1[s] += t[s];
            for k in 0..4 {
                g.0[s][k] += m[s][k];
            }
        }
    }
}

fn select_top_k_orthogonal(
    actions: &[[u8; 3]],
    legal: &[usize],
    eq: &[f64],
    br: &[f64],
    k: usize,
) -> Vec<usize> {
    let core_mask = |t: [u8; 3]| -> u8 {
        (1u8 << (t[0] - 1)) | (1u8 << (t[1] - 1)) | (1u8 << (t[2] - 1))
    };
    let mut order: Vec<usize> = (0..legal.len()).collect();
    order.sort_by(|&i, &j| {
        let si = (eq[i] >= 0.01, eq[i] + 0.05 * br[i]);
        let sj = (eq[j] >= 0.01, eq[j] + 0.05 * br[j]);
        sj.0.cmp(&si.0).then_with(|| sj.1.partial_cmp(&si.1).unwrap())
    });

    // Pass 1: select up to `k` distinct 3-mon cores so lead permutations of a single core
    // cannot crowd out alternative 3-mon combinations.
    let target = k.min(legal.len());
    let mut sup: Vec<usize> = Vec::with_capacity(target + 2);
    let mut seen_masks: BTreeSet<u8> = BTreeSet::new();
    for &idx in &order {
        if sup.len() >= target {
            break;
        }
        let a = legal[idx];
        let m = core_mask(actions[a]);
        if seen_masks.insert(m) {
            sup.push(a);
        }
    }
    for &idx in &order {
        if sup.len() >= target {
            break;
        }
        let a = legal[idx];
        if !sup.contains(&a) {
            sup.push(a);
        }
    }

    // Pass 2 (Leave-One-Out Anti-Monopoly Guarantee): if any slot `s` appears in 100% of `sup`,
    // append the highest-ranked legal action excluding `s` so `rem_rows` is never empty.
    if !sup.is_empty() {
        for s1 in 1u8..=6u8 {
            if sup.iter().all(|&a| actions[a].contains(&s1)) {
                if let Some(&idx) = order.iter().find(|&&i| !actions[legal[i]].contains(&s1)) {
                    let a_alt = legal[idx];
                    if !sup.contains(&a_alt) {
                        sup.push(a_alt);
                    }
                }
            }
        }
    }
    sup
}

/// Evaluate a candidate 6-mon party against a weighted opponent gauntlet via Two-Stage
/// Forced-Preview 6v6 Core-Nash (`RolloutAgent` screen -> top-k + leave-one-out coverage
/// support -> `OpenAgent` with `Bayesian-Nash-6v6` hidden bench).
/// When `cfg.inherited = Some((mut_slot, parent_subgames))` is provided for a single-slot
/// mutation, inherits the parent's exact Stage-2 rows excluding `mut_slot` (`R_{-ms}(P)`)
/// and evaluates only the newly mutated slot's rows (`R_{+ms}(C)`), guaranteeing exact
/// minimax subgame monotonicity (`p_{ms}^*(C) = 0 => V^*(C) = V_{-ms}^*(P) <= V^*(P)`).
fn core_nash_eval_weighted(
    dex: &Dex,
    cand: &[PokemonSet],
    opps: &[Vec<PokemonSet>],
    weights: &[f64],
    cfg: &CoreNashGauntletCfg<'_>,
) -> CoreNashEvalResult {
    use nc2000_bot::preview::{legal_action_indices, preview_actions, MatrixEst, RolloutAgent};
    use nc2000_bot::smmcts::{solve_rm_plus, SelRule};
    use nc2000_bot::{OpenAgent, RmConfig};

    if opps.is_empty() {
        return CoreNashEvalResult {
            eval: nc2000_bot::teamgen::EvalResult {
                score: 0.5,
                per_opponent: Vec::new(),
                games: 0,
                slot_picks: [0.0; 6],
                slot_wins: [0.0; 6],
                core_stats: BTreeMap::new(),
            },
            slot_marginals: [0.0; 6],
            move_dqs: [[0.0; 4]; 6],
            move_raw_sum: [[0.0; 4]; 6],
            move_raw_turns: [0; 6],
            opp_move_dqs: Vec::new(),
            subgames: Vec::new(),
        };
    }

    let actions = preview_actions();
    let levels_a: Vec<u8> = cand.iter().map(|s| s.level).collect();
    let legal_a = legal_action_indices(&actions, &levels_a);

    let w_sum: f64 = weights.iter().sum::<f64>().max(1e-12);
    let mut total_score = 0.0f64;
    let mut per_opponent = Vec::with_capacity(opps.len());
    let mut opp_move_dqs = Vec::with_capacity(opps.len());
    let mut subgames = Vec::with_capacity(opps.len());
    let mut slot_picks = [0.0f64; 6];
    let mut slot_wins = [0.0f64; 6];
    let mut slot_marginals = [0.0f64; 6];
    let mut move_dqs = [[0.0f64; 4]; 6];
    let move_acc = Arc::new(std::sync::Mutex::new(([[0.0f64; 4]; 6], [0u32; 6])));
    let mut core_stats: BTreeMap<u8, (f64, f64)> = BTreeMap::new();
    let mut total_games = 0u32;

    for (m, opp) in opps.iter().enumerate() {
        let w_m = weights.get(m).copied().unwrap_or(1.0) / w_sum;
        let opp_seed = cfg.seed ^ ((m as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let levels_b: Vec<u8> = opp.iter().map(|s| s.level).collect();
        let legal_b = legal_action_indices(&actions, &levels_b);

        let inherited_opt = cfg
            .inherited
            .and_then(|(ms, subs)| subs.get(m).map(|ps| (ms, ps)));

        let (sup_a, sup_b, p_a_s1, p_b_s1, mat) = if let Some((mut_slot, p_sub)) = inherited_opt {
            // Monotonic Subgame Inheritance:
            // 1. Inherit all parent Stage-2 rows `R_{-ms}(P)` that do NOT contain `mut_slot` verbatim.
            let ms1 = (mut_slot + 1) as u8;
            let inherited_ia: Vec<usize> = (0..p_sub.sup_a.len())
                .filter(|&ia| !actions[p_sub.sup_a[ia]].contains(&ms1))
                .collect();
            let sup_b = p_sub.sup_b.clone();
            let kb = sup_b.len();
            let p_b_s1 = p_sub.p_b_s1.clone();

            // 2. Screen candidate's legal rows containing `mut_slot` (`R_{+ms}(C)`) against `sup_b`.
            let legal_a_mut: Vec<usize> = legal_a
                .iter()
                .copied()
                .filter(|&a| actions[a].contains(&ms1))
                .collect();
            let new_sup_a: Vec<usize> = if legal_a_mut.is_empty() {
                Vec::new()
            } else {
                let mut screen_jobs =
                    Vec::with_capacity(legal_a_mut.len() * kb * cfg.screen_games as usize);
                for (ia_m, &a) in legal_a_mut.iter().enumerate() {
                    for (ib, &b) in sup_b.iter().enumerate() {
                        push_core_jobs(
                            &mut screen_jobs,
                            ia_m * kb + ib,
                            actions[a],
                            actions[b],
                            cfg.screen_games.max(1),
                            opp_seed ^ 0x5C4E,
                        );
                    }
                }
                let screen_res = run_forced_preview_games(
                    dex,
                    cand,
                    opp,
                    &screen_jobs,
                    cfg.threads,
                    cfg.max_turns,
                    &|_, _, _, sa, sb| {
                        (
                            Box::new(RolloutAgent::new(0.2, sa)),
                            Box::new(RolloutAgent::new(0.2, sb)),
                        )
                    },
                    false,
                );
                let mut screen_mut = MatrixEst::new(legal_a_mut.len(), kb);
                for (cell, sc) in screen_res {
                    screen_mut.record(cell, sc);
                }
                let sm_ref = &screen_mut;
                let sub_m: Vec<f64> = (0..legal_a_mut.len())
                    .flat_map(|ia_m| (0..kb).map(move |ib| sm_ref.at(ia_m, ib)))
                    .collect();
                let (e0_m, e1_m) = solve_rm_plus(&sub_m, [legal_a_mut.len(), kb], 4_000);
                let u_a_m: Vec<f64> = (0..legal_a_mut.len())
                    .map(|ia_m| {
                        (0..kb)
                            .zip(&e1_m)
                            .map(|(ib, &wb)| screen_mut.at(ia_m, ib) * wb)
                            .sum()
                    })
                    .collect();
                let k_mut = (cfg.support_k / 2).max(2).min(legal_a_mut.len());
                let core_mask = |t: [u8; 3]| -> u8 {
                    (1u8 << (t[0] - 1)) | (1u8 << (t[1] - 1)) | (1u8 << (t[2] - 1))
                };
                let mut order: Vec<usize> = (0..legal_a_mut.len()).collect();
                order.sort_by(|&i, &j| {
                    let si = (e0_m[i] >= 0.01, e0_m[i] + 0.05 * u_a_m[i]);
                    let sj = (e0_m[j] >= 0.01, e0_m[j] + 0.05 * u_a_m[j]);
                    sj.0.cmp(&si.0).then_with(|| sj.1.partial_cmp(&si.1).unwrap())
                });
                let mut picked = Vec::with_capacity(k_mut);
                let mut seen_masks = BTreeSet::new();
                for &idx in &order {
                    if picked.len() >= k_mut {
                        break;
                    }
                    let a = legal_a_mut[idx];
                    if seen_masks.insert(core_mask(actions[a])) {
                        picked.push(a);
                    }
                }
                for &idx in &order {
                    if picked.len() >= k_mut {
                        break;
                    }
                    let a = legal_a_mut[idx];
                    if !picked.contains(&a) {
                        picked.push(a);
                    }
                }
                picked
            };

            let mut sup_a: Vec<usize> = inherited_ia.iter().map(|&ia| p_sub.sup_a[ia]).collect();
            for &a in &new_sup_a {
                if !sup_a.contains(&a) {
                    sup_a.push(a);
                }
            }
            let n_inh = inherited_ia.len();
            let ka = sup_a.len();
            let mut p_a_s1 = p_sub.p_a_s1.clone();
            let share_new = 1.0 / (ka.max(1) as f64);
            for &a in &sup_a[n_inh..] {
                p_a_s1[a] = share_new;
            }

            // 3. Simulate Stage-2 OpenAgent games ONLY for the newly added `sup_a[n_inh..]` rows!
            let n_new = ka.saturating_sub(n_inh);
            let mut refine_jobs = Vec::with_capacity(n_new * kb * cfg.refine_games as usize);
            for ia_new in 0..n_new {
                let ia = n_inh + ia_new;
                for ib in 0..kb {
                    push_core_jobs(
                        &mut refine_jobs,
                        ia_new * kb + ib,
                        actions[sup_a[ia]],
                        actions[sup_b[ib]],
                        cfg.refine_games.max(2),
                        opp_seed ^ 0x2222,
                    );
                }
            }
            let rm_cfg = RmConfig {
                iterations: cfg.agent_iters,
                c: cfg.c,
                rule: SelRule::Ucb,
                ..RmConfig::default()
            };
            let pa_ref = &p_a_s1;
            let pb_ref = &p_b_s1;
            let smoothing = cfg.smoothing;
            let move_acc_ref = &move_acc;
            let opp_acc = Arc::new(std::sync::Mutex::new(([[0.0f64; 4]; 6], [0u32; 6])));
            let opp_acc_ref = &opp_acc;
            let refine_res = run_forced_preview_games(
                dex,
                cand,
                opp,
                &refine_jobs,
                cfg.threads,
                cfg.max_turns,
                &|b, side_a, side_b, sa, sb| {
                    let mut oa = OpenAgent::new(rm_cfg.clone(), None, sa)
                        .with_opponent_pick_policy(pb_ref.clone(), smoothing);
                    let mut ob = OpenAgent::new(rm_cfg.clone(), None, sb)
                        .with_opponent_pick_policy(pa_ref.clone(), smoothing);
                    oa.init_at_preview(b, side_a);
                    ob.init_at_preview(b, side_b);
                    (
                        Box::new(TelemetryOpenAgent {
                            inner: oa,
                            sink: move_acc_ref.clone(),
                        }),
                        Box::new(TelemetryOpenAgent {
                            inner: ob,
                            sink: opp_acc_ref.clone(),
                        }),
                    )
                },
                true,
            );
            total_games += refine_jobs.len() as u32;
            {
                let og = opp_acc.lock().unwrap();
                let mut o_mdq = [[0.0f64; 4]; 6];
                for s in 0..6 {
                    let turns = (og.1[s] as f64).max(1.0);
                    for k in 0..4 {
                        o_mdq[s][k] = og.0[s][k] / turns;
                    }
                }
                opp_move_dqs.push(o_mdq);
            }

            let mut mat = MatrixEst::new(ka, kb);
            for (r_idx, &old_ia) in inherited_ia.iter().enumerate() {
                for ib in 0..kb {
                    let v = p_sub.mat_v[old_ia * kb + ib];
                    mat.record(r_idx * kb + ib, v);
                }
            }
            for (cell, sc) in refine_res {
                let ia_new = cell / kb;
                let ib = cell % kb;
                mat.record((n_inh + ia_new) * kb + ib, sc);
            }
            (sup_a, sup_b, p_a_s1, p_b_s1, mat)
        } else {
            // Full Two-Stage Evaluation with Orthogonal Leave-One-Out Support
            let mut screen_jobs =
                Vec::with_capacity(legal_a.len() * legal_b.len() * cfg.screen_games as usize);
            for &a in &legal_a {
                for &b in &legal_b {
                    push_core_jobs(
                        &mut screen_jobs,
                        a * 60 + b,
                        actions[a],
                        actions[b],
                        cfg.screen_games.max(1),
                        opp_seed ^ 0x5C4E,
                    );
                }
            }
            let screen_res = run_forced_preview_games(
                dex,
                cand,
                opp,
                &screen_jobs,
                cfg.threads,
                cfg.max_turns,
                &|_, _, _, sa, sb| {
                    (
                        Box::new(RolloutAgent::new(0.2, sa)),
                        Box::new(RolloutAgent::new(0.2, sb)),
                    )
                },
                false,
            );
            let mut screen = MatrixEst::new(60, 60);
            for (cell, sc) in screen_res {
                screen.record(cell, sc);
            }
            let screen_ref = &screen;
            let sub: Vec<f64> = legal_a
                .iter()
                .flat_map(|&a| legal_b.iter().map(move |&b| screen_ref.at(a, b)))
                .collect();
            let (e0, e1) = solve_rm_plus(&sub, [legal_a.len(), legal_b.len()], 8_000);

            let u_a: Vec<f64> = legal_a
                .iter()
                .map(|&a| {
                    legal_b
                        .iter()
                        .zip(&e1)
                        .map(|(&b, &wb)| screen.at(a, b) * wb)
                        .sum()
                })
                .collect();
            let u_b: Vec<f64> = legal_b
                .iter()
                .map(|&b| {
                    legal_a
                        .iter()
                        .zip(&e0)
                        .map(|(&a, &wa)| (1.0 - screen.at(a, b)) * wa)
                        .sum()
                })
                .collect();

            let sup_a = select_top_k_orthogonal(&actions, &legal_a, &e0, &u_a, cfg.support_k);
            let sup_b = select_top_k_orthogonal(&actions, &legal_b, &e1, &u_b, cfg.support_k);

            let mut p_a_s1 = vec![0.0f64; 60];
            let mut p_b_s1 = vec![0.0f64; 60];
            for (ia, &a) in legal_a.iter().enumerate() {
                p_a_s1[a] = e0[ia];
            }
            for (ib, &b) in legal_b.iter().enumerate() {
                p_b_s1[b] = e1[ib];
            }

            let ka = sup_a.len();
            let kb = sup_b.len();
            let mut refine_jobs = Vec::with_capacity(ka * kb * cfg.refine_games as usize);
            for ia in 0..ka {
                for ib in 0..kb {
                    push_core_jobs(
                        &mut refine_jobs,
                        ia * kb + ib,
                        actions[sup_a[ia]],
                        actions[sup_b[ib]],
                        cfg.refine_games.max(2),
                        opp_seed ^ 0x2222,
                    );
                }
            }
            let rm_cfg = RmConfig {
                iterations: cfg.agent_iters,
                c: cfg.c,
                rule: SelRule::Ucb,
                ..RmConfig::default()
            };
            let pa_ref = &p_a_s1;
            let pb_ref = &p_b_s1;
            let smoothing = cfg.smoothing;
            let move_acc_ref = &move_acc;
            let opp_acc = Arc::new(std::sync::Mutex::new(([[0.0f64; 4]; 6], [0u32; 6])));
            let opp_acc_ref = &opp_acc;
            let refine_res = run_forced_preview_games(
                dex,
                cand,
                opp,
                &refine_jobs,
                cfg.threads,
                cfg.max_turns,
                &|b, side_a, side_b, sa, sb| {
                    let mut oa = OpenAgent::new(rm_cfg.clone(), None, sa)
                        .with_opponent_pick_policy(pb_ref.clone(), smoothing);
                    let mut ob = OpenAgent::new(rm_cfg.clone(), None, sb)
                        .with_opponent_pick_policy(pa_ref.clone(), smoothing);
                    oa.init_at_preview(b, side_a);
                    ob.init_at_preview(b, side_b);
                    (
                        Box::new(TelemetryOpenAgent {
                            inner: oa,
                            sink: move_acc_ref.clone(),
                        }),
                        Box::new(TelemetryOpenAgent {
                            inner: ob,
                            sink: opp_acc_ref.clone(),
                        }),
                    )
                },
                true,
            );
            total_games += refine_jobs.len() as u32;
            {
                let og = opp_acc.lock().unwrap();
                let mut o_mdq = [[0.0f64; 4]; 6];
                for s in 0..6 {
                    let turns = (og.1[s] as f64).max(1.0);
                    for k in 0..4 {
                        o_mdq[s][k] = og.0[s][k] / turns;
                    }
                }
                opp_move_dqs.push(o_mdq);
            }

            let mut mat = MatrixEst::new(ka, kb);
            for (cell, sc) in refine_res {
                mat.record(cell, sc);
            }
            (sup_a, sup_b, p_a_s1, p_b_s1, mat)
        };

        let ka = sup_a.len();
        let kb = sup_b.len();
        let (x_star, y_star) = solve_rm_plus(&mat.v, [ka, kb], 10_000);
        let v_star: f64 = (0..ka)
            .map(|ia| {
                (0..kb)
                    .map(|ib| mat.at(ia, ib) * x_star[ia] * y_star[ib])
                    .sum::<f64>()
            })
            .sum();

        total_score += w_m * v_star;
        per_opponent.push(v_star);

        for ia in 0..ka {
            let p_row = x_star[ia];
            if p_row <= 1e-9 {
                continue;
            }
            let ev_row: f64 = (0..kb).map(|ib| mat.at(ia, ib) * y_star[ib]).sum();
            let triple = actions[sup_a[ia]];
            let mut mask = 0u8;
            for &s1 in &triple {
                let s = (s1 - 1) as usize;
                slot_picks[s] += w_m * p_row;
                slot_wins[s] += w_m * p_row * ev_row;
                mask |= 1u8 << s;
            }
            let entry = core_stats.entry(mask).or_insert((0.0, 0.0));
            entry.0 += w_m * p_row;
            entry.1 += w_m * p_row * ev_row;
        }

        // Exact Leave-One-Out Marginal Contribution \Delta V_{-s}^{(m)} from Stage-2 matrix
        for s in 0..6usize {
            let s1 = (s + 1) as u8;
            let rem_rows: Vec<usize> = (0..ka)
                .filter(|&ia| !actions[sup_a[ia]].contains(&s1))
                .collect();
            let v_minus_s = if !rem_rows.is_empty() {
                let mat_ref = &mat;
                let sub_m: Vec<f64> = rem_rows
                    .iter()
                    .flat_map(|&ia| (0..kb).map(move |ib| mat_ref.at(ia, ib)))
                    .collect();
                let (xr, yr) = solve_rm_plus(&sub_m, [rem_rows.len(), kb], 8_000);
                (0..rem_rows.len())
                    .map(|ir| {
                        (0..kb)
                            .map(|ib| sub_m[ir * kb + ib] * xr[ir] * yr[ib])
                            .sum::<f64>()
                    })
                    .sum()
            } else {
                v_star
            };
            let dv = (v_star - v_minus_s).max(0.0);
            slot_marginals[s] += w_m * dv;
        }

        subgames.push(OpponentStage2Subgame {
            sup_a,
            sup_b,
            p_a_s1,
            p_b_s1,
            mat_v: mat.v,
        });
    }

    let (move_raw_sum, move_raw_turns) = {
        let g = move_acc.lock().unwrap();
        for s in 0..6 {
            let turns = (g.1[s] as f64).max(1.0);
            for k in 0..4 {
                move_dqs[s][k] = g.0[s][k] / turns;
            }
        }
        (g.0, g.1)
    };

    CoreNashEvalResult {
        eval: nc2000_bot::teamgen::EvalResult {
            score: total_score,
            per_opponent,
            games: total_games as usize,
            slot_picks,
            slot_wins,
            core_stats,
        },
        slot_marginals,
        move_dqs,
        move_raw_sum,
        move_raw_turns,
        opp_move_dqs,
        subgames,
    }
}

struct FlatCoreNashCand<'a> {
    sid: String,
    sets: Vec<PokemonSet>,
    inh_bench: Option<(usize, &'a [OpponentStage2Subgame])>,
    inh_eco: Option<(usize, &'a [OpponentStage2Subgame])>,
}

fn combine_single_opp_core_nash(
    parts: Vec<CoreNashEvalResult>,
    weights: &[f64],
    fallback_score: f64,
) -> (
    nc2000_bot::teamgen::EvalResult,
    [f64; 6],
    [[f64; 4]; 6],
    Vec<OpponentStage2Subgame>,
) {
    if parts.is_empty() {
        return (
            nc2000_bot::teamgen::EvalResult {
                score: fallback_score,
                per_opponent: Vec::new(),
                games: 0,
                slot_picks: [0.0; 6],
                slot_wins: [0.0; 6],
                core_stats: BTreeMap::new(),
            },
            [0.0; 6],
            [[0.0; 4]; 6],
            Vec::new(),
        );
    }
    let w_sum: f64 = weights.iter().sum::<f64>().max(1e-12);
    let mut total_score = 0.0f64;
    let mut per_opponent = Vec::with_capacity(parts.len());
    let mut subgames = Vec::with_capacity(parts.len());
    let mut slot_picks = [0.0f64; 6];
    let mut slot_wins = [0.0f64; 6];
    let mut slot_marginals = [0.0f64; 6];
    let mut move_sum = [[0.0f64; 4]; 6];
    let mut move_turns = [0u32; 6];
    let mut core_stats: BTreeMap<u8, (f64, f64)> = BTreeMap::new();
    let mut total_games = 0usize;

    for (m, mut r) in parts.into_iter().enumerate() {
        let w_m = weights.get(m).copied().unwrap_or(1.0) / w_sum;
        total_score += w_m * r.eval.score;
        per_opponent.push(r.eval.score);
        total_games += r.eval.games;
        for s in 0..6 {
            slot_picks[s] += w_m * r.eval.slot_picks[s];
            slot_wins[s] += w_m * r.eval.slot_wins[s];
            slot_marginals[s] += w_m * r.slot_marginals[s];
            move_turns[s] += r.move_raw_turns[s];
            for k in 0..4 {
                move_sum[s][k] += r.move_raw_sum[s][k];
            }
        }
        for (mask, (p, w)) in r.eval.core_stats {
            let entry = core_stats.entry(mask).or_insert((0.0, 0.0));
            entry.0 += w_m * p;
            entry.1 += w_m * w;
        }
        if let Some(sub) = r.subgames.pop() {
            subgames.push(sub);
        }
    }
    let mut move_dqs = [[0.0f64; 4]; 6];
    for s in 0..6 {
        let turns = (move_turns[s] as f64).max(1.0);
        for k in 0..4 {
            move_dqs[s][k] = move_sum[s][k] / turns;
        }
    }
    (
        nc2000_bot::teamgen::EvalResult {
            score: total_score,
            per_opponent,
            games: total_games,
            slot_picks,
            slot_wins,
            core_stats,
        },
        slot_marginals,
        move_dqs,
        subgames,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn core_nash_eval_flat_pass(
    dex: &Dex,
    cands: &[FlatCoreNashCand<'_>],
    wb_teams: &[Vec<PokemonSet>],
    wb_weights: &[f64],
    wb_seed: u64,
    eco_opps: &[Vec<PokemonSet>],
    eco_weights: &[f64],
    eco_seed: u64,
    screen_games: u32,
    refine_games: u32,
    support_k: usize,
    agent_iters: u32,
    agent_c: f64,
    max_turns: u16,
    threads: usize,
) -> Vec<(
    String,
    nc2000_bot::teamgen::EvalResult,
    [f64; 6],
    [[f64; 4]; 6],
    Vec<OpponentStage2Subgame>,
    nc2000_bot::teamgen::EvalResult,
    [f64; 6],
    [[f64; 4]; 6],
    Vec<OpponentStage2Subgame>,
)> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    if cands.is_empty() {
        return Vec::new();
    }
    let n_wb = wb_teams.len();
    let n_eco = eco_opps.len();
    let opps_per_cand = n_wb + n_eco;
    let total_tasks = cands.len() * opps_per_cand;
    const OPP_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

    let cursor = AtomicUsize::new(0);
    let mut task_results: Vec<(usize, CoreNashEvalResult)> = Vec::with_capacity(total_tasks);
    let num_workers = threads.max(1).min(total_tasks.max(1));
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..num_workers {
            let cursor = &cursor;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let t = cursor.fetch_add(1, Ordering::Relaxed);
                    if t >= total_tasks {
                        break;
                    }
                    let ci = t / opps_per_cand;
                    let sub_i = t % opps_per_cand;
                    let item = &cands[ci];
                    let res = if sub_i < n_wb {
                        let m = sub_i;
                        let adj_seed = wb_seed ^ ((m as u64 + 1).wrapping_mul(OPP_MIX)) ^ OPP_MIX;
                        let inh = item
                            .inh_bench
                            .and_then(|(ms, subs)| subs.get(m).map(|ps| (ms, std::slice::from_ref(ps))));
                        core_nash_eval_weighted(
                            dex,
                            &item.sets,
                            std::slice::from_ref(&wb_teams[m]),
                            &[1.0],
                            &CoreNashGauntletCfg {
                                screen_games,
                                refine_games,
                                support_k,
                                agent_iters,
                                c: agent_c,
                                smoothing: 0.15,
                                max_turns,
                                seed: adj_seed,
                                threads: 1,
                                inherited: inh,
                            },
                        )
                    } else {
                        let m = sub_i - n_wb;
                        let adj_seed = eco_seed ^ ((m as u64 + 1).wrapping_mul(OPP_MIX)) ^ OPP_MIX;
                        let inh = item
                            .inh_eco
                            .and_then(|(ms, subs)| subs.get(m).map(|ps| (ms, std::slice::from_ref(ps))));
                        core_nash_eval_weighted(
                            dex,
                            &item.sets,
                            std::slice::from_ref(&eco_opps[m]),
                            &[1.0],
                            &CoreNashGauntletCfg {
                                screen_games,
                                refine_games,
                                support_k,
                                agent_iters,
                                c: agent_c,
                                smoothing: 0.15,
                                max_turns,
                                seed: adj_seed,
                                threads: 1,
                                inherited: inh,
                            },
                        )
                    };
                    local.push((t, res));
                }
                local
            }));
        }
        for h in handles {
            task_results.extend(h.join().expect("core_nash_eval_flat_pass worker panicked"));
        }
    });
    task_results.sort_by_key(|(t, _)| *t);

    let mut iter = task_results.into_iter().map(|(_, r)| r);
    let mut out = Vec::with_capacity(cands.len());
    for item in cands {
        let mut wb_parts = Vec::with_capacity(n_wb);
        for _ in 0..n_wb {
            wb_parts.push(iter.next().unwrap());
        }
        let mut eco_parts = Vec::with_capacity(n_eco);
        for _ in 0..n_eco {
            eco_parts.push(iter.next().unwrap());
        }
        let (bench_res, bench_marg, bench_mdq, bench_sub) =
            combine_single_opp_core_nash(wb_parts, wb_weights, 0.5);
        let (eco_res, eco_marg, eco_mdq, eco_sub) =
            combine_single_opp_core_nash(eco_parts, eco_weights, bench_res.score);
        out.push((
            item.sid.clone(),
            bench_res,
            bench_marg,
            bench_mdq,
            bench_sub,
            eco_res,
            eco_marg,
            eco_mdq,
            eco_sub,
        ));
    }
    out
}

/// Core-Nash Decomposition & Shadow-Context Verification (`meta_eco core-nash`):
/// 1. Screens all legal 60x60 canonical lead+core cells between Team A and Team B.
/// 2. Selects the top-k support cores per side via RM+ equilibrium + Best-Response ranking.
/// 3. Evaluates the k x k support matrix under three information regimes:
///    - Regime 0 (`Pure-3v3`): Only the 3 picked mons exist (`[1,2,3]`), zero shadow bench uncertainty.
///    - Regime 1 (`Uniform-6v6`): Full 6-mon open sheet, forced Turn 0 pick, uniform belief over unrevealed bench.
///    - Regime 2 (`Bayesian-Nash-6v6`): Full 6-mon open sheet, forced Turn 0 pick, hidden bench sampled from Stage-1 Nash policy `(p_a^(1), p_b^(1))`.
/// 4. Computes exact per-slot Nash equilibrium pick rate `p_i*` and leave-one-out marginal value `\Delta V_{-i}`.
/// 5. Runs a CRN-paired head-to-head duel: Nash-Core-Picker + `OpenAgent` vs standard Turn-0 `OpenAgent`.
fn cmd_core_nash(args: &[String]) {
    use nc2000_bot::preview::{
        legal_action_indices, preview_actions, solve_pair, MatrixEst, PairSolution, RolloutAgent,
    };
    use nc2000_bot::smmcts::{solve_rm_plus, SelRule};
    use nc2000_bot::{play_game, Agent, GameResult, OpenAgent, RmConfig};
    use nc2000_engine::battle::{Outcome, SearchChoice};
    use nc2000_engine::state::Battle;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let pool_file = flag(args, "--pool").unwrap_or_else(|| {
        root.join("data/team-pool-v2/team-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let pv: Value = serde_json::from_str(&std::fs::read_to_string(&pool_file).unwrap()).unwrap();
    let teams_arr = pv["teams"].as_array().expect("pool teams array");

    let team_a_spec = flag(args, "--team-a").unwrap_or_else(|| "sample-12".to_string());
    let team_b_spec = flag(args, "--team-b").unwrap_or_else(|| "sample-07".to_string());

    let find_team = |spec: &str| -> (String, Vec<Value>, Vec<PokemonSet>) {
        if let Some(t) = teams_arr.iter().find(|t| t["id"].as_str() == Some(spec)) {
            let id = t["id"].as_str().unwrap().to_string();
            let canon = gen
                .canonize(&dex, t["sets"].as_array().unwrap())
                .expect("canonize");
            let sets = to_sets(&canon).unwrap();
            return (id, canon, sets);
        }
        if let Ok(idx) = spec.parse::<usize>() {
            let t = &teams_arr[idx];
            let id = t["id"].as_str().unwrap_or(spec).to_string();
            let canon = gen
                .canonize(&dex, t["sets"].as_array().unwrap())
                .expect("canonize");
            let sets = to_sets(&canon).unwrap();
            return (id, canon, sets);
        }
        panic!("Team '{spec}' not found in {pool_file}");
    };

    let (id_a, canon_a, sets_a) = find_team(&team_a_spec);
    let (id_b, canon_b, sets_b) = find_team(&team_b_spec);
    let mirror = id_a == id_b;

    let screen_games: u32 = num(args, "--screen-games", 8);
    let refine_games: u32 = num(args, "--refine-games", 8);
    let support_k: usize = num(args, "--support", 6);
    let battle_iters: u32 = num(args, "--iters", 400);
    let duel_games: u32 = num(args, "--duel-games", 64);
    let c_val: f64 = num(args, "--c", 0.4);
    let smoothing: f64 = num(args, "--smoothing", 0.15);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let seed: u64 = num(args, "--seed", 2026102977);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let actions = preview_actions();
    let levels_a: Vec<u8> = sets_a.iter().map(|s| s.level).collect();
    let levels_b: Vec<u8> = sets_b.iter().map(|s| s.level).collect();
    let legal_a = legal_action_indices(&actions, &levels_a);
    let legal_b = legal_action_indices(&actions, &levels_b);

    println!(
        "=== Core-Nash Two-Layer Decomposition & Approximation Verification ===\n  Matchup: {} ({} legal actions) vs {} ({} legal actions)\n  Budgets: screen={}g/cell, refine={}g/cell @ open:{}:{}, support k={}",
        id_a,
        legal_a.len(),
        id_b,
        legal_b.len(),
        screen_games,
        refine_games,
        battle_iters,
        c_val,
        support_k
    );

    // ---- Stage 1: Full-width screening (legal_a x legal_b)
    let t0 = std::time::Instant::now();
    let mut screen_jobs = Vec::new();
    for &a in &legal_a {
        for &b in &legal_b {
            if mirror && b <= a {
                continue;
            }
            push_core_jobs(
                &mut screen_jobs,
                a * 60 + b,
                actions[a],
                actions[b],
                screen_games,
                seed ^ 0x5C4E,
            );
        }
    }
    let screen_res = run_forced_preview_games(
        &dex,
        &sets_a,
        &sets_b,
        &screen_jobs,
        threads,
        max_turns,
        &|_, _, _, sa, sb| {
            (
                Box::new(RolloutAgent::new(0.2, sa)),
                Box::new(RolloutAgent::new(0.2, sb)),
            )
        },
        false,
    );
    let mut screen = MatrixEst::new(60, 60);
    for (cell, sc) in screen_res {
        screen.record(cell, sc);
    }
    if mirror {
        for a in 0..60 {
            screen.v[a * 60 + a] = 0.5;
            screen.n[a * 60 + a] = screen_games;
            for b in (a + 1)..60 {
                screen.v[b * 60 + a] = 1.0 - screen.v[a * 60 + b];
                screen.n[b * 60 + a] = screen.n[a * 60 + b];
            }
        }
    }

    // Solve RM+ on legal_a x legal_b screen matrix
    let screen_ref = &screen;
    let sub: Vec<f64> = legal_a
        .iter()
        .flat_map(|&a| legal_b.iter().map(move |&b| screen_ref.at(a, b)))
        .collect();
    let (e0, e1) = solve_rm_plus(&sub, [legal_a.len(), legal_b.len()], 20_000);
    let screen_val: f64 = (0..legal_a.len())
        .map(|ia| {
            (0..legal_b.len())
                .map(|ib| screen.at(legal_a[ia], legal_b[ib]) * e0[ia] * e1[ib])
                .sum::<f64>()
        })
        .sum();

    let u_a: Vec<f64> = legal_a
        .iter()
        .map(|&a| {
            legal_b
                .iter()
                .zip(&e1)
                .map(|(&b, &p)| screen.at(a, b) * p)
                .sum()
        })
        .collect();
    let u_b: Vec<f64> = legal_b
        .iter()
        .map(|&b| {
            legal_a
                .iter()
                .zip(&e0)
                .map(|(&a, &p)| (1.0 - screen.at(a, b)) * p)
                .sum()
        })
        .collect();

    // Build support of size `support_k`: include positive-equilibrium-mass actions first, then top BR
    let select_sup = |u: &[f64], eq: &[f64], legal: &[usize], k: usize| -> Vec<usize> {
        let mut ord_eq: Vec<usize> = (0..legal.len()).collect();
        ord_eq.sort_by(|&a, &b| eq[b].total_cmp(&eq[a]).then(u[b].total_cmp(&u[a])));
        let mut sup = Vec::new();
        for &idx in &ord_eq {
            if sup.len() >= k {
                break;
            }
            if eq[idx] >= 0.02 {
                sup.push(legal[idx]);
            }
        }
        let mut ord_br: Vec<usize> = (0..legal.len()).collect();
        ord_br.sort_by(|&a, &b| u[b].total_cmp(&u[a]));
        for &idx in &ord_br {
            if sup.len() >= k {
                break;
            }
            let act = legal[idx];
            if !sup.contains(&act) {
                sup.push(act);
            }
        }
        sup.sort_unstable();
        sup
    };

    let s0 = select_sup(&u_a, &e0, &legal_a, support_k);
    let s1 = if mirror {
        s0.clone()
    } else {
        select_sup(&u_b, &e1, &legal_b, support_k)
    };
    let (k0, k1) = (s0.len(), s1.len());
    println!(
        "  [Stage 1 Screen] {:.2}s ({} games) | Screen Nash Value = {:.4}",
        t0.elapsed().as_secs_f64(),
        screen_jobs.len(),
        screen_val
    );

    let fmt_triple = |t: [u8; 3], canon: &[Value]| -> String {
        let lead = canon[(t[0] - 1) as usize]["species"]
            .as_str()
            .unwrap_or("?");
        let b1 = canon[(t[1] - 1) as usize]["species"]
            .as_str()
            .unwrap_or("?");
        let b2 = canon[(t[2] - 1) as usize]["species"]
            .as_str()
            .unwrap_or("?");
        format!("{lead}*+{b1}+{b2}")
    };

    println!(
        "    Support A ({}): {}",
        k0,
        s0.iter()
            .map(|&a| fmt_triple(actions[a], &canon_a))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "    Support B ({}): {}",
        k1,
        s1.iter()
            .map(|&b| fmt_triple(actions[b], &canon_b))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ---- Stage 2A: Pure-3v3 (Context-Free 3-mon Roster, zero shadow bench)
    let t_3v3 = std::time::Instant::now();
    let mut mat_3v3 = MatrixEst::new(k0, k1);
    let mut jobs_3v3 = Vec::new();
    for ri in 0..k0 {
        for ci in 0..k1 {
            push_core_jobs(
                &mut jobs_3v3,
                ri * k1 + ci,
                [1, 2, 3],
                [1, 2, 3],
                refine_games,
                seed ^ 0x3333,
            );
        }
    }
    // Run Pure-3v3 where each cell uses 3-mon subsets so neither side sees the other 3 bench mons
    let cursor = AtomicUsize::new(0);
    let mut res_3v3: Vec<(usize, (usize, f64))> = Vec::with_capacity(jobs_3v3.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads.max(1) {
            let (cursor, jobs_ref, s0_ref, s1_ref, actions_ref, sets_a_ref, sets_b_ref, dex_ref) = (
                &cursor, &jobs_3v3, &s0, &s1, &actions, &sets_a, &sets_b, &dex,
            );
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let n = cursor.fetch_add(1, Ordering::Relaxed);
                    if n >= jobs_ref.len() {
                        break;
                    }
                    let job = &jobs_ref[n];
                    let ri = job.cell / k1;
                    let ci = job.cell % k1;
                    let sub_a = subset_3_sets(sets_a_ref, actions_ref[s0_ref[ri]]);
                    let sub_b = subset_3_sets(sets_b_ref, actions_ref[s1_ref[ci]]);
                    let rm_cfg = RmConfig {
                        iterations: battle_iters,
                        rule: SelRule::Ucb,
                        c: c_val,
                        ..Default::default()
                    };
                    let mut p1 = OpenAgent::new(rm_cfg.clone(), None, job.agent_seed ^ 0xA24B_AED4_963E_E407);
                    let mut p2 = OpenAgent::new(rm_cfg, None, job.agent_seed ^ 0x9FB2_1C65_1E98_DF25);
                    let (team1, team2) = if job.swap {
                        (&sub_b, &sub_a)
                    } else {
                        (&sub_a, &sub_b)
                    };
                    let mut b = Battle::from_fixture(dex_ref, &job.battle_seed, team1, team2).unwrap();
                    b.set_log_enabled(true);
                    p1.init_at_preview(&b, 0);
                    p2.init_at_preview(&b, 1);
                    b.apply_choices(
                        dex_ref,
                        [
                            Some(SearchChoice::Team([1, 2, 3])),
                            Some(SearchChoice::Team([1, 2, 3])),
                        ],
                    )
                    .unwrap();
                    let res = play_game(dex_ref, &mut b, &mut [&mut p1, &mut p2], max_turns).unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let score = if job.swap { 1.0 - p1_score } else { p1_score };
                    out.push((n, (job.cell, score)));
                }
                out
            }));
        }
        let mut all = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        res_3v3 = all;
    });
    for (_, (cell, sc)) in res_3v3 {
        mat_3v3.record(cell, sc);
    }
    let sol_3v3 = solve_pair(&mat_3v3, &[s0.clone(), s1.clone()], 0.05, 50_000);
    println!(
        "  [Regime 0: Pure-3v3 (Context-Free)]   {:.2}s | Nash Value = {:.4}",
        t_3v3.elapsed().as_secs_f64(),
        sol_3v3.value
    );

    // ---- Stage 2B: Uniform-6v6 (6-mon Forced-Preview, Uniform hidden bench belief)
    let t_u6 = std::time::Instant::now();
    let mut jobs_6v6 = Vec::new();
    for (ri, &a) in s0.iter().enumerate() {
        for (ci, &b) in s1.iter().enumerate() {
            push_core_jobs(
                &mut jobs_6v6,
                ri * k1 + ci,
                actions[a],
                actions[b],
                refine_games,
                seed ^ 0x3333,
            );
        }
    }
    let res_u6 = run_forced_preview_games(
        &dex,
        &sets_a,
        &sets_b,
        &jobs_6v6,
        threads,
        max_turns,
        &|b, side_a, side_b, sa, sb| {
            let rm_cfg = RmConfig {
                iterations: battle_iters,
                rule: SelRule::Ucb,
                c: c_val,
                ..Default::default()
            };
            let mut ag_a = OpenAgent::new(rm_cfg.clone(), None, sa);
            let mut ag_b = OpenAgent::new(rm_cfg, None, sb);
            ag_a.init_at_preview(b, side_a);
            ag_b.init_at_preview(b, side_b);
            (Box::new(ag_a), Box::new(ag_b))
        },
        true,
    );
    let mut mat_u6 = MatrixEst::new(k0, k1);
    for (cell, sc) in res_u6 {
        mat_u6.record(cell, sc);
    }
    let sol_u6: PairSolution = solve_pair(&mat_u6, &[s0.clone(), s1.clone()], 0.05, 50_000);
    println!(
        "  [Regime 1: Uniform-6v6 Shadow Bench]  {:.2}s | Nash Value = {:.4}",
        t_u6.elapsed().as_secs_f64(),
        sol_u6.value
    );

    // ---- Stage 2C: Bayesian-Nash-6v6 (Hidden bench conditioned on Stage-2B Nash policies p_a, p_b)
    let t_bn6 = std::time::Instant::now();
    let pol_a_prior = sol_u6.p_a.clone();
    let pol_b_prior = sol_u6.p_b.clone();
    let res_bn6 = run_forced_preview_games(
        &dex,
        &sets_a,
        &sets_b,
        &jobs_6v6,
        threads,
        max_turns,
        &|b, side_a, side_b, sa, sb| {
            let rm_cfg = RmConfig {
                iterations: battle_iters,
                rule: SelRule::Ucb,
                c: c_val,
                ..Default::default()
            };
            // Row agent A conditions on Column B's Nash pick distribution `pol_b_prior`;
            // Column agent B conditions on Row A's Nash pick distribution `pol_a_prior`.
            let mut ag_a = OpenAgent::new(rm_cfg.clone(), None, sa)
                .with_opponent_pick_policy(pol_b_prior.clone(), smoothing);
            let mut ag_b = OpenAgent::new(rm_cfg, None, sb)
                .with_opponent_pick_policy(pol_a_prior.clone(), smoothing);
            ag_a.init_at_preview(b, side_a);
            ag_b.init_at_preview(b, side_b);
            (Box::new(ag_a), Box::new(ag_b))
        },
        true,
    );
    let mut mat_bn6 = MatrixEst::new(k0, k1);
    for (cell, sc) in res_bn6 {
        mat_bn6.record(cell, sc);
    }
    let sol_bn6: PairSolution = solve_pair(&mat_bn6, &[s0.clone(), s1.clone()], 0.05, 50_000);
    println!(
        "  [Regime 2: Bayesian-Nash-6v6 Bench]   {:.2}s | Nash Value = {:.4}",
        t_bn6.elapsed().as_secs_f64(),
        sol_bn6.value
    );

    // ---- Compare Approximation Regimes (Cell MAE, Value diff, Strategy L1/Cosine)
    let cell_mae = |m1: &MatrixEst, m2: &MatrixEst| -> f64 {
        m1.v.iter()
            .zip(m2.v.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / (m1.v.len().max(1) as f64)
    };
    let l1_dist = |p: &[f64], q: &[f64]| -> f64 {
        p.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() * 0.5
    };
    println!("\n=== Approximation Fidelity Diagnostics ===");
    println!(
        "  Pure-3v3      vs Uniform-6v6 : |ΔValue| = {:+.4}, Cell MAE = {:.4}, TV(p_a) = {:.3}, TV(p_b) = {:.3}",
        sol_3v3.value - sol_u6.value,
        cell_mae(&mat_3v3, &mat_u6),
        l1_dist(&sol_3v3.p_a, &sol_u6.p_a),
        l1_dist(&sol_3v3.p_b, &sol_u6.p_b),
    );
    println!(
        "  Uniform-6v6   vs BayesNash-6v6: |ΔValue| = {:+.4}, Cell MAE = {:.4}, TV(p_a) = {:.3}, TV(p_b) = {:.3}",
        sol_u6.value - sol_bn6.value,
        cell_mae(&mat_u6, &mat_bn6),
        l1_dist(&sol_u6.p_a, &sol_bn6.p_a),
        l1_dist(&sol_u6.p_b, &sol_bn6.p_b),
    );
    println!(
        "  Pure-3v3      vs BayesNash-6v6: |ΔValue| = {:+.4}, Cell MAE = {:.4}, TV(p_a) = {:.3}, TV(p_b) = {:.3}",
        sol_3v3.value - sol_bn6.value,
        cell_mae(&mat_3v3, &mat_bn6),
        l1_dist(&sol_3v3.p_a, &sol_bn6.p_a),
        l1_dist(&sol_3v3.p_b, &sol_bn6.p_b),
    );

    // ---- Per-Slot Equilibrium Pick Share `p_i*` & Leave-One-Out Marginal Value `\Delta V_{-i}`
    let print_slot_nash_analysis =
        |label: &str, canon: &[Value], sup: &[usize], mat: &MatrixEst, sol: &PairSolution, is_row: bool| {
            println!("\n=== Per-Slot Nash Equilibrium & Marginal Value: {} ===", label);
            for slot in 0..canon.len().min(6) {
                let s_idx = (slot + 1) as u8;
                let sp = canon[slot]["species"].as_str().unwrap_or("?");
                let lv = canon[slot]["level"].as_u64().unwrap_or(50);
                let pol = if is_row { &sol.p_a } else { &sol.p_b };
                let mut pick_share = 0.0;
                let mut lead_share = 0.0;
                for (a_idx, &pr) in pol.iter().enumerate() {
                    if pr <= 0.0 {
                        continue;
                    }
                    let tr = actions[a_idx];
                    if tr.contains(&s_idx) {
                        pick_share += pr;
                    }
                    if tr[0] == s_idx {
                        lead_share += pr;
                    }
                }
                // Leave-one-out: solve the subgame where `slot` cannot be picked by this side
                let kept_indices: Vec<usize> = (0..sup.len())
                    .filter(|&i| !actions[sup[i]].contains(&s_idx))
                    .collect();
                let delta_v = if kept_indices.is_empty() {
                    // Mandatory ace (every support core contains this mon)
                    f64::NAN
                } else if is_row {
                    let sub_v: Vec<f64> = kept_indices
                        .iter()
                        .flat_map(|&ri| (0..mat.cols).map(move |ci| mat.at(ri, ci)))
                        .collect();
                    let (s_a, s_b) = solve_rm_plus(&sub_v, [kept_indices.len(), mat.cols], 20_000);
                    let v_without: f64 = (0..kept_indices.len())
                        .map(|ia| {
                            (0..mat.cols)
                                .map(|ib| sub_v[ia * mat.cols + ib] * s_a[ia] * s_b[ib])
                                .sum::<f64>()
                        })
                        .sum();
                    sol.value - v_without
                } else {
                    let sub_v: Vec<f64> = (0..mat.rows)
                        .flat_map(|ri| kept_indices.iter().map(move |&ci| mat.at(ri, ci)))
                        .collect();
                    let (s_a, s_b) = solve_rm_plus(&sub_v, [mat.rows, kept_indices.len()], 20_000);
                    let v_a_without: f64 = (0..mat.rows)
                        .map(|ia| {
                            (0..kept_indices.len())
                                .map(|ib| sub_v[ia * kept_indices.len() + ib] * s_a[ia] * s_b[ib])
                                .sum::<f64>()
                        })
                        .sum();
                    (1.0 - sol.value) - (1.0 - v_a_without)
                };
                let dv_str = if delta_v.is_nan() {
                    "MANDATORY_ACE".to_string()
                } else {
                    format!("{:+.4} ({:+.1}pt)", delta_v, delta_v * 100.0)
                };
                println!(
                    "  slot {}: {:12} L{:2} | Nash Pick={:5.1}% (Lead={:4.1}%) | Marginal ΔV={}",
                    slot,
                    sp,
                    lv,
                    pick_share * 100.0,
                    lead_share * 100.0,
                    dv_str
                );
            }
        };

    print_slot_nash_analysis(&id_a, &canon_a, &s0, &mat_bn6, &sol_bn6, true);
    print_slot_nash_analysis(&id_b, &canon_b, &s1, &mat_bn6, &sol_bn6, false);

    // ---- Stage 3: Live Duel Verification (Nash-Core-Picker vs Standard Turn-0 OpenAgent)
    if duel_games > 0 {
        let t_duel = std::time::Instant::now();
        let n_duel = (duel_games + duel_games % 2) as usize;
        let mut duel_jobs = Vec::with_capacity(n_duel);
        let mut rng = SplitMix64::new(seed ^ 0xD0E1_CAFE);
        let mut b_seed = String::new();
        for g in 0..n_duel {
            if g % 2 == 0 {
                b_seed = rng.battle_seed();
            }
            duel_jobs.push((
                g % 2 == 0,
                b_seed.clone(),
                rng.next(),
                rng.next(),
            ));
        }

        let cursor = AtomicUsize::new(0);
        let mut duel_scores = vec![0.0; n_duel];
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..threads.max(1) {
                let (cursor, duel_jobs_ref, sets_a_ref, sets_b_ref, sol_ref, actions_ref, dex_ref) = (
                    &cursor, &duel_jobs, &sets_a, &sets_b, &sol_bn6, &actions, &dex,
                );
                handles.push(scope.spawn(move || {
                    let mut local = Vec::new();
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= duel_jobs_ref.len() {
                            break;
                        }
                        let (a_is_p1, ref bseed, sa, sb) = duel_jobs_ref[i];
                        let rm_cfg = RmConfig {
                            iterations: battle_iters,
                            rule: SelRule::Ucb,
                            c: c_val,
                            ..Default::default()
                        };
                        // Side A uses solved Nash preview equilibrium + Bayesian-Nash bench belief;
                        // Side B uses standard Turn-0 MCTS preview + uniform bench belief.
                        let mut rng_a = SplitMix64::new(sa ^ 0x7777);
                        let u = rng_a.next_f64();
                        let mut acc = 0.0;
                        let mut pick_idx = sol_ref.argmax_a;
                        for (idx, &pr) in sol_ref.p_a.iter().enumerate() {
                            acc += pr;
                            if u < acc {
                                pick_idx = idx;
                                break;
                            }
                        }
                        let nash_choice = SearchChoice::Team(actions_ref[pick_idx]);

                        let mut agent_a = OpenAgent::new(rm_cfg.clone(), None, sa)
                            .with_opponent_pick_policy(sol_ref.p_b.clone(), smoothing);
                        let mut agent_b = OpenAgent::new(rm_cfg, None, sb);

                        let (t1, t2) = if a_is_p1 {
                            (sets_a_ref.as_slice(), sets_b_ref.as_slice())
                        } else {
                            (sets_b_ref.as_slice(), sets_a_ref.as_slice())
                        };
                        let mut b = Battle::from_fixture(dex_ref, bseed, t1, t2).unwrap();
                        b.set_log_enabled(true);

                        // Turn 0 preview: Side A plays `nash_choice`, Side B calls `agent_b.choose`
                        let side_a = if a_is_p1 { 0 } else { 1 };
                        let side_b = 1 - side_a;
                        agent_a.init_at_preview(&b, side_a);
                        let choices_b = b.legal_choices(dex_ref, side_b);
                        let b_choice = agent_b.choose(&b, dex_ref, side_b, &choices_b);
                        let joint = if a_is_p1 {
                            [Some(nash_choice), Some(b_choice)]
                        } else {
                            [Some(b_choice), Some(nash_choice)]
                        };
                        b.apply_choices(dex_ref, joint).unwrap();

                        let res = if a_is_p1 {
                            play_game(dex_ref, &mut b, &mut [&mut agent_a, &mut agent_b], max_turns)
                        } else {
                            play_game(dex_ref, &mut b, &mut [&mut agent_b, &mut agent_a], max_turns)
                        }
                        .unwrap();
                        let p1_sc = match res {
                            GameResult::Outcome(Outcome::P1Win) => 1.0,
                            GameResult::Outcome(Outcome::P2Win) => 0.0,
                            GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                        };
                        let sc = if a_is_p1 { p1_sc } else { 1.0 - p1_sc };
                        local.push((i, sc));
                    }
                    local
                }));
            }
            for h in handles {
                for (i, sc) in h.join().unwrap() {
                    duel_scores[i] = sc;
                }
            }
        });
        let mean_duel: f64 = duel_scores.iter().sum::<f64>() / (n_duel as f64);
        println!(
            "\n=== Live Duel Verification ({} CRN games, {:.1}s) ===",
            n_duel,
            t_duel.elapsed().as_secs_f64()
        );
        println!(
            "  {} (Core-Nash Picker + BayesBench) vs {} (Standard Turn-0 OpenAgent): {:.4} ({:.1}%) [Predicted Nash Value = {:.4}]",
            id_a,
            id_b,
            mean_duel,
            mean_duel * 100.0,
            sol_bn6.value
        );
    }
}

fn cmd_verify_1gen(args: &[String]) {
    let t_start = std::time::Instant::now();
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let n_parents: usize = num(args, "--parents", 6);
    let buds_per_parent: usize = num(args, "--buds-per-parent", 2);
    let screen_games: u32 = num(args, "--screen-games", 2);
    let refine_games: u32 = num(args, "--refine-games", 2);
    let support_k: usize = num(args, "--support", 3);
    let oos_games: u32 = num(args, "--oos-games", 4);
    let max_turns: u16 = num(args, "--max-turns", 300);
    let base_seed: u64 = num(args, "--seed", 20261029);
    let out_file = flag(args, "--out").map(PathBuf::from);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        });
    let (agent_iters, agent_c, blind_pool, open_sheet) = parse_agent_spec(args, &root, &dex, &gen);

    let bench_file = flag(args, "--bench-file").unwrap_or_else(|| {
        root.join("data/team-pool-v2/team-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let (all_bench_teams, all_bench_weights) = load_gauntlet_file(&dex, &gen, &bench_file);
    let slice_len = all_bench_teams.len().min(10);
    let bench_teams: Vec<Vec<PokemonSet>> = all_bench_teams.iter().take(slice_len).cloned().collect();
    let mut bench_weights: Vec<f64> = all_bench_weights.iter().take(slice_len).copied().collect();
    let bw_sum: f64 = bench_weights.iter().sum::<f64>().max(1e-12);
    for w in &mut bench_weights {
        *w /= bw_sum;
    }

    let oos_file = root
        .join("data/meta-nash-v3/pool-artifact.json")
        .to_string_lossy()
        .into_owned();
    let (oos_teams, oos_weights) = load_gauntlet_file(&dex, &gen, &oos_file);

    // Load top `n_parents` distinct founder teams from `bench_file`
    let bv: Value = serde_json::from_str(&std::fs::read_to_string(&bench_file).unwrap()).unwrap();
    let mut parents: Vec<Strain> = Vec::new();
    let mut seen_keys = BTreeSet::new();
    if let Some(arr) = bv["teams"].as_array() {
        for (idx, t) in arr.iter().enumerate() {
            if parents.len() >= n_parents {
                break;
            }
            let Some(sets) = t["sets"].as_array() else {
                continue;
            };
            let Some(canon) = gen.canonize(&dex, sets) else {
                continue;
            };
            if !seen_keys.insert(team_key(&canon)) {
                continue;
            }
            let origin = t["id"]
                .as_str()
                .map(String::from)
                .unwrap_or_else(|| format!("founder-{idx}"));
            parents.push(Strain {
                id: format!("P{idx:02}({origin})"),
                parent_id: origin,
                niche: "verify".into(),
                anchors: String::new(),
                max_staples: String::new(),
                born_wave: 0,
                last_op: "founder".into(),
                abundance: 1.0 / (n_parents as f64),
                cum_games: 0.0,
                cum_wins: 0.0,
                bench_games: 0.0,
                bench_wins: 0.0,
                eco_games: 0.0,
                eco_wins: 0.0,
                last_wave_fit: 0.5,
                parent_prior: 0.5,
                bayes_fit: 0.5,
                lcb95: 0.0,
                slot_picks: [0.0; 6],
                slot_wins: [0.0; 6],
                slot_marginals: [0.0; 6],
                slot_wars: [0.0; 6],
                move_dqs: [[0.0; 4]; 6],
                last_wave_slot_picks: [0.0; 6],
                last_wave_slot_marginals: [0.0; 6],
                effective_cores: 1.0,
                symbiont_of: None,
                mix_alpha: 1.0,
                deme_bonus: 0.0,
                team: canon,
            });
        }
    }

    let w_games = (refine_games as usize * support_k * support_k * bench_teams.len()) as f64;
    let inner_threads = (threads / 3).max(2);
    let seed_w1 = base_seed ^ 0x1111_2222_3333_4444;
    let seed_w2 = base_seed ^ 0x5555_6666_7777_8888;
    let seed_oos = base_seed ^ 0x9999_AAAA_BBBB_CCCC;

    eprintln!(
        "=== Step 1: Wave 1 Parent Calibration ({} parents vs {} bench teams, k={}, iters={}) ===",
        parents.len(),
        bench_teams.len(),
        support_k,
        agent_iters
    );
    let dex_ref = &dex;
    let bt_ref = &bench_teams;
    let bw_ref = &bench_weights;
    let opp_mdq_sink: std::sync::Mutex<Vec<Vec<[[f64; 4]; 6]>>> = std::sync::Mutex::new(Vec::new());
    let opp_sink_ref = &opp_mdq_sink;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for p in &mut parents {
            handles.push(scope.spawn(move || {
                let cand = to_sets(&p.team).unwrap();
                let r = core_nash_eval_weighted(
                    dex_ref,
                    &cand,
                    bt_ref,
                    bw_ref,
                    &CoreNashGauntletCfg {
                        screen_games,
                        refine_games,
                        support_k,
                        agent_iters,
                        c: agent_c,
                        smoothing: 0.15,
                        max_turns,
                        seed: seed_w1,
                        threads: inner_threads,
                        inherited: None,
                    },
                );
                p.cum_games = w_games;
                p.cum_wins = r.eval.score * w_games;
                p.last_wave_fit = r.eval.score;
                p.bayes_fit = r.eval.score;
                for i in 0..6 {
                    p.slot_picks[i] = r.eval.slot_picks[i] * w_games;
                    p.slot_wins[i] = r.eval.slot_wins[i] * w_games;
                    p.slot_marginals[i] = r.slot_marginals[i] * w_games;
                }
                p.move_dqs = r.move_dqs;
                opp_sink_ref.lock().unwrap().push(r.opp_move_dqs);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });

    let (mu_50, mu_55, beta_l) = recompute_population_wars(&mut parents);
    eprintln!(
        "  Endogenous Replacement Level: mu(L50)={:+.2}pt, mu(L55)={:+.2}pt, beta_L={:+.2}pt/Lv",
        mu_50 * 100.0,
        mu_55 * 100.0,
        beta_l * 100.0
    );
    for p in &parents {
        let slots: Vec<String> = (0..6)
            .map(|i| {
                let sp = p.team[i]["species"].as_str().unwrap_or("?");
                let lv = p.team[i]["level"].as_u64().unwrap_or(50);
                let pk = p.slot_picks[i] / w_games * 100.0;
                let war = p.slot_wars[i] * 100.0;
                format!("{sp}(L{lv},p={pk:.0}%,WAR={war:+.1}pt)")
            })
            .collect();
        eprintln!(
            "  {:18} V*(W1)={:.3} | {}",
            p.id,
            p.last_wave_fit,
            slots.join(" | ")
        );
    }

    // Step 2: Spawn `buds_per_parent` Targeted children (WAR + move dQ Stochastic Resonance)
    // and `buds_per_parent` Untargeted Control children (uniform random slot & move) per parent.
    let free_constraint = NicheConstraint::default();
    let mut rng_targ = SplitMix64(base_seed ^ 0xABCD_0001);
    let mut rng_ctrl = SplitMix64(base_seed ^ 0xABCD_0002);

    #[derive(Clone)]
    struct ChildCand {
        id: String,
        parent_idx: usize,
        arm: &'static str,
        op: String,
        mut_slot: Option<usize>,
        team: Vec<Value>,
        w2_score: f64,
        w2_picks: [f64; 6],
        w2_marginals: [f64; 6],
        w2_wars: [f64; 6],
        w2_move_dqs: [[f64; 4]; 6],
    }

    let mut endo_pool = build_endogenous_pool(&parents);
    for opp_list in opp_mdq_sink.into_inner().unwrap() {
        for (m, o_mdq) in opp_list.iter().enumerate() {
            if let Some(opp_team) = bench_teams.get(m) {
                for (s, pset) in opp_team.iter().take(6).enumerate() {
                    endo_pool.record_pokemon_set_moves(pset, &o_mdq[s]);
                }
            }
        }
    }
    let mut children: Vec<ChildCand> = Vec::new();
    for (p_idx, p) in parents.iter().enumerate() {
        let mut seen_t = BTreeSet::new();
        seen_t.insert(team_key(&p.team));
        for b in 0..buds_per_parent {
            for _ in 0..16 {
                if let Some(prop) = gen.propose_marginal_targeted(
                    &dex,
                    &p.team,
                    &p.slot_picks,
                    &p.slot_wins,
                    &p.slot_wars,
                    Some(&p.move_dqs),
                    Some(&endo_pool),
                    &mut rng_targ,
                    &free_constraint,
                    150,
                ) {
                    if seen_t.insert(team_key(&prop.team)) {
                        children.push(ChildCand {
                            id: format!("T{p_idx:02}_{b}"),
                            parent_idx: p_idx,
                            arm: "targeted",
                            op: format!("{}@{}", prop.op.name(), prop.slot),
                            mut_slot: Some(prop.slot),
                            team: prop.team,
                            w2_score: 0.0,
                            w2_picks: [0.0; 6],
                            w2_marginals: [0.0; 6],
                            w2_wars: [0.0; 6],
                            w2_move_dqs: [[0.0; 4]; 6],
                        });
                        break;
                    }
                }
            }
        }
        let mut seen_c = BTreeSet::new();
        seen_c.insert(team_key(&p.team));
        for b in 0..buds_per_parent {
            for _ in 0..16 {
                if let Some(prop) = gen.propose(&dex, &p.team, &mut rng_ctrl) {
                    if seen_c.insert(team_key(&prop.team)) {
                        children.push(ChildCand {
                            id: format!("C{p_idx:02}_{b}"),
                            parent_idx: p_idx,
                            arm: "untargeted",
                            op: format!("{}@{}", prop.op.name(), prop.slot),
                            mut_slot: Some(prop.slot),
                            team: prop.team,
                            w2_score: 0.0,
                            w2_picks: [0.0; 6],
                            w2_marginals: [0.0; 6],
                            w2_wars: [0.0; 6],
                            w2_move_dqs: [[0.0; 4]; 6],
                        });
                        break;
                    }
                }
            }
        }
    }

    eprintln!(
        "\n=== Step 2: Wave 2 CRN-Paired Evaluation ({} parents + {} children on seed_w2) ===",
        parents.len(),
        children.len()
    );
    let mut parent_w2_scores = vec![0.0f64; parents.len()];
    let mut parent_w2_marginals = vec![[0.0f64; 6]; parents.len()];
    let mut parent_w2_wars = vec![[0.0f64; 6]; parents.len()];
    let mut parent_w2_subgames: Vec<Vec<OpponentStage2Subgame>> = vec![Vec::new(); parents.len()];
    std::thread::scope(|scope| {
        let mut p_handles = Vec::new();
        for (idx, p) in parents.iter().enumerate() {
            let team_c = p.team.clone();
            p_handles.push(scope.spawn(move || {
                let cand = to_sets(&team_c).unwrap();
                let r = core_nash_eval_weighted(
                    dex_ref,
                    &cand,
                    bt_ref,
                    bw_ref,
                    &CoreNashGauntletCfg {
                        screen_games,
                        refine_games,
                        support_k,
                        agent_iters,
                        c: agent_c,
                        smoothing: 0.15,
                        max_turns,
                        seed: seed_w2,
                        threads: inner_threads,
                        inherited: None,
                    },
                );
                (idx, r.eval.score, r.slot_marginals, r.subgames)
            }));
        }
        for h in p_handles {
            let (idx, sc, marg, subs) = h.join().unwrap();
            parent_w2_scores[idx] = sc;
            parent_w2_marginals[idx] = marg;
            parent_w2_subgames[idx] = subs;
        }
    });
    let p_subs_ref = &parent_w2_subgames;
    std::thread::scope(|scope| {
        let mut c_handles = Vec::new();
        for c in &mut children {
            c_handles.push(scope.spawn(move || {
                let cand = to_sets(&c.team).unwrap();
                let inh = c
                    .mut_slot
                    .filter(|&ms| ms < 6 && !c.op.starts_with("level-tweak"))
                    .map(|ms| (ms, p_subs_ref[c.parent_idx].as_slice()));
                let r = core_nash_eval_weighted(
                    dex_ref,
                    &cand,
                    bt_ref,
                    bw_ref,
                    &CoreNashGauntletCfg {
                        screen_games,
                        refine_games,
                        support_k,
                        agent_iters,
                        c: agent_c,
                        smoothing: 0.15,
                        max_turns,
                        seed: seed_w2,
                        threads: inner_threads,
                        inherited: inh,
                    },
                );
                c.w2_score = r.eval.score;
                c.w2_picks = r.eval.slot_picks;
                c.w2_marginals = r.slot_marginals;
                c.w2_move_dqs = r.move_dqs;
            }));
        }
        for h in c_handles {
            h.join().unwrap();
        }
    });

    // Compute W2 WARs using the population's W1 endogenous replacement level `(mu_50, mu_55, beta_l)`
    for (idx, p) in parents.iter().enumerate() {
        for i in 0..6 {
            let lv = p.team[i]["level"].as_u64().unwrap_or(50) as f64;
            let rep = mu_50 + beta_l * (lv - 50.0);
            parent_w2_wars[idx][i] = parent_w2_marginals[idx][i] - rep;
        }
    }
    for c in &mut children {
        for i in 0..6 {
            let lv = c.team[i]["level"].as_u64().unwrap_or(50) as f64;
            let rep = mu_50 + beta_l * (lv - 50.0);
            c.w2_wars[i] = c.w2_marginals[i] - rep;
        }
    }

    let mut targ_dv = Vec::new();
    let mut ctrl_dv = Vec::new();
    let mut targ_dwar = Vec::new();
    let mut ctrl_dwar = Vec::new();

    for c in &children {
        let p_sc = parent_w2_scores[c.parent_idx];
        let dv = c.w2_score - p_sc;
        let dwar = c
            .mut_slot
            .filter(|&s| s < 6)
            .map(|s| c.w2_wars[s] - parent_w2_wars[c.parent_idx][s])
            .unwrap_or(0.0);
        let p_id = &parents[c.parent_idx].id;
        let slot_desc = if let Some(s) = c.mut_slot.filter(|&s| s < 6) {
            let old_sp = parents[c.parent_idx].team[s]["species"].as_str().unwrap_or("?");
            let new_sp = c.team[s]["species"].as_str().unwrap_or("?");
            let old_moves: Vec<&str> = parents[c.parent_idx].team[s]["moves"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let new_moves: Vec<&str> = c.team[s]["moves"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let diff_detail = if old_sp != new_sp {
                format!("{old_sp}->{new_sp}")
            } else if old_moves != new_moves {
                let dropped: Vec<&str> = old_moves.iter().copied().filter(|m| !new_moves.contains(m)).collect();
                let added: Vec<&str> = new_moves.iter().copied().filter(|m| !old_moves.contains(m)).collect();
                format!("{old_sp}[{:?}->{:?}]", dropped, added)
            } else {
                format!("{old_sp}->{new_sp}")
            };
            format!(
                "{diff_detail} (pick={:.0}%, WAR: {:+.1}pt -> {:+.1}pt, dWAR={:+.1}pt)",
                c.w2_picks[s] * 100.0,
                parent_w2_wars[c.parent_idx][s] * 100.0,
                c.w2_wars[s] * 100.0,
                dwar * 100.0
            )
        } else {
            "multi-slot".into()
        };
        eprintln!(
            "  [{:10}] {:6} (from {:18}, op={:16}): V*(P)={:.3} -> V*(C)={:.3} | dV={:+.2}pt | {}",
            c.arm,
            c.id,
            p_id,
            c.op,
            p_sc,
            c.w2_score,
            dv * 100.0,
            slot_desc
        );
        if c.arm == "targeted" {
            targ_dv.push(dv);
            targ_dwar.push(dwar);
        } else {
            ctrl_dv.push(dv);
            ctrl_dwar.push(dwar);
        }
    }

    let mean = |xs: &[f64]| -> f64 { xs.iter().sum::<f64>() / (xs.len().max(1) as f64) };
    let hit_rate = |xs: &[f64], thresh: f64| -> f64 {
        (xs.iter().filter(|&&x| x >= thresh).count() as f64) / (xs.len().max(1) as f64)
    };

    eprintln!("\n=== Summary of Criterion 1 (Causality) & Criterion 2 (Proposal Efficiency) ===");
    eprintln!(
        "  Targeted   (n={:2}): mean dV(C-P)={:+.2}pt | hit(dV>+1.0pt)={:.1}% | hit(dV>0)={:.1}% | mean dWAR_slot={:+.2}pt",
        targ_dv.len(),
        mean(&targ_dv) * 100.0,
        hit_rate(&targ_dv, 0.01) * 100.0,
        hit_rate(&targ_dv, 0.0001) * 100.0,
        mean(&targ_dwar) * 100.0
    );
    eprintln!(
        "  Untargeted (n={:2}): mean dV(C-P)={:+.2}pt | hit(dV>+1.0pt)={:.1}% | hit(dV>0)={:.1}% | mean dWAR_slot={:+.2}pt",
        ctrl_dv.len(),
        mean(&ctrl_dv) * 100.0,
        hit_rate(&ctrl_dv, 0.01) * 100.0,
        hit_rate(&ctrl_dv, 0.0001) * 100.0,
        mean(&ctrl_dwar) * 100.0
    );

    // Step 3: Criterion 3 (Out-of-Sample Full Turn-0 OpenAgent Verification on Top Surviving Children)
    let mut ranked_targ: Vec<&ChildCand> = children
        .iter()
        .filter(|c| {
            c.arm == "targeted"
                && (c.op.starts_with("level-tweak")
                    || c.mut_slot.is_none_or(|s| s < 6 && c.w2_picks[s] >= 0.08))
        })
        .collect();
    ranked_targ.sort_by(|a, b| {
        let dva = a.w2_score - parent_w2_scores[a.parent_idx];
        let dvb = b.w2_score - parent_w2_scores[b.parent_idx];
        dvb.partial_cmp(&dva).unwrap()
    });

    let top_verify = ranked_targ.into_iter().take(3).collect::<Vec<_>>();
    eprintln!(
        "\n=== Step 3: Criterion 3 Out-of-Sample Full Turn-0 OpenAgent Verification ({} games/opp vs {} Meta-Nash-v3 teams) ===",
        oos_games,
        oos_teams.len()
    );
    let mut oos_records = Vec::new();
    for c in top_verify {
        let p = &parents[c.parent_idx];
        let p_sets = to_sets(&p.team).unwrap();
        let c_sets = to_sets(&c.team).unwrap();
        let pair_seed = seed_oos ^ ((c.parent_idx as u64 + 1) * 0x9E37_79B9_7F4A_7C15);
        let p_oos = gauntlet_eval_weighted(
            &dex,
            &p_sets,
            &oos_teams,
            &oos_weights,
            &WeightedEvalCfg {
                games_per_opponent: oos_games,
                agent_iters,
                c: agent_c,
                max_turns,
                seed: pair_seed,
                threads,
                blind_pool: blind_pool.clone(),
                open_sheet,
            },
        )
        .score;
        let c_oos = gauntlet_eval_weighted(
            &dex,
            &c_sets,
            &oos_teams,
            &oos_weights,
            &WeightedEvalCfg {
                games_per_opponent: oos_games,
                agent_iters,
                c: agent_c,
                max_turns,
                seed: pair_seed,
                threads,
                blind_pool: blind_pool.clone(),
                open_sheet,
            },
        )
        .score;
        let in_dv = c.w2_score - parent_w2_scores[c.parent_idx];
        let oos_dv = c_oos - p_oos;
        eprintln!(
            "  {} (from {}, op={}): In-Sample dV={:+.2}pt | OOS Turn-0 OpenAgent: Parent={:.3} -> Child={:.3} (OOS dWR={:+.2}pt)",
            c.id,
            p.id,
            c.op,
            in_dv * 100.0,
            p_oos,
            c_oos,
            oos_dv * 100.0
        );
        oos_records.push(json!({
            "child_id": c.id,
            "parent_id": p.id,
            "op": c.op,
            "in_sample_dv": in_dv,
            "parent_oos_wr": p_oos,
            "child_oos_wr": c_oos,
            "oos_dwr": oos_dv,
        }));
    }

    if let Some(path) = out_file {
        let out = json!({
            "elapsed_secs": t_start.elapsed().as_secs_f64(),
            "replacement_level": {
                "mu_50": mu_50,
                "mu_55": mu_55,
                "beta_l": beta_l,
            },
            "targeted": {
                "count": targ_dv.len(),
                "mean_dv": mean(&targ_dv),
                "hit_rate_1pt": hit_rate(&targ_dv, 0.01),
                "hit_rate_pos": hit_rate(&targ_dv, 0.0001),
                "mean_dwar_slot": mean(&targ_dwar),
            },
            "untargeted": {
                "count": ctrl_dv.len(),
                "mean_dv": mean(&ctrl_dv),
                "hit_rate_1pt": hit_rate(&ctrl_dv, 0.01),
                "hit_rate_pos": hit_rate(&ctrl_dv, 0.0001),
                "mean_dwar_slot": mean(&ctrl_dwar),
            },
            "oos_verification": oos_records,
        });
        write_json_atomic(&path, &out);
    }
}

fn cmd_audit_elites(args: &[String]) {
    let t_start = std::time::Instant::now();
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let elites_file = flag(args, "--elites").unwrap_or_else(|| "/tmp/kodoku-endo-v1/kodoku-elites.json".to_string());
    let bench_file = flag(args, "--bench").unwrap_or_else(|| {
        root.join("data/meta-nash-v3/pool-artifact.json")
            .to_string_lossy()
            .into_owned()
    });
    let catalog_file = flag(args, "--catalog").unwrap_or_else(|| {
        root.join("data/team-pool-v2/team-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let belief_file = flag(args, "--belief").unwrap_or_else(|| {
        root.join("data/belief-pool-v3/belief-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let out_file = flag(args, "--out").map(PathBuf::from);
    let open200_games: u32 = num(args, "--open200-games", 40);
    let open2000_games: u32 = num(args, "--open2000-games", 16);
    let blind2000_games: u32 = num(args, "--blind2000-games", 16);
    let heldout_games: u32 = num(args, "--heldout-games", 8);
    let max_turns: u16 = num(args, "--max-turns", 250);
    let seed: u64 = num(args, "--seed", 20261029);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let (bench_teams, bench_weights) = load_gauntlet_file(&dex, &gen, &bench_file);
    let bv: Value = serde_json::from_str(&std::fs::read_to_string(&bench_file).unwrap()).unwrap();
    let bench_ids: BTreeSet<String> = bv["teams"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["id"].as_str().map(String::from))
        .collect();

    let cv: Value = serde_json::from_str(&std::fs::read_to_string(&catalog_file).unwrap()).unwrap();
    let mut parent_map: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut heldout_teams: Vec<Vec<PokemonSet>> = Vec::new();
    for t in cv["teams"].as_array().unwrap() {
        let id = t["id"].as_str().unwrap().to_string();
        let raw = t["sets"].as_array().unwrap().clone();
        let canon = gen.canonize(&dex, &raw).unwrap();
        parent_map.insert(id.clone(), canon.clone());
        if !bench_ids.contains(&id) {
            heldout_teams.push(to_sets(&canon).unwrap());
        }
    }
    for extra_rel in [
        "data/eco-pool/master-pool.json",
        "data/community-rentals-v0/teams.json",
        "data/eco-r17/pool-artifact.json",
        "data/team-inventory-v1/eval/teams.json",
    ] {
        let p = root.join(extra_rel);
        if let Ok(txt) = std::fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                if let Some(arr) = v["teams"].as_array() {
                    for t in arr {
                        if let (Some(id), Some(sets)) = (t["id"].as_str(), t["sets"].as_array()) {
                            if let Some(canon) = gen.canonize(&dex, sets) {
                                parent_map.entry(id.to_string()).or_insert(canon);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut ledger_dirs: Vec<PathBuf> = Vec::new();
    if let Some(parent_dir) = PathBuf::from(&elites_file).parent() {
        ledger_dirs.push(parent_dir.to_path_buf());
    }
    if let Some(extra_ldir) = flag(args, "--ledger-dir") {
        ledger_dirs.push(PathBuf::from(extra_ldir));
    }
    for ldir in ledger_dirs {
        if let Ok(entries) = std::fs::read_dir(&ldir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(ltxt) = std::fs::read_to_string(&p) {
                    if let Ok(lv) = serde_json::from_str::<Value>(&ltxt) {
                        for key in ["active_strains", "archived_strains"] {
                            if let Some(arr) = lv[key].as_array() {
                                for s in arr {
                                    if let (Some(id), Some(team)) = (s["id"].as_str(), s["team"].as_array()) {
                                        if let Some(canon) = gen.canonize(&dex, team) {
                                            parent_map.entry(id.to_string()).or_insert(canon);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let heldout_weights = vec![1.0 / (heldout_teams.len().max(1) as f64); heldout_teams.len()];

    let bp_text = std::fs::read_to_string(&belief_file).unwrap();
    let raw_mp: MetaPool = serde_json::from_str(&bp_text).expect("parse belief-pool");
    let mp = normalize_belief_pool(&dex, &gen, &raw_mp);
    let blind_pool = Some(Arc::new(mp));

    let top_n: usize = num(args, "--top", 5);
    let ev: Value = serde_json::from_str(&std::fs::read_to_string(&elites_file).unwrap()).unwrap();
    let evolved: Vec<&Value> = ev["teams"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["born_wave"].as_u64().unwrap_or(0) > 0)
        .take(top_n)
        .collect();

    eprintln!(
        "=== Adversarial Verification of {} Evolved Strains ===",
        evolved.len()
    );
    eprintln!(
        "  Regimes: open:200 ({}g/opp = {}g), open:2000 ({}g/opp = {}g), blind:2000 ({}g/opp = {}g), HeldOut-19 open:200 ({}g/opp = {}g)",
        open200_games,
        open200_games as usize * bench_teams.len(),
        open2000_games,
        open2000_games as usize * bench_teams.len(),
        blind2000_games,
        blind2000_games as usize * bench_teams.len(),
        heldout_games,
        heldout_games as usize * heldout_teams.len(),
    );

    let mut parent_cache_200: BTreeMap<String, nc2000_bot::teamgen::EvalResult> = BTreeMap::new();
    let mut parent_cache_2000: BTreeMap<String, nc2000_bot::teamgen::EvalResult> = BTreeMap::new();
    let mut parent_cache_blind: BTreeMap<String, nc2000_bot::teamgen::EvalResult> = BTreeMap::new();
    let mut parent_cache_heldout: BTreeMap<String, nc2000_bot::teamgen::EvalResult> = BTreeMap::new();

    let mut records = Vec::new();
    for t in evolved.iter() {
        let cid = t["id"].as_str().unwrap();
        let pid = t["parent_id"].as_str().unwrap();
        let op = t["last_op"].as_str().unwrap_or("?");
        let mut_slot: Option<usize> = op
            .split_once('@')
            .and_then(|(_, s)| s.parse::<usize>().ok());
        let c_raw = t["sets"].as_array().unwrap().clone();
        let c_canon = gen.canonize(&dex, &c_raw).unwrap();
        let Some(p_canon) = parent_map.get(pid).cloned() else {
            eprintln!("  [skip parent-diff] {cid}: parent {pid} not found in catalog/ledgers");
            continue;
        };
        let c_sets = to_sets(&c_canon).unwrap();
        let p_sets = to_sets(&p_canon).unwrap();

        let pair_seed = seed;

        // 1. open:200 vs Meta-Nash-v3 (CRN-paired)
        let p_open200 = parent_cache_200
            .entry(pid.to_string())
            .or_insert_with(|| {
                gauntlet_eval_weighted(
                    &dex,
                    &p_sets,
                    &bench_teams,
                    &bench_weights,
                    &WeightedEvalCfg {
                        games_per_opponent: open200_games,
                        agent_iters: 200,
                        c: 0.5,
                        max_turns,
                        seed: pair_seed ^ 0x1111,
                        threads,
                        blind_pool: None,
                        open_sheet: true,
                    },
                )
            })
            .clone();
        let c_open200 = gauntlet_eval_weighted(
            &dex,
            &c_sets,
            &bench_teams,
            &bench_weights,
            &WeightedEvalCfg {
                games_per_opponent: open200_games,
                agent_iters: 200,
                c: 0.5,
                max_turns,
                seed: pair_seed ^ 0x1111,
                threads,
                blind_pool: None,
                open_sheet: true,
            },
        );

        // 2. open:2000 vs Meta-Nash-v3 (CRN-paired, 10x search budget)
        let p_open2000 = parent_cache_2000
            .entry(pid.to_string())
            .or_insert_with(|| {
                gauntlet_eval_weighted(
                    &dex,
                    &p_sets,
                    &bench_teams,
                    &bench_weights,
                    &WeightedEvalCfg {
                        games_per_opponent: open2000_games,
                        agent_iters: 2000,
                        c: 0.5,
                        max_turns,
                        seed: pair_seed ^ 0x2222,
                        threads,
                        blind_pool: None,
                        open_sheet: true,
                    },
                )
            })
            .clone();
        let c_open2000 = gauntlet_eval_weighted(
            &dex,
            &c_sets,
            &bench_teams,
            &bench_weights,
            &WeightedEvalCfg {
                games_per_opponent: open2000_games,
                agent_iters: 2000,
                c: 0.5,
                max_turns,
                seed: pair_seed ^ 0x2222,
                threads,
                blind_pool: None,
                open_sheet: true,
            },
        );

        // 3. blind:2000 vs Meta-Nash-v3 (CRN-paired, hidden info)
        let p_blind2000 = parent_cache_blind
            .entry(pid.to_string())
            .or_insert_with(|| {
                gauntlet_eval_weighted(
                    &dex,
                    &p_sets,
                    &bench_teams,
                    &bench_weights,
                    &WeightedEvalCfg {
                        games_per_opponent: blind2000_games,
                        agent_iters: 2000,
                        c: 0.5,
                        max_turns,
                        seed: pair_seed ^ 0x3333,
                        threads,
                        blind_pool: blind_pool.clone(),
                        open_sheet: false,
                    },
                )
            })
            .clone();
        let c_blind2000 = gauntlet_eval_weighted(
            &dex,
            &c_sets,
            &bench_teams,
            &bench_weights,
            &WeightedEvalCfg {
                games_per_opponent: blind2000_games,
                agent_iters: 2000,
                c: 0.5,
                max_turns,
                seed: pair_seed ^ 0x3333,
                threads,
                blind_pool: blind_pool.clone(),
                open_sheet: false,
            },
        );

        // 4. Held-Out 19 Teams (open:200, CRN-paired)
        let p_heldout = parent_cache_heldout
            .entry(pid.to_string())
            .or_insert_with(|| {
                gauntlet_eval_weighted(
                    &dex,
                    &p_sets,
                    &heldout_teams,
                    &heldout_weights,
                    &WeightedEvalCfg {
                        games_per_opponent: heldout_games,
                        agent_iters: 200,
                        c: 0.5,
                        max_turns,
                        seed: pair_seed ^ 0x4444,
                        threads,
                        blind_pool: None,
                        open_sheet: true,
                    },
                )
            })
            .clone();
        let c_heldout = gauntlet_eval_weighted(
            &dex,
            &c_sets,
            &heldout_teams,
            &heldout_weights,
            &WeightedEvalCfg {
                games_per_opponent: heldout_games,
                agent_iters: 200,
                c: 0.5,
                max_turns,
                seed: pair_seed ^ 0x4444,
                threads,
                blind_pool: None,
                open_sheet: true,
            },
        );

        let (p_pick_200, c_pick_200, p_pick_2000, c_pick_2000) = if let Some(s) = mut_slot {
            (
                p_open200.slot_picks[s],
                c_open200.slot_picks[s],
                p_open2000.slot_picks[s],
                c_open2000.slot_picks[s],
            )
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        eprintln!(
            "  {:12} (from {:13}, {:14}):\n    open:200   P={:.3} -> C={:.3} (dWR={:+.2}pt) | mut_pick: {:.1}% -> {:.1}%\n    open:2000  P={:.3} -> C={:.3} (dWR={:+.2}pt) | mut_pick: {:.1}% -> {:.1}%\n    blind:2000 P={:.3} -> C={:.3} (dWR={:+.2}pt)\n    HeldOut-19 P={:.3} -> C={:.3} (dWR={:+.2}pt)",
            cid,
            pid,
            op,
            p_open200.score,
            c_open200.score,
            (c_open200.score - p_open200.score) * 100.0,
            p_pick_200 * 100.0,
            c_pick_200 * 100.0,
            p_open2000.score,
            c_open2000.score,
            (c_open2000.score - p_open2000.score) * 100.0,
            p_pick_2000 * 100.0,
            c_pick_2000 * 100.0,
            p_blind2000.score,
            c_blind2000.score,
            (c_blind2000.score - p_blind2000.score) * 100.0,
            p_heldout.score,
            c_heldout.score,
            (c_heldout.score - p_heldout.score) * 100.0,
        );

        records.push(json!({
            "child_id": cid,
            "parent_id": pid,
            "op": op,
            "mut_slot": mut_slot,
            "open200": {
                "parent_wr": p_open200.score,
                "child_wr": c_open200.score,
                "dwr": c_open200.score - p_open200.score,
                "parent_mut_pick": p_pick_200,
                "child_mut_pick": c_pick_200,
            },
            "open2000": {
                "parent_wr": p_open2000.score,
                "child_wr": c_open2000.score,
                "dwr": c_open2000.score - p_open2000.score,
                "parent_mut_pick": p_pick_2000,
                "child_mut_pick": c_pick_2000,
            },
            "blind2000": {
                "parent_wr": p_blind2000.score,
                "child_wr": c_blind2000.score,
                "dwr": c_blind2000.score - p_blind2000.score,
            },
            "heldout19_open200": {
                "parent_wr": p_heldout.score,
                "child_wr": c_heldout.score,
                "dwr": c_heldout.score - p_heldout.score,
            },
        }));
    }

    if let Some(path) = out_file {
        write_json_atomic(
            &path,
            &json!({
                "elapsed_secs": t_start.elapsed().as_secs_f64(),
                "records": records,
            }),
        );
    }
}

fn eval_blind_asymmetric_prior(
    dex: &Dex,
    candidate: &[PokemonSet],
    gauntlet: &[Vec<PokemonSet>],
    weights: &[f64],
    cand_pool: Arc<MetaPool>,
    opp_pool: Arc<MetaPool>,
    games_per_opponent: u32,
    agent_iters: u32,
    c: f64,
    max_turns: u16,
    threads: usize,
    seed: u64,
) -> nc2000_bot::teamgen::EvalResult {
    use nc2000_bot::runner::{play_game, GameResult};
    use nc2000_bot::smmcts::{RmConfig, SelRule};
    use nc2000_bot::{Agent, BlindAgent};
    use nc2000_engine::battle::Outcome;
    use nc2000_engine::state::Battle;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Job {
        opp: usize,
        battle_seed: String,
        cand_seed: u64,
        opp_seed: u64,
        cand_is_p1: bool,
    }

    let games = (games_per_opponent + games_per_opponent % 2) as usize;
    let mut jobs = Vec::with_capacity(gauntlet.len() * games);
    for opp in 0..gauntlet.len() {
        let mut rng =
            SplitMix64::new(seed ^ (opp as u64 + 1).wrapping_mul(0x9FB2_1C65_1E98_DF25));
        let mut battle_seed = String::new();
        for g in 0..games {
            if g % 2 == 0 {
                battle_seed = rng.battle_seed();
            }
            jobs.push(Job {
                opp,
                battle_seed: battle_seed.clone(),
                cand_seed: rng.next(),
                opp_seed: rng.next(),
                cand_is_p1: g % 2 == 0,
            });
        }
    }

    let cursor = AtomicUsize::new(0);
    let mut results: Vec<(usize, f64, u8)> = Vec::with_capacity(jobs.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads.max(1) {
            let (jobs, cursor, cand_pool, opp_pool) = (&jobs, &cursor, &cand_pool, &opp_pool);
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, (usize, f64, u8))> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let job = &jobs[i];
                    let rm_cfg = RmConfig {
                        iterations: agent_iters,
                        rule: SelRule::Ucb,
                        c,
                        ..Default::default()
                    };
                    let mut cand_agent: Box<dyn Agent> =
                        Box::new(BlindAgent::new(rm_cfg.clone(), cand_pool.clone(), None, job.cand_seed));
                    let mut opp_agent: Box<dyn Agent> =
                        Box::new(BlindAgent::new(rm_cfg, opp_pool.clone(), None, job.opp_seed));
                    let (t1, t2) = if job.cand_is_p1 {
                        (candidate, gauntlet[job.opp].as_slice())
                    } else {
                        (gauntlet[job.opp].as_slice(), candidate)
                    };
                    let mut b = Battle::from_fixture(dex, &job.battle_seed, t1, t2).unwrap();
                    b.set_log_enabled(true);
                    let res = if job.cand_is_p1 {
                        play_game(
                            dex,
                            &mut b,
                            &mut [cand_agent.as_mut(), opp_agent.as_mut()],
                            max_turns,
                        )
                    } else {
                        play_game(
                            dex,
                            &mut b,
                            &mut [opp_agent.as_mut(), cand_agent.as_mut()],
                            max_turns,
                        )
                    }
                    .unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let score = if job.cand_is_p1 { p1_score } else { 1.0 - p1_score };
                    let cand_side = if job.cand_is_p1 { 0 } else { 1 };
                    let mut cmask = 0u8;
                    for &slot in &b.sides[cand_side].party {
                        if (slot as usize) < 6 {
                            cmask |= 1 << slot;
                        }
                    }
                    out.push((i, (job.opp, score, cmask)));
                }
                out
            }));
        }
        let mut all: Vec<(usize, (usize, f64, u8))> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all
            .into_iter()
            .map(|(i, (_, s, cm))| (jobs[i].opp, s, cm))
            .collect();
    });

    let wsum: f64 = if weights.len() == gauntlet.len() {
        weights.iter().sum::<f64>().max(1e-12)
    } else {
        gauntlet.len().max(1) as f64
    };
    let mut per: Vec<(f64, usize)> = vec![(0.0, 0); gauntlet.len()];
    let mut slot_picks = [0.0f64; 6];
    let mut slot_wins = [0.0f64; 6];
    let mut core_stats: BTreeMap<u8, (f64, f64)> = BTreeMap::new();
    for &(opp, s, cm) in &results {
        per[opp].0 += s;
        per[opp].1 += 1;
        let opp_w = if weights.len() == gauntlet.len() {
            weights[opp] / wsum
        } else {
            1.0 / wsum
        };
        let w_game = opp_w / (games.max(1) as f64);
        let entry = core_stats.entry(cm).or_insert((0.0, 0.0));
        entry.0 += w_game;
        entry.1 += w_game * s;
        for slot in 0..6 {
            if (cm & (1 << slot)) != 0 {
                slot_picks[slot] += w_game;
                slot_wins[slot] += w_game * s;
            }
        }
    }
    let per_opponent: Vec<f64> = per.iter().map(|&(s, n)| s / n.max(1) as f64).collect();
    let score = if weights.len() == gauntlet.len() {
        per_opponent
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| s * w)
            .sum::<f64>()
            / wsum
    } else {
        per_opponent.iter().sum::<f64>() / per_opponent.len().max(1) as f64
    };
    nc2000_bot::teamgen::EvalResult {
        score,
        per_opponent,
        games: results.len(),
        slot_picks,
        slot_wins,
        core_stats,
    }
}

fn cmd_sibling_mix(args: &[String]) {
    let t_start = std::time::Instant::now();
    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    let elites_file = flag(args, "--elites")
        .unwrap_or_else(|| "/tmp/kodoku-clean-v1/kodoku-elites.json".to_string());
    let child_id = flag(args, "--child").unwrap_or_else(|| "kodoku-s0018".to_string());
    let bench_file = flag(args, "--bench").unwrap_or_else(|| {
        root.join("data/meta-nash-v3/pool-artifact.json")
            .to_string_lossy()
            .into_owned()
    });
    let catalog_file = flag(args, "--catalog").unwrap_or_else(|| {
        root.join("data/team-pool-v2/team-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let belief_file = flag(args, "--belief").unwrap_or_else(|| {
        root.join("data/belief-pool-v3/belief-pool.json")
            .to_string_lossy()
            .into_owned()
    });
    let out_file = flag(args, "--out").map(PathBuf::from);
    let games_per_opp: u32 = num(args, "--games", 16);
    let iters: u32 = num(args, "--iters", 2000);
    let max_turns: u16 = num(args, "--max-turns", 250);
    let seed: u64 = num(args, "--seed", 20261029);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let (bench_teams, bench_weights) = load_gauntlet_file(&dex, &gen, &bench_file);
    let bv: Value = serde_json::from_str(&std::fs::read_to_string(&bench_file).unwrap()).unwrap();
    let cv: Value = serde_json::from_str(&std::fs::read_to_string(&catalog_file).unwrap()).unwrap();
    let mut parent_map: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for src in [&bv, &cv] {
        for t in src["teams"].as_array().unwrap() {
            let id = t["id"].as_str().unwrap().to_string();
            let raw = t["sets"].as_array().unwrap().clone();
            let canon = gen.canonize(&dex, &raw).unwrap();
            parent_map.insert(id, canon);
        }
    }
    let ledger_file = PathBuf::from(&elites_file)
        .parent()
        .map(|p| p.join("kodoku-ledger.json"))
        .unwrap_or_else(|| PathBuf::from("/tmp/kodoku-clean-v1/kodoku-ledger.json"));
    if let Ok(ltxt) = std::fs::read_to_string(&ledger_file) {
        if let Ok(lv) = serde_json::from_str::<Value>(&ltxt) {
            for key in ["active_strains", "archived_strains"] {
                if let Some(arr) = lv[key].as_array() {
                    for s in arr {
                        if let (Some(id), Some(team)) = (s["id"].as_str(), s["team"].as_array()) {
                            if let Some(canon) = gen.canonize(&dex, team) {
                                parent_map.entry(id.to_string()).or_insert(canon);
                            }
                        }
                    }
                }
            }
        }
    }

    let ev: Value = serde_json::from_str(&std::fs::read_to_string(&elites_file).unwrap()).unwrap();
    let child_val = ev["teams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"].as_str() == Some(child_id.as_str()))
        .expect("child_id not found in elites");
    let pid = child_val["parent_id"].as_str().unwrap();
    let op = child_val["last_op"].as_str().unwrap_or("?");
    let mut_slot: usize = op
        .split_once('@')
        .and_then(|(_, s)| s.parse::<usize>().ok())
        .unwrap_or(0);
    let c_raw = child_val["sets"].as_array().unwrap().clone();
    let c_canon = gen.canonize(&dex, &c_raw).unwrap();
    let p_canon = parent_map.get(pid).expect("parent not found").clone();
    let c_sets = to_sets(&c_canon).unwrap();
    let p_sets = to_sets(&p_canon).unwrap();

    let bp_text = std::fs::read_to_string(&belief_file).unwrap();
    let raw_mp: MetaPool = serde_json::from_str(&bp_text).expect("parse belief-pool");
    let base_mp = normalize_belief_pool(&dex, &gen, &raw_mp);
    let cand_pool = Arc::new(base_mp.clone());
    let deme_sig = preview_sig_sets(&p_sets);

    let alphas: Vec<f64> = flag(args, "--alphas")
        .map(|s| s.split(',').map(|x| x.trim().parse::<f64>().unwrap()).collect())
        .unwrap_or_else(|| vec![0.0, 0.0001, 0.15, 0.25, 0.40, 0.60, 0.9999]);

    eprintln!(
        "=== Same-Signature Deme Equilibrium Sweep: Parent={} vs Child={} ({}) ===",
        pid, child_id, op
    );
    eprintln!(
        "  Regime: blind:{} ({}g/opp = {}g CRN-paired vs {} Meta-Nash-v3 teams)",
        iters,
        games_per_opp,
        games_per_opp as usize * bench_teams.len(),
        bench_teams.len()
    );

    let pair_seed = seed ^ 0x3333;
    let mut sweep_records = Vec::new();
    for &alpha in &alphas {
        let mut mp = base_mp.clone();
        // Remove any existing candidate in `mp` that collides with P/C's preview signature,
        // then insert P with weight `0.5 * (1 - alpha)` and C with weight `0.5 * alpha`.
        mp.teams.retain(|t| preview_sig_sets(&t.sets) != deme_sig);
        if alpha <= 0.0 {
            mp.teams.push(nc2000_bot::preview::MetaTeam {
                id: pid.to_string(),
                sets: p_sets.clone(),
                weight: Some(0.5),
            });
        } else if alpha >= 1.0 {
            mp.teams.push(nc2000_bot::preview::MetaTeam {
                id: child_id.clone(),
                sets: c_sets.clone(),
                weight: Some(0.5),
            });
        } else {
            mp.teams.push(nc2000_bot::preview::MetaTeam {
                id: pid.to_string(),
                sets: p_sets.clone(),
                weight: Some((0.5 * (1.0 - alpha)).max(1e-9)),
            });
            mp.teams.push(nc2000_bot::preview::MetaTeam {
                id: child_id.clone(),
                sets: c_sets.clone(),
                weight: Some((0.5 * alpha).max(1e-9)),
            });
        }
        let opp_pool = Arc::new(mp);

        // Sanity check that `Belief` sees both P and C alive at preview when `0 < alpha < 1`
        {
            let tb = nc2000_engine::state::Battle::from_fixture(&dex, "1,2,3,4", &p_sets, &c_sets).unwrap();
            let obs_p = nc2000_bot::Observer::new(&tb, 1); // observing side 0 (P)
            let bel_p = nc2000_bot::Belief::new(&dex, &opp_pool, &obs_p);
            let obs_c = nc2000_bot::Observer::new(&tb, 0); // observing side 1 (C)
            let bel_c = nc2000_bot::Belief::new(&dex, &opp_pool, &obs_c);
            let expected_alive = if alpha <= 0.0 || alpha >= 1.0 { 1 } else { 2 };
            assert_eq!(
                bel_p.candidate_count(),
                expected_alive,
                "preview alive count for P must equal {}",
                expected_alive
            );
            assert_eq!(
                bel_c.candidate_count(),
                expected_alive,
                "preview alive count for C must equal {}",
                expected_alive
            );
        }

        let p_res = eval_blind_asymmetric_prior(
            &dex,
            &p_sets,
            &bench_teams,
            &bench_weights,
            cand_pool.clone(),
            opp_pool.clone(),
            games_per_opp,
            iters,
            0.5,
            max_turns,
            threads,
            pair_seed,
        );
        let c_res = eval_blind_asymmetric_prior(
            &dex,
            &c_sets,
            &bench_teams,
            &bench_weights,
            cand_pool.clone(),
            opp_pool,
            games_per_opp,
            iters,
            0.5,
            max_turns,
            threads,
            pair_seed,
        );
        let group_wr = (1.0 - alpha) * p_res.score + alpha * c_res.score;
        let gap = c_res.score - p_res.score;
        eprintln!(
            "  alpha={:6.4} | V_blind(P|a)={:.4} (pick={:.1}%) | V_blind(C|a)={:.4} (pick={:.1}%) | C-P={:+.2}pt | V_deme(a)={:.4}",
            alpha,
            p_res.score,
            p_res.slot_picks[mut_slot] * 100.0,
            c_res.score,
            c_res.slot_picks[mut_slot] * 100.0,
            gap * 100.0,
            group_wr
        );
        sweep_records.push(json!({
            "alpha": alpha,
            "parent_wr": p_res.score,
            "child_wr": c_res.score,
            "child_minus_parent": gap,
            "deme_wr": group_wr,
            "parent_mut_pick": p_res.slot_picks[mut_slot],
            "child_mut_pick": c_res.slot_picks[mut_slot],
        }));
    }

    if let Some(path) = out_file {
        write_json_atomic(
            &path,
            &json!({
                "elapsed_secs": t_start.elapsed().as_secs_f64(),
                "parent_id": pid,
                "child_id": child_id,
                "op": op,
                "sweep": sweep_records,
            }),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sub = args
        .first()
        .map(String::as_str)
        .unwrap_or("evolve-island");
    match sub {
        "evolve-island" => cmd_evolve_island(&args[1..]),
        "extract-pool" => cmd_extract_pool(&args[1..]),
        "extract-nash" => cmd_extract_nash(&args[1..]),
        "verify-nash" => cmd_verify_nash(&args[1..]),
        "update-ring" => cmd_update_ring(&args[1..]),
        "update-pool" => cmd_update_pool(&args[1..]),
        "tomography" => cmd_tomography(&args[1..]),
        "kodoku" => cmd_kodoku(&args[1..]),
        "core-nash" => cmd_core_nash(&args[1..]),
        "verify-1gen" => cmd_verify_1gen(&args[1..]),
        "audit-elites" => cmd_audit_elites(&args[1..]),
        "sibling-mix" => cmd_sibling_mix(&args[1..]),
        other => panic!(
            "unknown subcommand {other} (expected evolve-island | extract-pool | extract-nash | verify-nash | update-ring | update-pool | tomography | kodoku | core-nash | verify-1gen | audit-elites | sibling-mix)"
        ),
    }
}
