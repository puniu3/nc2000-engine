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

fn parse_agent_spec(args: &[String], root: &Path) -> (u32, f64, Option<Arc<MetaPool>>) {
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
            "blind" => {
                let bpath = flag(args, "--belief-pool")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| root.join("data/belief-pool-v3/belief-pool.json"));
                let btext = std::fs::read_to_string(&bpath)
                    .unwrap_or_else(|e| panic!("read belief-pool {}: {e}", bpath.display()));
                let mp: MetaPool = serde_json::from_str(&btext).expect("parse belief-pool");
                (it, c, Some(Arc::new(mp)))
            }
            "skuct" => (it, c, None),
            other => panic!("unsupported --agent kind {other} (expected blind or skuct)"),
        }
    } else {
        (1000, 0.4, None)
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
    let (agent_iters, agent_c, blind_pool) = parse_agent_spec(args, &root);

    let niches_spec = flag(args, "--niches").unwrap_or_else(|| "rand||".into());
    let niches = parse_niches(&niches_spec);

    let mut ledger: IslandLedger = if resume && ledger_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&ledger_path).unwrap()).unwrap()
    } else {
        let mut rng = SplitMix64::new(base_seed ^ 0xEC05_757E_u64);
        let seed_dir = flag(args, "--seed-dir")
            .filter(|s| !s.is_empty() && s != "none")
            .map(PathBuf::from);
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
                &mut ledger,
                &mut rng,
            );
        }

        // Build co-evolutionary opponent pool from top strains of each founder lineage (up to 4).
        // In Wave 1 (before any strain has cum_games > 0), use the initial founder teams so
        // local island co-evolution starts from Wave 1.
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
        let mut vet_niches: BTreeSet<String> = BTreeSet::new();
        for vi in vet_indices {
            if chosen_vets.len() >= 4 {
                break;
            }
            if vet_niches.insert(ledger.active_strains[vi].niche.clone()) {
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

        let bench_games_per_opp = (eval_games / 2).max(4);
        let bench_n_games = (bench_games_per_opp as usize * bench_teams.len()) as f64;
        let eco_n_games = (eval_games as usize * eco_opps.len()) as f64;

        // Evaluate every active strain against Living Island Ecosystem (primary) and Benchmark (secondary)
        for s in &mut ledger.active_strains {
            let cand = to_sets(&s.team).expect("valid strain team");
            let bench_fit = gauntlet_eval_weighted(
                &dex,
                &cand,
                &bench_teams,
                &bench_weights,
                &WeightedEvalCfg {
                    games_per_opponent: bench_games_per_opp,
                    agent_iters,
                    c: agent_c,
                    max_turns,
                    seed: wave_seed,
                    threads,
                    blind_pool: blind_pool.clone(),
                },
            )
            .score;
            s.bench_games += bench_n_games;
            s.bench_wins += bench_fit * bench_n_games;

            let wave_fit = if !eco_opps.is_empty() && env_self_weight > 0.0 {
                let eco_fit = gauntlet_eval_weighted(
                    &dex,
                    &cand,
                    &eco_opps,
                    &eco_weights,
                    &WeightedEvalCfg {
                        games_per_opponent: eval_games,
                        agent_iters,
                        c: agent_c,
                        max_turns,
                        seed: wave_seed ^ 0x5555_AAAA_5555_AAAA,
                        threads,
                        blind_pool: blind_pool.clone(),
                    },
                )
                .score;
                s.eco_games += eco_n_games;
                s.eco_wins += eco_fit * eco_n_games;
                (1.0 - env_self_weight) * bench_fit + env_self_weight * eco_fit
            } else {
                bench_fit
            };

            let total_w_games = bench_n_games + if eco_opps.is_empty() { 0.0 } else { eco_n_games };
            s.cum_games += total_w_games;
            s.cum_wins += wave_fit * total_w_games;
            s.last_wave_fit = wave_fit;
            s.recompute_stats(prior_games);
        }

        // Within-Niche Boltzmann-Replicator Update on continuous abundance w_i:
        // Each niche maintains its own abundance share (1 / num_niches of the island) so
        // warm-started niches never starve de-novo anchor niches of mutations, and within
        // each niche damped replicator dynamics w_i <- w_i^0.5 * exp(eta * (u_i - u_bar_niche))
        // lets stronger offspring overtake parents without the 3.55x founder handicap.
        let mut niche_groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (idx, s) in ledger.active_strains.iter().enumerate() {
            niche_groups.entry(s.niche.clone()).or_default().push(idx);
        }
        let num_niches = niche_groups.len().max(1) as f64;
        for indices in niche_groups.values() {
            let mut w_sum = 0.0;
            let mut u_bar = 0.0;
            for &i in indices {
                let s = &ledger.active_strains[i];
                let u = 0.35 * s.last_wave_fit + 0.65 * s.bayes_fit;
                u_bar += s.abundance * u;
                w_sum += s.abundance;
            }
            if w_sum > 0.0 {
                u_bar /= w_sum;
            }
            let mut new_niche_w = 0.0;
            for &i in indices {
                let s = &mut ledger.active_strains[i];
                let u = 0.35 * s.last_wave_fit + 0.65 * s.bayes_fit;
                s.abundance = (s.abundance.max(1e-4).sqrt() * (eta * (u - u_bar)).exp()).max(1e-6);
                new_niche_w += s.abundance;
            }
            for &i in indices {
                ledger.active_strains[i].abundance =
                    (ledger.active_strains[i].abundance / new_niche_w.max(1e-9)) / num_niches;
            }
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

fn bud_next_wave_by_niche(
    dex: &Dex,
    gen: &TeamGen,
    island: &str,
    born_wave: u32,
    carry: usize,
    buds: usize,
    ledger: &mut IslandLedger,
    rng: &mut SplitMix64,
) {
    let num_niches = ledger.niches.len().max(1);
    let carry_per_niche = (carry / num_niches).max(2);
    let buds_per_niche = (buds / num_niches).max(1);

    // Group active strains by niche, sorted by composite score (70% bayes_fit + 30% abundance)
    let mut by_niche: BTreeMap<String, Vec<Strain>> = BTreeMap::new();
    for s in ledger.active_strains.drain(..) {
        by_niche.entry(s.niche.clone()).or_default().push(s);
    }

    let mut next_active: Vec<Strain> = Vec::new();
    let mut seen_keys: BTreeSet<String> = BTreeSet::new();

    for niche_spec in &ledger.niches {
        let mut group = by_niche.remove(&niche_spec.name).unwrap_or_default();
        group.sort_by(|a, b| {
            let qa = 0.75 * a.bayes_fit + 0.25 * a.lcb95.max(a.bayes_fit - 0.12);
            let qb = 0.75 * b.bayes_fit + 0.25 * b.lcb95.max(b.bayes_fit - 0.12);
            qb.partial_cmp(&qa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut survivors: Vec<Strain> = Vec::new();
        for s in group {
            let k = team_key(&s.team);
            if survivors.len() < carry_per_niche && seen_keys.insert(k) {
                survivors.push(s);
            } else {
                ledger.archived_strains.push(s);
            }
        }

        // If a niche somehow had no survivors, seed a fresh one
        if survivors.is_empty() {
            let (anchors, constraint) =
                build_constraint(dex, &niche_spec.anchors, &niche_spec.max_staples);
            if let Some(team) = gen.random_team_constrained(dex, rng, &anchors, &constraint, 300) {
                let id = format!("{island}-s{:04}", ledger.next_strain_seq);
                ledger.next_strain_seq += 1;
                survivors.push(Strain {
                    id,
                    parent_id: "reseed".into(),
                    niche: niche_spec.name.clone(),
                    anchors: niche_spec.anchors.clone(),
                    max_staples: niche_spec.max_staples.clone(),
                    born_wave,
                    last_op: "reseed".into(),
                    abundance: 1.0 / (num_niches as f64),
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
                    team,
                });
            }
        }

        // Bud `buds_per_niche` mutant offspring for this niche without penalizing the parent's abundance
        let num_parents = survivors.len();
        let mut niche_buds: Vec<Strain> = Vec::new();
        for b_idx in 0..buds_per_niche {
            let p_idx = b_idx % num_parents;
            let parent = &survivors[p_idx];
            let (_, constraint) = build_constraint(dex, &parent.anchors, &parent.max_staples);
            if let Some(prop) = gen.propose_constrained(dex, &parent.team, rng, &constraint, 150) {
                let k = team_key(&prop.team);
                if seen_keys.insert(k) {
                    let id = format!("{island}-s{:04}", ledger.next_strain_seq);
                    ledger.next_strain_seq += 1;
                    niche_buds.push(Strain {
                        id,
                        parent_id: parent.id.clone(),
                        niche: parent.niche.clone(),
                        anchors: parent.anchors.clone(),
                        max_staples: parent.max_staples.clone(),
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
                        team: prop.team,
                    });
                }
            }
        }
        survivors.extend(niche_buds);
        let nw: f64 = survivors.iter().map(|s| s.abundance.max(1e-4)).sum();
        for s in &mut survivors {
            s.abundance = (s.abundance.max(1e-4) / nw.max(1e-9)) / (num_niches as f64);
        }
        next_active.extend(survivors);
    }

    // Archive any remaining ungrouped strains
    for (_k, rem) in by_niche {
        ledger.archived_strains.extend(rem);
    }
    ledger.active_strains = next_active;
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
            if s.cum_games < min_games {
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
    // Sort by composite ecological quality: blend local ecosystem Bayes winrate,
    // external benchmark Bayes winrate, and LCB95 so long-generation-time islands
    // are not penalized by sample count and weak-island self-play inflation is controlled.
    let strain_quality = |s: &Strain| -> f64 {
        let bench_bayes = if s.bench_games > 0.0 {
            (s.bench_wins + 16.0 * 0.40) / (s.bench_games + 16.0)
        } else {
            s.bayes_fit
        };
        0.45 * bench_bayes + 0.35 * s.bayes_fit + 0.20 * s.lcb95
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

fn cmd_extract_pool(args: &[String]) {
    let in_dir = PathBuf::from(flag(args, "--in").expect("--in DIR"));
    let out_file = PathBuf::from(flag(args, "--out").expect("--out FILE"));
    let min_games: f64 = num(args, "--min-games", 40.0);
    let min_bayes: f64 = num(args, "--min-bayes", 0.40);
    let top_k: usize = num(args, "--top", 24);
    let max_jaccard: f64 = num(args, "--max-jaccard", 0.50);

    let all = collect_all_strains(&in_dir, min_games);
    let mut selected: Vec<Strain> = Vec::new();
    let mut selected_species: Vec<BTreeSet<String>> = Vec::new();
    let mut seen_niches: BTreeSet<String> = BTreeSet::new();

    // Pass 1: take top validated representative of each distinct niche that meets `min_bayes`
    for s in &all {
        if selected.len() >= top_k {
            break;
        }
        if s.bayes_fit < min_bayes {
            continue;
        }
        let sp = s.species_set();
        if !seen_niches.contains(&s.niche)
            && selected_species
                .iter()
                .all(|prev| jaccard(&sp, prev) <= max_jaccard)
        {
            seen_niches.insert(s.niche.clone());
            selected_species.push(sp);
            selected.push(s.clone());
        }
    }
    // Pass 2: fill remaining slots with highest quality strains honoring max_jaccard
    for s in &all {
        if selected.len() >= top_k {
            break;
        }
        if selected.iter().any(|x| x.id == s.id) {
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
    let (agent_iters, agent_c, blind_pool) = parse_agent_spec(args, &root);

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

    // 2. Out-of-Sample Entry Gate: only admit evolved challengers that strictly beat Old-Nash
    // (`score_vs_old >= min_gate_wr`) on independent verification seeds!
    let pv: Value = serde_json::from_str(&std::fs::read_to_string(&pool_file).unwrap()).unwrap();
    eprintln!(
        "=== Step 1: Out-of-Sample Challenger Entry Gate vs Old-Nash (min_gate_wr={min_gate_wr:.3}, {gate_games} games/opp) ==="
    );
    for (c_idx, t) in pv["teams"]
        .as_array()
        .unwrap()
        .iter()
        .take(top_evolved)
        .enumerate()
    {
        let id = t["id"].as_str().unwrap().to_string();
        let raw = t["sets"].as_array().unwrap().clone();
        let canon = gen.canonize(&dex, &raw).unwrap();
        let cand_sets = to_sets(&canon).unwrap();
        let vs_old = gauntlet_eval_weighted(
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
            },
        )
        .score;
        let passed = vs_old >= min_gate_wr;
        eprintln!(
            "  challenger {:20} vs Old-Nash = {:.4} ({:.1}%) -> {}",
            id,
            vs_old,
            vs_old * 100.0,
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
    let (agent_iters, agent_c, blind_pool) = parse_agent_spec(args, &root);

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

    println!(
        "=== Out-of-Sample Head-to-Head Verification ({} new vs {} old, {} games/pair) ===",
        new_teams.len(),
        old_teams.len(),
        eval_games
    );
    let mut total_ev = 0.0;
    for (i, (cand, &w_i)) in new_teams.iter().zip(new_weights.iter()).enumerate() {
        let score_i = gauntlet_eval_weighted(
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
            },
        )
        .score;
        total_ev += w_i * score_i;
        println!(
            "  {:22} (w_new={:.4}) vs Old-Nash: {:.4} ({:.1}%)",
            new_ids.get(i).map(String::as_str).unwrap_or("?"),
            w_i,
            score_i,
            score_i * 100.0
        );
    }
    println!(
        "==> E[New-Nash vs Old-Nash] (out-of-sample) = {:.4} ({:.2}%)",
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
                    .or_else(|| t["drawWeight"].as_f64())
                    .or_else(|| t["lcb95"].as_f64())
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
        other => panic!(
            "unknown subcommand {other} (expected evolve-island | extract-pool | extract-nash | verify-nash | update-ring)"
        ),
    }
}
