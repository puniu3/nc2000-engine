//! M17c anchor gate — zero-noise eval regression against the certified
//! endgame anchors mined by `endgame_exactness_corpus`.
//!
//! Reads a complete merged M17e v3 artifact, re-derives
//! each position deterministically through `bot::corpus` (same seed, same
//! fabrication), evaluates a candidate `EvalWeights` config, and reports
//! proven bracket violations (eval outside [exact−w/2, exact+w/2] by more
//! than eps). No solving, no playouts — seconds, bit-reproducible, so a
//! candidate eval change can be gated on "violation count × margin must
//! drop" before the statistical gates (`eval_calibration --ab`,
//! `eval_ab_duel`) run.
//!
//! Artifact identity is fail-closed: complete shard coverage and row sets,
//! solver build, runtime data, corpus content, selection/solver config, and
//! every complete 128-bit reconstructed state must match. Shards, legacy
//! CSVs, custom sweeps, and stale merged artifacts are rejected.
//!
//! Usage: anchor_gate [--artifact tmp/m17e-merged-v3.json]
//!                    [--corpus tmp/corpus-spectator] [--race F]
//!                    [--leaf-alpha F] [--metric raw|leaf] [--eps 0.02]
//!                    [--list] [--moves]

use std::collections::HashMap;

use nc2000_bot::corpus::{
    corpus_files, corpus_fingerprint, load_battle, load_sources, reconstruct,
};
use nc2000_bot::eval::{eval01, eval_leaf, EvalWeights};
use nc2000_bot::m17e_artifact::{
    runtime_data_fingerprints, solver_build_fingerprint, validate_merged, MergedArtifact, Row,
};
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn validate_args(args: &[String]) {
    const VALUES: &[&str] = &[
        "--artifact",
        "--corpus",
        "--eps",
        "--race",
        "--leaf-alpha",
        "--metric",
    ];
    const FLAGS: &[&str] = &["--list", "--moves", "--help"];
    let mut index = 1;
    while index < args.len() {
        let key = args[index].as_str();
        if VALUES.contains(&key) {
            assert!(index + 1 < args.len(), "{key} requires a value");
            index += 2;
        } else if FLAGS.contains(&key) {
            index += 1;
        } else {
            panic!("unknown argument {key}; M17e v3 requires --artifact, not legacy --csv");
        }
    }
}

fn parse_artifact(text: &str, path: &str) -> Result<MergedArtifact, String> {
    let artifact: MergedArtifact =
        serde_json::from_str(text).map_err(|error| format!("{path}: {error}"))?;
    validate_merged(&artifact).map_err(|error| format!("{path}: {error}"))?;
    Ok(artifact)
}

fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side]
        .party
        .iter()
        .filter(|&&s| !b.sides[side].roster[s as usize].fainted)
        .count()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    validate_args(&args);
    if args.iter().any(|arg| arg == "--help") {
        println!("anchor_gate --artifact tmp/m17e-merged-v3.json [--corpus DIR] [--metric raw|leaf] [--eps F]");
        return;
    }
    let artifact_path = arg_s(&args, "--artifact", "tmp/m17e-merged-v3.json");
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let eps: f64 = arg_s(&args, "--eps", "0.02").parse().unwrap();
    let race: f64 = args
        .iter()
        .position(|a| a == "--race")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(EvalWeights::default().race);
    let leaf_alpha: f64 = arg_s(&args, "--leaf-alpha", "1").parse().unwrap();
    let metric = arg_s(&args, "--metric", "raw");
    assert!(
        metric == "raw" || metric == "leaf",
        "--metric must be raw or leaf"
    );
    let list = args.iter().any(|a| a == "--list");

    let weights = EvalWeights {
        race,
        leaf_alpha,
        ..EvalWeights::default()
    };

    let text = std::fs::read_to_string(&artifact_path)
        .unwrap_or_else(|error| panic!("{artifact_path}: {error}"));
    let artifact = parse_artifact(&text, &artifact_path).unwrap_or_else(|error| panic!("{error}"));
    assert!(
        artifact.run.is_formal(),
        "{artifact_path}: only the exact formal M17e v3 solver/selection profile is gateable"
    );

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let files = corpus_files(&root.join(&corpus));
    let current_corpus_fingerprint = corpus_fingerprint(&files);
    let current_runtime_data =
        runtime_data_fingerprints(&root).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        artifact.run.solver_build_fingerprint,
        solver_build_fingerprint(),
        "solver/build fingerprint mismatch; regenerate the formal sweep"
    );
    assert_eq!(
        artifact.run.runtime_data, current_runtime_data,
        "runtime Dex/meta/rentals/learnsets mismatch; regenerate the formal sweep"
    );
    assert_eq!(
        artifact.run.corpus_fingerprint, current_corpus_fingerprint,
        "corpus fingerprint mismatch; regenerate the formal sweep"
    );
    assert_eq!(
        artifact.run.corpus_count,
        files.len(),
        "corpus count mismatch; regenerate the formal sweep"
    );
    println!(
        "anchors: {} from {artifact_path}; {} complete shard(s), corpus {}; build {}; race {race}; metric {metric}; leaf alpha {leaf_alpha}",
        artifact.rows.len(),
        artifact.merge.shards.len(),
        current_corpus_fingerprint,
        solver_build_fingerprint(),
    );

    let anchors = &artifact.rows;
    let mut by_battle: HashMap<usize, Vec<&Row>> = HashMap::new();
    for a in anchors {
        by_battle.entry(a.battle).or_default().push(a);
    }

    let mut evaluated = 0usize;
    let mut viols: Vec<(f64, &Row, f64)> = Vec::new();
    let mut tight_absdev = 0.0f64;
    let mut tight_n = 0usize;

    let mut battles: Vec<usize> = by_battle.keys().cloned().collect();
    battles.sort();
    for bi in battles {
        let path = files
            .get(bi)
            .unwrap_or_else(|| panic!("anchor battle {bi} is outside corpus"));
        let cb = load_battle(path);
        for a in &by_battle[&bi] {
            let d = cb.decisions.get(a.decision).unwrap_or_else(|| {
                panic!(
                    "b{} d{}: source decision missing; regenerate anchors",
                    a.battle, a.decision
                )
            });
            assert!(
                d.side == a.side && d.turn == a.turn,
                "b{} d{}: source decision coordinate mismatch (CSV s{} T{}, corpus s{} T{}); regenerate anchors",
                a.battle,
                a.decision,
                a.side,
                a.turn,
                d.side,
                d.turn
            );
            let b = reconstruct(
                &dex,
                &src,
                &pool_path,
                &cb.lines,
                &cb.evidence,
                d,
                artifact.run.selection.reconstruction_seed,
            )
            .unwrap_or_else(|| {
                panic!(
                    "b{} d{}: reconstruction failed; regenerate anchors",
                    a.battle, a.decision
                )
            });
            let expected_state = u128::from_str_radix(&a.state_key128, 16)
                .unwrap_or_else(|error| panic!("invalid validated state key: {error}"));
            assert_eq!(
                b.state_key128(),
                expected_state,
                "b{} d{}: reconstructed state_key128 mismatch; regenerate anchors",
                a.battle,
                a.decision
            );
            let total_hp: u64 = b
                .sides
                .iter()
                .flat_map(|s| s.party.iter().map(|&sl| s.roster[sl as usize].hp as u64))
                .sum();
            assert!(
                alive(&b, 0) == a.alive0 && alive(&b, 1) == a.alive1 && total_hp == a.total_hp,
                "b{} d{}: state diagnostics mismatch; regenerate anchors",
                a.battle,
                a.decision
            );
            evaluated += 1;
            let e = if metric == "leaf" {
                eval_leaf(&b, &dex, &weights)
            } else {
                eval01(&b, &dex, &weights)
            };
            let (lo, hi) = (a.exact - a.width / 2.0, a.exact + a.width / 2.0);
            let v = (e - hi).max(lo - e).max(0.0);
            if v > eps {
                viols.push((v, a, e));
            }
            if a.width <= 0.05 {
                tight_absdev += (e - a.exact).abs();
                tight_n += 1;
            }
            if list {
                let needs = b.needs_choice();
                let (nc0, nc1) = {
                    let mut probe = b.clone();
                    (
                        probe.legal_choices(&dex, 0).len(),
                        probe.legal_choices(&dex, 1).len(),
                    )
                };
                print!("  needs {needs:?} legal {nc0}/{nc1}");
                let rm = nc2000_bot::eval::race_margin(&b, &dex, &weights);
                let diag = match (b.active_id(0), b.active_id(1)) {
                    (Some(i0), Some(i1)) => {
                        let (q0, q1) = (b.poke(i0), b.poke(i1));
                        format!(
                            "st {:?}/{:?} spe {}/{} ehf {:.2}/{:.2}",
                            q0.status,
                            q1.status,
                            b.get_pokemon_action_speed(&dex, i0),
                            b.get_pokemon_action_speed(&dex, i1),
                            nc2000_bot::eval::best_hit_fraction(&b, &dex, i0, i1, true),
                            nc2000_bot::eval::best_hit_fraction(&b, &dex, i1, i0, true),
                        )
                    }
                    _ => String::new(),
                };
                println!(
                    "  b{} s{} T{}: eval {e:.3} race_margin {rm:+.2} [{diag}] vs [{lo:.3},{hi:.3}]  {}",
                    a.battle, a.side, a.turn, a.desc
                );
                if args.iter().any(|x| x == "--moves") {
                    for s in 0..2usize {
                        if let Some(id) = b.active_id(s) {
                            let p = b.poke(id);
                            let ms: Vec<String> = p
                                .move_slots
                                .iter()
                                .map(|m| {
                                    format!(
                                        "{}(pp{}{})",
                                        dex.moves.key(m.id),
                                        m.pp,
                                        if m.disabled { ",dis" } else { "" }
                                    )
                                })
                                .collect();
                            println!(
                                "      s{s} {}: {}",
                                dex.species.get(p.species).name,
                                ms.join(" ")
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        evaluated,
        anchors.len(),
        "not every anchor was evaluated; refusing partial gate"
    );

    viols.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let total_margin: f64 = viols.iter().map(|(v, _, _)| v).sum();
    println!(
        "\nevaluated {evaluated}: violations {} (eps {eps}), total margin {total_margin:.3}, tight MAE {:.4} (n {tight_n})",
        viols.len(),
        tight_absdev / tight_n.max(1) as f64
    );
    for (v, a, e) in viols.iter().take(10) {
        println!(
            "  margin {v:.3}: eval {e:.3} vs [{:.3},{:.3}]  {}",
            a.exact - a.width / 2.0,
            a.exact + a.width / 2.0,
            a.desc
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_legacy_or_shard_artifacts() {
        let legacy = "battle,side,turn,human,exact\n";
        assert!(parse_artifact(legacy, "old.csv").is_err());
        let shard = r#"{"schema":"nc2000-m17e-exactness-shard-v3"}"#;
        assert!(parse_artifact(shard, "shard.json").is_err());
    }

    #[test]
    fn serde_rejects_duplicate_or_unknown_top_level_fields() {
        let duplicate = r#"{"schema":"a","schema":"b","run":{},"merge":{},"rows":[]}"#;
        let error = serde_json::from_str::<MergedArtifact>(duplicate).unwrap_err();
        assert!(error.to_string().contains("duplicate field"), "{error}");
        let unknown = r#"{"schema":"a","run":{},"merge":{},"rows":[],"extra":1}"#;
        assert!(serde_json::from_str::<MergedArtifact>(unknown).is_err());
    }
}
