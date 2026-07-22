//! M17c anchor gate — zero-noise eval regression against the certified
//! endgame anchors mined by `endgame_exactness_corpus`.
//!
//! Reads the anchor CSV (battle, side, turn, exact, width, …), re-derives
//! each position deterministically through `bot::corpus` (same seed, same
//! fabrication), evaluates a candidate `EvalWeights` config, and reports
//! proven bracket violations (eval outside [exact−w/2, exact+w/2] by more
//! than eps). No solving, no playouts — seconds, bit-reproducible, so a
//! candidate eval change can be gated on "violation count × margin must
//! drop" before the statistical gates (`eval_calibration --ab`,
//! `eval_ab_duel`) run.
//!
//! Anchor staleness guard: each reconstructed position's alive counts and
//! total HP are checked against the CSV row; a mismatch (importer or
//! fabrication drift since the anchors were solved) skips the row loudly —
//! regenerate the anchors when that starts happening.
//!
//! Usage: anchor_gate [--csv tmp/eec-all.csv] [--corpus tmp/corpus-spectator]
//!                    [--race F] [--leaf-alpha F] [--metric raw|leaf]
//!                    [--eps 0.02] [--list]

use std::collections::HashMap;

use nc2000_bot::corpus::{corpus_files, load_battle, load_sources, reconstruct};
use nc2000_bot::eval::{eval01, eval_leaf, EvalWeights};
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

struct Anchor {
    battle: usize,
    side: usize,
    turn: u16,
    exact: f64,
    width: f64,
    alive0: usize,
    alive1: usize,
    total_hp: u64,
    desc: String,
}

fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side].party.iter().filter(|&&s| !b.sides[side].roster[s as usize].fainted).count()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let csv_path = arg_s(&args, "--csv", "tmp/eec-all.csv");
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
    assert!(metric == "raw" || metric == "leaf", "--metric must be raw or leaf");
    let list = args.iter().any(|a| a == "--list");

    let weights = EvalWeights { race, leaf_alpha, ..EvalWeights::default() };

    // ---- parse anchors (CSV fields quoted only in the trailing desc)
    let text = std::fs::read_to_string(&csv_path).unwrap_or_else(|e| panic!("{csv_path}: {e}"));
    let mut anchors: Vec<Anchor> = Vec::new();
    for line in text.lines().skip(1) {
        let p: Vec<&str> = line.splitn(12, ',').collect();
        if p.len() < 12 {
            continue;
        }
        anchors.push(Anchor {
            battle: p[0].parse().unwrap(),
            side: p[1].parse().unwrap(),
            turn: p[2].parse().unwrap(),
            exact: p[4].parse().unwrap(),
            width: p[5].parse().unwrap(),
            alive0: p[8].parse().unwrap(),
            alive1: p[9].parse().unwrap(),
            total_hp: p[10].parse().unwrap(),
            desc: p[11].trim_matches('"').to_string(),
        });
    }
    println!("anchors: {} from {csv_path}; race {race}; metric {metric}; leaf alpha {leaf_alpha}", anchors.len());

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let files = corpus_files(&root.join(&corpus));

    let mut by_battle: HashMap<usize, Vec<&Anchor>> = HashMap::new();
    for a in &anchors {
        by_battle.entry(a.battle).or_default().push(a);
    }

    let mut evaluated = 0usize;
    let mut stale = 0usize;
    let mut viols: Vec<(f64, &Anchor, f64)> = Vec::new();
    let mut tight_absdev = 0.0f64;
    let mut tight_n = 0usize;

    let mut battles: Vec<usize> = by_battle.keys().cloned().collect();
    battles.sort();
    for bi in battles {
        let Some(path) = files.get(bi) else { continue };
        let cb = load_battle(path);
        for a in &by_battle[&bi] {
            let Some(d) = cb.decisions.iter().find(|d| d.side == a.side && d.turn == a.turn)
            else {
                stale += 1;
                println!("STALE (no decision): b{} s{} T{}", a.battle, a.side, a.turn);
                continue;
            };
            let Some(b) = reconstruct(&dex, &src, &pool_path, &cb.lines, &cb.evidence, d, 1) else {
                stale += 1;
                println!("STALE (reconstruct failed): b{} s{} T{}", a.battle, a.side, a.turn);
                continue;
            };
            let total_hp: u64 = b
                .sides
                .iter()
                .flat_map(|s| s.party.iter().map(|&sl| s.roster[sl as usize].hp as u64))
                .sum();
            if alive(&b, 0) != a.alive0 || alive(&b, 1) != a.alive1 || total_hp != a.total_hp {
                stale += 1;
                println!(
                    "STALE (state drift): b{} s{} T{} — regenerate anchors",
                    a.battle, a.side, a.turn
                );
                continue;
            }
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
                            println!("      s{s} {}: {}", dex.species.get(p.species).name, ms.join(" "));
                        }
                    }
                }
            }
        }
    }

    viols.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let total_margin: f64 = viols.iter().map(|(v, _, _)| v).sum();
    println!(
        "\nevaluated {evaluated} (stale {stale}): violations {} (eps {eps}), total margin {total_margin:.3}, tight MAE {:.4} (n {tight_n})",
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
