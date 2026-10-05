use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use nc2000_bot::bounds::{BoundConfig, BoundSolver, Bounds, ChanceDiagnostics, FixedPolicySolver, RootPayoffs};
use nc2000_bot::player::action_input;
use nc2000_bot::{Agent, MaxDamageAgent, RmConfig, SkuctSearch};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    corpus: String,
    corpus_sha256: String,
    baseline: String,
    baseline_sha256: String,
    reference_ms: u64,
    node_budget: usize,
    work_budget: usize,
    cell_cap: usize,
    tolerance: f64,
    iterations: Vec<u32>,
    search_seeds: usize,
    seed: u64,
    threads: usize,
    limit: Option<usize>,
    interpretation: String,
    #[serde(default)]
    targets: TargetSet,
    #[serde(default)]
    diagnostics: bool,
    #[serde(default)]
    certified_action_pruning: bool,
    fixed_policy_side: Option<usize>,
    fixed_first_action: Option<String>,
    fixed_priority_actions: Option<Vec<String>>,
    #[serde(default)]
    aggregate_lazy_children: bool,
    #[serde(default)]
    chance_fair_period: usize,
    resource_closure_nodes: Option<usize>,
    resource_closure_period: Option<usize>,
    #[serde(default)]
    directed_policy_refinement: bool,
    #[serde(default)]
    target_aware_refinement: bool,
    witness_preparation: Option<WitnessPreparation>,
    descendant_preparation: Option<DescendantPreparation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DescendantPreparation {
    root_state_key: String,
    actions: [String; 2],
    semantic_key: String,
    mass: f64,
    enumeration_cap: usize,
    reference_ms: u64,
    work_budget: usize,
    node_budget: usize,
    tolerance: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessPreparation {
    sides: Vec<usize>,
    reference_ms: u64,
    work_budget: usize,
    node_budget: usize,
    tolerance: f64,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TargetSet {
    #[default]
    AllActions,
    Archived,
}

#[derive(Deserialize)]
struct Decision {
    key: String,
    choices: [Option<usize>; 2],
}

#[derive(Deserialize)]
struct Game {
    schema: String,
    game: usize,
    seed: String,
    teams: [Vec<PokemonSet>; 2],
    decisions: Vec<Decision>,
    samples: Vec<usize>,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bound(b: Bounds) -> Value {
    json!({"lo": b.lo, "hi": b.hi})
}

fn chance_diagnostic(cell: &ChanceDiagnostics, dex: &Dex) -> Value {
    json!({
        "visits":cell.visits,"runs":cell.runs,
        "pending_mass":cell.pending_mass,"pending_scripts":cell.pending_scripts,
        "resolved_edges":cell.resolved_edges,"unique_children":cell.unique_children,
        "child_uncertainty":cell.child_uncertainty,
        "largest_edge_uncertainty":cell.largest_edge_uncertainty,
        "largest_child_uncertainty":cell.largest_child_uncertainty,
        "expanded_children":cell.expanded_children,"max_child_runs":cell.max_child_runs,
        "total_child_runs":cell.total_child_runs,
        "largest_children":cell.largest_children.iter().map(|child| json!({
            "key":child.key,"mass":child.mass,"bounds":bound(child.bounds),
            "uncertainty":child.uncertainty,"runs":child.runs,"visits":child.visits,
            "resource_rank":child.resource_rank.map(|r| json!({"pp_total":r.pp_total,
                "healing_pp":r.healing_pp,"turns_remaining":r.turns_remaining})),
            "state":child.battle.as_ref().map(|b| b.essence(dex)),
            "actions":child.actions.iter().map(|actions| actions.iter().map(|a| a.map(|a|action_input(dex,a))).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "cells":child.cells.iter().copied().map(bound).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

fn assessment(matrix: &RootPayoffs, side: usize, policy: &[f64]) -> Value {
    json!({"policy":policy,"guarantee":bound(matrix.policy_value(side,policy)),
        "loss":bound(matrix.policy_loss(side,policy))})
}

fn archived_policy(record: &Value, names: &[String]) -> Vec<f64> {
    let mut policy = vec![0.0; names.len()];
    for item in record["policy"].as_array().unwrap() {
        let index = names.iter().position(|name| *name == item["action"].as_str().unwrap()).unwrap();
        let probability = item["probability"].as_f64().unwrap();
        assert!(probability.is_finite() && probability >= 0.0);
        policy[index] += probability;
    }
    assert!((policy.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    policy
}

fn reconstruct(dex: &Dex, game: &Game, decision: usize) -> Battle {
    let mut battle = Battle::from_fixture(dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
    for record in &game.decisions[..decision] {
        assert_eq!(battle.state_key128().to_string(), record.key);
        let choices = [0, 1].map(|side| battle.legal_choices(dex, side));
        battle
            .apply_choices(
                dex,
                [0, 1].map(|side| record.choices[side].map(|index| choices[side][index])),
            )
            .unwrap();
    }
    assert_eq!(
        battle.state_key128().to_string(),
        game.decisions[decision].key
    );
    battle
}

fn audit(
    dex: &Dex,
    game: &Game,
    decision: usize,
    archived: &[Value],
    protocol: &Protocol,
) -> Value {
    let mut battle = reconstruct(dex, game, decision);
    let legal = [0, 1].map(|side| battle.legal_choices(dex, side));
    let names = legal.clone().map(|aa| {
        aa.into_iter()
            .map(|a| action_input(dex, a))
            .collect::<Vec<_>>()
    });
    if let Some(side) = protocol.fixed_policy_side {
        assert!(protocol.witness_preparation.is_none());
        assert!(protocol.descendant_preparation.is_none());
        assert!(side < 2 && !legal[side].is_empty());
        let priority_names = protocol.fixed_priority_actions.as_ref().unwrap_or(&names[side]);
        assert_eq!(priority_names.len(), names[side].len());
        assert!(names[side].iter().all(|name| priority_names.iter().filter(|p| *p == name).count() == 1));
        let priority: Vec<_> = priority_names.iter().map(|name|
            legal[side][names[side].iter().position(|n| n == name).unwrap()]).collect();
        let cfg = BoundConfig {
            time_budget: Some(Duration::from_millis(protocol.reference_ms)),
            work_budget: protocol.work_budget,
            node_budget: protocol.node_budget,
            cell_cap: protocol.cell_cap,
            eps: protocol.tolerance,
            certified_action_pruning: protocol.certified_action_pruning,
            aggregate_lazy_children: protocol.aggregate_lazy_children,
            chance_fair_period: protocol.chance_fair_period,
            resource_closure_nodes: protocol.resource_closure_nodes,
            resource_closure_period: protocol.resource_closure_period.unwrap_or(1),
            directed_policy_refinement: protocol.directed_policy_refinement,
            target_aware_refinement: protocol.target_aware_refinement,
            ..BoundConfig::default()
        };
        let mut solver = if let Some(first) = &protocol.fixed_first_action {
            let index = names[side].iter().position(|name| name == first).unwrap();
            FixedPolicySolver::new_with_first_action(dex, cfg, side, priority,
                &battle, legal[side][index])
        } else {
            FixedPolicySolver::new(dex, cfg, side, priority)
        };
        let started = Instant::now();
        let report = solver.solve(&battle);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let graph = solver.graph_diagnostics();
        let matrix = solver.restricted_root_payoffs(&battle).unwrap();
        let stats = solver.stats();
        return json!({
            "schema":"nc2000-fixed-policy-bound-v1",
            "game":game.game,"decision":decision,"state_key":battle.state_key128().to_string(),
            "side":side,"priority_actions":priority_names,
            "fixed_first_action":protocol.fixed_first_action,
            "guaranteed_score":bound(report.guaranteed_score),
            "unrestricted_game_value":bound(report.unrestricted_game_value),
            "restricted_matrix":{
                "actions":matrix.actions.map(|actions|actions.into_iter().map(|a|a.map(|a|action_input(dex,a))).collect::<Vec<_>>()),
                "cells":matrix.cells.iter().map(|cell|bound(*cell)).collect::<Vec<_>>(),
                "value":bound(matrix.value),
            },
            "root_chance":solver.root_chance_diagnostics(&battle).unwrap().iter().map(|c| chance_diagnostic(c, dex)).collect::<Vec<_>>(),
            "trial_depth_counts":stats.trial_depth_counts,"depth_limit_trials":stats.depth_limit_trials,
            "eager_failures":stats.eager_failures,"lazy_edge_merges":stats.lazy_edge_merges,
            "reference":{"stop":format!("{:?}",report.stop),"runs":report.runs,
                "elapsed_ms":elapsed_ms,"nodes":solver.node_count()},
            "graph":{"unexpanded_nodes":graph.unexpanded_nodes,"initialized_nodes":graph.initialized_nodes,
                "stored_battles":graph.stored_battles,"resolved_edges":graph.resolved_edges,
                "unique_child_edges":graph.unique_child_edges},
        });
    }
    assert!(protocol.fixed_first_action.is_none());
    assert!(protocol.fixed_priority_actions.is_none());
    let mut targets = Vec::new();
    let archived: [Option<&Value>; 2] = [0, 1].map(|side| {
        if legal[side].is_empty() {
            return None;
        }
        let record = archived.iter().find(|r|
            r["game"] == game.game && r["decision"] == decision && r["side"] == side
        ).expect("every actionable endpoint must have an archived baseline");
        assert_eq!(record["state_key"], battle.state_key128().to_string());
        Some(record)
    });
    for side in 0..2 {
        if legal[side].is_empty() {
            continue;
        }
        if matches!(protocol.targets, TargetSet::Archived) {
            targets.push((side, archived_policy(archived[side].unwrap(), &names[side])));
            continue;
        }
        for action in 0..legal[side].len() {
            let mut p = vec![0.0; legal[side].len()];
            p[action] = 1.0;
            targets.push((side, p));
        }
        targets.push((
            side,
            vec![1.0 / legal[side].len() as f64; legal[side].len()],
        ));
    }
    let cfg = BoundConfig {
            time_budget: Some(Duration::from_millis(protocol.reference_ms)),
            work_budget: protocol.work_budget,
            node_budget: protocol.node_budget,
            cell_cap: protocol.cell_cap,
            certified_action_pruning: protocol.certified_action_pruning,
            aggregate_lazy_children: protocol.aggregate_lazy_children,
            chance_fair_period: protocol.chance_fair_period,
            resource_closure_nodes: protocol.resource_closure_nodes,
            resource_closure_period: protocol.resource_closure_period.unwrap_or(1),
            directed_policy_refinement: protocol.directed_policy_refinement,
            target_aware_refinement: protocol.target_aware_refinement,
            ..BoundConfig::default()
        };
    let mut solver = BoundSolver::new(dex, cfg.clone());
    let mut preparation = Vec::new();
    let mut preparation_runs = 0;
    let mut preparation_ms = 0.0;
    let mut imported_certificates = 0;
    let mut descendant_preparation = Vec::new();
    if let Some(prep) = &protocol.descendant_preparation {
        assert!(protocol.witness_preparation.is_none());
        assert_eq!(battle.state_key128().to_string(), prep.root_state_key);
        assert!(!battle.damage_bookkeeping_observable(dex));
        assert!(prep.reference_ms > 0 && prep.work_budget > 0 && prep.node_budget > 0);
        assert!(prep.tolerance.is_finite() && prep.tolerance > 0.0);
        let started = Instant::now();
        let choices = [0, 1].map(|side| Some(legal[side][names[side].iter()
            .position(|name| *name == prep.actions[side]).unwrap()]));
        let step = nc2000_engine::battle::enumerate::enumerate_step(dex, &battle, choices,
            prep.enumeration_cap).unwrap();
        assert!((step.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-10);
        let key: u128 = prep.semantic_key.parse().unwrap();
        let matching: Vec<_> = step.leaves.iter().filter(|leaf|
            leaf.battle.state_key128_without_damage_bookkeeping() == key).collect();
        let mass: f64 = matching.iter().map(|leaf| leaf.prob).sum();
        assert!(!matching.is_empty() && (mass - prep.mass).abs() < 1e-12);
        let child = &matching[0].battle;
        assert!(child.outcome().is_none() && !child.damage_bookkeeping_observable(dex));
        let mut dedicated = BoundSolver::new(dex, BoundConfig {
            time_budget: Some(Duration::from_millis(prep.reference_ms)),
            work_budget: prep.work_budget,
            node_budget: prep.node_budget,
            eps: prep.tolerance,
            ..cfg.clone()
        });
        let report = dedicated.solve(child, None);
        let matrix = dedicated.root_payoffs(child).unwrap();
        let nodes = dedicated.node_count();
        let root_chance: Vec<_> = dedicated.root_chance_diagnostics(child).unwrap().iter()
            .map(|c| chance_diagnostic(c, dex)).collect();
        let certificates = dedicated.into_value_certificates();
        let certificate_count = certificates.len();
        imported_certificates = solver.import_value_certificates(certificates);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        preparation_runs += step.runs + report.runs;
        preparation_ms += elapsed_ms;
        descendant_preparation.push(json!({"semantic_key":prep.semantic_key,
            "actions":prep.actions,"mass":mass,"matching_leaves":matching.len(),
            "state":child.essence(dex),"enumeration_runs":step.runs,
            "value":bound(report.bounds),"stop":format!("{:?}",report.stop),
            "runs":report.runs,"nodes":nodes,"elapsed_ms":elapsed_ms,
            "certificates":certificate_count,"root_chance":root_chance,
            "matrix":{"actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|action_input(dex,a))).collect::<Vec<_>>()),
                "cells":matrix.cells.iter().copied().map(bound).collect::<Vec<_>>(),"value":bound(matrix.value)}}));
        eprintln!("descendant={} mass={mass} value=[{},{}] runs={} nodes={nodes}",
            prep.semantic_key, report.bounds.lo, report.bounds.hi, report.runs);
    }
    if let Some(prep) = &protocol.witness_preparation {
        assert!(prep.reference_ms > 0 && prep.work_budget > 0 && prep.node_budget > 0);
        assert!(prep.tolerance.is_finite() && prep.tolerance > 0.0);
        for &side in &prep.sides {
            assert!(side < 2);
            let started = Instant::now();
            let mut witness = FixedPolicySolver::new(dex, BoundConfig {
                time_budget: Some(Duration::from_millis(prep.reference_ms)),
                work_budget: prep.work_budget,
                node_budget: prep.node_budget,
                eps: prep.tolerance,
                ..cfg.clone()
            }, side, legal[side].clone());
            let report = witness.solve(&battle);
            let nodes = witness.node_count();
            let certificates = witness.into_value_certificates();
            let certificate_count = certificates.len();
            imported_certificates = solver.import_value_certificates(certificates);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            preparation_runs += report.runs;
            preparation_ms += elapsed_ms;
            preparation.push(json!({"side":side,"priority_actions":names[side],
                "guaranteed_score":bound(report.guaranteed_score),
                "unrestricted_game_value":bound(report.unrestricted_game_value),
                "stop":format!("{:?}",report.stop),"runs":report.runs,"nodes":nodes,
                "elapsed_ms":elapsed_ms,"certificates":certificate_count}));
        }
    }
    let started = Instant::now();
    let solved = solver.solve_policy_losses(&battle, &targets, protocol.tolerance);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let matrix = solver.root_payoffs(&battle).unwrap();
    for side in 0..2 {
        let expected: Vec<_> = if legal[side].is_empty() {
            vec![None]
        } else {
            legal[side].iter().copied().map(Some).collect()
        };
        assert_eq!(matrix.actions[side], expected);
    }
    let mut endpoints = Vec::new();
    let mut discriminating = false;
    for side in 0..2 {
        if legal[side].is_empty() {
            continue;
        }
        let archived = archived[side].unwrap();
        let p = archived_policy(archived, &names[side]);
        let pure: Vec<_> = (0..p.len())
            .map(|a| {
                let mut policy = vec![0.0; p.len()];
                policy[a] = 1.0;
                assessment(&matrix, side, &policy)
            })
            .collect();
        let all_good = pure
            .iter()
            .all(|r| r["loss"]["hi"].as_f64().unwrap() <= protocol.tolerance);
        let has_good = pure
            .iter()
            .any(|r| r["loss"]["hi"].as_f64().unwrap() <= protocol.tolerance);
        let has_bad = pure
            .iter()
            .any(|r| r["loss"]["lo"].as_f64().unwrap() > protocol.tolerance);
        discriminating |= has_good && has_bad;
        let greedy = MaxDamageAgent::conformant().choose(&battle, dex, side, &legal[side]);
        let greedy_index = legal[side].iter().position(|&a| a == greedy).unwrap();
        let oracle = solved.strategies.as_ref().map(|ss| {
            let p: Vec<_> = matrix.actions[side]
                .iter()
                .map(|a| {
                    ss[side]
                        .actions
                        .iter()
                        .find(|(b, _)| a == b)
                        .map_or(0.0, |(_, mass)| *mass)
                })
                .collect();
            assessment(&matrix, side, &p)
        });
        endpoints.push(
            json!({"side":side,"actions":names[side],"forced":p.len()==1,
            "all_actions_good":all_good,"has_good_action":has_good,"has_bad_action":has_bad,
            "archived":assessment(&matrix,side,&p),"archived_choice":archived["choice"],
            "archived_iterations":archived["iterations"],"pure_actions":pure,
            "uniform":assessment(&matrix,side,&vec![1.0/p.len() as f64;p.len()]),
            "max_damage":pure[greedy_index],"oracle":oracle}),
        );
    }
    let mut search_records = Vec::new();
    if discriminating {
        for seed_index in 0..protocol.search_seeds {
            let seed = protocol.seed
                ^ (game.game as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ (decision as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                ^ (seed_index as u64).wrapping_mul(0x94d049bb133111eb);
            let mut search = SkuctSearch::new(
                &battle,
                dex,
                RmConfig {
                    c: 0.4,
                    ..Default::default()
                },
                seed,
            );
            let started = Instant::now();
            for &iterations in &protocol.iterations {
                search.step(dex, iterations - search.iterations());
                let choices: [Option<String>; 2] =
                    [0, 1].map(|s| search.best(s).map(|a| action_input(dex, a)));
                search_records.push(json!({"seed_index":seed_index,"seed":seed,"iterations":iterations,
                    "choices":choices,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"nodes":search.node_count()}));
            }
        }
        for endpoint in &mut endpoints {
            let side = endpoint["side"].as_u64().unwrap() as usize;
            let rows: Vec<_> = protocol
                .iterations
                .iter()
                .map(|&iterations| {
                    let mut p = vec![0.0; legal[side].len()];
                    for row in search_records
                        .iter()
                        .filter(|r| r["iterations"] == iterations)
                    {
                        let choice = row["choices"][side].as_str().unwrap();
                        p[names[side].iter().position(|a| a == choice).unwrap()] +=
                            1.0 / protocol.search_seeds as f64;
                    }
                    json!({"iterations":iterations,"empirical_mixture":assessment(&matrix,side,&p)})
                })
                .collect();
            endpoint["budget_ladder"] = json!(rows);
        }
    }
    let diagnostics = protocol.diagnostics.then(|| {
        let stats = &solver.stats;
        let graph = solver.graph_diagnostics();
        json!({
            "graph": {
                "unexpanded_nodes": graph.unexpanded_nodes,
                "initialized_nodes": graph.initialized_nodes,
                "stored_battles": graph.stored_battles,
                "resolved_edges": graph.resolved_edges,
                "unique_child_edges": graph.unique_child_edges,
            },
            "trial_depth_counts": stats.trial_depth_counts,
            "depth_limit_trials": stats.depth_limit_trials,
            "trials": stats.trials,
            "sufficient_reply_skips": stats.sufficient_reply_skips,
            "target_filtered_trials": stats.target_filtered_trials,
            "lower_directed_trials": stats.lower_directed_trials,
            "upper_directed_trials": stats.upper_directed_trials,
            "resource_closure_visits": stats.resource_closure_visits,
            "first_resource_closure_nodes": stats.first_resource_closure_nodes,
            "two_sided_roots": stats.two_sided_roots,
            "two_sided_invalidations": stats.two_sided_invalidations,
            "eager_runs": stats.eager_runs,
            "eager_failures": stats.eager_failures,
            "reused_nodes": stats.reused_nodes,
            "dominated_rows": stats.dominated_rows,
            "dominated_cols": stats.dominated_cols,
            "avoided_cells": stats.avoided_cells,
            "lazy_edge_merges": stats.lazy_edge_merges,
            "root_chance": solver.root_chance_diagnostics(&battle).unwrap().iter().map(|c| chance_diagnostic(c, dex)).collect::<Vec<_>>(),
        })
    });
    json!({"game":game.game,"decision":decision,"state_key":battle.state_key128().to_string(),
        "diagnostics":diagnostics,
        "preparation":preparation,"total_runs":preparation_runs+solved.runs,
        "descendant_preparation":descendant_preparation,
        "total_elapsed_ms":preparation_ms+elapsed_ms,
        "turn":battle.turn,"alive":([0,1].map(|side|battle.sides[side].party.iter().filter(|&&i| !battle.sides[side].roster[i as usize].fainted).count())),
        "reference":{"value":bound(matrix.value),"cells":matrix.cells.iter().map(|b|bound(*b)).collect::<Vec<_>>(),
            "actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|action_input(dex,a))).collect::<Vec<_>>()),
            "stop":format!("{:?}",solved.stop),"runs":solved.runs,"elapsed_ms":elapsed_ms,
            "imported_certificates":imported_certificates,"certificate_hits":solver.stats.certificate_hits,
            "nodes":solver.node_count(),"closed":solver.closed_count()},
        "discriminating":discriminating,"endpoints":endpoints,"search_records":search_records})
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "endgame_policy_audit PROTOCOL.json OUTPUT_DIRECTORY"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-policy-gap-v1");
    assert!(protocol.threads > 0 && protocol.search_seeds > 0 && protocol.reference_ms > 0);
    assert!(protocol.iterations.windows(2).all(|w| w[0] < w[1]));
    let corpus = std::fs::read(&protocol.corpus).unwrap();
    let baseline = std::fs::read(&protocol.baseline).unwrap();
    assert_eq!(sha(&corpus), protocol.corpus_sha256);
    assert_eq!(sha(&baseline), protocol.baseline_sha256);
    let games: Vec<Game> = BufReader::new(corpus.as_slice())
        .lines()
        .map(|s| serde_json::from_str(&s.unwrap()).unwrap())
        .collect();
    let archived: Vec<Value> = serde_json::from_slice(&baseline).unwrap();
    let mut jobs = Vec::new();
    for (i, game) in games.iter().enumerate() {
        assert_eq!(game.schema, "nc2000-selfplay-endgames-v1");
        for &decision in &game.samples {
            jobs.push((i, decision));
        }
    }
    if let Some(limit) = protocol.limit {
        jobs.truncate(limit);
    }
    let directory = Path::new(&args[2]);
    std::fs::create_dir(directory).unwrap();
    let git = |args: &[&str]| {
        String::from_utf8(
            std::process::Command::new("git")
                .args(args)
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string()
    };
    let sources = [
        "crates/bot/src/bounds.rs",
        "crates/bot/examples/endgame_policy_audit.rs",
    ];
    let manifest = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),
        "protocol_sha256":sha(&bytes),"revision":git(&["rev-parse","HEAD"]),"dirty":git(&["status","--porcelain"]),
        "binary_sha256":sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
        "source_hashes":sources.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "expected_positions":jobs.len(),"source_games":games.len(),
        "interpretation":protocol.interpretation});
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let dex = conformance::load_dex();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol.threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(game, decision)) = jobs.get(index) else {
                    break;
                };
                let game = &games[game];
                let row = audit(&dex, game, decision, &archived, &protocol);
                std::fs::write(
                    directory.join(format!("game-{:03}-d{decision:04}.json", game.game)),
                    serde_json::to_vec_pretty(&row).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "{}/{} game={} decision={} stop={} reference_ms={:.0} ladder={}",
                    index + 1,
                    jobs.len(),
                    game.game,
                    decision,
                    row["reference"]["stop"],
                    row["reference"]["elapsed_ms"].as_f64().unwrap(),
                    row["discriminating"]
                );
            });
        }
    });
}
