#[path = "counter_mix_probe/bond.rs"]
mod bond;
use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_bot::exact::solve_matrix_full;
use nc2000_bot::root_policy::{analyze_bayesian_root, RootCell, RootType};
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig, SkuctSearch};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{validate_team, Learnsets};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

#[derive(Clone, Default, serde::Serialize)]
struct ProbeTrace {
    samples: u32,
    terminal: u32,
    terminal_reward: f64,
    expanded_leaf: u32,
    capped_leaf: u32,
}

#[derive(Clone, Default, serde::Serialize)]
struct CutoffCell {
    requests: u32,
    replacements: u32,
    runs: usize,
    max_error: f64,
}

fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn set(species: &str, moves: &[&str], trained: bool) -> PokemonSet {
    let ev = if trained { 255 } else { 4 };
    serde_json::from_value(
        json!({"name":species,"species":species,"ability":"", "item":"",
        "level":50,"happiness":255,"moves":moves,
        "evs":{"hp":ev,"atk":ev,"def":ev,"spa":ev,"spd":ev,"spe":ev},
        "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}}),
    )
    .unwrap()
}
fn step(
    dex: &Dex,
    b: &mut Battle,
    obs: &mut [Observer; 2],
    beliefs: &mut [Belief; 2],
    actions: [Option<SearchChoice>; 2],
    records: &mut Vec<Value>,
) {
    let indices = [0, 1].map(|s| {
        let legal = b.legal_choices(dex, s);
        match actions[s] {
            Some(action) => Some(legal.iter().position(|a| *a == action).unwrap_or_else(|| {
                panic!(
                    "illegal prefix side={s} turn={} wanted={action:?} legal={legal:?} state={}",
                    b.turn,
                    b.essence(dex)
                )
            })),
            None => {
                assert!(legal.is_empty());
                None
            }
        }
    });
    records.push(json!({"key":b.state_key128().to_string(),"choices":indices}));
    b.apply_choices(dex, actions).unwrap();
    for s in 0..2 {
        obs[s].observe(b, dex);
        beliefs[s].sync(dex, &obs[s]);
    }
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (6..=11).contains(&args.len()),
        "output seed search-seeds iterations search-seed-base [family] [probe-iterations] [adaptive|typed|fresh|fresh-trace|fresh-profile|fresh-cutoff|fresh-cutoff-focus] [hp-buckets] [ucb|exp3|exp3-mean]"
    );
    let family = args.get(6).map(String::as_str).unwrap_or("counter");
    assert!(matches!(family, "counter" | "bond"));
    let out = std::path::Path::new(&args[1]);
    assert!(!out.exists());
    let seed = &args[2];
    let count: usize = args[3].parse().unwrap();
    let iterations: u32 = args[4].parse().unwrap();
    let seed_base: u64 = args[5].parse().unwrap();
    let probe_iterations: u32 = args.get(7).map(|s| s.parse().unwrap()).unwrap_or(0);
    assert!(probe_iterations < iterations);
    let typed = args.get(8).map(String::as_str) == Some("typed");
    let trace = args.get(8).map(String::as_str) == Some("fresh-trace");
    let cutoff_focus = args.get(8).map(String::as_str) == Some("fresh-cutoff-focus");
    assert!(!cutoff_focus || family == "bond");
    let cutoff = cutoff_focus || args.get(8).map(String::as_str) == Some("fresh-cutoff");
    let profile = cutoff || args.get(8).map(String::as_str) == Some("fresh-profile");
    let fresh = trace || profile || args.get(8).map(String::as_str) == Some("fresh");
    let adaptive = typed || args.get(8).map(String::as_str) == Some("adaptive");
    let hp_buckets: i64 = args.get(9).map(|s| s.parse().unwrap()).unwrap_or(16);
    let continuation_exp3_ix = match args.get(10).map(String::as_str).unwrap_or("ucb") {
        "ucb" => false,
        "exp3" | "exp3-mean" => true,
        mode => panic!("unknown continuation mode {mode}"),
    };
    let continuation_exp3_mean = args.get(10).is_some_and(|mode| mode == "exp3-mean");
    let dex = conformance::load_dex();
    let teams: [Vec<PokemonSet>; 2] = if family == "bond" {
        bond::teams()
    } else {
        [0, 1].map(|side| {
            vec![
                set("Electrode", &["Explosion"], true),
                set("Raikou", &["Thunderbolt"], false),
                if side == 0 {
                    set("Starmie", &["Return", "Surf"], true)
                } else {
                    set("Wobbuffet", &["Counter", "Mirror Coat", "Safeguard"], true)
                },
                set("Snorlax", &["Body Slam"], true),
                set("Marowak", &["Earthquake"], true),
                set("Heracross", &["Megahorn"], true),
            ]
        })
    };
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let validations = teams
        .each_ref()
        .map(|team| validate_team(&dex, &learnsets, &serde_json::to_string(team).unwrap()));
    assert!(
        validations.iter().all(|v| v["ok"] == true),
        "{validations:?}"
    );
    let mut b = Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(true);
    let mut obs = [0, 1].map(|s| Observer::new(&b, s));
    let mut beliefs = [0, 1].map(|s| Belief::pinned_from_battle(&b, &obs[s]));
    let mut records = Vec::new();
    let mv = |name: &str| SearchChoice::Move(dex.moves.id(name).unwrap());
    if family == "bond" {
        bond::prefix(&dex, &mut b, &mut obs, &mut beliefs, &mut records);
    } else {
        let prefix = [
            [SearchChoice::Team([1, 2, 3]), SearchChoice::Team([2, 1, 3])],
            [mv("explosion"), mv("thunderbolt")],
            [SearchChoice::Switch(2), SearchChoice::Switch(2)],
            [mv("thunderbolt"), mv("explosion")],
            [SearchChoice::Switch(3), SearchChoice::Switch(3)],
            [mv("return"), mv("mirrorcoat")],
            [mv("surf"), mv("safeguard")],
            [mv("return"), mv("counter")],
        ];
        for actions in prefix {
            step(
                &dex,
                &mut b,
                &mut obs,
                &mut beliefs,
                actions.map(Some),
                &mut records,
            );
        }
    }
    assert!(b.outcome().is_none());
    assert_eq!(
        [0, 1].map(|s| b.sides[s].pokemon_left),
        if family == "bond" { [1, 2] } else { [1, 1] }
    );
    let decision = records.len();
    records.push(json!({"key":b.state_key128().to_string(),"choices":[0,0]}));
    let corpus = json!({"game":3000,"seed":seed,"teams":teams,"decisions":records,"samples":[decision],"outcome":null,"prefix_only":true});
    std::fs::create_dir(out).unwrap();
    std::fs::write(
        out.join("corpus.jsonl"),
        serde_json::to_string(&corpus).unwrap() + "\n",
    )
    .unwrap();
    eprintln!("root {}", b.essence(&dex));
    let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
    let mut targets = Vec::new();
    for s in 0..2 {
        for i in 0..actions[s].len() {
            let mut p = vec![0.; actions[s].len()];
            p[i] = 1.;
            targets.push((s, p));
        }
    }
    let mut solver = BoundSolver::new(
        &dex,
        BoundConfig {
            time_budget: Some(Duration::from_secs(60)),
            work_budget: 10_000_000,
            eps: 1e-10,
            certified_action_pruning: false,
            fold_closed_nodes: false,
            ..Default::default()
        },
    );
    let started = Instant::now();
    solver.solve_policy_losses(&b, &targets, 1e-8);
    let payoffs = solver.root_payoffs(&b).unwrap();
    let lower: Vec<f64> = payoffs.cells.iter().map(|c| c.lo).collect();
    let upper: Vec<f64> = payoffs.cells.iter().map(|c| c.hi).collect();
    let low = solve_matrix_full(&lower, actions[0].len(), actions[1].len());
    let high = solve_matrix_full(&upper, actions[0].len(), actions[1].len());
    let proof = json!({"value":[payoffs.value.lo,payoffs.value.hi],"cells":payoffs.cells.iter().map(|c|[c.lo,c.hi]).collect::<Vec<_>>(),
        "lower_mix":[low.x,low.y],"upper_mix":[high.x,high.y],"elapsed_seconds":started.elapsed().as_secs_f64(),
        "runs":solver.stats.runs});
    eprintln!("proof {proof}");
    let mut rows = Vec::new();
    for index in 0..count {
        let search_seed = seed_base ^ (index as u64).wrapping_mul(0x94d049bb133111eb);
        let mut full = SkuctSearch::new(
            &b,
            &dex,
            RmConfig {
                c: 0.4,
                hp_buckets,
                continuation_exp3_ix,
                continuation_exp3_mean,
                ..Default::default()
            },
            search_seed,
        );
        full.step(&dex, iterations);
        for s in 0..2 {
            rows.push(json!({"mode":"full","side":s,"seed_index":index,"seed":search_seed,"iterations":iterations,
                "choice":full.best(s).map(|a|a.to_input(&dex)),"visits":full.visits(s),"means":full.means(s)}));
            let search_started = Instant::now();
            let mut pinned = BlindSearch::new(
                &b,
                &dex,
                RmConfig {
                    c: 0.4,
                    hp_buckets,
                    continuation_exp3_ix,
                    continuation_exp3_mean,
                    ..Default::default()
                },
                s,
                search_seed,
            );
            let mut fresh_evidence = None;
            let probes_used = if fresh {
                pinned.step(&dex, &beliefs[s], &obs[s], iterations - probe_iterations);
                let (eligible, warm_types) = pinned.root_policy_evidence();
                assert_eq!(warm_types.len(), 1, "fresh fixture probes require one known type");
                let warm_joint = pinned.root_matrix().into_iter().map(|(own, reply, samples, mean)|
                    json!({"own":own,"reply":reply.to_input(&dex),"samples":samples,"mean":mean})).collect::<Vec<_>>();
                let warm_visits = pinned.visits().to_vec();
                let replies = actions[1-s].len();
                let schedule: Vec<_> = (0..actions[s].len()).filter(|&own| eligible[own])
                    .flat_map(|own| (0..replies).map(move |reply| (own, reply))).collect();
                assert!(!schedule.is_empty() && probe_iterations > 0);
                let mut cells = vec![(0u32, 0.); actions[s].len()*replies];
                let mut traces = vec![ProbeTrace::default(); cells.len()];
                let mut cutoff_cells = vec![CutoffCell::default(); cells.len()];
                let mut continuation = cutoff.then(|| BoundSolver::new(&dex,BoundConfig {
                    work_budget:1000,node_budget:12000,eps:0.005,..Default::default()
                }));
                let mut solver_ms = 0.;
                for k in 0..probe_iterations {
                    let (own, reply) = schedule[k as usize % schedule.len()];
                    let r = if let Some(solver) = continuation.as_mut() {
                        let stats = &mut cutoff_cells[own*replies+reply];
                        pinned.step_forced_joint_with_root_child_value(&dex,&beliefs[s],&obs[s],own,actions[1-s][reply],
                            &mut |child| {
                                if cutoff_focus {
                                    let pair = if s == 0 { [actions[s][own],actions[1-s][reply]] }
                                        else { [actions[1-s][reply],actions[s][own]] };
                                    if pair != [SearchChoice::Move(dex.moves.id("curse").unwrap()),
                                        SearchChoice::Move(dex.moves.id("destinybond").unwrap())] { return None; }
                                }
                                stats.requests += 1;
                                let started = Instant::now();
                                let mut bounds = solver.peek(child);
                                if !bounds.is_some_and(|b| b.width()<=0.005) && solver.stats.runs<40000 {
                                    solver.cfg.work_budget = (40000-solver.stats.runs).min(1000);
                                    let report = solver.solve(child,None);
                                    stats.runs += report.runs;
                                    bounds = Some(report.bounds);
                                }
                                solver_ms += started.elapsed().as_secs_f64()*1000.;
                                bounds.filter(|b| b.width()<=0.005).map(|b| {
                                    stats.replacements += 1;
                                    stats.max_error = stats.max_error.max(b.width()*0.5);
                                    b.mid()
                                })
                            })
                    } else if trace {
                        let stats = &mut traces[own*replies+reply];
                        pinned.step_forced_joint_observed(&dex, &beliefs[s], &obs[s], own, actions[1-s][reply],
                            &mut |event| match event {
                                nc2000_bot::smmcts::SearchTrace::Leaf { rollout, .. } => {
                                    if rollout { stats.expanded_leaf += 1; } else { stats.capped_leaf += 1; }
                                }
                                nc2000_bot::smmcts::SearchTrace::Result { battle, reward0 } => {
                                    stats.samples += 1;
                                    if battle.outcome().is_some() {
                                        stats.terminal += 1;
                                        stats.terminal_reward += if s==0 { reward0 } else { 1.-reward0 };
                                    }
                                }
                                _ => {}
                            })
                    } else { pinned.step_forced_joint(&dex, &beliefs[s], &obs[s], own, actions[1-s][reply]) };
                    let cell = &mut cells[own*replies+reply];
                    cell.0 += 1;
                    cell.1 += if s == 0 { r } else { 1.-r };
                }
                assert_eq!(pinned.root_policy_evidence().1.len(), 1);
                assert!(pinned.root_replies().iter().all(|&(_, n)| n == iterations));
                let phase = [RootType { samples: probe_iterations, replies,
                    cells: cells.into_iter().enumerate().filter(|(_, (n, _))| *n > 0)
                        .map(|(i, (samples, sum))| RootCell { own:i/replies, reply:i%replies,
                            samples, mean:sum/f64::from(samples) }).collect() }];
                let (_, cumulative) = pinned.root_policy_evidence();
                fresh_evidence = Some(json!({"warm_joint":warm_joint,"warm_visits":warm_visits,
                    "phase_types":phase,"eligible":eligible,
                    "phase_analysis":analyze_bayesian_root(&eligible,probe_iterations,&phase),
                    "cumulative_analysis":analyze_bayesian_root(&eligible,iterations,&cumulative)}));
                if trace { fresh_evidence.as_mut().unwrap()["probe_trace"] = json!(traces); }
                if profile {
                    fresh_evidence.as_mut().unwrap()["profile"] = json!({
                        "milliseconds":search_started.elapsed().as_secs_f64()*1000.,
                        "solver_ms":solver_ms,"solver_runs":continuation.as_ref().map_or(0,|s|s.stats.runs),
                        "solver_nodes":continuation.as_ref().map_or(0,|s|s.node_count()+s.closed_count()),
                        "cells":cutoff_cells});
                }
                probe_iterations
            } else if typed {
                pinned.finish_with_typed_mixed_probes(&dex, &beliefs[s], &obs[s], iterations, probe_iterations)
            } else if adaptive {
                pinned.finish_with_mixed_probes(
                    &dex,
                    &beliefs[s],
                    &obs[s],
                    iterations,
                    probe_iterations,
                )
            } else {
                pinned.step(&dex, &beliefs[s], &obs[s], iterations - probe_iterations);
                pinned.step_joint_probes(&dex, &beliefs[s], &obs[s], probe_iterations);
                probe_iterations
            };
            let mut play_rng = nc2000_bot::rng::SplitMix64::new(
                search_seed ^ (s as u64 + 1).wrapping_mul(0xa24baed4963ee407),
            );
            let play_draw = play_rng.next_f64();
            let mixed = pinned.mixed_policy();
            let mut row = json!({"mode":"pinned","side":s,"seed_index":index,"seed":search_seed,"iterations":iterations,
                "choice":pinned.best().map(|a|a.to_input(&dex)),"visits":pinned.visits(),"joint_probe_visits":pinned.joint_probe_visits(),"means":pinned.means(),
                "dominated":pinned.dominated(),"play_draw":play_draw,"probes_used":probes_used,
                "candidate_choice":pinned.sample_mixed(play_draw).map(|a|a.to_input(&dex)),
                "candidate_policy":mixed.as_ref().map(|m|m.policy.clone()),
                "candidate_estimated_advantage":mixed.map(|m|m.estimated_advantage),
                "joint":pinned.root_matrix().into_iter().map(|(own,reply,samples,mean)|json!({"own":own,"reply":reply.to_input(&dex),"samples":samples,"mean":mean})).collect::<Vec<_>>(),
                "reply_availability":pinned.root_replies().iter().map(|&(reply,samples)|json!({"reply":reply.to_input(&dex),"samples":samples})).collect::<Vec<_>>() });
            if let Some(evidence) = fresh_evidence { row["fresh_evidence"] = evidence; }
            rows.push(row);
        }
        eprintln!("seed {index} complete");
    }
    let paths = [
        "crates/bot/examples/counter_mix_probe.rs",
        "crates/bot/examples/counter_mix_probe/bond.rs",
        "crates/bot/src/bounds.rs",
        "crates/bot/src/exact.rs",
        "crates/bot/src/root_policy.rs",
        "crates/bot/src/blind.rs",
        "crates/bot/src/smmcts.rs",
        "crates/bot/src/exp3.rs",
        "crates/bot/src/belief.rs",
        "crates/bot/src/observe.rs",
        "crates/bot/src/eval.rs",
        "crates/bot/src/agent.rs",
        "crates/bot/src/rng.rs",
        "data/gen2stadium2.json",
        "data/learnsets-gen2.json",
    ];
    let mut output = json!({"schema":"nc2000-counter-mix-v1","family":family,"state":b.essence(&dex),"state_key":b.state_key128().to_string(),
        "validations":validations,"actions":actions.map(|aa|aa.into_iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),
        "proof":proof,"rows":rows,"log":b.log,"source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "search_seed_base":seed_base,"search_seeds":count,"iterations":iterations,"probe_iterations":probe_iterations,"adaptive_probes":adaptive,"typed_probes":typed,
        "binary_sha256":sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
        "interpretation":"Legal prefix; fixed known loadouts. Pinned beliefs reconstructed from preview. No shipped-pool or human calibration claim. Matrix continuation values bound the first-action distribution with optimal subsequent play."});
    if fresh { output["fresh_probes"] = json!(true); }
    if trace { output["trace_probes"] = json!(true); }
    if continuation_exp3_ix { output["continuation_exp3_ix"] = json!(true); }
    if continuation_exp3_mean { output["continuation_exp3_mean"] = json!(true); }
    if profile { output["continuation_cutoff"] = json!(cutoff); }
    if cutoff_focus { output["continuation_cutoff_focus"] = json!(true); }
    if hp_buckets != 16 { output["hp_buckets"] = json!(hp_buckets); }
    std::fs::write(
        out.join("records.json"),
        serde_json::to_vec_pretty(&output).unwrap(),
    )
    .unwrap();
}
