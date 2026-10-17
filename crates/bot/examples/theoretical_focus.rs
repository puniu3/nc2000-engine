use nc2000_bot::bounds::{BoundConfig, BoundSolver, FixedPolicySolver};
use nc2000_bot::player::action_input;
use nc2000_engine::battle::SearchChoice;
use serde_json::{json, Value};
use std::{
    io::Write,
    time::{Duration, Instant},
};
#[allow(dead_code)]
mod historical {
    include!("theoretical_headroom.rs");
    pub fn roots(dex: &Dex) -> Vec<(String, Battle, [Option<SearchChoice>; 2])> {
        let base = std::path::Path::new("findings/deployed-endurance-v1");
        let all: Vec<Value> = std::fs::read_to_string(base.join("games/g4.jsonl"))
            .unwrap()
            .lines()
            .map(|x| serde_json::from_str(x).unwrap())
            .collect();
        let teams: [Vec<nc2000_engine::battle::PokemonSet>; 2] =
            serde_json::from_value(all[0]["teams"].clone()).unwrap();
        let mut b = Battle::from_fixture(
            dex,
            all[0]["job"]["seed"].as_str().unwrap(),
            &teams[0],
            &teams[1],
        )
        .unwrap();
        let mut out = Vec::new();
        let mut fork = None;
        for r in all.iter().filter(|r| r["type"] == "decision") {
            assert_eq!(
                normalized(state_view_json(&b, dex)),
                normalized(r["state"].clone())
            );
            assert_eq!(b.prng.seed_str(), r["seed"].as_str().unwrap());
            let p = picks(dex, &mut b, r);
            if b.turn == 10 {
                fork = Some(b.clone());
            }
            if [10, 53].contains(&b.turn) {
                out.push((format!("g4-t{}", b.turn), b.clone(), p));
            }
            b.apply_choices(dex, p).unwrap();
        }
        for trial in [4, 12] {
            let v: Value = serde_json::from_slice(
                &std::fs::read(base.join(format!("continuations/g4-d12-baseline-{trial}.json")))
                    .unwrap(),
            )
            .unwrap();
            let mut b = fork.as_ref().unwrap().clone();
            b.reseed(v["chanceSeed"].as_u64().unwrap());
            b.set_log_enabled(false);
            b.set_log_enabled(true);
            for r in v["trace"].as_array().unwrap() {
                assert_eq!(
                    normalized(state_view_json(&b, dex)),
                    normalized(r["state"].clone())
                );
                let p = picks(dex, &mut b, r);
                if b.turn == 500 {
                    out.push((format!("loop{trial}-t500"), b.clone(), p));
                    break;
                }
                b.apply_choices(dex, p).unwrap();
            }
        }
        assert_eq!(out.len(), 4);
        out
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "config output root-name-or-all");
    let config: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let dex = conformance::load_dex();
    let mut output = std::fs::File::create_new(&args[2]).unwrap();
    for (name, mut b, chosen) in historical::roots(&dex) {
        if args[3] != "all" && args[3] != name {
            continue;
        }
        b.set_log_enabled(false);
        let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
        let policies: [Vec<f64>; 2] = [0, 1].map(|s| {
            legal[s]
                .iter()
                .map(|a| f64::from(Some(*a) == chosen[s]))
                .collect()
        });
        let cfg = BoundConfig {
            work_budget: config["work"].as_u64().unwrap() as usize,
            node_budget: config["nodes"].as_u64().unwrap() as usize,
            time_budget: Some(Duration::from_millis(
                config["milliseconds"].as_u64().unwrap(),
            )),
            cell_cap: 256,
            retain_eager_prefix: true,
            trial_depth: config["depth"].as_u64().unwrap() as usize,
            chance_fair_period: config["fair_period"].as_u64().unwrap_or(0) as usize,
            probability_first_eager: config["priority"].as_bool().unwrap_or(false),
            chance_material_order: config["material"].as_bool().unwrap_or(false),
            material_descent_period: if config["material"] == true { 4 } else { 0 },
            dead_quick_claw_quotient: config["quotient"].as_bool().unwrap_or(false),
            ..Default::default()
        };
        if config["mode"] == "chance-audit" {
            let e =
                nc2000_engine::battle::enumerate::enumerate_step_partial(&dex, &b, chosen, 10000);
            assert!(e.pending.is_empty());
            let raw_leaves = e.leaves.len();
            let mut unique = std::collections::BTreeMap::new();
            for leaf in e.leaves {
                let entry = unique
                    .entry(leaf.battle.state_key128())
                    .or_insert((0.0, leaf.battle));
                entry.0 += leaf.prob;
            }
            assert_eq!(unique.len(), 2);
            let leaves:Vec<_>=unique.into_values().map(|(prob,battle)|{let mut masked=battle.clone();masked.quick_claw_roll=false;json!({"probability":prob,"state_key":battle.state_key128().to_string(),"quick_claw_roll":battle.quick_claw_roll,"roll_observable":battle.quick_claw_roll_observable(&dex),"key_after_clearing_only_quick_claw_roll":masked.state_key128().to_string()})}).collect();
            assert_eq!(
                leaves[0]["key_after_clearing_only_quick_claw_roll"],
                leaves[1]["key_after_clearing_only_quick_claw_roll"]
            );
            writeln!(output,"{}",json!({"name":name,"state_key":b.state_key128().to_string(),"runs":e.runs,"raw_leaves":raw_leaves,"leaves":leaves})).unwrap();
            continue;
        }
        if config["mode"] == "fixed-counterpolicy" {
            let side = config["side"].as_u64().unwrap() as usize;
            let priority: Vec<_> = config["priority_moves"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| SearchChoice::Move(dex.moves.id(n.as_str().unwrap()).unwrap()))
                .chain([SearchChoice::Switch(2)])
                .collect();
            let first = chosen[side].unwrap();
            let reply = chosen[1 - side].unwrap();
            let mut solver =
                FixedPolicySolver::new_with_first_action(&dex, cfg, side, priority, &b, first)
                    .with_rest_threshold(90)
                    .with_rest_on_wake();
            let started = Instant::now();
            let r = solver.solve_reply(&b, reply, None);
            let d = solver.root_chance_diagnostics(&b).unwrap();
            let rec = json!({"name":name,"config":config,"state_key":b.state_key128().to_string(),"fixed_side":side,"fixed_first":action_input(&dex,first),"free_side_first":action_input(&dex,reply),"cell_p0":[r.bounds.lo,r.bounds.hi],"stop":format!("{:?}",r.stop),"runs":r.runs,"total_runs":solver.stats().runs,"nodes":solver.node_count(),"stats":format!("{:?}",solver.stats()),"depth_histogram":solver.stats().trial_depth_counts,"graph":format!("{:?}",solver.graph_diagnostics()),"root_diagnostics":d.iter().map(|d|json!({"pending_mass":d.pending_mass,"child_uncertainty":d.child_uncertainty,"children":d.unique_children})).collect::<Vec<_>>(),"elapsed_seconds":started.elapsed().as_secs_f64()});
            writeln!(output, "{rec}").unwrap();
            output.flush().unwrap();
            eprintln!(
                "{name} fixed side={side} bounds={} stop={} runs={} elapsed={:.1}",
                rec["cell_p0"],
                rec["stop"],
                r.runs,
                started.elapsed().as_secs_f64()
            );
            continue;
        }
        let mut solver = BoundSolver::new(&dex, cfg);
        let started = Instant::now();
        let targets = vec![(0, policies[0].clone()), (1, policies[1].clone())];
        let mut checkpoints = Vec::new();
        let (bounds, stop, runs) = if config["mode"] == "target" {
            let side = if name == "g4-t10" { 1 } else { 0 };
            let mut last = None;
            for pass in 0..config["passes"].as_u64().unwrap_or(1) {
                let r = solver.solve_policy_loss_threshold(&b, side, &policies[side], 1e-6);
                let stop = format!("{:?}", r.stop);
                checkpoints.push(json!({"pass":pass,"value":[r.value_p0.lo,r.value_p0.hi],"loss":[r.policy_loss.lo,r.policy_loss.hi],"stop":stop,"total_runs":solver.stats.runs,"nodes":solver.node_count(),"closed":solver.closed_count(),"seconds":started.elapsed().as_secs_f64()}));
                eprintln!(
                    "{name} pass={pass} value={:?} loss={:?} runs={} nodes={} closed={}",
                    r.value_p0,
                    r.policy_loss,
                    solver.stats.runs,
                    solver.node_count(),
                    solver.closed_count()
                );
                last = Some((r.value_p0, stop.clone(), r.runs));
                if !["TimeBudget", "WorkExhausted"].contains(&stop.as_str()) {
                    break;
                }
            }
            last.unwrap()
        } else {
            let r = solver.solve_policy_losses(&b, &targets, 0.001);
            (r.bounds, format!("{:?}", r.stop), r.runs)
        };
        let m = solver.root_payoffs(&b).unwrap();
        let diag = solver.root_chance_diagnostics(&b).unwrap();
        let ds:Vec<_>=diag.into_iter().enumerate().map(|(i,d)|json!({"index":i,"runs":d.runs,"visits":d.visits,"pending_mass":d.pending_mass,"child_uncertainty":d.child_uncertainty,"unique_children":d.unique_children,"expanded_children":d.expanded_children,"total_child_runs":d.total_child_runs,"largest_children":d.largest_children.into_iter().map(|c|json!({"key":c.key,"mass":c.mass,"bounds":[c.bounds.lo,c.bounds.hi],"runs":c.runs,"turn":c.battle.as_ref().map(|b|b.turn)})).collect::<Vec<_>>()})).collect();
        let correlation = solver
            .root_correlation_for_policies(&b, policies.clone())
            .map(|c| c.loss_bounds());
        let rec = json!({"name":name,"config":config,"root_essence":b.essence(&dex),"state_key":b.state_key128().to_string(),"actions":legal.each_ref().map(|v|v.iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>()),"selected":chosen.map(|a|a.map(|a|action_input(&dex,a))),"cells":m.cells.iter().map(|v|[v.lo,v.hi]).collect::<Vec<_>>(),"value":[bounds.lo,bounds.hi],"loss":policies.iter().enumerate().map(|(s,p)|{let x=m.policy_loss(s,p);[x.lo,x.hi]}).collect::<Vec<_>>(),"correlated":correlation,"checkpoints":checkpoints,"stop":stop,"runs":runs,"total_runs":solver.stats.runs,"nodes":solver.node_count(),"closed":solver.closed_count(),"stats":format!("{:?}",solver.stats),"depth_histogram":solver.stats.trial_depth_counts,"depth_limit_trials":solver.stats.depth_limit_trials,"diagnostics":ds,"elapsed_seconds":started.elapsed().as_secs_f64()});
        writeln!(output, "{rec}").unwrap();
        output.flush().unwrap();
        eprintln!(
            "{name} {} runs={} depthlimit={} elapsed={:.1}",
            rec["stop"],
            solver.stats.runs,
            solver.stats.depth_limit_trials,
            started.elapsed().as_secs_f64()
        );
    }
}
