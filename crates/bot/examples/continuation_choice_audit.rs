use nc2000_bot::{bounds::{BoundConfig, BoundSolver}, smmcts::SearchTrace, Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{battle::{PokemonSet, SearchChoice}, state::Battle};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, time::Duration};

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>; 2] }
#[derive(Deserialize)]
struct Game { seed: String, teams: [Vec<PokemonSet>; 2], decisions: Vec<Decision> }
#[derive(Deserialize)]
struct Config { side: usize, seed: u64, iterations: u32, probes: u32, focus: [String; 2] }

struct Child {
    depth: usize,
    battle: Battle,
    actions: [Vec<SearchChoice>; 2],
    counts: [Vec<u32>; 2],
    rewards: [Vec<f64>; 2],
    terminal: [Vec<u32>; 2],
    nodes: std::collections::BTreeSet<usize>,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "input.json output.json certificate-seconds");
    let input: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let game: Game = serde_json::from_str(input["corpus"].as_str().unwrap()).unwrap();
    let cfg: Config = serde_json::from_value(input["config"].clone()).unwrap();
    let seconds: u64 = args[3].parse().unwrap();
    let dex = conformance::load_dex();
    let mut battle = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
    let mut observers = [0, 1].map(|s| Observer::new(&battle, s));
    let mut beliefs = [0, 1].map(|s| Belief::pinned_from_battle(&battle, &observers[s]));
    for (i, decision) in game.decisions.iter().enumerate() {
        assert_eq!(battle.state_key128().to_string(), decision.key);
        if i+1 == game.decisions.len() { break; }
        let legal = [0, 1].map(|s| battle.legal_choices(&dex, s));
        battle.apply_choices(&dex, [0, 1].map(|s| decision.choices[s].map(|a| legal[s][a]))).unwrap();
        for s in 0..2 { observers[s].observe(&battle, &dex); beliefs[s].sync(&dex, &observers[s]); }
    }
    let focus = [0, 1].map(|s| battle.legal_choices(&dex, s).into_iter()
        .find(|a| a.to_input(&dex) == cfg.focus[s]).unwrap());
    let mut search = BlindSearch::new(&battle, &dex, RmConfig { c: 0.4, ..Default::default() }, cfg.side, cfg.seed);
    let mut children: BTreeMap<u128, Child> = BTreeMap::new();
    let mut focused = [[0u32; 3]; 2];
    let mut focus_reward = [0.; 2];
    let mut warm = Value::Null;
    let mut probe_enabled = false;
    for iteration in 0..cfg.iterations {
        let phase = usize::from(iteration >= cfg.iterations-cfg.probes);
        if iteration == cfg.iterations-cfg.probes {
            probe_enabled = search.estimated_mix_gain().is_some_and(|v| v > 0.02) && search.mixed_policy().is_none();
            warm = json!({"visits":search.visits(), "means":search.means(), "probe_enabled":probe_enabled});
        }
        let selected = if phase == 1 && probe_enabled {
            let eligible = search.root_policy_evidence().0;
            let joint = search.root_matrix();
            let mut selected = None;
            for (own, allowed) in eligible.iter().enumerate() {
                if !allowed { continue; }
                for &(reply, available) in search.root_replies() {
                    if available != search.iterations() { continue; }
                    let count = joint.iter().find(|c| c.0 == own && c.1 == reply).map_or(0, |c| c.2);
                    if selected.as_ref().is_none_or(|&(_, _, n)| count < n) { selected = Some((own, reply, count)); }
                }
            }
            selected
        } else { None };
        let mut depth = 0;
        let mut active = false;
        let mut chain = false;
        let mut path = Vec::new();
        let mut trace = |event: SearchTrace<'_>| match event {
            SearchTrace::Choice { battle, node, actions, chosen, .. } => {
                if depth == 0 {
                    active = chosen == focus.map(Some);
                    chain = active;
                    if active { focused[phase][0] += 1; }
                } else if chain {
                    let key = battle.state_key128();
                    let child = children.entry(key).or_insert_with(|| {
                        let len = actions[0].len()*actions[1].len();
                        Child { depth, battle: battle.clone(), actions: actions.clone(), counts: [vec![0; len], vec![0; len]],
                            rewards: [vec![0.; len], vec![0.; len]], terminal: [vec![0; len], vec![0; len]],
                            nodes: Default::default() }
                    });
                    assert_eq!(&child.actions, actions);
                    assert_eq!(child.depth, depth);
                    child.nodes.insert(node);
                    let indices = [0, 1].map(|s| actions[s].iter().position(|a| Some(*a) == chosen[s]).unwrap());
                    path.push((key, indices[0]*actions[1].len()+indices[1]));
                    chain = chosen == focus.map(Some);
                }
                depth += 1;
            }
            SearchTrace::Leaf { rollout, .. } => {
                if active { focused[phase][if rollout {1} else {2}] += 1; }
            }
            SearchTrace::Result { battle, reward0 } => {
                if active { focus_reward[phase] += reward0; }
                for &(key, cell) in &path {
                    let child = children.get_mut(&key).unwrap();
                    child.counts[phase][cell] += 1;
                    child.rewards[phase][cell] += reward0;
                    child.terminal[phase][cell] += u32::from(battle.outcome().is_some());
                }
            }
        };
        if let Some((own, reply, _)) = selected {
            search.step_forced_joint_observed(&dex, &beliefs[cfg.side], &observers[cfg.side], own, reply, &mut trace);
        } else {
            search.step_observed(&dex, &beliefs[cfg.side], &observers[cfg.side], 1, &mut trace);
        }
    }
    let final_row = json!({"visits":search.visits(), "joint_probe_visits":search.joint_probe_visits(),
        "means":search.means(), "dominated":search.dominated(),
        "choice":search.best().map(|a|a.to_input(&dex)),
        "candidate_policy":search.mixed_policy().map(|m|m.policy),
        "joint":search.root_matrix().into_iter().map(|(own,reply,samples,mean)|
            json!({"own":own,"reply":reply.to_input(&dex),"samples":samples,"mean":mean})).collect::<Vec<_>>()});
    for (key, value) in final_row.as_object().unwrap() { assert_eq!(&input["expected"][key], value, "trace changed {key}"); }
    let mut ordered: Vec<_> = children.into_iter().collect();
    ordered.sort_by_key(|(_, c)| (c.depth, std::cmp::Reverse(c.counts.iter().flatten().sum::<u32>())));
    eprintln!("trace parity passed; {} repeated-focus states", ordered.len());
    let mut output = Vec::new();
    let mut solver = BoundSolver::new(&dex, BoundConfig { work_budget:10_000_000,
        time_budget:Some(Duration::from_secs(seconds)), eps:1e-8,
        certified_action_pruning:false, fold_closed_nodes:false, ..Default::default() });
    for (key, child) in ordered {
        let targets: Vec<_> = (0..2).flat_map(|s| (0..child.actions[s].len()).map(move |a| (s, a)))
            .map(|(s,a)| (s,(0..child.actions[s].len()).map(|i|f64::from(i==a)).collect())).collect();
        let before = solver.stats.runs;
        let result = solver.solve_policy_losses(&child.battle, &targets, 1e-8);
        let matrix = solver.root_payoffs(&child.battle).unwrap();
        eprintln!("child {key} samples={} width={}", child.counts.iter().flatten().sum::<u32>(), result.bounds.width());
        output.push(json!({"depth":child.depth,"state_key":key.to_string(), "state":child.battle.essence(&dex), "nodes":child.nodes,
            "actions":child.actions.map(|a|a.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),
            "counts":child.counts, "reward0_sums":child.rewards, "terminal_counts":child.terminal,
            "value":[matrix.value.lo,matrix.value.hi], "cells":matrix.cells.iter().map(|b|[b.lo,b.hi]).collect::<Vec<_>>(),
            "runs":solver.stats.runs-before,"stop":format!("{:?}",result.stop)}));
    }
    std::fs::write(&args[2], serde_json::to_vec(&json!({"warm":warm,"final":final_row,"focused_counts":focused,
        "focus_reward0_sums":focus_reward,"children":output})).unwrap()).unwrap();
}
