#![allow(dead_code)]

include!("../policy_witness.rs");

pub struct AliasTask {
    pub battle: Battle,
    pub depth: usize,
    pub key: Value,
    pub choices: [Option<SearchChoice>; 2],
    pub representative: [Option<SearchChoice>; 2],
}

pub fn solve(
    dex: &Dex,
    battle: Battle,
    config: Value,
    threshold: f64,
    merge: bool,
    fast: bool,
    ignored_root_action: Option<&str>,
) -> (Value, Vec<AliasTask>) {
    let cfg: Config = serde_json::from_value(config).unwrap();
    assert!(cfg.target_side < 2 && cfg.target_side == cfg.policy_side && cfg.guarded_keys);
    assert!(cfg.horizon > 0 && cfg.work_budget > 0 && cfg.cell_cap > 0);
    assert!(threshold.is_finite() && (0.0..=1.0).contains(&threshold));
    assert!(!cfg.material_order && !cfg.share_root_budget && !cfg.stop_if_asleep);
    assert!(cfg.minimum_target_living.is_none());
    let mut w = Witness {
        dex,
        cfg,
        runs: 0,
        memo: HashMap::new(),
        nodes: Vec::new(),
        prior: HashMap::new(),
        prioritize: false,
    };
    let mut audits = Vec::new();
    let root = visit(
        &mut w,
        battle,
        0,
        threshold,
        merge,
        fast,
        ignored_root_action,
        &mut audits,
    );
    (
        json!({"config":w.cfg,"runs":w.runs,"root":root,"bounds":w.nodes[root]["bounds"],"nodes":w.nodes}),
        audits,
    )
}

pub fn upper(dex: &Dex, battle: Battle, config: Value) -> Value {
    let cfg: Config = serde_json::from_value(config).unwrap();
    let mut w = Witness {
        dex,
        cfg,
        runs: 0,
        memo: HashMap::new(),
        nodes: Vec::new(),
        prior: HashMap::new(),
        prioritize: false,
    };
    let root = w.visit(battle, 0, w.cfg.work_budget);
    json!({"config":w.cfg,"runs":w.runs,"root":root,"bounds":w.nodes[root]["bounds"],"nodes":w.nodes})
}

fn visit(
    w: &mut Witness<'_>,
    mut battle: Battle,
    depth: usize,
    threshold: f64,
    merge: bool,
    fast: bool,
    ignored_root_action: Option<&str>,
    audits: &mut Vec<AliasTask>,
) -> usize {
    let dex = w.dex;
    let cache_key = w.key(&battle, depth);
    if let Some(&id) = w.memo.get(&cache_key) {
        return id;
    }
    if battle.outcome().is_some() || depth >= w.cfg.horizon || w.runs >= w.cfg.work_budget {
        return w.visit(battle, depth, w.cfg.work_budget);
    }
    let legal = [0, 1].map(|s| {
        if battle.needs_choice()[s] {
            battle
                .legal_choices(dex, s)
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>()
        } else {
            vec![None]
        }
    });
    let selected = [0, 1].map(|s| w.selected(&battle, s, &legal[s], depth));
    let mut bounds = [1f64, 1f64];
    let mut branches: Vec<Value> = Vec::new();
    let mut representatives: Vec<[Option<SearchChoice>; 2]> = Vec::new();
    for &a in &selected[0] {
        for &b in &selected[1] {
            let choices = [a, b];
            if depth == 0
                && ignored_root_action.is_some_and(|ignored| {
                    choices[1 - w.cfg.target_side].is_some_and(|a| a.to_input(dex) == ignored)
                })
            {
                branches.push(json!({"choices":choices.map(|a|a.map(|a|a.to_input(dex))),"runs":0,"pending":1.,"continuation_pending_mass":0.,"children":[],"bounds":[0.,1.]}));
                representatives.push(choices);
                bounds[0] = 0.;
                continue;
            }
            let alias = if merge {
                representatives.iter().position(|prior| {
                    let side = w.cfg.target_side;
                    if prior[side] != choices[side] {
                        return false;
                    }
                    let (Some(a), Some(b)) = (prior[1 - side], choices[1 - side]) else {
                        return false;
                    };
                    let mut pair = vec![a, b];
                    nc2000_bot::smmcts::collapse_sleep_choices(&battle, dex, 1 - side, &mut pair);
                    pair == [a]
                })
            } else {
                None
            };
            if let Some(index) = alias {
                let mut branch = branches[index].clone();
                branch["choices"] = json!(choices.map(|a| a.map(|a| a.to_input(dex))));
                branch["runs"] = json!(0);
                branch["alias_of"] = json!(index);
                audits.push(AliasTask {
                    battle: battle.clone(),
                    depth,
                    key: json!([cache_key.0, cache_key.1, cache_key.2.to_string()]),
                    choices,
                    representative: representatives[index],
                });
                for k in 0..2 {
                    bounds[k] = bounds[k].min(branch["bounds"][k].as_f64().unwrap());
                }
                branches.push(branch);
                representatives.push(choices);
                continue;
            }
            let remaining = w.cfg.work_budget.saturating_sub(w.runs);
            let mut children = Vec::new();
            let mut branch = [0., 1.];
            let mut pending = 1.;
            let mut continuation_pending = 0.;
            let mut executions = 0;
            if remaining > 0 {
                let mut step =
                    enumerate(dex, &battle, choices, w.cfg.cell_cap.min(remaining), fast);
                executions = step.runs;
                w.runs += executions;
                pending = step.pending.iter().map(|p| p.mass).sum::<f64>();
                branch = [0., pending];
                let pad = 1e-12 * (step.leaves.len() + 2) as f64;
                if depth == 0 && threshold < 1. {
                    let weights = nc2000_bot::eval::EvalWeights::default();
                    step.leaves.sort_by_cached_key(|leaf| {
                        let value = nc2000_bot::eval::eval01(&leaf.battle, dex, &weights);
                        let own = if w.cfg.target_side == 0 {
                            value
                        } else {
                            1. - value
                        };
                        std::cmp::Reverse((leaf.prob * own).to_bits())
                    });
                }
                for leaf in step.leaves {
                    if depth == 0 && branch[0] - pad > threshold {
                        continuation_pending += leaf.prob;
                        branch[1] += leaf.prob;
                    } else {
                        let child = visit(
                            w,
                            leaf.battle,
                            depth + 1,
                            threshold,
                            merge,
                            fast,
                            ignored_root_action,
                            audits,
                        );
                        for k in 0..2 {
                            branch[k] += leaf.prob * w.nodes[child]["bounds"][k].as_f64().unwrap();
                        }
                        children.push(json!({"mass":leaf.prob,"node":child}));
                    }
                }
                pending += continuation_pending;
                branch = [(branch[0] - pad).max(0.), (branch[1] + pad).min(1.)];
            }
            for k in 0..2 {
                bounds[k] = bounds[k].min(branch[k]);
            }
            branches.push(json!({"choices":choices.map(|a|a.map(|a|a.to_input(dex))),
                "runs":executions,"pending":pending,"continuation_pending_mass":continuation_pending,
                "children":children,"bounds":branch}));
            representatives.push(choices);
        }
    }
    let (damage, claw, semantic, _) = w.key(&battle, depth);
    let names = |aa: [Vec<Option<SearchChoice>>; 2]| {
        aa.map(|v| {
            v.into_iter()
                .map(|a| a.map(|a| a.to_input(dex)))
                .collect::<Vec<_>>()
        })
    };
    let root = w.nodes.len();
    w.nodes.push(
        json!({"state_key":battle.state_key128().to_string(),"depth":depth,
        "cache_key":[damage,claw,semantic.to_string()],"kind":"decision","maximizing":false,
        "legal":names(legal),"selected":names(selected),"branches":branches,"bounds":bounds}),
    );
    w.memo.insert(cache_key, root);
    root
}

fn enumerate(
    dex: &Dex,
    battle: &Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
    fast: bool,
) -> nc2000_engine::battle::enumerate::PartialStepEnum {
    if !fast {
        return enumerate_step_partial(dex, battle, choices, cap);
    }
    use nc2000_engine::battle::enumerate::{
        enumerate_step_with_runner, run_scripted, ChancePrefix, PartialStepEnum,
    };
    let mut runs = 0;
    let mut runner = |script: &[usize]| {
        let (next, trace) = run_scripted(dex, battle, choices, script);
        runs += 1;
        (next, trace, true)
    };
    let result = enumerate_step_with_runner(dex, battle, choices, cap, &mut runner);
    assert!(runs <= cap);
    match result {
        Some(step) => {
            assert_eq!(runs, step.runs);
            PartialStepEnum {
                leaves: step.leaves,
                leaf_scripts: None,
                pending: Vec::new(),
                runs,
                exhausted: false,
            }
        }
        None => PartialStepEnum {
            leaves: Vec::new(),
            leaf_scripts: Some(Vec::new()),
            pending: vec![ChancePrefix {
                script: Vec::new(),
                mass: 1.,
            }],
            runs,
            exhausted: true,
        },
    }
}
