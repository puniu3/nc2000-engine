#![allow(dead_code)]

include!("../policy_witness.rs");

pub fn solve(dex: &Dex, mut battle: Battle, config: Value, threshold: f64, order: &str) -> Value {
    let cfg: Config = serde_json::from_value(config).unwrap();
    assert!(cfg.target_side < 2 && cfg.target_side == cfg.policy_side && cfg.guarded_keys);
    assert!(cfg.horizon > 0 && cfg.work_budget > 0 && cfg.cell_cap > 0);
    assert!(threshold.is_finite() && (0.0..=1.0).contains(&threshold));
    assert!(battle.outcome().is_none());
    assert!(["native", "eval"].contains(&order));
    let mut w = Witness {
        dex,
        cfg,
        runs: 0,
        memo: HashMap::new(),
        nodes: Vec::new(),
        prior: HashMap::new(),
        prioritize: false,
    };
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
    let selected = [0, 1].map(|s| w.selected(&battle, s, &legal[s], 0));
    let mut bounds = [1f64, 1f64];
    let mut branches = Vec::new();
    for &a in &selected[0] {
        for &b in &selected[1] {
            let choices = [a, b];
            let remaining = w.cfg.work_budget.saturating_sub(w.runs);
            let mut children = Vec::new();
            let mut branch = [0., 1.];
            let mut pending = 1.;
            let mut continuation_pending = 0.;
            let mut executions = 0;
            if remaining > 0 {
                let mut step =
                    enumerate_step_partial(dex, &battle, choices, w.cfg.cell_cap.min(remaining));
                executions = step.runs;
                w.runs += executions;
                pending = step.pending.iter().map(|p| p.mass).sum::<f64>();
                branch = [0., pending];
                let pad = 1e-12 * (step.leaves.len() + 2) as f64;
                if order == "eval" {
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
                    if branch[0] - pad > threshold {
                        continuation_pending += leaf.prob;
                        branch[1] += leaf.prob;
                    } else {
                        let child = w.visit(leaf.battle, 1, w.cfg.work_budget);
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
        }
    }
    let (damage, claw, semantic, _) = w.key(&battle, 0);
    let names = |aa: [Vec<Option<SearchChoice>>; 2]| {
        aa.map(|v| {
            v.into_iter()
                .map(|a| a.map(|a| a.to_input(dex)))
                .collect::<Vec<_>>()
        })
    };
    let root = w.nodes.len();
    w.nodes.push(
        json!({"state_key":battle.state_key128().to_string(),"depth":0,
        "cache_key":[damage,claw,semantic.to_string()],"kind":"decision","maximizing":false,
        "legal":names(legal),"selected":names(selected),"branches":branches,"bounds":bounds}),
    );
    json!({"config":w.cfg,"runs":w.runs,"root":root,"bounds":bounds,"nodes":w.nodes})
}
