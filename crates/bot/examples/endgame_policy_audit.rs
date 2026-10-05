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

#[path="endgame_policy_audit/hidden_selection.rs"]
mod hidden_selection;

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
    trial_depth: Option<usize>,
    descend_floor: Option<f64>,
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
    frontier_key_census: bool,
    hp_translation_audit_limit: Option<usize>,
    #[serde(default)]
    hp_transition_capacity: usize,
    #[serde(default)]
    hp_transition_eager: bool,
    #[serde(default)]
    certified_action_pruning: bool,
    fixed_policy_side: Option<usize>,
    fixed_first_action: Option<String>,
    fixed_priority_actions: Option<Vec<String>>,
    fixed_reply_action: Option<String>,
    fixed_reply_threshold: Option<f64>,
    fixed_rest_percent: Option<u8>,
    fixed_recover_percent: Option<u8>,
    fixed_recover_last_foes: Option<Vec<String>>,
    #[serde(default)]
    fixed_rest_after_boom: bool,
    #[serde(default)]
    fixed_rest_on_wake: bool,
    #[serde(default)]
    fixed_sleep_talk: bool,
    #[serde(default)]
    root_census_only: bool,
    #[serde(default)]
    fixed_frozen_switch: bool,
    #[serde(default)]
    aggregate_lazy_children: bool,
    #[serde(default)]
    chance_fair_period: usize,
    #[serde(default)]
    chance_status_groups: bool,
    #[serde(default)]
    chance_material_order: bool,
    #[serde(default)]
    material_descent_period: usize,
    #[serde(default)]
    dead_quick_claw_quotient: bool,
    resource_closure_nodes: Option<usize>,
    resource_closure_period: Option<usize>,
    #[serde(default)]
    directed_policy_refinement: bool,
    #[serde(default)]
    target_aware_refinement: bool,
    witness_preparation: Option<WitnessPreparation>,
    descendant_preparation: Option<DescendantPreparation>,
    hidden_selection: Option<hidden_selection::Config>,
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
    replacement_semantic_key: Option<String>,
    #[serde(default)]
    quick_claw_kernel_audit: bool,
    #[serde(default)]
    quotient_quick_claw: bool,
    lower_threshold: Option<f64>,
    fixed_policy_side: Option<usize>,
    fixed_priority_actions: Option<Vec<String>>,
    fixed_rest_percent: Option<u8>,
    #[serde(default)]
    fixed_sleep_talk: bool,
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

fn visit_effect_states(battle: &mut Battle, mut visit: impl FnMut(&mut nc2000_engine::state::EffectState)) {
    visit(&mut battle.format_data);
    visit(&mut battle.field.weather_state);
    for (_,state) in &mut battle.field.pseudo_weather { visit(state); }
    for side in &mut battle.sides {
        for (_,state) in &mut side.side_conditions { visit(state); }
        for (_,state) in &mut side.slot_conditions { visit(state); }
        for pokemon in &mut side.roster {
            visit(&mut pokemon.status_state);visit(&mut pokemon.item_state);
            for (_,state) in &mut pokemon.volatiles { visit(state); }
        }
    }
}

fn project_frontier_state(battle: &mut Battle, name: &str) {
    use nc2000_engine::state::MoveResult;
    if matches!(name,"effect_order_rank" | "combined_history") {
        let mut orders=vec![0,battle.effect_order];
        visit_effect_states(battle,|state|orders.push(state.effect_order));
        orders.sort_unstable();orders.dedup();
        battle.effect_order=orders.binary_search(&battle.effect_order).unwrap() as u32;
        visit_effect_states(battle,|state|state.effect_order=orders.binary_search(&state.effect_order).unwrap() as u32);
    }
    if matches!(name,"unused_history" | "move_history" | "combined_history") {
        battle.last_move_id=None;battle.last_successful_move_this_turn=None;
        for side in &mut battle.sides { for p in &mut side.roster {
            p.last_move_used=None;p.move_last_turn_result=MoveResult::Undef;p.active_move_actions=0;
        }}
    }
    if matches!(name,"move_history" | "combined_history") {
        for side in &mut battle.sides {
            side.last_move=None;
            for p in &mut side.roster {
                p.last_move=None;p.last_move_encore=None;p.last_move_target_loc=None;
                p.move_this_turn=None;p.move_this_turn_result=MoveResult::Undef;
            }
        }
    }
    if name=="turn_counters" {
        battle.turn=1;
        for side in &mut battle.sides { for p in &mut side.roster { p.active_turns=i32::from(p.active_turns>0); }}
    }
    if name=="pp" {
        for side in &mut battle.sides { for p in &mut side.roster {
            for slot in &mut p.move_slots { slot.pp=0; }
            for slot in &mut p.base_move_slots { slot.pp=0; }
        }}
    }
    if name=="hp" {
        for side in &mut battle.sides { for p in &mut side.roster { if p.hp>0 {p.hp=p.maxhp;} }}
    }
}

fn frontier_key_census<'a>(battles: impl Iterator<Item=&'a Battle>, dex: &Dex, protocol: &Protocol) -> Value {
    use std::collections::{BTreeMap,HashMap};
    let started=Instant::now();
    let names=["baseline","effect_order_rank","unused_history","move_history","combined_history","turn_counters","pp","hp"];
    let mut groups: Vec<HashMap<(bool,bool,u128),usize>>=names.iter().map(|_|HashMap::new()).collect();
    let mut requests=BTreeMap::<String,usize>::new();
    let mut retained=0;
    for battle in battles {
        retained+=1;
        *requests.entry(battle.request_state.as_str().to_string()).or_default()+=1;
        let damage=!battle.damage_bookkeeping_observable(dex);
        let claw=protocol.dead_quick_claw_quotient && !battle.quick_claw_roll_observable(dex);
        for (name,group) in names.iter().zip(groups.iter_mut()) {
            let mut projected=battle.clone();
            project_frontier_state(&mut projected,name);
            let key=if claw {projected.state_key128_without_quick_claw_roll(dex,damage)}
                else if damage {projected.state_key128_without_damage_bookkeeping()}
                else {projected.state_key128()};
            *group.entry((damage,claw,key)).or_default()+=1;
        }
    }
    assert_eq!(groups[0].len(),retained);
    let projections: BTreeMap<_,_>=names.into_iter().zip(groups).map(|(name,groups)| {
        (name,json!({"unique":groups.len(),"redundant_states":retained-groups.len(),
            "largest_class":groups.values().copied().max().unwrap_or(0)}))
    }).collect();
    json!({"schema":"nc2000-retained-key-ablation-v1","retained":retained,"requests":requests,
        "projections":projections,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
        "interpretation":"Read-only ablation of retained concrete battles after solving; expanded nodes whose battles were released are excluded. Projections are diagnostic, not certified equivalences. In particular HP, PP, turn and move-history deletion can change the game and are never used by the solver."})
}

fn hp_translation_audit<'a>(battles: impl Iterator<Item=&'a Battle>, dex: &Dex, limit: usize) -> Value {
    use nc2000_bot::hp_transition::{HpTransition,HpTransitionKey};
    use std::collections::HashMap;
    assert!(limit>0);
    let started=Instant::now();
    let mut states=battles.map(|b|(b.state_key128(),b)).collect::<Vec<_>>();
    states.sort_unstable_by_key(|(key,_)|*key);
    let available=states.len();
    let stride=available.div_ceil(limit).max(1);
    let mut cache=HashMap::<HpTransitionKey,HpTransition>::new();
    let mut samples=0;let mut candidates=0;let mut eligible=0;let mut translated=0;let mut changed_hp=0;let mut rejected_hp=0;
    let mut max_cache=0;
    let key=|b:&Battle| {
        let damage=!b.damage_bookkeeping_observable(dex);
        let claw=!b.quick_claw_roll_observable(dex);
        let value=if claw {b.state_key128_without_quick_claw_roll(dex,damage)}
            else if damage {b.state_key128_without_damage_bookkeeping()} else {b.state_key128()};
        (damage,claw,value)
    };
    for (_,state) in states.into_iter().step_by(stride).take(limit) {
        samples+=1;
        let mut probe=state.clone();
        let choices=[0,1].map(|side|probe.legal_choices(dex,side));
        for &a in &choices[0] { for &b in &choices[1] {
            candidates+=1;
            let choices=[Some(a),Some(b)];
            let Some(input)=HpTransitionKey::new(dex,state,choices,&[]) else {continue;};
            eligible+=1;
            let (actual,trace,entry)=HpTransition::run(dex,state,choices,&[]);
            if let Some(old)=cache.get(&input) {
                if let Some((predicted,predicted_trace))=old.translate(dex,state,choices,&[]) {
                    assert_eq!(key(&predicted),key(&actual),"translated semantic state mismatch");
                    assert_eq!(predicted_trace.len(),trace.len());
                    for (a,b) in predicted_trace.iter().zip(&trace) {
                        assert_eq!((a.label,&a.counts,a.chosen),(b.label,&b.counts,b.chosen));
                    }
                    translated+=1;
                    changed_hp+=usize::from(old.input_hp()!=[0,1].map(|side|state.poke(state.active_id(side).unwrap()).hp));
                } else {rejected_hp+=1;}
            }
            if let Some(entry)=entry {cache.entry(input).or_insert(entry);}
            max_cache=max_cache.max(cache.len());
        }}
    }
    json!({"schema":"nc2000-hp-translation-audit-v1","available":available,"sampled":samples,"stride":stride,
        "candidate_steps":candidates,"eligible_steps":eligible,"engine_runs":eligible,"translated":translated,
        "changed_hp":changed_hp,"rejected_hp_boundary":rejected_hp,"cache_entries":max_cache,
        "semantic_mismatches":0,"trace_mismatches":0,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
        "interpretation":"Read-only audit after solving, using one empty-prefix scripted representative for every eligible legal joint action in sampled retained states. Every predicted result is checked against a fresh engine run; all chance labels, weights and choices must match. This is reuse potential for this sampled workload, not measured search speedup or a complete-kernel audit."})
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
    let hidden_selection = protocol.hidden_selection.as_ref().map(|cfg| {
        assert!(protocol.fixed_policy_side.is_some());
        let (variant, evidence) = hidden_selection::replay(dex, game, decision, cfg);
        battle = variant;
        evidence
    });
    let legal = [0, 1].map(|side| battle.legal_choices(dex, side));
    let names = legal.clone().map(|aa| {
        aa.into_iter()
            .map(|a| action_input(dex, a))
            .collect::<Vec<_>>()
    });
    if protocol.root_census_only {
        let started = Instant::now();
        let mut actions = legal.clone().map(|aa| if aa.is_empty() { vec![None] }
            else { aa.into_iter().map(Some).collect() });
        if let Some(side) = protocol.fixed_policy_side {
            let first = protocol.fixed_first_action.as_ref().unwrap();
            let index = names[side].iter().position(|name| name == first).unwrap();
            actions[side] = vec![Some(legal[side][index])];
        }
        let mut runs = 0;
        let mut cells = Vec::new();
        for &a in &actions[0] { for &b in &actions[1] {
            let step = nc2000_engine::battle::enumerate::enumerate_step(dex, &battle,
                [a, b], protocol.cell_cap).unwrap();
            runs += step.runs;
            let mut groups = std::collections::BTreeMap::<String, Value>::new();
            for leaf in &step.leaves {
                let next = &leaf.battle;
                let class = json!({"outcome":format!("{:?}",next.outcome()),
                    "needs_choice":next.needs_choice(),"alive":[next.sides[0].pokemon_left,next.sides[1].pokemon_left],
                    "status":([0,1].map(|side| next.active_id(side).map(|id|next.poke(id).status.as_str())))});
                let hp = [0,1].map(|side| next.active_id(side).map_or(0, |id| next.poke(id).hp));
                let group = groups.entry(class.to_string()).or_insert_with(|| json!({
                    "class":class,"mass":0.0,"leaves":0,"hp_ranges":hp.map(|h|[h,h]),
                    "first_semantic_key":next.state_key128_without_damage_bookkeeping().to_string()}));
                group["mass"] = json!(group["mass"].as_f64().unwrap() + leaf.prob);
                group["leaves"] = json!(group["leaves"].as_u64().unwrap() + 1);
                for side in 0..2 {
                    group["hp_ranges"][side][0] = json!(group["hp_ranges"][side][0].as_i64().unwrap().min(i64::from(hp[side])));
                    group["hp_ranges"][side][1] = json!(group["hp_ranges"][side][1].as_i64().unwrap().max(i64::from(hp[side])));
                }
            }
            assert!((groups.values().map(|g|g["mass"].as_f64().unwrap()).sum::<f64>() - 1.0).abs() < 1e-10);
            cells.push(json!({"actions":([a,b].map(|a|a.map(|a|action_input(dex,a)))),
                "runs":step.runs,"groups":groups.into_values().collect::<Vec<_>>()}));
        }}
        return json!({"schema":"nc2000-root-transition-census-v1","game":game.game,"decision":decision,
            "state_key":battle.state_key128().to_string(),"cells":cells,
            "reference":{"stop":"CensusOnly","runs":runs,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0}});
    }
    if let Some(side) = protocol.fixed_policy_side {
        assert!(protocol.witness_preparation.is_none());
        assert!(protocol.descendant_preparation.is_none());
        assert!(side < 2 && !legal[side].is_empty());
        let priority_names = protocol.fixed_priority_actions.as_ref().unwrap_or(&names[side]);
        assert!(priority_names.iter().all(|name| names[side].contains(name)
            && priority_names.iter().filter(|p| *p == name).count() == 1));
        let priority: Vec<_> = priority_names.iter().map(|name|
            legal[side][names[side].iter().position(|n| n == name).unwrap()]).collect();
        let cfg = BoundConfig {
            time_budget: Some(Duration::from_millis(protocol.reference_ms)),
            work_budget: protocol.work_budget,
            node_budget: protocol.node_budget,
            cell_cap: protocol.cell_cap,
            trial_depth: protocol.trial_depth.unwrap_or_else(|| BoundConfig::default().trial_depth),
            descend_floor: protocol.descend_floor.unwrap_or_else(|| BoundConfig::default().descend_floor),
            eps: protocol.tolerance,
            certified_action_pruning: protocol.certified_action_pruning,
            aggregate_lazy_children: protocol.aggregate_lazy_children,
            chance_fair_period: protocol.chance_fair_period,
            chance_status_groups: protocol.chance_status_groups,
            chance_material_order: protocol.chance_material_order,
            material_descent_period: protocol.material_descent_period,
            dead_quick_claw_quotient: protocol.dead_quick_claw_quotient,
            hp_transition_capacity: protocol.hp_transition_capacity,
            hp_transition_eager: protocol.hp_transition_eager,
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
        if protocol.fixed_sleep_talk { solver = solver.with_sleep_talk(); }
        if protocol.fixed_frozen_switch { solver = solver.with_frozen_switch(); }
        if let Some(percent)=protocol.fixed_rest_percent { solver=solver.with_rest_threshold(percent); }
        if let Some(percent)=protocol.fixed_recover_percent { solver=solver.with_recover_threshold(percent); }
        if let Some(species)=&protocol.fixed_recover_last_foes {
            solver=solver.with_recover_against_last(species.iter().map(|name|dex.species.id(name).unwrap()).collect());
        }
        if protocol.fixed_rest_after_boom { solver=solver.with_rest_after_boom(); }
        if protocol.fixed_rest_on_wake { solver=solver.with_rest_on_wake(); }
        let started = Instant::now();
        if let Some(reply) = &protocol.fixed_reply_action {
            let reply_choice = legal[1-side][names[1-side].iter().position(|name|name==reply).unwrap()];
            let report = solver.solve_reply(&battle,reply_choice,protocol.fixed_reply_threshold);
            let elapsed_ms=started.elapsed().as_secs_f64()*1000.0;
            let matrix = solver.restricted_root_payoffs(&battle).unwrap();
            let census=protocol.frontier_key_census.then(||frontier_key_census(solver.retained_battles(),dex,protocol));
            let hp_audit=protocol.hp_translation_audit_limit.map(|limit|hp_translation_audit(solver.retained_battles(),dex,limit));
            return json!({"schema":"nc2000-fixed-reply-bound-v1","game":game.game,"decision":decision,
                "hidden_selection":hidden_selection,
                "state_key":battle.state_key128().to_string(),"side":side,"priority_actions":priority_names,
                "fixed_first_action":protocol.fixed_first_action,"reply_action":reply,
                "fixed_rest_percent":protocol.fixed_rest_percent,
                "fixed_recover_percent":protocol.fixed_recover_percent,
                "fixed_recover_last_foes":protocol.fixed_recover_last_foes,
                "fixed_rest_after_boom":protocol.fixed_rest_after_boom,
                "fixed_rest_on_wake":protocol.fixed_rest_on_wake,
                "reply_value":bound(report.bounds),"threshold":protocol.fixed_reply_threshold,
                "frontier_key_census":census,
                "hp_translation_audit":hp_audit,
                "refinement":{"trials":solver.stats().trials,
                    "trial_depth_counts":solver.stats().trial_depth_counts,"depth_limit_trials":solver.stats().depth_limit_trials,
                    "eager_failures":solver.stats().eager_failures,"lp_solves":solver.stats().lp_solves,
                    "material_descent_bypasses":solver.stats().material_descent_bypasses,
                    "fixed_waking_rest_nodes":solver.stats().fixed_waking_rest_nodes,
                    "resource_closure_visits":solver.stats().resource_closure_visits,
                    "lower_directed_trials":solver.stats().lower_directed_trials,"upper_directed_trials":solver.stats().upper_directed_trials},
                "hp_transition_cache":{"capacity":protocol.hp_transition_capacity,
                    "eligible":solver.stats().hp_transition_eligible,"hits":solver.stats().hp_transition_hits,
                    "eager_enabled":protocol.hp_transition_eager,
                    "eager_hits":solver.stats().hp_transition_eager_hits,"eager_eligible":solver.stats().hp_transition_eager_eligible,
                    "rejected_hp":solver.stats().hp_transition_rejected,"peak":solver.stats().hp_transition_peak,
                    "expansions":solver.stats().expansions,"eager_runs":solver.stats().eager_runs},
                "restricted_matrix":{"actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|action_input(dex,a))).collect::<Vec<_>>()),
                    "cells":matrix.cells.iter().copied().map(bound).collect::<Vec<_>>(),"value":bound(matrix.value)},
                "reference":{"stop":format!("{:?}",report.stop),"runs":report.runs,
                    "nodes":solver.node_count(),"elapsed_ms":elapsed_ms},
                "root_chance":solver.root_chance_diagnostics(&battle).unwrap().iter().map(|c|chance_diagnostic(c,dex)).collect::<Vec<_>>()});
        }
        assert!(protocol.fixed_reply_threshold.is_none());
        let report = solver.solve(&battle);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let graph = solver.graph_diagnostics();
        let matrix = solver.restricted_root_payoffs(&battle).unwrap();
        let stats = solver.stats();
        return json!({
            "schema":"nc2000-fixed-policy-bound-v1",
            "hidden_selection":hidden_selection,
            "game":game.game,"decision":decision,"state_key":battle.state_key128().to_string(),
            "side":side,"priority_actions":priority_names,
            "fixed_first_action":protocol.fixed_first_action,
            "fixed_sleep_talk":protocol.fixed_sleep_talk,
            "fixed_frozen_switch":protocol.fixed_frozen_switch,
            "fixed_rest_percent":protocol.fixed_rest_percent,
            "fixed_recover_percent":protocol.fixed_recover_percent,
            "fixed_recover_last_foes":protocol.fixed_recover_last_foes,
            "fixed_rest_after_boom":protocol.fixed_rest_after_boom,
            "fixed_rest_on_wake":protocol.fixed_rest_on_wake,
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
    assert!(protocol.fixed_reply_action.is_none() && protocol.fixed_reply_threshold.is_none());
    assert!(protocol.fixed_priority_actions.is_none());
    assert!(!protocol.fixed_sleep_talk);
    assert!(!protocol.fixed_frozen_switch);
    assert!(protocol.fixed_rest_percent.is_none());
    assert!(protocol.fixed_recover_percent.is_none());
    assert!(protocol.fixed_recover_last_foes.is_none());
    assert!(!protocol.fixed_rest_after_boom);
    assert!(!protocol.fixed_rest_on_wake);
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
            trial_depth: protocol.trial_depth.unwrap_or_else(|| BoundConfig::default().trial_depth),
            descend_floor: protocol.descend_floor.unwrap_or_else(|| BoundConfig::default().descend_floor),
            certified_action_pruning: protocol.certified_action_pruning,
            aggregate_lazy_children: protocol.aggregate_lazy_children,
            chance_fair_period: protocol.chance_fair_period,
            chance_status_groups: protocol.chance_status_groups,
            chance_material_order: protocol.chance_material_order,
            material_descent_period: protocol.material_descent_period,
            dead_quick_claw_quotient: protocol.dead_quick_claw_quotient,
            hp_transition_capacity: protocol.hp_transition_capacity,
            hp_transition_eager: protocol.hp_transition_eager,
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
        if prep.quotient_quick_claw {
            assert!(protocol.dead_quick_claw_quotient);
            assert!(step.leaves.iter().all(|leaf| !leaf.battle.quick_claw_roll_observable(dex)
                && !leaf.battle.damage_bookkeeping_observable(dex)));
        }
        let successor_key = |state: &Battle| if prep.quotient_quick_claw {
            state.state_key128_without_quick_claw_roll(dex,true)
        } else { state.state_key128_without_damage_bookkeeping() };
        let mut successor_distribution = std::collections::BTreeMap::<String,f64>::new();
        for leaf in &step.leaves {
            *successor_distribution.entry(successor_key(&leaf.battle).to_string()).or_default() += leaf.prob;
        }
        let matching: Vec<_> = step.leaves.iter().filter(|leaf|
            successor_key(&leaf.battle) == key).collect();
        let mass: f64 = matching.iter().map(|leaf| leaf.prob).sum();
        assert!(!matching.is_empty() && (mass - prep.mass).abs() < 1e-12);
        let mut child = matching[0].battle.clone();
        let mut replacement = Value::Null;
        let mut enumeration_runs = step.runs;
        if let Some(selected) = &prep.replacement_semantic_key {
            let needs = child.needs_choice();
            assert_ne!(needs[0], needs[1]);
            let legal = [0, 1].map(|side| child.legal_choices(dex, side));
            assert!(legal.iter().all(|actions| actions.len() <= 1));
            let actions = legal.map(|actions| actions.first().copied());
            assert!(actions.iter().flatten().any(|action|
                matches!(action, nc2000_engine::battle::SearchChoice::Switch(_))));
            let next = nc2000_engine::battle::enumerate::enumerate_step(dex, &child,
                actions, prep.enumeration_cap).unwrap();
            assert!((next.leaves.iter().map(|leaf| leaf.prob).sum::<f64>() - 1.0).abs() < 1e-10);
            let selected: u128 = selected.parse().unwrap();
            let mut distribution = std::collections::BTreeMap::<String, f64>::new();
            let mut quotient_distribution = std::collections::BTreeMap::<String, f64>::new();
            for leaf in &next.leaves {
                assert!(!leaf.battle.damage_bookkeeping_observable(dex));
                *distribution.entry(leaf.battle.state_key128_without_damage_bookkeeping().to_string()).or_default() += leaf.prob;
                if prep.quotient_quick_claw {
                    assert!(!leaf.battle.quick_claw_roll_observable(dex));
                    *quotient_distribution.entry(successor_key(&leaf.battle).to_string()).or_default() += leaf.prob;
                }
            }
            let first = &next.leaves.iter().find(|leaf|
                leaf.battle.state_key128_without_damage_bookkeeping() == selected).unwrap().battle;
            enumeration_runs += next.runs;
            replacement = json!({"actions":actions.map(|a|a.map(|a|action_input(dex,a))),
                "runs":next.runs,"leaves":next.leaves.len(),"distribution":distribution,
                "mass":distribution[&selected.to_string()],
                "quotient_distribution":quotient_distribution,
                "quotient_semantic_key":prep.quotient_quick_claw.then(||successor_key(first).to_string()),
                "semantic_key":first.state_key128_without_damage_bookkeeping().to_string()});
            child = first.clone();
        }
        let child = &child;
        assert!(child.outcome().is_none() && !child.damage_bookkeeping_observable(dex));
        let mut quick_claw_kernel = Value::Null;
        if prep.quick_claw_kernel_audit {
            let mut alternate = child.clone();
            alternate.quick_claw_roll = !alternate.quick_claw_roll;
            let mut probe = child.clone();
            let choices = [0,1].map(|side|probe.legal_choices(dex,side));
            assert!([0,1].into_iter().all(|side|alternate.legal_choices(dex,side) == choices[side]));
            let mut cells = Vec::new();
            for &a in &choices[0] { for &b in &choices[1] {
                let mut kernels = Vec::new();
                let mut runs = 0;
                for state in [child,&alternate] {
                    let step = nc2000_engine::battle::enumerate::enumerate_step(dex,state,
                        [Some(a),Some(b)],prep.enumeration_cap).unwrap();
                    runs += step.runs;
                    let mut kernel = std::collections::BTreeMap::<String,f64>::new();
                    for leaf in &step.leaves {
                        assert!(!leaf.battle.damage_bookkeeping_observable(dex));
                        let key = leaf.battle.outcome().map_or_else(
                            ||leaf.battle.state_key128_without_damage_bookkeeping().to_string(),
                            |outcome|format!("terminal:{outcome:?}"));
                        *kernel.entry(key).or_default() += leaf.prob;
                    }
                    assert!((kernel.values().sum::<f64>() - 1.0).abs() < 1e-10);
                    kernels.push(kernel);
                }
                let difference: f64 = kernels[0].keys().chain(kernels[1].keys())
                    .cloned().collect::<std::collections::BTreeSet<_>>().iter()
                    .map(|key|(kernels[0].get(key).unwrap_or(&0.0)-kernels[1].get(key).unwrap_or(&0.0)).abs()).sum();
                enumeration_runs += runs;
                cells.push(json!({"actions":[action_input(dex,a),action_input(dex,b)],
                    "runs":runs,"states":kernels.iter().map(|k|k.len()).collect::<Vec<_>>(),"l1_difference":difference}));
            }}
            quick_claw_kernel = json!({"semantic_keys":[child.state_key128_without_damage_bookkeeping().to_string(),
                alternate.state_key128_without_damage_bookkeeping().to_string()],"cells":cells});
        }
        let dedicated_config = BoundConfig {
            time_budget: Some(Duration::from_millis(prep.reference_ms)),
            work_budget: prep.work_budget,
            node_budget: prep.node_budget,
            eps: prep.tolerance,
            ..cfg.clone()
        };
        assert!(prep.lower_threshold.is_none_or(|threshold|threshold.is_finite() && (0.0..=1.0).contains(&threshold)));
        let (value, stop, runs, matrix, nodes, root_chance, certificates, strategy) = if let Some(side) = prep.fixed_policy_side {
            assert!(side < 2 && prep.lower_threshold.is_none());
            let mut probe = child.clone();
            let legal = probe.legal_choices(dex,side);
            let priority = prep.fixed_priority_actions.as_ref().map_or_else(||legal.clone(),|names|
                names.iter().map(|name|*legal.iter().find(|&&action|action_input(dex,action)==*name)
                    .expect("descendant priority action must be legal at its root")).collect());
            assert!(!priority.is_empty());
            let priority_names: Vec<_> = priority.iter().map(|a|action_input(dex,*a)).collect();
            let mut dedicated = FixedPolicySolver::new(dex,dedicated_config,side,priority);
            if prep.fixed_sleep_talk { dedicated = dedicated.with_sleep_talk(); }
            if let Some(percent)=prep.fixed_rest_percent { dedicated=dedicated.with_rest_threshold(percent); }
            let report = dedicated.solve(child);
            let matrix = dedicated.restricted_root_payoffs(child).unwrap();
            let nodes = dedicated.node_count();
            let chance: Vec<_> = dedicated.root_chance_diagnostics(child).unwrap().iter()
                .map(|c|chance_diagnostic(c,dex)).collect();
            (report.unrestricted_game_value,report.stop,report.runs,matrix,nodes,chance,
                dedicated.into_value_certificates(),json!({"side":side,"priority_actions":priority_names,
                    "sleep_talk":prep.fixed_sleep_talk,
                    "rest_percent":prep.fixed_rest_percent,
                    "guaranteed_score":bound(report.guaranteed_score)}))
        } else {
            assert!(prep.fixed_priority_actions.is_none() && !prep.fixed_sleep_talk);
            assert!(prep.fixed_rest_percent.is_none());
            let mut dedicated = BoundSolver::new(dex,dedicated_config);
            let report = dedicated.solve(child,prep.lower_threshold.map(|t|(t,t)));
            let matrix = dedicated.root_payoffs(child).unwrap();
            let nodes = dedicated.node_count();
            let chance: Vec<_> = dedicated.root_chance_diagnostics(child).unwrap().iter()
                .map(|c|chance_diagnostic(c,dex)).collect();
            (report.bounds,report.stop,report.runs,matrix,nodes,chance,dedicated.into_value_certificates(),Value::Null)
        };
        let certificate_count = certificates.len();
        imported_certificates = solver.import_value_certificates(certificates);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        preparation_runs += enumeration_runs + runs;
        preparation_ms += elapsed_ms;
        descendant_preparation.push(json!({"semantic_key":prep.semantic_key,
            "quotient_quick_claw":prep.quotient_quick_claw,"successor_distribution":successor_distribution,
            "lower_threshold":prep.lower_threshold,"strategy":strategy,
            "actions":prep.actions,"mass":mass,"matching_leaves":matching.len(),
            "state":child.essence(dex),"enumeration_runs":enumeration_runs,"replacement":replacement,
            "one_sided_resource_class":format!("{:?}",nc2000_bot::stall::classify_one_sided_heal(child,dex)),
            "two_sided_resource_class":format!("{:?}",nc2000_bot::stall::classify_two_sided_heal(child,dex)),
            "quick_claw_roll":child.quick_claw_roll,
            "quick_claw_kernel":quick_claw_kernel,
            "value":bound(value),"stop":format!("{:?}",stop),
            "runs":runs,"nodes":nodes,"elapsed_ms":elapsed_ms,
            "certificates":certificate_count,"root_chance":root_chance,
            "matrix":{"actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|action_input(dex,a))).collect::<Vec<_>>()),
                "cells":matrix.cells.iter().copied().map(bound).collect::<Vec<_>>(),"value":bound(matrix.value)}}));
        eprintln!("descendant={} mass={mass} value=[{},{}] runs={} nodes={nodes}",
            prep.semantic_key, value.lo, value.hi, runs);
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
    assert!(protocol.trial_depth.is_none_or(|depth| depth > 0));
    assert!(protocol.descend_floor.is_none_or(|floor|floor.is_finite() && (0.0..=1.0).contains(&floor)));
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
        "crates/bot/src/hp_transition.rs",
        "crates/bot/examples/endgame_policy_audit.rs",
        "crates/bot/examples/endgame_policy_audit/hidden_selection.rs",
        "crates/engine/src/battle/search.rs",
        "crates/engine/src/battle/enumerate.rs",
        "crates/engine/src/battle/pokemon.rs",
        "crates/engine/src/battle/moveexec.rs",
        "crates/engine/src/battle/events.rs",
        "crates/engine/src/battle/conditions.rs",
        "crates/engine/src/battle/actions.rs",
        "crates/engine/src/battle/turn.rs",
        "crates/engine/src/battle/mod.rs",
        "crates/engine/src/state.rs",
        "crates/engine/src/prng.rs",
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
