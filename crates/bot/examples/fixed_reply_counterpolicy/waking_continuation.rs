use super::*;
use nc2000_bot::bounds::{BoundConfig,BoundSolver,FixedPolicySolver};
use std::collections::HashSet;
use std::time::Duration;

#[derive(Clone,Copy,Default,PartialEq,Eq,Deserialize,Serialize)]
#[serde(rename_all="snake_case")]
enum Selection { #[default] First,LowPp }

#[derive(Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    samples:usize,
    seed:u64,
    positions:usize,
    work_budget:usize,
    time_budget_ms:u64,
    #[serde(default)]
    selection:Selection,
    #[serde(default)]
    unrestricted_work_budget:usize,
}

struct Fork {battle:Battle,trial:usize,prefix:Vec<[Option<SearchChoice>;2]>}

fn live_pp(b:&Battle)->i64 {
    b.sides.iter().map(|side|side.party.iter().map(|&slot|&side.roster[slot as usize])
        .filter(|p|p.hp>0 && !p.fainted).flat_map(|p|p.move_slots.iter()).map(|slot|i64::from(slot.pp)).sum::<i64>()).sum()
}

fn collect(dex:&Dex,root:&Battle,cfg:&Config)->(Vec<Fork>,usize) {
    let response=CounterResponse {return_when:ReturnWhen::SelfLow,ice_first:false,finish_low:true};
    let mut forks=Vec::new();let mut keys=HashSet::new();let mut divergent_trials=0;
    for trial in 0..cfg.samples {
        let mut b=root.clone();b.set_log_enabled(false);b.reseed(seed(cfg.seed,trial));
        let mut prefix=Vec::new();
        let mut candidate:Option<Fork>=None;
        while b.outcome().is_none() {
            assert!(prefix.len()<2048);
            let first=prefix.is_empty();
            let old=choose_response(dex,&mut b,0,first,OwnPolicy::AfterBoom75,response);
            let new=choose_response(dex,&mut b,0,first,OwnPolicy::AfterBoomWake75,response);
            if old!=new {
                assert_eq!(new,Some(SearchChoice::Move(dex.moves.id("rest").unwrap())));
                if candidate.as_ref().is_none_or(|saved|live_pp(&b)<live_pp(&saved.battle)) {
                    candidate=Some(Fork {battle:b.clone(),trial,prefix:prefix.clone()});
                }
                if cfg.selection==Selection::First {break;}
            }
            let other=choose_response(dex,&mut b,1,first,OwnPolicy::AfterBoom75,response);
            let pair=[if cfg.selection==Selection::LowPp {new} else {old},other];
            b.apply_choices(dex,pair).unwrap();prefix.push(pair);
        }
        if let Some(candidate)=candidate {
            divergent_trials+=1;
            if keys.insert(candidate.battle.state_key128()) {
                forks.push(candidate);
                if cfg.selection==Selection::LowPp {forks.sort_by_key(|fork|(live_pp(&fork.battle),fork.trial));}
                forks.truncate(cfg.positions);
            }
        }
    }
    (forks,divergent_trials)
}

fn replay(dex:&Dex,root:&Battle,cfg:&Config,fork:&Fork)->Battle {
    let mut b=root.clone();b.set_log_enabled(false);b.reseed(seed(cfg.seed,fork.trial));
    for &pair in &fork.prefix {b.apply_choices(dex,pair).unwrap();}
    assert_eq!(b.state_key128(),fork.battle.state_key128());
    assert_eq!(b.essence(dex),fork.battle.essence(dex));
    b
}

fn solve(dex:&Dex,b:&Battle,cfg:&Config,waking:bool)->Value {
    let config=BoundConfig {
        work_budget:cfg.work_budget,time_budget:Some(Duration::from_millis(cfg.time_budget_ms)),
        node_budget:120000,cell_cap:4096,eps:0.01,certified_action_pruning:true,
        chance_fair_period:32,aggregate_lazy_children:true,resource_closure_nodes:Some(90000),
        dead_quick_claw_quotient:true,chance_material_order:true,..BoundConfig::default()
    };
    let priority=["bodyslam","fireblast","rest","selfdestruct"].map(|name|SearchChoice::Move(dex.moves.id(name).unwrap())).to_vec();
    let mut solver=FixedPolicySolver::new(dex,config.clone(),0,priority).with_rest_threshold(75).with_rest_after_boom();
    if waking {solver=solver.with_rest_on_wake();}
    let started=Instant::now();let report=solver.solve(b);let elapsed_ms=started.elapsed().as_secs_f64()*1000.0;
    let matrix=solver.restricted_root_payoffs(b).unwrap();
    let nodes=solver.node_count();let waking_nodes=solver.stats().fixed_waking_rest_nodes;
    let certificates=solver.into_value_certificates();let count=certificates.len();
    let mut target=BoundSolver::new(dex,BoundConfig {work_budget:0,..config});
    assert_eq!(target.import_value_certificates(certificates),count);
    let imported=target.solve(b,None);
    assert!(imported.bounds.lo+1e-12>=report.guaranteed_score.lo);
    assert_eq!(imported.bounds.hi,1.0);
    json!({"waking":waking,"guaranteed_score":{"lo":report.guaranteed_score.lo,"hi":report.guaranteed_score.hi},
        "runs":report.runs,"stop":format!("{:?}",report.stop),"nodes":nodes,"elapsed_ms":elapsed_ms,
        "waking_rest_nodes":waking_nodes,"certificates":count,"import_hits":target.stats.certificate_hits,
        "imported_unrestricted_value":{"lo":imported.bounds.lo,"hi":imported.bounds.hi},
        "root_actions":matrix.actions.map(|actions|actions.into_iter().map(|a|a.map(|a|nc2000_bot::player::action_input(dex,a))).collect::<Vec<_>>()),
        "root_cells":matrix.cells.into_iter().map(|v|json!({"lo":v.lo,"hi":v.hi})).collect::<Vec<_>>()})
}

fn solve_unrestricted(dex:&Dex,b:&Battle,cfg:&Config)->Value {
    let mut solver=BoundSolver::new(dex,BoundConfig {
        work_budget:cfg.unrestricted_work_budget,time_budget:Some(Duration::from_millis(cfg.time_budget_ms)),
        node_budget:120000,cell_cap:4096,eps:0.001,certified_action_pruning:true,
        chance_fair_period:32,aggregate_lazy_children:true,dead_quick_claw_quotient:true,
        ..BoundConfig::default()
    });
    let rest=Some(SearchChoice::Move(dex.moves.id("rest").unwrap()));
    let legal=b.clone().legal_choices(dex,0);
    let policy=legal.iter().map(|&a|if Some(a)==rest {1.0} else {0.0}).collect::<Vec<_>>();
    let started=Instant::now();let report=solver.solve_policy_losses(b,&[(0,policy.clone())],0.001);
    let matrix=solver.root_payoffs(b).unwrap();
    let loss=matrix.policy_loss(0,&policy);
    json!({"value":{"lo":report.bounds.lo,"hi":report.bounds.hi},"rest_loss":{"lo":loss.lo,"hi":loss.hi},
        "runs":report.runs,"stop":format!("{:?}",report.stop),"nodes":solver.node_count(),
        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
        "root_actions":matrix.actions.map(|actions|actions.into_iter().map(|a|a.map(|a|nc2000_bot::player::action_input(dex,a))).collect::<Vec<_>>()),
        "root_cells":matrix.cells.into_iter().map(|v|json!({"lo":v.lo,"hi":v.hi})).collect::<Vec<_>>()})
}

pub(super) fn execute(args:&[String]) {
    assert_eq!(args.len(),5,"corpus continuations config output");
    let cfg:Config=serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();
    assert!(cfg.samples>0 && cfg.positions>0 && cfg.time_budget_ms>0 && cfg.work_budget>0);
    let raw=std::fs::read(&args[1]).unwrap();let dex=conformance::load_dex();
    let game=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Game>(line).unwrap())
        .find(|game|game.game==17).unwrap();
    let root=reconstruct(&dex,&game,21);
    let (forks,divergent_trials)=collect(&dex,&root,&cfg);assert!(!forks.is_empty());
    let mut positions=Vec::new();
    for fork in forks {
        let b=replay(&dex,&root,&cfg,&fork);
        let state=json!({"key":b.state_key128().to_string(),"trial":fork.trial,"turn":b.turn,"live_pp":live_pp(&b),"essence":b.essence(&dex),
            "prefix":fork.prefix.iter().map(|pair|pair.map(|a|a.map(|a|nc2000_bot::player::action_input(&dex,a)))).collect::<Vec<_>>()});
        let mut arms=Vec::new();
        for waking in [false,true] {
            let arm=solve(&dex,&b,&cfg,waking);
            eprintln!("trial={} waking={} value={} runs={}",fork.trial,waking,arm["guaranteed_score"],arm["runs"]);
            arms.push(arm);
        }
        let optimal=(cfg.unrestricted_work_budget>0).then(||solve_unrestricted(&dex,&b,&cfg));
        if let Some(optimal)=&optimal {eprintln!("trial={} unrestricted={}",fork.trial,optimal);}
        positions.push(json!({"state":state,"arms":arms,"unrestricted":optimal}));
    }
    let record=json!({"schema":"nc2000-waking-continuations-v1","config":cfg,"manifest":manifest(&dex,&raw),
        "root_key":root.state_key128().to_string(),"divergent_trials":divergent_trials,"positions":positions,
        "interpretation":"Selection first takes the first distinct first-divergence states in baseline-policy rollouts. Selection low_pp follows the waking policy to termination, retains each trial's divergence with least living selected-party PP, then takes the globally lowest-PP distinct states with trial-order ties. Both use the concrete self-low / finish-low response only to find reachable states. Selected states are replayed exactly, then each continuation policy is bounded against all future legal opponent choices. Sampling selects states only, never their certified probabilities or values. Projected lower certificates are imported into an unrestricted zero-work solver at each same state; no upper fixed-policy bound transfers. This does not establish a root-game guarantee: coverage and propagation back to the original root remain necessary."});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&record).unwrap()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replayed_waking_fork_retains_both_opponent_actions_and_conservative_import() {
        let dex=conformance::load_dex();let mut b=crate::tests::battle(&dex,false);
        let own=b.active_id(0).unwrap();b.poke_mut(own).hp=b.poke(own).maxhp/2;
        b.poke_mut(own).status=Status::Slp;b.poke_mut(own).status_state.set_int(DK::Time,1);
        b.refresh_poke_mask(&dex,own);b.turn=1000;
        let cfg=Config {samples:1,seed:1,positions:1,work_budget:100000,time_budget_ms:60000,selection:Selection::First,unrestricted_work_budget:100000};
        b.reseed(seed(cfg.seed,0));
        let fork=Fork {battle:b.clone(),trial:0,prefix:vec![]};let restored=replay(&dex,&b,&cfg,&fork);
        let legal=restored.clone().legal_choices(&dex,1).into_iter()
            .map(|a|json!(nc2000_bot::player::action_input(&dex,a))).collect::<Vec<_>>();
        for waking in [false,true] {
            let arm=solve(&dex,&restored,&cfg,waking);
            assert_eq!(arm["root_actions"][1],json!(legal));
            assert_eq!(arm["root_actions"][0],json!([if waking {"move rest"} else {"move bodyslam"}]));
            assert!(arm["guaranteed_score"]["hi"].as_f64().unwrap()-arm["guaranteed_score"]["lo"].as_f64().unwrap()<=0.01);
        }
        let optimal=solve_unrestricted(&dex,&restored,&cfg);
        assert_eq!(optimal["root_actions"][1],json!(legal));
        assert_eq!(optimal["root_actions"][0].as_array().unwrap().len(),restored.clone().legal_choices(&dex,0).len());
        assert!(optimal["value"]["hi"].as_f64().unwrap()-optimal["value"]["lo"].as_f64().unwrap()<=0.001);
    }
}
