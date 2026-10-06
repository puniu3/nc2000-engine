use nc2000_engine::{battle::{Outcome, PokemonSet, SearchChoice}, battle::enumerate::enumerate_step_partial, dex::Dex, state::Battle};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::BufRead};

type Key = (bool, bool, String);
#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>; 2] }
#[derive(Deserialize)]
struct Game { game: usize, seed: String, teams: [Vec<PokemonSet>; 2], decisions: Vec<Decision> }
#[derive(Deserialize)]
struct Step { choices: [Option<String>; 2], child: Key }
#[derive(Deserialize)]
struct Target { id: usize, cache_key: Key, path: Vec<Step> }
#[derive(Deserialize)]
struct Config {
    game: usize, decision: usize, guarded_keys: bool, target_side: usize, policy_side: usize,
    priority: Vec<String>, opponent_priority: BTreeMap<String, Vec<String>>,
    rest_at_percent: Option<i64>, finishing_hp: Option<i32>,
}
#[derive(Deserialize)]
struct Manifest {
    config: Config, targets: Vec<Target>, cap: usize,
    #[serde(default)]
    solve_budgets: Vec<usize>,
    #[serde(default)]
    solve_side: Option<usize>,
    #[serde(default)]
    solve_priority: Vec<String>,
    #[serde(default)]
    solve_node_budget: Option<usize>,
    #[serde(default)]
    solve_reply_order: Vec<String>,
}

fn key(dex: &Dex, b: &Battle, guarded: bool) -> Key {
    let damage=guarded && !b.damage_bookkeeping_observable(dex);
    let claw=guarded && !b.quick_claw_roll_observable(dex);
    let semantic=if claw {b.state_key128_without_quick_claw_roll(dex,damage)}
        else if damage {b.state_key128_without_damage_bookkeeping()} else {b.state_key128()};
    (damage,claw,semantic.to_string())
}

fn legal(dex: &Dex, b: &mut Battle) -> [Vec<Option<SearchChoice>>; 2] {
    [0,1].map(|s|if b.needs_choice()[s] {b.legal_choices(dex,s).into_iter().map(Some).collect()} else {vec![None]})
}

fn names(dex: &Dex, choices: &[Option<SearchChoice>]) -> Vec<Option<String>> {
    choices.iter().map(|c|c.map(|c|c.to_input(dex))).collect()
}

fn main() {
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4,"corpus manifest output");
    let raw=std::fs::read(&args[2]).unwrap();let m:Manifest=serde_json::from_slice(&raw).unwrap();let cfg=&m.config;
    assert!(cfg.target_side<2 && cfg.policy_side==1-cfg.target_side && m.cap>0);
    assert!(cfg.rest_at_percent.is_none() && cfg.finishing_hp.is_none());
    let dex=conformance::load_dex();
    let game=std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines()
        .map(|l|serde_json::from_str::<Game>(&l.unwrap()).unwrap()).find(|g|g.game==cfg.game).unwrap();
    let mut root=Battle::from_fixture(&dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();
    for d in game.decisions.iter().take(cfg.decision) {
        assert_eq!(root.state_key128().to_string(),d.key);
        let aa=[0,1].map(|s|root.legal_choices(&dex,s));
        root.apply_choices(&dex,[0,1].map(|s|d.choices[s].map(|i|aa[s][i]))).unwrap();
    }
    assert_eq!(root.state_key128().to_string(),game.decisions[cfg.decision].key);
    let mut results=Vec::new();let mut replay_runs=0;let mut probe_runs=0;
    for target in m.targets {
        assert!(!target.path.is_empty());let mut b=root.clone();let mut trace=Vec::new();
        for step in &target.path {
            let aa=legal(&dex,&mut b);
            let choices=[0,1].map(|s| {
                let index=names(&dex,&aa[s]).iter().position(|a|a==&step.choices[s]).unwrap();aa[s][index]
            });
            let found=enumerate_step_partial(&dex,&b,choices,m.cap);replay_runs+=found.runs;
            b=found.leaves.into_iter().find(|l|key(&dex,&l.battle,cfg.guarded_keys)==step.child)
                .expect("recorded child must be reachable under its guarded identity").battle;
            trace.push(json!({"runs":found.runs,"cache_key":key(&dex,&b,cfg.guarded_keys),
                "state_key":b.state_key128().to_string()}));
        }
        assert_eq!(key(&dex,&b,cfg.guarded_keys),target.cache_key);assert!(b.outcome().is_none());
        let aa=legal(&dex,&mut b);assert_eq!(key(&dex,&b,cfg.guarded_keys),target.cache_key);
        let mut selected=aa.clone();let side=cfg.policy_side;
        if aa[side]!=[None] {
            let contextual=b.active_id(1-side).and_then(|id|cfg.opponent_priority.get(dex.species.key(b.poke(id).species)));
            let labels=names(&dex,&aa[side]);
            let choice=contextual.into_iter().flatten().chain(&cfg.priority)
                .find_map(|name|labels.iter().position(|a|a.as_ref()==Some(name))).unwrap_or(0);
            selected[side]=vec![aa[side][choice]];
        }
        let mut probes=Vec::new();
        for &a in &selected[0] {for &c in &selected[1] {
            let choices=[a,c];let step=enumerate_step_partial(&dex,&b,choices,m.cap);probe_runs+=step.runs;
            let leaves:Vec<Value>=step.leaves.iter().map(|l| {
                let value=l.battle.outcome().map(|o|match o {
                    Outcome::Tie=>0.5,Outcome::P1Win=>f64::from(cfg.target_side==0),Outcome::P2Win=>f64::from(cfg.target_side==1),
                });
                json!({"mass":l.prob,"cache_key":key(&dex,&l.battle,cfg.guarded_keys),
                    "state_key":l.battle.state_key128().to_string(),"terminal":value})
            }).collect();
            probes.push(json!({"choices":choices.map(|a|a.map(|a|a.to_input(&dex))),"runs":step.runs,
                "pending":step.pending.iter().map(|p|p.mass).sum::<f64>(),"complete":!step.exhausted,
                "pending_scripts":step.pending.len(),"leaves":leaves}));
        }}
        let active=[0,1].map(|s|b.active_id(s).map(|id|json!({"species":dex.species.key(b.poke(id).species),
            "hp":b.poke(id).hp,"speed":b.poke(id).speed,"stored_stats":b.poke(id).stored_stats})));
        let mut result=json!({"id":target.id,"cache_key":target.cache_key,"trace":trace,"state":b.essence(&dex),
            "active":active,"legal":aa.each_ref().map(|a|names(&dex,a)),
            "selected":selected.each_ref().map(|a|names(&dex,a)),"probes":probes});
        if !m.solve_budgets.is_empty() {
            use nc2000_bot::bounds::{BoundConfig, FixedPolicySolver};
            assert_eq!((cfg.policy_side,cfg.target_side),(0,1));
            assert!((0..2).all(|s|b.sides[s].pokemon_left==1 && b.active_id(s).is_some_and(|id|b.poke(id).hp>0)));
            assert_eq!(active[0].as_ref().unwrap()["species"],json!("tyranitar"));
            let foe=active[1].as_ref().unwrap()["species"].as_str().unwrap();
            let foe_moves=match foe {
                "poliwrath"=>["hydropump","submission","return","bellydrum"],
                "snorlax"=>["bodyslam","earthquake","curse","selfdestruct"],
                _=>panic!("unsupported subgame opponent"),
            };
            for (s,moves) in [(0,["rockslide","crunch","fireblast","dynamicpunch"]),
                (1,foe_moves)] {
                assert!(b.poke(b.active_id(s).unwrap()).move_slots.iter().all(|slot|moves.contains(&dex.moves.key(slot.id))));
            }
            let solve_side=m.solve_side.unwrap_or(side);assert!(solve_side<2);
            assert!(!m.solve_priority.is_empty() || solve_side==side);
            let context=cfg.opponent_priority.get(dex.species.key(b.poke(b.active_id(1-side).unwrap()).species));
            let priority_names:Vec<_>=if m.solve_priority.is_empty() {context.into_iter().flatten().chain(&cfg.priority).cloned().collect()}
                else {m.solve_priority.clone()};
            let priority=priority_names.iter().filter_map(|name|name.strip_prefix("move "))
                .map(|name|SearchChoice::Move(dex.moves.id(name).unwrap())).collect();
            let mut rounds=Vec::new();
            let bound_cfg=BoundConfig {work_budget:m.solve_budgets[0],node_budget:m.solve_node_budget.unwrap_or(40000),eps:1e-6,
                dead_quick_claw_quotient:true,retain_eager_prefix:true,aggregate_lazy_children:true,
                chance_fair_period:8,..BoundConfig::default()};
            let mut solver=FixedPolicySolver::new(&dex,bound_cfg,solve_side,priority);
            for &budget in &m.solve_budgets {
                assert!(budget>0 && budget==m.solve_budgets[0]);
                if !m.solve_reply_order.is_empty() {
                    let name=&m.solve_reply_order[rounds.len()%m.solve_reply_order.len()];
                    let reply=SearchChoice::Move(dex.moves.id(name.strip_prefix("move ").unwrap()).unwrap());
                    let report=solver.solve_reply(&b,reply,None);let graph=solver.graph_diagnostics();
                    let matrix=solver.restricted_root_payoffs(&b).unwrap();
                    assert_eq!(matrix.actions[solve_side].len(),1);
                    let bounds=if solve_side==0 {[matrix.value.lo,matrix.value.hi]}
                        else {[1.0-matrix.value.hi,1.0-matrix.value.lo]};
                    let cells:Vec<_>=matrix.actions[1-solve_side].iter().zip(&matrix.cells)
                        .map(|(a,v)|json!({"reply":a.map(|a|a.to_input(&dex)),"side_zero_bounds":[v.lo,v.hi]})).collect();
                    rounds.push(json!({"budget":budget,"runs":report.runs,"total_runs":solver.stats().runs,
                        "fixed_side_bounds":bounds,"reply":name,"reply_side_zero_bounds":[report.bounds.lo,report.bounds.hi],
                        "root_cells":cells,"stop":format!("{:?}",report.stop),"nodes":solver.node_count(),
                        "unexpanded_nodes":graph.unexpanded_nodes}));
                    continue;
                }
                let report=solver.solve(&b);let graph=solver.graph_diagnostics();
                rounds.push(json!({"budget":budget,"runs":report.runs,"total_runs":solver.stats().runs,
                    "fixed_side_bounds":[report.guaranteed_score.lo,report.guaranteed_score.hi],
                    "stop":format!("{:?}",report.stop),"nodes":solver.node_count(),
                    "unexpanded_nodes":graph.unexpanded_nodes}));
            }
            result["subgame"]=json!(rounds);
            result["subgame_side"]=json!(solve_side);
            result["subgame_priority"]=json!(priority_names);
        }
        results.push(result);
    }
    let output=json!({"manifest_sha256":format!("{:x}",Sha256::digest(raw)),
        "corpus_sha256":format!("{:x}",Sha256::digest(std::fs::read(&args[1]).unwrap())),
        "root_state_key":root.state_key128().to_string(),"replay_runs":replay_runs,"probe_runs":probe_runs,"targets":results});
    std::fs::write(&args[3],serde_json::to_vec(&output).unwrap()).unwrap();
    eprintln!("targets={} replay_runs={replay_runs} probe_runs={probe_runs}",output["targets"].as_array().unwrap().len());
}
