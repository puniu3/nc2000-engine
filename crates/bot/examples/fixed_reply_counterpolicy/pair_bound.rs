use super::*;
use std::collections::{BTreeMap,BinaryHeap};
use nc2000_engine::battle::enumerate::enumerate_step;

#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    response: CounterResponse,
    width: usize,
    steps: usize,
    cell_cap: usize,
    work_budget: usize,
    lazy: bool,
    #[serde(default)]
    hp_storage_census: bool,
    #[serde(default)]
    guarded_keys: bool,
    #[serde(default)]
    packed_hp: bool,
    #[serde(default="state_cap")]
    packed_state_cap: usize,
    #[serde(default)]
    healing_capacity: usize,
}

fn state_cap()->usize {100_000}

type Frontier = BTreeMap<u128,(Battle,f64)>;

fn partial_step(dex:&Dex,battle:&Battle,choices:[Option<SearchChoice>;2],cap:usize,cache:&mut nc2000_bot::hp_transition::HpHealingCache)->(Frontier,f64,usize) {
    let mut pending=BinaryHeap::from([(1.0f64.to_bits(),Vec::<usize>::new())]);
    let mut leaves=Frontier::new();let mut runs=0;let mut observations=0;
    let mut runner=cache.prepare(dex,battle,choices);
    while observations<cap {
        let Some((mass,script))=pending.pop() else {break;};
        let (leaf,trace,executed)=runner.run(&script);runs+=usize::from(executed);observations+=1;
        let mut prefix=1.0;let mut siblings=0.0;
        for (p,draw) in trace.iter().enumerate() {
            if p>=script.len() {
                for (class,&count) in draw.counts.iter().enumerate() {
                    if class!=draw.chosen && count>0 {
                        let mut branch=trace[..p].iter().map(|draw|draw.chosen).collect::<Vec<_>>();
                        branch.push(class);
                        let probability=prefix*count as f64/(1u64<<32) as f64;
                        pending.push((probability.to_bits(),branch));siblings+=probability;
                    }
                }
            }
            prefix*=draw.prob();
        }
        assert!((prefix+siblings-f64::from_bits(mass)).abs()<1e-10);
        leaves.entry(leaf.state_key128()).and_modify(|(_,mass)|*mass+=prefix).or_insert((leaf,prefix));
    }
    let unresolved=pending.into_iter().map(|(mass,_)|f64::from_bits(mass)).sum();
    (leaves,unresolved,runs)
}

fn analyze(dex:&Dex,root:&Battle,cfg:&Config,
    mut choose:impl FnMut(&mut Battle,bool)->[Option<SearchChoice>;2])->Value {
    assert!(cfg.width>0 && cfg.width<=100_000 && cfg.steps>0 && cfg.steps<=2048 && cfg.cell_cap>0);
    assert!(cfg.healing_capacity<=100_000 && (cfg.healing_capacity==0 || cfg.lazy));
    let mut healing=nc2000_bot::hp_transition::HpHealingCache::new(cfg.healing_capacity);
    let started=Instant::now();
    assert!(cfg.packed_state_cap>0 && cfg.packed_state_cap<=1_000_000);
    let mut frontier=super::retained::Retained::new(cfg.packed_hp,cfg.guarded_keys);
    frontier.insert(dex,root.clone(),1.0);
    let (mut wins,mut ties,mut losses,mut dropped,mut failed)=(0.0,0.0,0.0,0.0,0.0);
    let mut runs=0;let mut rows=Vec::new();let mut peak_frontier=1;let mut peak_templates=1;
    for depth in 0..cfg.steps {
        let mut next=super::retained::Retained::new(cfg.packed_hp,cfg.guarded_keys);let mut failures=0;
        let mut census=super::hp_storage::Census::default();
        for (mut battle,mass) in frontier.into_states() {
            if let Some(outcome)=battle.outcome() {
                match outcome {Outcome::P1Win=>wins+=mass,Outcome::Tie=>ties+=mass,Outcome::P2Win=>losses+=mass};
                continue;
            }
            let cap=cfg.cell_cap.min(cfg.work_budget-runs);
            if cap==0 {failed+=mass;continue;}
            let choices=choose(&mut battle,depth==0);
            let (leaves,unresolved,work)=if cfg.lazy {
                partial_step(dex,&battle,choices,cap,&mut healing)
            } else if let Some(step)=enumerate_step(dex,&battle,choices,cap) {
                let mut leaves=Frontier::new();
                for leaf in step.leaves {
                    leaves.entry(leaf.battle.state_key128()).and_modify(|(_,mass)|*mass+=leaf.prob)
                        .or_insert((leaf.battle,leaf.prob));
                }
                (leaves,0.0,step.runs)
            } else {(Frontier::new(),1.0,cap)};
            runs+=work;failed+=mass*unresolved;failures+=usize::from(unresolved>0.0);
            let total:f64=leaves.values().map(|(_,prob)|prob).sum::<f64>()+unresolved;
            assert!((total-1.0).abs()<1e-10);
            for (_, (leaf,prob)) in leaves {
                let probability=mass*prob;
                if let Some(outcome)=leaf.outcome() {
                    match outcome {Outcome::P1Win=>wins+=probability,Outcome::Tie=>ties+=probability,Outcome::P2Win=>losses+=probability};
                } else {
                    if cfg.hp_storage_census {census.observe(dex,&leaf,probability,cfg.response);}
                    next.insert(dex,leaf,probability);
                }
            }
            peak_frontier=peak_frontier.max(next.len());
            peak_templates=peak_templates.max(next.templates());
            if next.templates()>cfg.width*2 || (cfg.packed_hp && next.len()>cfg.packed_state_cap*2) {dropped+=next.prune(cfg.width,cfg.packed_state_cap);}
        }
        dropped+=next.prune(cfg.width,cfg.packed_state_cap);
        let unresolved=next.mass();
        let unknown=dropped+failed+unresolved;
        let mass_error=(wins+ties+losses+unknown-1.0).abs();
        assert!(mass_error<1e-9);
        rows.push(json!({"depth":depth+1,"runs":runs,"wins":wins,"ties":ties,"losses":losses,
            "dropped_mass":dropped,"failed_mass":failed,"frontier_mass":unresolved,"unknown_mass":unknown,
            "frontier_states":next.len(),"frontier_templates":next.templates(),"enumeration_failures":failures,"mass_error":mass_error,
            "hp_storage_census":cfg.hp_storage_census.then(||census.report(cfg.width)),
            "score":{"lo":(wins+0.5*ties-1e-9).max(0.0),"hi":(wins+0.5*ties+unknown+1e-9).min(1.0)}}));
        frontier=next;
        if frontier.len()==0 || runs==cfg.work_budget {break;}
    }
    json!({"score":rows.last().unwrap()["score"],"layers":rows,"runs":runs,
        "peak_frontier":peak_frontier,"peak_templates":peak_templates,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
        "numerical_envelope":1e-9,"healing_cache":healing.stats()})
}

pub(super) fn execute(args:&[String]) {
    assert_eq!(args.len(),5,"corpus certify config.json output.json");
    let raw=std::fs::read(&args[1]).unwrap();
    let cfg:Config=serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();
    let game=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Game>(line).unwrap())
        .find(|game|game.game==17).unwrap();
    let dex=conformance::load_dex();let root=reconstruct(&dex,&game,21);
    let report=analyze(&dex,&root,&cfg,|b,first|[0,1].map(|side|
        choose_response(&dex,b,side,first,OwnPolicy::AfterBoom75,cfg.response)));
    let result=json!({"schema":"nc2000-fixed-pair-enumeration-v1","state_key":root.state_key128().to_string(),
        "config":cfg,"report":report,"manifest":manifest(&dex,&raw),
        "interpretation":"Side-zero score for two fixed full-state policies after fixed first Body Slam / switch-2 actions. Every expanded step enumerates engine chance outcomes. Discarded frontier mass, incomplete enumerations and the final frontier remain in [0,1]. Default state aggregation uses full state_key128; optional guarded keys omit only unobservable damage history and Quick Claw rolls. Packed storage keeps each exact HP vector and its mass separately under a shared guarded template, and applies the width limit to templates plus a separate concrete-state cap. No sampled value, heuristic terminal value or unrestricted-game certificate transfer is used. This pair's upper bound also bounds the fixed side-zero policy's worst-case score from above; it does not bound the unrestricted game value from above. Optional healing reuse admits only a strictly faster, below-full-HP Rest or restricted Sleep Talk user, and rejects changed HP-dependent control flow. Work counts actual engine executions; each cell still caps scripted observations. Floating-point probability arithmetic uses the reported numerical envelope."});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!("{}",serde_json::to_string(&result["report"]).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(width:usize,steps:usize,cap:usize,work:usize)->Config {
        Config {response:CounterResponse {return_when:ReturnWhen::Always,ice_first:false,finish_low:false},
            width,steps,cell_cap:cap,work_budget:work,lazy:false,hp_storage_census:false,guarded_keys:false,packed_hp:false,packed_state_cap:state_cap(),healing_capacity:0}
    }
    fn exhaustive(dex:&Dex,b:&Battle)->f64 {
        if let Some(outcome)=b.outcome() {return match outcome {Outcome::P1Win=>1.0,Outcome::Tie=>0.5,Outcome::P2Win=>0.0};}
        let mut b=b.clone();let choices=[0,1].map(|s|b.needs_choice()[s].then(||b.legal_choices(dex,s)[0]));
        enumerate_step(dex,&b,choices,100_000).unwrap().leaves.iter().map(|leaf|leaf.prob*exhaustive(dex,&leaf.battle)).sum()
    }
    #[test]
    fn truncated_mass_encloses_exhaustive_two_step_value() {
        let dex=conformance::load_dex();let mut b=super::super::tests::battle(&dex,false);
        b.turn=999;
        for side in 0..2 {let id=b.active_id(side).unwrap();b.poke_mut(id).hp=80;}
        let exact=exhaustive(&dex,&b);
        assert!(exact>0.0 && exact<1.0);
        for (lazy,packed) in [(false,false),(true,false),(false,true),(true,true)] {
        for mut cfg in [config(1,1,1,1),config(1,2,100_000,200_000),config(10_000,2,100_000,1_000_000)] {
            cfg.lazy=lazy;cfg.packed_hp=packed;cfg.guarded_keys=packed;
            let report=analyze(&dex,&b,&cfg,|b,_|[0,1].map(|s|b.needs_choice()[s].then(||b.legal_choices(&dex,s)[0])));
            assert!(report["score"]["lo"].as_f64().unwrap()<=exact);
            assert!(report["score"]["hi"].as_f64().unwrap()>=exact);
            assert!(report["runs"].as_u64().unwrap()<=cfg.work_budget as u64);
            if cfg.cell_cap==1 {assert!(report["layers"][0]["failed_mass"].as_f64().unwrap()>0.0);}
            if !packed && cfg.width==1 && cfg.cell_cap>1 {assert!(report["layers"][0]["dropped_mass"].as_f64().unwrap()>0.0);}
            if cfg.width==10_000 {assert!(report["score"]["hi"].as_f64().unwrap()-report["score"]["lo"].as_f64().unwrap()<3e-9);}
        }
    }
    }
    #[test]
    fn terminal_states_have_no_unknown_mass_or_engine_work() {
        let dex=conformance::load_dex();let mut b=super::super::tests::battle(&dex,false);
        b.turn=1000;
        let choices=[0,1].map(|s|b.legal_choices(&dex,s)[0]);b.apply_choices(&dex,choices.map(Some)).unwrap();
        assert!(b.outcome().is_some());
        let report=analyze(&dex,&b,&config(1,1,1,1),|_,_|panic!("terminal needs no policy"));
        assert_eq!(report["runs"],0);
        assert_eq!(report["layers"][0]["unknown_mass"],0.0);
    }
}
