use super::*;
use std::cmp::Reverse;
use std::collections::HashMap;
use super::super::retained::{key,Key as StateKey};

type Key=(usize,StateKey);
struct Cell {shape:StateKey,hp:Vec<i32>,mass:f64}
struct Template {battle:Battle,users:usize}

#[derive(Default)]
struct Agenda {
    cells:BTreeMap<Key,Cell>,
    templates:HashMap<StateKey,Template>,
    heap:BinaryHeap<(u64,Reverse<Key>)>,
    merges:usize,
    heap_rebuilds:usize,
}

impl Agenda {
    fn insert(&mut self,dex:&Dex,depth:usize,b:Battle,mass:f64) {
        assert!(mass.is_finite() && mass>=0.0);
        if mass==0.0 {return;}
        let state=(depth,key(dex,&b,true));
        let mass=if let Some(cell)=self.cells.get_mut(&state) {
            cell.mass+=mass;self.merges+=1;cell.mass
        } else {
            let mut template=b;
            let hp=template.sides.iter().flat_map(|s|s.roster.iter().map(|p|p.hp)).collect();
            for side in &mut template.sides {for p in &mut side.roster {if p.hp>0 {p.hp=p.maxhp;}}}
            let shape=key(dex,&template,true);
            self.templates.entry(shape).and_modify(|t|t.users+=1).or_insert(Template {battle:template,users:1});
            self.cells.insert(state,Cell {shape,hp,mass});mass
        };
        self.heap.push((mass.to_bits(),Reverse(state)));
        if self.heap.len()>self.cells.len()*4+1024 {
            self.heap=self.cells.iter().map(|(key,cell)|(cell.mass.to_bits(),Reverse(*key))).collect();self.heap_rebuilds+=1;
        }
    }
    fn pop(&mut self)->Option<(usize,Battle,f64)> {
        while let Some((mass,Reverse(key)))=self.heap.pop() {
            if self.cells.get(&key).is_none_or(|cell|cell.mass.to_bits()!=mass) {continue;}
            let cell=self.cells.remove(&key).unwrap();
            let template=self.templates.get_mut(&cell.shape).unwrap();let mut battle=template.battle.clone();
            template.users-=1;if template.users==0 {self.templates.remove(&cell.shape);}
            let mut hp=cell.hp.into_iter();
            for side in &mut battle.sides {for p in &mut side.roster {p.hp=hp.next().unwrap();}}
            assert!(hp.next().is_none());return Some((key.0,battle,cell.mass));
        }
        assert!(self.cells.is_empty());None
    }
    fn mass(&self)->f64 {self.cells.values().map(|cell|cell.mass).sum()}
}

#[derive(Default)]
struct Totals {wins:f64,ties:f64,losses:f64,failed:f64,horizon:f64,runs:usize,expansions:usize,max_depth:usize}

impl Totals {
    fn terminal(&mut self,outcome:Outcome,mass:f64) {
        match outcome {Outcome::P1Win=>self.wins+=mass,Outcome::Tie=>self.ties+=mass,Outcome::P2Win=>self.losses+=mass}
    }
    fn checkpoint(&self,agenda:&Agenda)->Value {
        let frontier=agenda.mass();let unknown=self.failed+self.horizon+frontier;
        let error=(self.wins+self.ties+self.losses+unknown-1.0).abs();assert!(error<1e-9);
        json!({"expansions":self.expansions,"runs":self.runs,"max_depth":self.max_depth,
            "wins":self.wins,"ties":self.ties,"losses":self.losses,"failed_mass":self.failed,"horizon_mass":self.horizon,
            "frontier_mass":frontier,"unknown_mass":unknown,"frontier_states":agenda.cells.len(),"frontier_templates":agenda.templates.len(),
            "mass_error":error,"score":{"lo":(self.wins+0.5*self.ties-1e-9).max(0.0),"hi":(self.wins+0.5*self.ties+unknown+1e-9).min(1.0)}})
    }
}

pub(super) fn analyze(dex:&Dex,root:&Battle,cfg:&Config,
    mut choose:impl FnMut(&mut Battle,bool)->[Option<SearchChoice>;2],started:Instant,
    mut stop:impl FnMut()->bool)->Value {
    assert!(cfg.lazy && cfg.packed_hp && cfg.guarded_keys && !cfg.hp_storage_census);
    let mut agenda=Agenda::default();let mut totals=Totals::default();
    if let Some(outcome)=root.outcome() {totals.terminal(outcome,1.0);} else {agenda.insert(dex,0,root.clone(),1.0);}
    let mut healing=nc2000_bot::hp_transition::HpHealingCache::new(cfg.healing_capacity);
    if cfg.reuse_attacks {healing=healing.with_attacks();}
    if cfg.reuse_switches {healing=healing.with_switches();}
    let mut kernels=super::super::switch_kernel::Cache::new(cfg.switch_kernel_capacity);
    let mut boom=BoomStats::default();let mut rows=Vec::new();let mut depths=vec![0usize;cfg.steps];
    let mut peak_frontier=agenda.cells.len();let mut peak_templates=agenda.templates.len();
    let mut time_exhausted=false;let mut memory_exhausted=false;
    while !agenda.cells.is_empty() && totals.runs<cfg.work_budget {
        time_exhausted=stop();
        memory_exhausted=agenda.cells.len()>cfg.packed_state_cap || agenda.templates.len()>cfg.width;
        if time_exhausted || memory_exhausted {break;}
        let (depth,mut battle,mass)=agenda.pop().unwrap();assert!(depth<cfg.steps);
        let choices=choose(&mut battle,depth==0);
        let mut eager_work=0;
        let eager=if cfg.boom_eager_cap>0 && super::super::terminal_step::admitted(dex,&battle,choices) {
            boom.attempts+=1;
            let (result,work,replacements)=super::super::terminal_step::enumerate(dex,&battle,choices,
                cfg.boom_eager_cap.min(cfg.work_budget-totals.runs),cfg.boom_terminal_quotient);
            eager_work=work;boom.runs+=work;boom.terminal_replacements+=replacements;
            boom.completed+=usize::from(result.is_some());result
        } else {None};
        let (leaves,unknown,work)=if let Some(step)=eager {(collect_step(step),0.0,eager_work)} else {
            let cap=cfg.cell_cap.min(cfg.work_budget-totals.runs-eager_work);
            let (leaves,unknown,work)=kernels.run(dex,&battle,choices,cap,&mut healing);(leaves,unknown,work+eager_work)
        };
        assert!((leaves.values().map(|(_,prob)|prob).sum::<f64>()+unknown-1.0).abs()<1e-10);
        totals.runs+=work;totals.failed+=mass*unknown;totals.expansions+=1;totals.max_depth=totals.max_depth.max(depth+1);depths[depth]+=1;
        for (_, (leaf,prob)) in leaves {
            let probability=mass*prob;
            if let Some(outcome)=leaf.outcome() {totals.terminal(outcome,probability);}
            else if depth+1==cfg.steps {totals.horizon+=probability;}
            else {agenda.insert(dex,depth+1,leaf,probability);}
        }
        peak_frontier=peak_frontier.max(agenda.cells.len());peak_templates=peak_templates.max(agenda.templates.len());
        if totals.expansions%128==0 {rows.push(totals.checkpoint(&agenda));}
    }
    let last=totals.checkpoint(&agenda);
    if rows.last()!=Some(&last) {rows.push(last.clone());}
    json!({"schedule":"best_first","score":last["score"],"checkpoints":rows,"runs":totals.runs,
        "expansions":totals.expansions,"expansions_by_depth":depths,"max_depth":totals.max_depth,
        "peak_frontier":peak_frontier,"peak_templates":peak_templates,"merges":agenda.merges,"heap_rebuilds":agenda.heap_rebuilds,
        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"time_exhausted":time_exhausted,"memory_exhausted":memory_exhausted,
        "numerical_envelope":1e-9,"healing_cache":healing.stats(),"boom_eager":boom,"switch_kernels":kernels.stats})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn agenda_preserves_mass_hp_depth_and_later_arrivals() {
        let dex=conformance::load_dex();let mut base=crate::tests::battle(&dex,false);
        let id=base.active_id(0).unwrap();base.poke_mut(id).hp=130;
        let mut other=base.clone();other.poke_mut(id).hp=70;
        let mut agenda=Agenda::default();
        agenda.insert(&dex,2,base.clone(),0.125);agenda.insert(&dex,2,base.clone(),0.125);
        agenda.insert(&dex,1,base.clone(),0.25);agenda.insert(&dex,4,other.clone(),0.5);
        assert_eq!(agenda.cells.len(),3);assert_eq!(agenda.templates.len(),1);assert_eq!(agenda.mass(),1.0);
        for (depth,b,mass) in [(4,other,0.5),(1,base.clone(),0.25),(2,base.clone(),0.25)] {
            let (found,restored,weight)=agenda.pop().unwrap();assert_eq!((found,weight),(depth,mass));assert_eq!(restored.state_key128(),b.state_key128());
        }
        assert!(agenda.pop().is_none());assert!(agenda.templates.is_empty());
        for _ in 0..2048 {agenda.insert(&dex,2,base.clone(),1.0/2048.0);}
        assert!(agenda.heap_rebuilds>0);assert_eq!(agenda.mass(),1.0);
        let (depth,b,mass)=agenda.pop().unwrap();assert_eq!((depth,mass),(2,1.0));assert_eq!(b.state_key128(),base.state_key128());
        assert!(agenda.pop().is_none());assert!(agenda.templates.is_empty());
    }
    #[test]
    fn best_first_bounds_enclose_exhaustive_value_at_every_stop() {
        let dex=conformance::load_dex();let mut b=crate::tests::battle(&dex,false);b.turn=999;
        for side in 0..2 {let id=b.active_id(side).unwrap();b.poke_mut(id).hp=80;}
        let exact=super::super::tests::exhaustive(&dex,&b);assert!(exact>0.0 && exact<1.0);
        for (cap,work,states,steps,cutoff) in [(1,1,100000,2,usize::MAX),(50,500,100000,2,usize::MAX),
            (100000,1000000,100000,2,usize::MAX),(50,1000000,1,2,usize::MAX),
            (50,1000000,100000,2,0),(50,1000000,100000,2,2),(100000,1000000,100000,1,usize::MAX)] {
            let mut cfg=super::super::tests::config(1000,steps,cap,work);
            cfg.best_first=true;cfg.lazy=true;cfg.guarded_keys=true;cfg.packed_hp=true;cfg.packed_state_cap=states;
            let mut calls=0;
            let report=super::super::analyze_until(&dex,&b,&cfg,|b,_|[0,1].map(|s|b.needs_choice()[s].then(||b.legal_choices(&dex,s)[0])),Instant::now(),||{calls+=1;calls>cutoff});
            for row in report["checkpoints"].as_array().unwrap() {
                assert!(row["score"]["lo"].as_f64().unwrap()<=exact);assert!(row["score"]["hi"].as_f64().unwrap()>=exact);
                let mass=["wins","ties","losses","failed_mass","horizon_mass","frontier_mass"].iter().map(|k|row[k].as_f64().unwrap()).sum::<f64>();
                assert!((mass-1.0).abs()<1e-9);
            }
            assert!(report["runs"].as_u64().unwrap()<=work as u64);
            assert_eq!(report["runs"],report["healing_cache"]["runs"]);
            if states==1 {assert_eq!(report["memory_exhausted"],true);}
            if cutoff==0 {assert_eq!(report["time_exhausted"],true);assert_eq!(report["runs"],0);}
            if cap==100000 && steps==2 {assert!(report["score"]["hi"].as_f64().unwrap()-report["score"]["lo"].as_f64().unwrap()<3e-9);}
            if steps==1 {assert!(report["checkpoints"].as_array().unwrap().last().unwrap()["horizon_mass"].as_f64().unwrap()>0.0);}
        }
    }
}
