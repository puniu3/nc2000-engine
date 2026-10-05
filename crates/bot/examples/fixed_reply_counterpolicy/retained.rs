use super::*;
use std::collections::{BTreeMap,BTreeSet};

pub(super) type Key=(bool,bool,u128);

pub(super) fn key(dex:&Dex,b:&Battle,guarded:bool)->Key {
    let damage=guarded && !b.damage_bookkeeping_observable(dex);
    let claw=guarded && !b.quick_claw_roll_observable(dex);
    let hash=if claw {b.state_key128_without_quick_claw_roll(dex,damage)}
        else if damage {b.state_key128_without_damage_bookkeeping()} else {b.state_key128()};
    (damage,claw,hash)
}

struct Cell { shape:Key,hp:Vec<i32>,mass:f64 }

pub(super) struct Retained {
    guarded:bool,
    packed:bool,
    full:BTreeMap<Key,(Battle,f64)>,
    templates:BTreeMap<Key,Battle>,
    cells:BTreeMap<Key,Cell>,
}

impl Retained {
    pub(super) fn new(packed:bool,guarded:bool)->Self {
        assert!(!packed || guarded);
        Self {guarded,packed,full:BTreeMap::new(),templates:BTreeMap::new(),cells:BTreeMap::new()}
    }
    pub(super) fn insert(&mut self,dex:&Dex,b:Battle,mass:f64) {
        let state=key(dex,&b,self.guarded);
        if !self.packed {
            self.full.entry(state).and_modify(|(_,p)|*p+=mass).or_insert((b,mass));return;
        }
        if let Some(cell)=self.cells.get_mut(&state) {cell.mass+=mass;return;}
        let mut template=b;
        let hp=template.sides.iter().flat_map(|s|s.roster.iter().map(|p|p.hp)).collect();
        for side in &mut template.sides {for p in &mut side.roster {if p.hp>0 {p.hp=p.maxhp;}}}
        let shape=key(dex,&template,self.guarded);
        self.templates.entry(shape).or_insert(template);
        self.cells.insert(state,Cell {shape,hp,mass});
    }
    pub(super) fn len(&self)->usize {if self.packed {self.cells.len()} else {self.full.len()}}
    pub(super) fn templates(&self)->usize {if self.packed {self.templates.len()} else {self.full.len()}}
    pub(super) fn mass(&self)->f64 {
        if self.packed {self.cells.values().map(|c|c.mass).sum()} else {self.full.values().map(|(_,mass)|mass).sum()}
    }
    pub(super) fn prune(&mut self,width:usize,state_cap:usize)->f64 {
        if !self.packed {
            if self.full.len()<=width {return 0.0;}
            let mut ordered=std::mem::take(&mut self.full).into_iter().collect::<Vec<_>>();
            ordered.sort_by(|a,b|b.1.1.total_cmp(&a.1.1).then_with(||a.0.cmp(&b.0)));
            let dropped=ordered[width..].iter().map(|(_,(_,mass))|mass).sum();
            self.full.extend(ordered.into_iter().take(width));return dropped;
        }
        let mut dropped=0.0;
        if self.templates.len()>width {
            let mut masses=BTreeMap::<Key,f64>::new();
            for cell in self.cells.values() {*masses.entry(cell.shape).or_default()+=cell.mass;}
            let mut ordered=masses.into_iter().collect::<Vec<_>>();
            ordered.sort_by(|a,b|b.1.total_cmp(&a.1).then_with(||a.0.cmp(&b.0)));
            let keep=ordered.into_iter().take(width).map(|(key,_)|key).collect::<BTreeSet<_>>();
            self.cells.retain(|_,cell|{if keep.contains(&cell.shape) {true} else {dropped+=cell.mass;false}});
        }
        if self.cells.len()>state_cap {
            let mut ordered=self.cells.iter().map(|(key,cell)|(*key,cell.mass)).collect::<Vec<_>>();
            ordered.sort_by(|a,b|b.1.total_cmp(&a.1).then_with(||a.0.cmp(&b.0)));
            let keep=ordered.into_iter().take(state_cap).map(|(key,_)|key).collect::<BTreeSet<_>>();
            self.cells.retain(|key,cell|{if keep.contains(key) {true} else {dropped+=cell.mass;false}});
        }
        let used=self.cells.values().map(|cell|cell.shape).collect::<BTreeSet<_>>();
        self.templates.retain(|key,_|used.contains(key));dropped
    }
    pub(super) fn sorted_masses(&self)->Vec<f64> {
        let mut masses=if self.packed {self.cells.values().map(|c|c.mass).collect::<Vec<_>>()} else {self.full.values().map(|(_,mass)|*mass).collect()};
        masses.sort_by(|a,b|b.total_cmp(a));masses
    }
    pub(super) fn into_states(self,mass_order:bool)->Box<dyn Iterator<Item=(Battle,f64)>> {
        if !self.packed {
            if !mass_order {return Box::new(self.full.into_values());}
            let mut cells=self.full.into_iter().collect::<Vec<_>>();
            cells.sort_by(|a,b|b.1.1.total_cmp(&a.1.1).then_with(||a.0.cmp(&b.0)));
            return Box::new(cells.into_iter().map(|(_,cell)|cell));
        }
        let templates=self.templates;
        let cells:Box<dyn Iterator<Item=Cell>>=if mass_order {
            let mut cells=self.cells.into_iter().collect::<Vec<_>>();
            cells.sort_by(|a,b|b.1.mass.total_cmp(&a.1.mass).then_with(||a.0.cmp(&b.0)));
            Box::new(cells.into_iter().map(|(_,cell)|cell))
        } else {Box::new(self.cells.into_values())};
        Box::new(cells.map(move|cell| {
            let mut b=templates[&cell.shape].clone();
            let mut hp=cell.hp.into_iter();
            for side in &mut b.sides {for p in &mut side.roster {p.hp=hp.next().unwrap();}}
            assert!(hp.next().is_none());(b,cell.mass)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::enumerate::run_scripted;
    #[test]
    fn probability_order_preserves_every_state_and_tie_order() {
        let dex=conformance::load_dex();let base=super::super::tests::battle(&dex,false);
        for packed in [false,true] {
            let mut retained=Retained::new(packed,true);let mut expected=Vec::new();
            for (hp,mass) in [(1,0.1),(70,0.2),(130,0.2),(200,0.5)] {
                let mut b=base.clone();let id=b.active_id(0).unwrap();b.poke_mut(id).hp=hp;
                expected.push((key(&dex,&b,true),b.state_key128(),mass));retained.insert(&dex,b,mass);
            }
            expected.sort_by(|a,b|b.2.total_cmp(&a.2).then_with(||a.0.cmp(&b.0)));
            assert_eq!(retained.sorted_masses(),vec![0.5,0.2,0.2,0.1]);
            let actual=retained.into_states(true).map(|(b,mass)|(key(&dex,&b,true),b.state_key128(),mass)).collect::<Vec<_>>();
            assert_eq!(actual,expected);
        }
    }
    #[test]
    fn packed_states_preserve_hp_actions_and_scripted_successors() {
        let dex=conformance::load_dex();let mut flat=Retained::new(false,true);let mut packed=Retained::new(true,true);
        let base=super::super::tests::battle(&dex,false);
        let response=CounterResponse {return_when:ReturnWhen::SelfLow,ice_first:false,finish_low:true};
        for hp0 in [1,70,130,200,250] {for hp1 in [1,70,130,200,250] {
            let mut b=base.clone();
            for (side,hp) in [hp0,hp1].into_iter().enumerate() {let id=b.active_id(side).unwrap();b.poke_mut(id).hp=hp.min(b.poke(id).maxhp);}
            b.quick_claw_roll=hp0%2==0;b.last_damage=i64::from(hp0);
            flat.insert(&dex,b.clone(),0.04);packed.insert(&dex,b,0.04);
        }}
        assert_eq!(packed.len(),25);assert_eq!(packed.templates(),1);
        let before=packed.mass();assert_eq!(packed.prune(1,100),0.0);assert_eq!(packed.mass(),before);
        for ((mut a,pa),(mut b,pb)) in flat.into_states(false).zip(packed.into_states(false)) {
            assert_eq!(pa,pb);assert_eq!(key(&dex,&a,true),key(&dex,&b,true));
            let aa=[0,1].map(|s|choose_response(&dex,&mut a,s,false,OwnPolicy::AfterBoom75,response));
            let ab=[0,1].map(|s|choose_response(&dex,&mut b,s,false,OwnPolicy::AfterBoom75,response));
            assert_eq!(aa,ab);
            for script in [vec![],vec![1],vec![0,1],vec![1,0,1]] {
                let (a,ta)=run_scripted(&dex,&a,aa,&script);let (b,tb)=run_scripted(&dex,&b,ab,&script);
                assert_eq!(key(&dex,&a,true),key(&dex,&b,true));
                assert_eq!(format!("{ta:?}"),format!("{tb:?}"));
            }
        }
    }
    #[test]
    fn packing_keeps_observable_damage_history_and_pruned_mass() {
        let dex=conformance::load_dex();let mut base=super::super::tests::battle(&dex,false);
        let id=base.active_id(0).unwrap();base.poke_mut(id).move_slots[0].id=dex.moves.id("counter").unwrap();
        let mut packed=Retained::new(true,true);
        for (hp,damage) in [(70,1),(130,2),(200,3)] {
            let mut b=base.clone();b.poke_mut(id).hp=hp;b.last_damage=damage;
            packed.insert(&dex,b,1.0/3.0);
        }
        assert_eq!(packed.templates(),3);
        let removed=packed.prune(2,1);
        assert_eq!(packed.len(),1);assert_eq!(packed.templates(),1);
        assert!((removed+packed.mass()-1.0).abs()<1e-12);
    }
}
