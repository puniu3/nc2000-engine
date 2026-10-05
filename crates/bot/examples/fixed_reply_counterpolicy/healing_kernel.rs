use super::*;
use super::pair_bound::Frontier;
use nc2000_bot::hp_transition::{HpHealingCache,HpTransitionKey};
use nc2000_engine::state::{PokeId,RequestState};
use std::collections::{HashMap,VecDeque};

type Key=(HpTransitionKey,usize);
struct Leaf {battle:Battle,mass:f64,damage:[i32;2],reset:Option<usize>}
struct Entry {active:[PokeId;2],required_hp:[i32;2],leaves:Vec<Leaf>,unknown:f64,observations:usize}

impl Entry {
    fn new(dex:&Dex,b:&Battle,choices:[Option<SearchChoice>;2],leaves:&Frontier,unknown:f64,observations:usize)->Option<Self> {
        let healer=(0..2).find(|&side|matches!(choices[side],Some(SearchChoice::Move(id)) if matches!(dex.moves.key(id),"rest"|"sleeptalk")))?;
        let active=[b.active_id(0)?,b.active_id(1)?];let mut required_hp=[1;2];let mut transforms=Vec::new();
        for (leaf,_) in leaves.values() {
            if leaf.outcome().is_some() || leaf.request_state!=RequestState::Move || leaf.needs_choice()!=[true,true] {return None;}
            let called=leaf.poke(active[healer]).last_move_used?;
            let reset=match dex.moves.key(called) {"rest"=>Some(healer),"hydropump"|"icebeam"=>None,_=>return None};
            let mut damage=[0;2];
            for side in 0..2 {
                let id=active[side];
                if leaf.active_id(side)!=Some(id) || leaf.poke(id).fainted || leaf.poke(id).hp<=0 {return None;}
                if reset!=Some(side) {
                    damage[side]=b.poke(id).hp-leaf.poke(id).hp;
                    if damage[side]<0 {return None;}
                    required_hp[side]=required_hp[side].max(damage[side]+1);
                }
            }
            transforms.push((damage,reset));
        }
        let stored=leaves.values().zip(transforms).map(|((leaf,mass),(damage,reset))|Leaf {battle:leaf.clone(),mass:*mass,damage,reset}).collect();
        Some(Self {active,required_hp,leaves:stored,unknown,observations})
    }
    fn translate(&self,b:&Battle)->Option<Frontier> {
        if (0..2).any(|side|b.poke(self.active[side]).hp<self.required_hp[side]) {return None;}
        let mut leaves=Frontier::new();
        for source in &self.leaves {
            let mut leaf=source.battle.clone();
            for side in 0..2 {if source.reset!=Some(side) {
                leaf.poke_mut(self.active[side]).hp=b.poke(self.active[side]).hp-source.damage[side];
            }}
            leaves.entry(leaf.state_key128()).and_modify(|(_,mass)|*mass+=source.mass).or_insert((leaf,source.mass));
        }
        Some(leaves)
    }
}

#[derive(Default,Serialize)]
pub(super) struct Stats {
    calls:usize,eligible:usize,hits:usize,rejected_hp:usize,rejected_source:usize,rejected_capacity:usize,
    evictions:usize,peak_leaves:usize,translated_leaves:usize,reset_leaves:usize,observations_avoided:usize,
}

pub(super) struct Cache {
    capacity:usize,leaves:usize,entries:HashMap<Key,Entry>,order:VecDeque<Key>,pub(super) stats:Stats,
}

impl Cache {
    pub(super) fn new(capacity:usize)->Self {
        Self {capacity,leaves:0,entries:HashMap::new(),order:VecDeque::new(),stats:Stats::default()}
    }
    pub(super) fn run(&mut self,dex:&Dex,b:&Battle,choices:[Option<SearchChoice>;2],cap:usize,
        healing:&mut HpHealingCache,switches:&mut super::switch_kernel::Cache)->(Frontier,f64,usize) {
        self.stats.calls+=1;
        let key=(self.capacity>0 && cap>0).then(||HpTransitionKey::new_healing(dex,b,choices,&[])).flatten().map(|key|(key,cap));
        self.stats.eligible+=usize::from(key.is_some());
        if let Some(entry)=key.as_ref().and_then(|key|self.entries.get(key)) {
            if let Some(leaves)=entry.translate(b) {
                self.stats.hits+=1;self.stats.translated_leaves+=entry.leaves.len();self.stats.observations_avoided+=entry.observations;
                self.stats.reset_leaves+=entry.leaves.iter().filter(|leaf|leaf.reset.is_some()).count();
                return (leaves,entry.unknown,0);
            }
            self.stats.rejected_hp+=1;
        }
        let before=key.as_ref().map(|_|healing.stats().observations).unwrap_or(0);
        let result=switches.run(dex,b,choices,cap,healing);
        if let Some(key)=key.filter(|key|!self.entries.contains_key(key)) {
            if result.0.is_empty() || result.0.len()>self.capacity {self.stats.rejected_capacity+=1;}
            else if let Some(entry)=Entry::new(dex,b,choices,&result.0,result.1,healing.stats().observations-before) {
                while self.leaves+entry.leaves.len()>self.capacity {
                    let old=self.order.pop_front().unwrap();self.leaves-=self.entries.remove(&old).unwrap().leaves.len();self.stats.evictions+=1;
                }
                self.leaves+=entry.leaves.len();self.stats.peak_leaves=self.stats.peak_leaves.max(self.leaves);
                self.order.push_back(key.clone());self.entries.insert(key,entry);
            } else {self.stats.rejected_source+=1;}
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pair_bound::partial_step;
    fn battle(dex:&Dex,healer:usize,sleep:i64)->(Battle,[Option<SearchChoice>;2]) {
        let mon=|species:&str,moves:&[&str],item:&str|PokemonSet {name:species.into(),species:species.into(),item:item.into(),ability:String::new(),
            moves:moves.iter().map(|m|(*m).into()).collect(),level:50,evs:None,ivs:None,gender:None,happiness:None};
        let a=vec![mon("Snorlax",&["Body Slam"],"Mint Berry")];
        let h=vec![mon("Vaporeon",&["Hydro Pump","Ice Beam","Sleep Talk","Rest"],"Bright Powder")];
        let teams=if healer==0 {[h,a]} else {[a,h]};
        let mut b=Battle::from_fixture(dex,"1,2,3,4",&teams[0],&teams[1]).unwrap();
        let preview=[0,1].map(|side|b.legal_choices(dex,side)[0]);b.apply_choices(dex,preview.map(Some)).unwrap();
        let id=b.active_id(healer).unwrap();b.poke_mut(id).hp-=1;
        if sleep>0 {
            b.poke_mut(id).status=Status::Slp;b.poke_mut(id).status_state.set_int(DK::Time,sleep);
            b.poke_mut(id).status_state.set_int(DK::StartTime,3);b.refresh_poke_mask(dex,id);
        }
        let choices=[0,1].map(|side|Some(SearchChoice::Move(dex.moves.id(if side!=healer {"bodyslam"} else if sleep>1 {"sleeptalk"} else {"rest"}).unwrap())));
        (b,choices)
    }
    fn equal(a:&(Frontier,f64,usize),b:&(Frontier,f64,usize)) {
        assert_eq!(a.0.len(),b.0.len());assert!((a.1-b.1).abs()<1e-12);
        for (key,(_,mass)) in &a.0 {assert!((mass-b.0[key].1).abs()<1e-12);}
    }
    #[test]
    fn healing_kernels_preserve_partial_maps_resets_and_ko_fallbacks() {
        let dex=conformance::load_dex();let mut hits=0;let mut fallbacks=0;
        for healer in 0..2 {for sleep in [0,1,3] {for cap in [1,7,4096] {
            let (base,choices)=battle(&dex,healer,sleep);let mut cache=Cache::new(10000);
            let mut scripts=HpHealingCache::new(0);let mut switches=crate::switch_kernel::Cache::new(0);
            let id=base.active_id(healer).unwrap();let foe=base.active_id(1-healer).unwrap();
            for (own,other) in [(base.poke(id).hp,base.poke(foe).hp),(base.poke(id).hp-1,base.poke(foe).hp),(1,base.poke(foe).hp),(base.poke(id).hp,1)] {
                let mut b=base.clone();b.poke_mut(id).hp=own;b.poke_mut(foe).hp=other;
                let expected=partial_step(&dex,&b,choices,cap,&mut HpHealingCache::new(0));
                let found=cache.run(&dex,&b,choices,cap,&mut scripts,&mut switches);equal(&found,&expected);
            }
            hits+=cache.stats.hits;fallbacks+=cache.stats.rejected_hp;
            assert!(cache.stats.hits>0);assert!(cache.stats.peak_leaves<=10000);
            if sleep<=1 {assert_eq!(cache.stats.hits,3);assert_eq!(cache.stats.translated_leaves,cache.stats.reset_leaves);}
            let mut b=base.clone();b.poke_mut(id).hp=b.poke(id).maxhp;
            let before=cache.stats.hits;let found=cache.run(&dex,&b,choices,cap,&mut scripts,&mut switches);
            equal(&found,&partial_step(&dex,&b,choices,cap,&mut HpHealingCache::new(0)));assert_eq!(cache.stats.hits,before);
        }}}
        assert!(hits>=40 && fallbacks>0);
    }
    #[test]
    fn healing_kernel_storage_limits_and_terminal_sources_fall_back() {
        let dex=conformance::load_dex();let (base,choices)=battle(&dex,1,0);
        for capacity in [0,1,2] {
            let mut cache=Cache::new(capacity);let mut scripts=HpHealingCache::new(0);let mut switches=crate::switch_kernel::Cache::new(0);
            for turn in [1,2,3,1] {
                let mut b=base.clone();b.turn=turn;
                let found=cache.run(&dex,&b,choices,1,&mut scripts,&mut switches);
                equal(&found,&partial_step(&dex,&b,choices,1,&mut HpHealingCache::new(0)));
                assert!(cache.stats.peak_leaves<=capacity);
            }
            if capacity>0 {assert!(cache.stats.evictions>0);}
            let found=cache.run(&dex,&base,choices,4096,&mut scripts,&mut switches);
            equal(&found,&partial_step(&dex,&base,choices,4096,&mut HpHealingCache::new(0)));
            if capacity>0 {assert!(cache.stats.rejected_capacity>0);}
        }
        let (mut b,choices)=battle(&dex,1,3);let id=b.active_id(1).unwrap();b.poke_mut(id).hp=1;
        let mut cache=Cache::new(10000);let mut scripts=HpHealingCache::new(0);let mut switches=crate::switch_kernel::Cache::new(0);
        let found=cache.run(&dex,&b,choices,4096,&mut scripts,&mut switches);
        equal(&found,&partial_step(&dex,&b,choices,4096,&mut HpHealingCache::new(0)));
        assert_eq!(cache.stats.rejected_source,1);assert!(cache.entries.is_empty());
    }
}
