use super::*;
use super::pair_bound::{partial_step,Frontier};
use nc2000_bot::hp_transition::{HpHealingCache,HpTransitionKey};
use nc2000_engine::state::PokeId;
use std::collections::{HashMap,VecDeque};

type Key=(HpTransitionKey,usize);

struct Entry {
    active:[PokeId;2],
    leaves:Vec<(Battle,f64)>,
    unknown:f64,
    observations:usize,
}

#[derive(Default,Serialize)]
pub(super) struct Stats {
    calls:usize,
    eligible:usize,
    hits:usize,
    rejected:usize,
    evictions:usize,
    peak_leaves:usize,
    translated_leaves:usize,
    observations_avoided:usize,
}

pub(super) struct Cache {
    capacity:usize,
    leaves:usize,
    entries:HashMap<Key,Entry>,
    order:VecDeque<Key>,
    pub(super) stats:Stats,
}

impl Cache {
    pub(super) fn new(capacity:usize)->Self {
        Self {capacity,leaves:0,entries:HashMap::new(),order:VecDeque::new(),stats:Stats::default()}
    }
    pub(super) fn run(&mut self,dex:&Dex,b:&Battle,choices:[Option<SearchChoice>;2],cap:usize,healing:&mut HpHealingCache)->(Frontier,f64,usize) {
        self.stats.calls+=1;
        let key=(self.capacity>0 && cap>0).then(||HpTransitionKey::new_switching(dex,b,choices,&[])).flatten().map(|key|(key,cap));
        self.stats.eligible+=usize::from(key.is_some());
        if let Some(entry)=key.as_ref().and_then(|key|self.entries.get(key)) {
            let mut leaves=Frontier::new();
            for (source,mass) in &entry.leaves {
                let mut leaf=source.clone();
                for id in entry.active {leaf.poke_mut(id).hp=b.poke(id).hp;}
                leaves.entry(leaf.state_key128()).and_modify(|(_,p)|*p+=mass).or_insert((leaf,*mass));
            }
            self.stats.hits+=1;self.stats.translated_leaves+=entry.leaves.len();self.stats.observations_avoided+=entry.observations;
            return (leaves,entry.unknown,0);
        }
        let before=key.as_ref().map(|_|healing.stats().observations).unwrap_or(0);
        let result=partial_step(dex,b,choices,cap,healing);
        if let Some(key)=key {
            let active=[b.active_id(0).unwrap(),b.active_id(1).unwrap()];
            let switcher=(0..2).find(|&side|matches!(choices[side],Some(SearchChoice::Switch(_)))).unwrap();
            let reusable=!result.0.is_empty() && result.0.len()<=self.capacity && result.0.values().all(|(leaf,_)|
                leaf.outcome().is_none() && leaf.active_id(switcher)!=Some(active[switcher])
                    && leaf.active_id(1-switcher)==Some(active[1-switcher])
                    && active.iter().all(|&id|leaf.poke(id).hp==b.poke(id).hp && !leaf.poke(id).fainted));
            if reusable {
                while self.leaves+result.0.len()>self.capacity {
                    let old=self.order.pop_front().unwrap();self.leaves-=self.entries.remove(&old).unwrap().leaves.len();self.stats.evictions+=1;
                }
                self.leaves+=result.0.len();self.stats.peak_leaves=self.stats.peak_leaves.max(self.leaves);
                self.order.push_back(key.clone());
                self.entries.insert(key,Entry {active,leaves:result.0.values().cloned().collect(),unknown:result.1,
                    observations:healing.stats().observations-before});
            } else {self.stats.rejected+=1;}
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn battle(dex:&Dex,switcher:usize,incoming_hp:i32)->(Battle,[Option<SearchChoice>;2]) {
        let mon=|species:&str,moves:&[&str],item:&str|PokemonSet {name:species.into(),species:species.into(),item:item.into(),ability:String::new(),
            moves:moves.iter().map(|m|(*m).into()).collect(),level:50,evs:None,ivs:None,gender:None,happiness:None};
        let attacker=vec![mon("Snorlax",&["Body Slam"],"Mint Berry")];
        let switching=vec![mon("Vaporeon",&["Hydro Pump","Rest"],"Bright Powder"),mon("Snorlax",&["Self-Destruct"],"Leftovers")];
        let teams=if switcher==0 {[switching,attacker]} else {[attacker,switching]};
        let mut b=Battle::from_fixture(dex,"1,2,3,4",&teams[0],&teams[1]).unwrap();
        let preview=[0,1].map(|side|b.legal_choices(dex,side)[0]);b.apply_choices(dex,preview.map(Some)).unwrap();
        let incoming=b.sides[switcher].party[1] as usize;b.sides[switcher].roster[incoming].hp=incoming_hp;
        let outgoing=b.active_id(switcher).unwrap();b.poke_mut(outgoing).status=Status::Slp;
        b.poke_mut(outgoing).status_state.set_int(DK::Time,1);b.poke_mut(outgoing).status_state.set_int(DK::StartTime,3);b.refresh_poke_mask(dex,outgoing);
        let choices=[0,1].map(|side|Some(if side==switcher {SearchChoice::Switch(2)} else {SearchChoice::Move(dex.moves.id("bodyslam").unwrap())}));
        (b,choices)
    }
    fn equal(a:&(Frontier,f64,usize),b:&(Frontier,f64,usize)) {
        assert_eq!(a.0.len(),b.0.len());assert!((a.1-b.1).abs()<1e-12);
        for (key,(_,mass)) in &a.0 {assert!((mass-b.0[key].1).abs()<1e-12);}
    }
    #[test]
    fn aggregated_switch_reuse_matches_partial_and_complete_distributions() {
        for side in 0..2 {for incoming_hp in [20,133,200] {
            let dex=conformance::load_dex();let (base,choices)=battle(&dex,side,incoming_hp);
            let mut cache=Cache::new(4096);let mut healing=HpHealingCache::new(0);
            for cap in [1,7,4096] {
                for hp in [[200,200],[1,200],[150,1],[1,1]] {
                    let mut b=base.clone();for side in 0..2 {let id=b.active_id(side).unwrap();b.poke_mut(id).hp=hp[side];}
                    let expected=partial_step(&dex,&b,choices,cap,&mut HpHealingCache::new(0));
                    let found=cache.run(&dex,&b,choices,cap,&mut healing);equal(&found,&expected);
                    if hp!=[200,200] {assert_eq!(found.2,0);} else {assert!(found.2>0);}
                    if cap==4096 {assert_eq!(found.1,0.0);} else {assert!(found.1>0.0);}
                }
            }
            assert_eq!(cache.stats.hits,9);assert!(cache.stats.observations_avoided>cache.stats.translated_leaves);
            let mut b=base.clone();let incoming=b.sides[side].party[1] as usize;b.sides[side].roster[incoming].hp-=1;
            let result=cache.run(&dex,&b,choices,4096,&mut healing);assert!(result.2>0);
        }}
    }
    #[test]
    fn kernel_capacity_eviction_and_rejected_paths_preserve_results() {
        let dex=conformance::load_dex();let (base,choices)=battle(&dex,1,133);
        for capacity in [0,1,2,100] {
            let mut cache=Cache::new(capacity);let mut healing=HpHealingCache::new(0);
            for cap in [0,1,7,1,7] {
                let found=cache.run(&dex,&base,choices,cap,&mut healing);
                let expected=partial_step(&dex,&base,choices,cap,&mut HpHealingCache::new(0));equal(&found,&expected);
                assert!(cache.stats.peak_leaves<=capacity);
            }
            if capacity==1 {assert!(cache.stats.rejected>0);}
            if capacity==2 {assert!(cache.stats.evictions>0);}
        }
        let mut b=base.clone();b.turn=1000;
        let mut cache=Cache::new(4096);let mut healing=HpHealingCache::new(0);
        let first=cache.run(&dex,&b,choices,7,&mut healing);let second=cache.run(&dex,&b,choices,7,&mut healing);
        equal(&first,&second);assert!(first.2>0 && second.2>0);assert_eq!(cache.stats.hits,0);assert_eq!(cache.stats.rejected,2);
    }
}
