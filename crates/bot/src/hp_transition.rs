use nc2000_engine::battle::{enumerate::run_scripted,SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::Draw;
use nc2000_engine::state::{Battle, PokeId, RequestState, Status};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HpTransitionKey {
    fingerprint: [u8; 16],
    state: u128,
    damage_omitted: bool,
    claw_omitted: bool,
    choices: [Option<SearchChoice>; 2],
    script: Vec<usize>,
}

pub struct HpTransition {
    key: HpTransitionKey,
    active: [PokeId; 2],
    damage: [i32; 2],
    result: Battle,
    trace: Vec<Draw>,
}

impl HpTransitionKey {
    pub fn new(dex: &Dex, battle: &Battle, choices: [Option<SearchChoice>; 2], script: &[usize]) -> Option<Self> {
        if !battle.started || battle.ended || battle.turn==0 || battle.mid_turn
            || battle.request_state!=RequestState::Move || battle.needs_choice()!=[true,true]
            || !battle.queue.is_empty() || !battle.faint_queue.is_empty()
            || !battle.event_stack.is_empty() || !battle.effect_stack.is_empty()
            || battle.active_move.is_some() || battle.active_pokemon.is_some() || battle.active_target.is_some()
            || battle.pending_boosts.is_some() || battle.field.weather.is_some()
            || battle.field.pseudo_weather.iter().any(|(id,_)|!matches!(dex.conds.key(*id),"maxtotallevel"|"stadiumsleepclause"|"freezeclausemod")) {
            return None;
        }
        for side in 0..2 {
            let team=&battle.sides[side];
            if !team.side_conditions.is_empty() || !team.slot_conditions.is_empty()
                || !team.choice.actions.is_empty() || !team.choice.switch_ins.is_empty()
                || team.choice.forced_switches_left!=0 || team.choice.forced_passes_left!=0 { return None; }
            let pokemon=battle.poke(battle.active_id(side)?);
            if pokemon.hp<=0 || pokemon.hp>pokemon.maxhp || pokemon.fainted || pokemon.transformed
                || !matches!(pokemon.status,Status::None|Status::Par|Status::Slp|Status::Frz)
                || pokemon.volatiles.iter().any(|(id,_)|dex.conds.key(*id)!="parspeeddrop")
                || pokemon.item.is_some_and(|item|!matches!(dex.items.key(item),"mintberry"|"miracleberry"|"brightpowder"|"quickclaw")) {
                return None;
            }
            let Some(SearchChoice::Move(id))=choices[side] else { return None; };
            if !matches!(dex.moves.key(id),"bodyslam"|"hydropump"|"icebeam"|"fireblast"|"thunderbolt"|"thunder"|"earthquake"|"seismictoss"|"tackle"|"splash")
                || !pokemon.move_slots.iter().any(|slot|slot.id==id && slot.pp>0 && !slot.disabled) {
                return None;
            }
        }
        let mut canonical=battle.clone();
        for side in 0..2 {
            let id=canonical.active_id(side)?;
            let pokemon=canonical.poke_mut(id);pokemon.hp=pokemon.maxhp;
        }
        let damage_omitted=!canonical.damage_bookkeeping_observable(dex);
        let claw_omitted=!canonical.quick_claw_roll_observable(dex);
        let state=if claw_omitted {canonical.state_key128_without_quick_claw_roll(dex,damage_omitted)}
            else if damage_omitted {canonical.state_key128_without_damage_bookkeeping()}
            else {canonical.state_key128()};
        Some(Self {fingerprint:dex.replay_fingerprint,state,damage_omitted,claw_omitted,choices,script:script.to_vec()})
    }
}

impl HpTransition {
    /// In the admitted move/item domain HP only decreases during the step.
    /// Both translated survivors must stay positive, so no KO-dependent
    /// control flow or capped damage can differ from the recorded trace.
    fn new(dex: &Dex, base: &Battle, choices: [Option<SearchChoice>; 2], script: &[usize], result: &Battle, trace: &[Draw]) -> Option<Self> {
        let key=HpTransitionKey::new(dex,base,choices,script)?;
        Self::from_key(key, base, result, trace)
    }

    fn from_key(key: HpTransitionKey, base: &Battle, result: &Battle, trace: &[Draw]) -> Option<Self> {
        if result.outcome().is_some() || result.needs_choice()!=[true,true] || result.request_state!=RequestState::Move { return None; }
        let active=[base.active_id(0)?,base.active_id(1)?];
        let mut damage=[0;2];
        for side in 0..2 {
            let id=active[side];
            if result.active_id(side)!=Some(id) || result.poke(id).fainted || result.poke(id).hp<=0 { return None; }
            damage[side]=base.poke(id).hp-result.poke(id).hp;
            if damage[side]<0 { return None; }
        }
        Some(Self {key,active,damage,result:result.clone(),trace:trace.to_vec()})
    }

    pub fn key(&self) -> &HpTransitionKey { &self.key }

    pub fn input_hp(&self) -> [i32;2] {
        [0,1].map(|side|self.result.poke(self.active[side]).hp+self.damage[side])
    }

    pub fn run(dex: &Dex, base: &Battle, choices: [Option<SearchChoice>; 2], script: &[usize]) -> (Battle,Vec<Draw>,Option<Self>) {
        let (result,trace)=run_scripted(dex,base,choices,script);
        let entry=Self::new(dex,base,choices,script,&result,&trace);
        (result,trace,entry)
    }

    pub fn translate(&self, dex: &Dex, target: &Battle, choices: [Option<SearchChoice>;2], script: &[usize]) -> Option<(Battle,Vec<Draw>)> {
        if HpTransitionKey::new(dex,target,choices,script).as_ref()!=Some(&self.key) { return None; }
        self.translate_hp(target)
    }

    fn translate_hp(&self, target: &Battle) -> Option<(Battle,Vec<Draw>)> {
        let hp=[0,1].map(|side|target.poke(self.active[side]).hp-self.damage[side]);
        if hp.iter().any(|&hp|hp<=0) { return None; }
        let mut result=self.result.clone();
        for side in 0..2 { result.poke_mut(self.active[side]).hp=hp[side]; }
        Some((result,self.trace.clone()))
    }
}


#[derive(Default)]
pub(crate) struct HpTransitionCache {
    entries: HashMap<HpTransitionKey, HpTransition>,
    order: VecDeque<HpTransitionKey>,
}

pub(crate) struct CachedStep {
    pub battle: Battle,
    pub trace: Vec<Draw>,
    pub eligible: bool,
    pub hit: bool,
    pub rejected_hp: bool,
}

impl HpTransitionCache {
    pub fn len(&self) -> usize { self.entries.len() }

    pub fn run(&mut self, dex: &Dex, base: &Battle, choices: [Option<SearchChoice>;2], script: &[usize], capacity: usize) -> CachedStep {
        let key=(capacity>0).then(||HpTransitionKey::new(dex,base,choices,script)).flatten();
        self.run_key(dex,base,choices,script,capacity,key)
    }

    pub fn enumerate(&mut self, dex: &Dex, base: &Battle, choices: [Option<SearchChoice>;2], cap: usize, capacity: usize,
        mut observe: impl FnMut(&CachedStep,usize)) -> Option<nc2000_engine::battle::enumerate::StepEnum> {
        let template=(capacity>0).then(||HpTransitionKey::new(dex,base,choices,&[])).flatten();
        nc2000_engine::battle::enumerate::enumerate_step_with_runner(dex,base,choices,cap,&mut |script| {
            let key=template.as_ref().map(|template| {
                let mut key=template.clone();key.script=script.to_vec();key
            });
            let step=self.run_key(dex,base,choices,script,capacity,key);
            observe(&step,self.len());
            (step.battle,step.trace,!step.hit)
        })
    }

    fn run_key(&mut self, dex: &Dex, base: &Battle, choices: [Option<SearchChoice>;2], script: &[usize], capacity: usize,
        key: Option<HpTransitionKey>) -> CachedStep {
        while self.entries.len()>capacity {
            self.entries.remove(&self.order.pop_front().unwrap());
        }
        let eligible=key.is_some();
        let mut rejected_hp=false;
        if let Some(entry)=key.as_ref().and_then(|key|self.entries.get(key)) {
            if let Some((battle,trace))=entry.translate_hp(base) {
                return CachedStep {battle,trace,eligible,hit:true,rejected_hp:false};
            }
            rejected_hp=true;
        }
        let (battle,trace)=run_scripted(dex,base,choices,script);
        if let Some(key)=key.filter(|key|!self.entries.contains_key(key)) {
            if let Some(entry)=HpTransition::from_key(key.clone(),base,&battle,&trace) {
                if self.entries.len()==capacity {
                    self.entries.remove(&self.order.pop_front().unwrap());
                }
                self.order.push_back(key.clone());
                self.entries.insert(key,entry);
            }
        }
        CachedStep {battle,trace,eligible,hit:false,rejected_hp}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::{EffectHandle,PokemonSet};
    use std::collections::HashMap;

    fn battle(dex: &Dex, moves: [&[&str];2], items: [&str;2]) -> Battle {
        let teams=["Snorlax","Vaporeon"].into_iter().enumerate().map(|(side,species)|PokemonSet {
            name:species.into(),species:species.into(),item:items[side].into(),ability:String::new(),
            moves:moves[side].iter().map(|name|(*name).into()).collect(),level:50,
            evs:None,ivs:None,gender:None,happiness:None,
        }).collect::<Vec<_>>();
        let mut b=Battle::from_fixture(dex,"1,2,3,4",&teams[..1],&teams[1..]).unwrap();
        let preview=[0,1].map(|side|b.legal_choices(dex,side).first().copied());
        b.apply_choices(dex,preview).unwrap();b
    }

    fn trace_signature(trace: &[Draw]) -> Vec<(&'static str,Vec<u64>,usize)> {
        trace.iter().map(|d|(d.label,d.counts.clone(),d.chosen)).collect()
    }

    #[test]
    fn translated_survivors_match_every_script_and_exhaustive_probability() {
        let dex=conformance::load_dex();
        let base=battle(&dex,[&["Seismic Toss"],&["Ice Beam"]],["Mint Berry","Bright Powder"]);
        let choices=["seismictoss","icebeam"].map(|name|Some(SearchChoice::Move(dex.moves.id(name).unwrap())));
        let mut target=base.clone();
        for side in 0..2 {let id=target.active_id(side).unwrap();target.poke_mut(id).hp-=30;}
        let exact=nc2000_engine::battle::enumerate::enumerate_step(&dex,&target,choices,100_000).unwrap();
        let mut expected=HashMap::<u128,f64>::new();
        for leaf in exact.leaves {*expected.entry(leaf.battle.state_key128()).or_default()+=leaf.prob;}
        let mut found=HashMap::<u128,f64>::new();
        let mut pending=vec![Vec::<usize>::new()];
        let mut count=0;
        let mut cache=HpTransitionCache::default();
        while let Some(script)=pending.pop() {
            let (_,trace,entry)=HpTransition::run(&dex,&base,choices,&script);
            let entry=entry.expect("both full-HP source mons survive");
            let (translated,translated_trace)=entry.translate(&dex,&target,choices,&script).unwrap();
            let source=cache.run(&dex,&base,choices,&script,4096);
            assert!(!source.hit && source.eligible);
            let cached=cache.run(&dex,&target,choices,&script,4096);
            assert!(cached.hit);
            assert_eq!(cached.battle.state_key128(),translated.state_key128());
            assert_eq!(trace_signature(&cached.trace),trace_signature(&translated_trace));
            let (actual,actual_trace)=run_scripted(&dex,&target,choices,&script);
            assert_eq!(translated.state_key128(),actual.state_key128());
            assert_eq!(trace_signature(&translated_trace),trace_signature(&actual_trace));
            let mut mass=1.0;
            for (i,draw) in trace.iter().enumerate() {
                if i>=script.len() {
                    for (choice,&weight) in draw.counts.iter().enumerate() {
                        if choice!=draw.chosen && weight>0 {
                            let mut next=trace[..i].iter().map(|d|d.chosen).collect::<Vec<_>>();next.push(choice);pending.push(next);
                        }
                    }
                }
                mass*=draw.counts[draw.chosen] as f64/4294967296.0;
            }
            *found.entry(translated.state_key128()).or_default()+=mass;
            count+=1;assert!(count<100_000);
        }
        assert!(count>39);
        assert_eq!(found.len(),expected.len());
        for (key,mass) in found {assert!((mass-expected[&key]).abs()<1e-10);}
    }

    #[test]
    fn hp_translation_rejects_ko_boundaries_and_changed_inputs() {
        let dex=conformance::load_dex();
        let base=battle(&dex,[&["Seismic Toss"],&["Seismic Toss"]],["",""]);
        let choices=[Some(SearchChoice::Move(dex.moves.id("seismictoss").unwrap()));2];
        let (_,_,entry)=HpTransition::run(&dex,&base,choices,&[]);let entry=entry.unwrap();
        for hp in [1,49,50,51] {
            let mut target=base.clone();let id=target.active_id(0).unwrap();target.poke_mut(id).hp=hp;
            let translated=entry.translate(&dex,&target,choices,&[]);
            assert_eq!(translated.is_some(),hp>50);
            if let Some((translated,_))=translated {
                assert_eq!(translated.state_key128(),run_scripted(&dex,&target,choices,&[]).0.state_key128());
            }
        }
        let mut changed=base.clone();let id=changed.active_id(0).unwrap();changed.poke_mut(id).move_slots[0].pp-=1;
        assert!(entry.translate(&dex,&changed,choices,&[]).is_none());
        assert!(entry.translate(&dex,&base,choices,&[0]).is_none());
        changed=base.clone();changed.turn+=1;
        assert!(entry.translate(&dex,&changed,choices,&[]).is_none());
    }

    #[test]
    fn eager_cache_matches_complete_kernels_and_accounts_for_caps() {
        use nc2000_engine::battle::enumerate::{enumerate_step,enumerate_step_with_runner};
        let dex=conformance::load_dex();
        let base=battle(&dex,[&["Seismic Toss"],&["Ice Beam"]],["Mint Berry","Bright Powder"]);
        let choices=["seismictoss","icebeam"].map(|name|Some(SearchChoice::Move(dex.moves.id(name).unwrap())));
        let mut cache=HpTransitionCache::default();
        let mut hits=0;
        for reduction in [0,20,40,1000] {
            let mut target=base.clone();
            for side in 0..2 {let id=target.active_id(side).unwrap();let p=target.poke_mut(id);p.hp=(p.hp-reduction).max(1);}
            let exact=enumerate_step(&dex,&target,choices,100_000).unwrap();
            let mut runs=0;
            let cached=enumerate_step_with_runner(&dex,&target,choices,100_000,&mut |script| {
                let step=cache.run(&dex,&target,choices,script,4096);
                runs+=usize::from(!step.hit);hits+=usize::from(step.hit);
                let actual=run_scripted(&dex,&target,choices,script);
                assert_eq!(step.battle.state_key128(),actual.0.state_key128());
                assert_eq!(trace_signature(&step.trace),trace_signature(&actual.1));
                (step.battle,step.trace,!step.hit)
            }).unwrap();
            assert_eq!(cached.runs,runs);
            let aggregate=|leaves:Vec<nc2000_engine::battle::enumerate::ChanceLeaf>| {
                let mut mass=HashMap::<u128,f64>::new();
                for leaf in leaves {*mass.entry(leaf.battle.state_key128()).or_default()+=leaf.prob;}
                mass
            };
            let mut prepared_cache=HpTransitionCache::default();
            let prepared=prepared_cache.enumerate(&dex,&target,choices,100_000,4096,|_,_|{}).unwrap();
            assert_eq!(prepared.runs,exact.runs);
            let prepared=aggregate(prepared.leaves);
            let found=aggregate(cached.leaves);let expected=aggregate(exact.leaves);
            assert_eq!(prepared.len(),expected.len());
            for (key,mass) in prepared {assert!((mass-expected[&key]).abs()<1e-10);}
            let warm=prepared_cache.enumerate(&dex,&target,choices,100_000,4096,|_,_|{}).unwrap();
            let warm=aggregate(warm.leaves);
            assert_eq!(warm.len(),expected.len());
            for (key,mass) in warm {assert!((mass-expected[&key]).abs()<1e-10);}
            assert_eq!(found.len(),expected.len());
            for (key,mass) in found {assert!((mass-expected[&key]).abs()<1e-10);}
        }
        assert!(hits>39);
        for cap in [0,1,7] {
            let mut empty=HpTransitionCache::default();let mut runs=0;
            let result=enumerate_step_with_runner(&dex,&base,choices,cap,&mut |script| {
                let step=empty.run(&dex,&base,choices,script,4096);
                runs+=usize::from(!step.hit);
                (step.battle,step.trace,!step.hit)
            });
            assert!(result.is_none());assert_eq!(runs,cap);
        }
        let mut warm=HpTransitionCache::default();
        for cap in [100_000,1] {
            let step=enumerate_step_with_runner(&dex,&base,choices,cap,&mut |script| {
                let step=warm.run(&dex,&base,choices,script,4096);
                (step.battle,step.trace,!step.hit)
            }).unwrap();
            if cap==1 {assert_eq!(step.runs,0);}
        }
    }

    #[test]
    fn bounded_cache_evicts_and_disables_without_changing_execution() {
        let dex=conformance::load_dex();
        let base=battle(&dex,[&["Seismic Toss"],&["Splash"]],["",""]);
        let choices=["seismictoss","splash"].map(|name|Some(SearchChoice::Move(dex.moves.id(name).unwrap())));
        let mut cache=HpTransitionCache::default();
        for turn in 1..5 {
            let mut target=base.clone();target.turn=turn;
            let step=cache.run(&dex,&target,choices,&[],2);
            assert!(!step.hit && step.eligible);
            assert_eq!(step.battle.state_key128(),run_scripted(&dex,&target,choices,&[]).0.state_key128());
            assert!(cache.len()<=2);
        }
        assert_eq!(cache.len(),2);
        assert!(!cache.run(&dex,&base,choices,&[],2).hit);
        assert!(cache.run(&dex,&base,choices,&[],2).hit);
        let disabled=cache.run(&dex,&base,choices,&[],0);
        assert!(!disabled.hit && !disabled.eligible);
        assert_eq!(cache.len(),0);
    }

    #[test]
    fn hp_translation_guards_healing_hp_sensitive_moves_and_volatiles() {
        let dex=conformance::load_dex();
        for moves in [[&["Rest"][..],&["Splash"][..]],[&["Reversal"][..],&["Splash"][..]],[&["Double-Edge"][..],&["Splash"][..]]] {
            let mut base=battle(&dex,moves,["",""]);
            let choices=[0,1].map(|side|base.legal_choices(&dex,side).first().copied());
            assert!(HpTransitionKey::new(&dex,&base,choices,&[]).is_none());
        }
        for item in ["Leftovers","Gold Berry","Berry","Focus Band"] {
            let mut base=battle(&dex,[&["Tackle"],&["Splash"]],[item,""]);
            let choices=[0,1].map(|side|base.legal_choices(&dex,side).first().copied());
            assert!(HpTransitionKey::new(&dex,&base,choices,&[]).is_none());
        }
        let mut base=battle(&dex,[&["Tackle"],&["Splash"]],["",""]);
        let choices=[0,1].map(|side|base.legal_choices(&dex,side).first().copied());
        let id=base.active_id(0).unwrap();
        base.add_volatile(&dex,id,"substitute",None,EffectHandle::None);
        assert!(HpTransitionKey::new(&dex,&base,choices,&[]).is_none());
    }
}
