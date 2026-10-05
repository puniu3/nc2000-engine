use nc2000_engine::battle::{enumerate::run_scripted,SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::Draw;
use nc2000_engine::state::{Battle, PokeId, RequestState, Status, DK};
use std::collections::{BTreeMap, HashMap, VecDeque};

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
        Self::new_domain(dex,battle,choices,script,None,None).ok()
    }

    pub fn new_healing(dex:&Dex,battle:&Battle,choices:[Option<SearchChoice>;2],script:&[usize])->Option<Self> {
        Self::healing_domain(dex,battle,choices,script).ok()
    }

    fn healing_domain(dex:&Dex,battle:&Battle,choices:[Option<SearchChoice>;2],script:&[usize])->Result<Self,&'static str> {
        let healer=healing_side(dex,choices).ok_or("no_single_healer")?;
        Self::new_domain(dex,battle,choices,script,Some(healer),None)
    }

    pub fn new_switching(dex:&Dex,battle:&Battle,choices:[Option<SearchChoice>;2],script:&[usize])->Option<Self> {
        Self::switching_domain(dex,battle,choices,script).ok()
    }

    fn switching_domain(dex:&Dex,battle:&Battle,choices:[Option<SearchChoice>;2],script:&[usize])->Result<Self,&'static str> {
        let switcher=switching_side(choices).ok_or("not_single_switch")?;
        Self::new_domain(dex,battle,choices,script,None,Some(switcher))
    }

    fn new_domain(dex:&Dex,battle:&Battle,choices:[Option<SearchChoice>;2],script:&[usize],healer:Option<usize>,switcher:Option<usize>)->Result<Self,&'static str> {
        if !battle.started || battle.ended || battle.turn==0 || battle.mid_turn
            || battle.request_state!=RequestState::Move || battle.needs_choice()!=[true,true]
            || !battle.queue.is_empty() || !battle.faint_queue.is_empty()
            || !battle.event_stack.is_empty() || !battle.effect_stack.is_empty()
            || battle.active_move.is_some() || battle.active_pokemon.is_some() || battle.active_target.is_some()
            || battle.pending_boosts.is_some() || battle.field.weather.is_some()
            || battle.field.pseudo_weather.iter().any(|(id,_)|!matches!(dex.conds.key(*id),"maxtotallevel"|"stadiumsleepclause"|"freezeclausemod")) {
            return Err("frame_or_field");
        }
        for side in 0..2 {
            let team=&battle.sides[side];
            if !team.side_conditions.is_empty() || !team.slot_conditions.is_empty()
                || !team.choice.actions.is_empty() || !team.choice.switch_ins.is_empty()
                || team.choice.forced_switches_left!=0 || team.choice.forced_passes_left!=0 { return Err("side_conditions_or_choice"); }
            let pokemon=battle.poke(battle.active_id(side).ok_or("missing_active")?);
            if pokemon.hp<=0 || pokemon.hp>pokemon.maxhp || pokemon.fainted || pokemon.transformed
                || !matches!(pokemon.status,Status::None|Status::Par|Status::Slp|Status::Frz) {return Err("health_status_or_transform");}
            if pokemon.volatiles.iter().any(|(id,_)|dex.conds.key(*id)!="parspeeddrop") {return Err("volatile");}
            if pokemon.item.is_some_and(|item|!matches!(dex.items.key(item),"mintberry"|"miracleberry"|"brightpowder"|"quickclaw")) {return Err("active_item");}
            if switcher==Some(side) {
                let Some(SearchChoice::Switch(pos))=choices[side] else {return Err("non_switch_choice");};
                if pokemon.trapped || pos<=1 {return Err("illegal_switch");}
                let slot=*team.party.get(pos as usize-1).ok_or("illegal_switch")?;
                let incoming=team.roster.get(slot as usize).ok_or("illegal_switch")?;
                if incoming.hp<=0 || incoming.hp>incoming.maxhp || incoming.fainted || incoming.transformed
                    || incoming.is_active || !incoming.volatiles.is_empty()
                    || !matches!(incoming.status,Status::None|Status::Par|Status::Slp|Status::Frz)
                    || incoming.item.is_some_and(|id|!matches!(dex.items.key(id),"mintberry"|"miracleberry"|"brightpowder"|"quickclaw"|"leftovers")) {
                    return Err("incoming_state");
                }
                continue;
            }
            let Some(SearchChoice::Move(id))=choices[side] else {return Err("non_move_choice");};
            if !pokemon.move_slots.iter().any(|slot|slot.id==id && slot.pp>0 && !slot.disabled) {return Err("unavailable_move");}
            if healer==Some(side) {
                if pokemon.hp==pokemon.maxhp {return Err("healer_full_hp");}
                if !pokemon.volatiles.is_empty() {return Err("healer_volatile");}
                if pokemon.item.is_some_and(|item|dex.items.key(item)!="brightpowder") {return Err("healer_item");}
                match dex.moves.key(id) {
                    "rest" if pokemon.status==Status::None || (pokemon.status==Status::Slp && pokemon.status_state.get_int(DK::Time)<=1)=>{},
                    "sleeptalk" if pokemon.status==Status::Slp && pokemon.status_state.get_int(DK::Time)>1=> {
                        if pokemon.move_slots.iter().any(|slot| {
                            let mv=dex.move_static(slot.id);
                            !mv.has_flag(dex,"nosleeptalk") && !mv.has_flag(dex,"charge")
                                && !matches!(dex.moves.key(slot.id),"hydropump"|"icebeam"|"rest")
                        }) {return Err("called_move_pool");}
                    },
                    _=>return Err("healer_status"),
                }
            } else if !matches!(dex.moves.key(id),"bodyslam"|"hydropump"|"icebeam"|"fireblast"|"thunderbolt"|"thunder"|"earthquake"|"seismictoss"|"tackle"|"splash") {
                return Err("opponent_move");
            }
        }
        if let Some(healer)=healer {
            if battle.quick_claw_roll_observable(dex) {return Err("quick_claw");}
            if battle.get_pokemon_action_speed(dex,battle.active_id(healer).ok_or("missing_active")?)
                <=battle.get_pokemon_action_speed(dex,battle.active_id(1-healer).ok_or("missing_active")?) {return Err("not_strictly_faster");}
        }
        let mut canonical=battle.clone();
        for side in 0..2 {
            let id=canonical.active_id(side).ok_or("missing_active")?;
            let pokemon=canonical.poke_mut(id);pokemon.hp=pokemon.maxhp;
        }
        let damage_omitted=!canonical.damage_bookkeeping_observable(dex);
        let claw_omitted=!canonical.quick_claw_roll_observable(dex);
        let state=if claw_omitted {canonical.state_key128_without_quick_claw_roll(dex,damage_omitted)}
            else if damage_omitted {canonical.state_key128_without_damage_bookkeeping()}
            else {canonical.state_key128()};
        Ok(Self {fingerprint:dex.replay_fingerprint,state,damage_omitted,claw_omitted,choices,script:script.to_vec()})
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


fn healing_side(dex:&Dex,choices:[Option<SearchChoice>;2])->Option<usize> {
    let healers=(0..2).filter(|&side|matches!(choices[side],Some(SearchChoice::Move(id))
        if matches!(dex.moves.key(id),"rest"|"sleeptalk"))).collect::<Vec<_>>();
    (healers.len()==1).then(||healers[0])
}

pub struct HpHealingTransition {
    key:HpTransitionKey,
    active:[PokeId;2],
    damage:[i32;2],
    reset:Option<usize>,
    result:Battle,
    trace:Vec<Draw>,
}

impl HpHealingTransition {
    pub fn run(dex:&Dex,base:&Battle,choices:[Option<SearchChoice>;2],script:&[usize])->(Battle,Vec<Draw>,Option<Self>) {
        let key=HpTransitionKey::new_healing(dex,base,choices,script);
        let (result,trace)=run_scripted(dex,base,choices,script);
        let entry=key.and_then(|key|Self::from_result(dex,key,base,choices,&result,&trace));
        (result,trace,entry)
    }

    fn from_result(dex:&Dex,key:HpTransitionKey,base:&Battle,choices:[Option<SearchChoice>;2],result:&Battle,trace:&[Draw])->Option<Self> {
        if result.outcome().is_some() || result.needs_choice()!=[true,true] || result.request_state!=RequestState::Move {return None;}
        let healer=healing_side(dex,choices)?;
        let active=[base.active_id(0)?,base.active_id(1)?];
        let called=result.poke(active[healer]).last_move_used?;
        let reset=match dex.moves.key(called) {"rest"=>Some(healer),"hydropump"|"icebeam"=>None,_=>return None};
        let mut damage=[0;2];
        for side in 0..2 {
            let id=active[side];
            if result.active_id(side)!=Some(id) || result.poke(id).fainted || result.poke(id).hp<=0 {return None;}
            if reset!=Some(side) {
                damage[side]=base.poke(id).hp-result.poke(id).hp;
                if damage[side]<0 {return None;}
            }
        }
        Some(Self {key,active,damage,reset,result:result.clone(),trace:trace.to_vec()})
    }

    pub fn key(&self)->&HpTransitionKey {&self.key}
    pub fn resets_hp(&self)->bool {self.reset.is_some()}

    /// The admitted healer acts first and starts below full HP. Rest resets
    /// its HP before the only opposing attack; attack branches only decrease
    /// HP, so strictly positive translated survivors preserve the draw trace.
    pub fn translate(&self,dex:&Dex,target:&Battle,choices:[Option<SearchChoice>;2],script:&[usize])->Option<(Battle,Vec<Draw>)> {
        if HpTransitionKey::new_healing(dex,target,choices,script).as_ref()!=Some(&self.key) {return None;}
        self.translate_hp(target)
    }

    fn translate_hp(&self,target:&Battle)->Option<(Battle,Vec<Draw>)> {
        let mut result=self.result.clone();
        for side in 0..2 {
            if self.reset!=Some(side) {
                let hp=target.poke(self.active[side]).hp-self.damage[side];
                if hp<=0 {return None;}
                result.poke_mut(self.active[side]).hp=hp;
            }
        }
        Some((result,self.trace.clone()))
    }
}


fn switching_side(choices:[Option<SearchChoice>;2])->Option<usize> {
    let sides=(0..2).filter(|&side|matches!(choices[side],Some(SearchChoice::Switch(_)))).collect::<Vec<_>>();
    (sides.len()==1).then(||sides[0])
}

struct HpSwitchTransition {
    original_active:[PokeId;2],
    result:Battle,
    trace:Vec<Draw>,
}

impl HpSwitchTransition {
    fn from_result(base:&Battle,choices:[Option<SearchChoice>;2],result:&Battle,trace:&[Draw])->Option<Self> {
        if result.outcome().is_some() {return None;}
        let side=switching_side(choices)?;
        let original_active=[base.active_id(0)?,base.active_id(1)?];
        if result.active_id(1-side)!=Some(original_active[1-side])
            || result.active_id(side)==Some(original_active[side]) {return None;}
        for id in original_active {
            if result.poke(id).hp!=base.poke(id).hp || result.poke(id).fainted {return None;}
        }
        Some(Self {original_active,result:result.clone(),trace:trace.to_vec()})
    }

    /// The key retains the incoming Pokemon's exact HP. Only the two original
    /// actives vary; admitted attacks cannot hit the outgoing Pokemon or
    /// change the attacker's HP, and neither original HP changes during reuse.
    fn translate_hp(&self,target:&Battle)->Option<(Battle,Vec<Draw>)> {
        let mut result=self.result.clone();
        for id in self.original_active {result.poke_mut(id).hp=target.poke(id).hp;}
        Some((result,self.trace.clone()))
    }
}


#[derive(Default,Clone,serde::Serialize)]
pub struct HpHealingStats {
    pub observations:usize,
    pub eligible:usize,
    pub attack_eligible:usize,
    pub switch_eligible:usize,
    pub hits:usize,
    pub attack_hits:usize,
    pub switch_hits:usize,
    pub reset_hits:usize,
    pub rejected_hp:usize,
    pub runs:usize,
    pub peak_entries:usize,
    pub exclusions:BTreeMap<&'static str,usize>,
}

enum HpSharedTransition {Healing(HpHealingTransition),Attack(HpTransition),Switch(HpSwitchTransition)}

impl HpSharedTransition {
    fn translate_hp(&self,target:&Battle)->Option<(Battle,Vec<Draw>)> {
        match self {Self::Healing(entry)=>entry.translate_hp(target),Self::Attack(entry)=>entry.translate_hp(target),Self::Switch(entry)=>entry.translate_hp(target)}
    }
    fn resets_hp(&self)->bool {matches!(self,Self::Healing(entry) if entry.resets_hp())}
}

pub struct HpHealingCache {
    capacity:usize,
    attacks:bool,
    switches:bool,
    entries:HashMap<HpTransitionKey,HpSharedTransition>,
    order:VecDeque<HpTransitionKey>,
    stats:HpHealingStats,
}

pub struct HpHealingRunner<'a> {
    cache:&'a mut HpHealingCache,
    dex:&'a Dex,
    base:&'a Battle,
    choices:[Option<SearchChoice>;2],
    key:Option<HpTransitionKey>,
    exclusion:Option<&'static str>,
    attack:bool,
    switch:bool,
}

impl HpHealingCache {
    pub fn new(capacity:usize)->Self {
        Self {capacity,attacks:false,switches:false,entries:HashMap::new(),order:VecDeque::new(),stats:HpHealingStats::default()}
    }
    pub fn with_attacks(mut self)->Self {
        assert_eq!(self.stats.observations,0);self.attacks=true;self
    }
    pub fn with_switches(mut self)->Self {
        assert_eq!(self.stats.observations,0);self.switches=true;self
    }
    pub fn stats(&self)->HpHealingStats {self.stats.clone()}
    pub fn prepare<'a>(&'a mut self,dex:&'a Dex,base:&'a Battle,choices:[Option<SearchChoice>;2])->HpHealingRunner<'a> {
        let mut attack=false;
        let (mut key,mut exclusion)=if self.capacity==0 {(None,Some("disabled"))} else {
            match HpTransitionKey::healing_domain(dex,base,choices,&[]) {
                Ok(key)=>(Some(key),None),
                Err(reason)=>match self.attacks.then(||HpTransitionKey::new_domain(dex,base,choices,&[],None,None)) {
                    Some(Ok(key))=>{attack=true;(Some(key),None)},
                    Some(Err(attack_reason))=> {
                        let reason=if reason=="no_single_healer" {
                            if choices.iter().any(|choice|matches!(choice,Some(SearchChoice::Move(id))
                                if matches!(dex.moves.key(*id),"selfdestruct"|"explosion"))) {"boom_choice"}
                            else {attack_reason}
                        } else {reason};
                        (None,Some(reason))
                    },
                    None=>(None,Some(reason)),
                },
            }
        };
        let mut switch=false;
        if key.is_none() && self.capacity>0 && self.switches {
            match HpTransitionKey::switching_domain(dex,base,choices,&[]) {
                Ok(found)=>{key=Some(found);exclusion=None;switch=true;},
                Err(reason)=>if choices.iter().any(|choice|matches!(choice,Some(SearchChoice::Switch(_)))) {exclusion=Some(reason);},
            }
        }
        HpHealingRunner {cache:self,dex,base,choices,key,exclusion,attack,switch}
    }
}

impl HpHealingRunner<'_> {
    pub fn run(&mut self,script:&[usize])->(Battle,Vec<Draw>,bool) {
        self.cache.stats.observations+=1;
        if let Some(reason)=self.exclusion {*self.cache.stats.exclusions.entry(reason).or_default()+=1;}
        let key=self.key.as_ref().map(|key| {let mut key=key.clone();key.script=script.to_vec();key});
        self.cache.stats.eligible+=usize::from(key.is_some());
        self.cache.stats.attack_eligible+=usize::from(self.attack);
        self.cache.stats.switch_eligible+=usize::from(self.switch);
        if let Some(entry)=key.as_ref().and_then(|key|self.cache.entries.get(key)) {
            if let Some((battle,trace))=entry.translate_hp(self.base) {
                self.cache.stats.hits+=1;self.cache.stats.reset_hits+=usize::from(entry.resets_hp());
                self.cache.stats.attack_hits+=usize::from(self.attack);
                self.cache.stats.switch_hits+=usize::from(self.switch);
                return (battle,trace,false);
            }
            self.cache.stats.rejected_hp+=1;
        }
        let (battle,trace)=run_scripted(self.dex,self.base,self.choices,script);
        self.cache.stats.runs+=1;
        if let Some(key)=key.filter(|key|!self.cache.entries.contains_key(key)) {
            let entry=if self.switch {HpSwitchTransition::from_result(self.base,self.choices,&battle,&trace).map(HpSharedTransition::Switch)}
                else if self.attack {HpTransition::from_key(key.clone(),self.base,&battle,&trace).map(HpSharedTransition::Attack)}
                else {HpHealingTransition::from_result(self.dex,key.clone(),self.base,self.choices,&battle,&trace).map(HpSharedTransition::Healing)};
            if let Some(entry)=entry {
                if self.cache.entries.len()==self.cache.capacity {
                    self.cache.entries.remove(&self.cache.order.pop_front().unwrap());
                }
                self.cache.order.push_back(key.clone());self.cache.entries.insert(key,entry);
                self.cache.stats.peak_entries=self.cache.stats.peak_entries.max(self.cache.entries.len());
            }
        }
        (battle,trace,true)
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

    fn healing_battle(dex:&Dex,healer:usize,mode:&str)->(Battle,[Option<SearchChoice>;2]) {
        let kit=&["Hydro Pump","Ice Beam","Sleep Talk","Rest"][..];
        let moves=if healer==0 {[kit,&["Seismic Toss"][..]]} else {[&["Seismic Toss"][..],kit]};
        let items=if healer==0 {["Bright Powder","Mint Berry"]} else {["Mint Berry","Bright Powder"]};
        let mut b=battle(dex,moves,items);
        let id=b.active_id(healer).unwrap();
        b.poke_mut(id).hp-=1;b.poke_mut(id).stored_stats[4]=200;
        b.update_speed(dex,id);
        if mode!="awake" {
            b.poke_mut(id).status=Status::Slp;
            b.poke_mut(id).status_state.set_int(DK::Time,if mode=="waking" {1} else {2});
        }
        b.refresh_poke_mask(dex,id);
        let choices=[0,1].map(|side|Some(SearchChoice::Move(dex.moves.id(
            if side!=healer {"seismictoss"} else if mode=="talk" {"sleeptalk"} else {"rest"}).unwrap())));
        (b,choices)
    }

    #[test]
    fn healing_translation_matches_complete_kernels_on_both_sides() {
        use nc2000_engine::battle::enumerate::{enumerate_step,enumerate_step_with_runner,ChanceLeaf};
        let dex=conformance::load_dex();let mut resets=0;let mut attacks=0;let mut rejected=0;
        let aggregate=|leaves:Vec<ChanceLeaf>| {
            let mut out=HashMap::<u128,f64>::new();
            for leaf in leaves {*out.entry(leaf.battle.state_key128()).or_default()+=leaf.prob;}
            out
        };
        for healer in 0..2 {for mode in ["awake","waking","talk"] {
            let (base,choices)=healing_battle(&dex,healer,mode);
            assert!(HpTransitionKey::new_healing(&dex,&base,choices,&[]).is_some());
            for healer_hp in [1,50,100] {for other_hp in [1,80,200] {
                let mut target=base.clone();
                for (side,hp) in [(healer,healer_hp),(1-healer,other_hp)] {
                    let id=target.active_id(side).unwrap();target.poke_mut(id).hp=hp;
                }
                let actual=enumerate_step(&dex,&target,choices,100_000).unwrap();
                let shared=enumerate_step_with_runner(&dex,&target,choices,100_000,&mut |script| {
                    let (_,_,entry)=HpHealingTransition::run(&dex,&base,choices,script);
                    let entry=entry.expect("high HP reference must survive");
                    let direct=run_scripted(&dex,&target,choices,script);
                    if let Some(found)=entry.translate(&dex,&target,choices,script) {
                        resets+=usize::from(entry.resets_hp());attacks+=usize::from(!entry.resets_hp());
                        assert_eq!(found.0.state_key128(),direct.0.state_key128());
                        assert_eq!(trace_signature(&found.1),trace_signature(&direct.1));
                        (found.0,found.1,false)
                    } else {rejected+=1;(direct.0,direct.1,true)}
                }).unwrap();
                let expected=aggregate(actual.leaves);let found=aggregate(shared.leaves);
                assert_eq!(found.len(),expected.len());
                for (key,mass) in found {assert!((mass-expected[&key]).abs()<1e-10);}
            }}
        }}
        assert!(resets>100 && attacks>100 && rejected>100);
    }

    #[test]
    fn healing_guards_full_hp_action_order_status_and_called_pool() {
        let dex=conformance::load_dex();
        for mode in ["awake","waking","talk"] {
            let (base,choices)=healing_battle(&dex,1,mode);
            let id=base.active_id(1).unwrap();let foe=base.active_id(0).unwrap();
            let rejects=|b:&Battle|assert!(HpTransitionKey::new_healing(&dex,b,choices,&[]).is_none());
            let mut b=base.clone();b.poke_mut(id).hp=b.poke(id).maxhp;rejects(&b);
            b=base.clone();b.poke_mut(id).stored_stats[4]=1;rejects(&b);
            b=base.clone();b.poke_mut(id).stored_stats[4]=b.poke(foe).stored_stats[4];rejects(&b);
            b=base.clone();b.poke_mut(id).status=Status::Par;rejects(&b);
            b=base.clone();b.poke_mut(id).status=Status::Frz;rejects(&b);
            for name in ["Quick Claw","Leftovers","Gold Berry","Mint Berry"] {
                b=base.clone();b.poke_mut(id).item=Some(dex.items.id(&name.to_lowercase().replace(' ',"")).unwrap());rejects(&b);
            }
            b=base.clone();b.poke_mut(foe).item=dex.known_items.quickclaw;rejects(&b);
            b=base.clone();b.poke_mut(id).status=Status::Slp;b.poke_mut(id).status_state.set_int(DK::Time,if mode=="talk" {1} else {2});rejects(&b);
            if mode=="talk" {
                b=base.clone();b.poke_mut(id).move_slots[0].id=dex.moves.id("selfdestruct").unwrap();rejects(&b);
            }
        }
    }


    #[test]
    fn prepared_healing_cache_matches_kernels_and_counts_actual_work() {
        use nc2000_engine::battle::enumerate::{enumerate_step,enumerate_step_with_runner};
        let dex=conformance::load_dex();let (base,choices)=healing_battle(&dex,1,"talk");
        for capacity in [0,3,4096] {
            let mut cache=HpHealingCache::new(capacity);
            for hp in [200,150,1,100] {
                let mut target=base.clone();let id=target.active_id(1).unwrap();target.poke_mut(id).hp=hp;
                let before=cache.stats().runs;
                let mut runner=cache.prepare(&dex,&target,choices);
                let found=enumerate_step_with_runner(&dex,&target,choices,100_000,&mut |script| {
                    let found=runner.run(script);let actual=run_scripted(&dex,&target,choices,script);
                    assert_eq!(found.0.state_key128(),actual.0.state_key128());
                    assert_eq!(trace_signature(&found.1),trace_signature(&actual.1));
                    found
                }).unwrap();
                assert_eq!(found.runs,cache.stats().runs-before);
                let expected=enumerate_step(&dex,&target,choices,100_000).unwrap();
                let aggregate=|leaves:Vec<nc2000_engine::battle::enumerate::ChanceLeaf>| {
                    let mut map=HashMap::<u128,f64>::new();
                    for leaf in leaves {*map.entry(leaf.battle.state_key128()).or_default()+=leaf.prob;}
                    map
                };
                let found=aggregate(found.leaves);let expected=aggregate(expected.leaves);
                assert_eq!(found.len(),expected.len());
                for (key,mass) in found {assert!((mass-expected[&key]).abs()<1e-10);}
            }
            let stats=cache.stats();assert!(stats.peak_entries<=capacity);
            assert_eq!(stats.observations,stats.runs+stats.hits);
            if capacity==0 {assert_eq!(stats.hits,0);} else if capacity==4096 {assert!(stats.hits>100 && stats.reset_hits>0 && stats.rejected_hp>0);}
        }
    }


    #[test]
    fn prepared_combined_cache_preserves_attack_kernels_and_boundaries() {
        use nc2000_engine::battle::enumerate::{enumerate_step,enumerate_step_with_runner};
        let dex=conformance::load_dex();
        let base=battle(&dex,[&["Seismic Toss"],&["Ice Beam"]],["Mint Berry","Bright Powder"]);
        let choices=["seismictoss","icebeam"].map(|name|Some(SearchChoice::Move(dex.moves.id(name).unwrap())));
        let mut cache=HpHealingCache::new(4096).with_attacks();
        for hp in [220,180,1,150] {
            let mut target=base.clone();
            for side in 0..2 {let id=target.active_id(side).unwrap();target.poke_mut(id).hp=hp;}
            let mut runner=cache.prepare(&dex,&target,choices);
            let found=enumerate_step_with_runner(&dex,&target,choices,100_000,&mut |script| {
                let found=runner.run(script);let actual=run_scripted(&dex,&target,choices,script);
                assert_eq!(found.0.state_key128(),actual.0.state_key128());
                assert_eq!(trace_signature(&found.1),trace_signature(&actual.1));found
            }).unwrap();
            let expected=enumerate_step(&dex,&target,choices,100_000).unwrap();
            let aggregate=|leaves:Vec<nc2000_engine::battle::enumerate::ChanceLeaf>| {
                let mut map=HashMap::<u128,f64>::new();
                for leaf in leaves {*map.entry(leaf.battle.state_key128()).or_default()+=leaf.prob;}
                map
            };
            let found=aggregate(found.leaves);let expected=aggregate(expected.leaves);
            assert_eq!(found.len(),expected.len());
            for (key,mass) in found {assert!((mass-expected[&key]).abs()<1e-10);}
        }
        let stats=cache.stats();assert!(stats.attack_hits>100 && stats.rejected_hp>0);
        assert_eq!(stats.attack_hits,stats.hits);assert_eq!(stats.attack_eligible,stats.eligible);
        assert_eq!(stats.observations,stats.hits+stats.runs);
    }


    fn switching_battle(dex:&Dex,switcher:usize,sleeping:bool)->(Battle,[Option<SearchChoice>;2]) {
        let mon=|species:&str,moves:&[&str],item:&str|PokemonSet {
            name:species.into(),species:species.into(),item:item.into(),ability:String::new(),
            moves:moves.iter().map(|m|(*m).into()).collect(),level:50,evs:None,ivs:None,gender:None,happiness:None};
        let attacker=vec![mon("Snorlax",&["Body Slam"],"Mint Berry")];
        let switching=vec![mon("Vaporeon",&["Hydro Pump","Rest"],"Bright Powder"),mon("Snorlax",&["Self-Destruct"],"Leftovers")];
        let teams=if switcher==0 {[switching,attacker]} else {[attacker,switching]};
        let mut b=Battle::from_fixture(dex,"1,2,3,4",&teams[0],&teams[1]).unwrap();
        let preview=[0,1].map(|side|b.legal_choices(dex,side).first().copied());b.apply_choices(dex,preview).unwrap();
        if sleeping {
            let id=b.active_id(switcher).unwrap();b.poke_mut(id).status=Status::Slp;
            b.poke_mut(id).status_state.set_int(DK::Time,1);b.poke_mut(id).status_state.set_int(DK::StartTime,3);
            b.refresh_poke_mask(dex,id);
        }
        let choices=[0,1].map(|side|Some(if side==switcher {SearchChoice::Switch(2)} else {SearchChoice::Move(dex.moves.id("bodyslam").unwrap())}));
        (b,choices)
    }

    #[test]
    fn prepared_switch_cache_matches_full_kernels_including_incoming_ko() {
        use nc2000_engine::battle::enumerate::{enumerate_step,enumerate_step_with_runner};
        let dex=conformance::load_dex();let mut hits=0;let mut replacement_hits=0;
        for switcher in 0..2 {for sleeping in [false,true] {for incoming_hp in [20,133,200] {
            let (mut base,choices)=switching_battle(&dex,switcher,sleeping);
            let incoming=base.sides[switcher].party[1] as usize;base.sides[switcher].roster[incoming].hp=incoming_hp;
            let mut cache=HpHealingCache::new(4096).with_attacks().with_switches();
            for hp in [[200,200],[1,200],[150,1],[1,1]] {
                let mut target=base.clone();
                for side in 0..2 {let id=target.active_id(side).unwrap();target.poke_mut(id).hp=hp[side];}
                let expected=enumerate_step(&dex,&target,choices,100_000).unwrap();
                let mut runner=cache.prepare(&dex,&target,choices);
                let found=enumerate_step_with_runner(&dex,&target,choices,100_000,&mut |script| {
                    let found=runner.run(script);let actual=run_scripted(&dex,&target,choices,script);
                    if !found.2 {hits+=1;replacement_hits+=usize::from(found.0.request_state==RequestState::Switch);}
                    assert_eq!(found.0.state_key128(),actual.0.state_key128());
                    assert_eq!(trace_signature(&found.1),trace_signature(&actual.1));found
                }).unwrap();
                let aggregate=|leaves:Vec<nc2000_engine::battle::enumerate::ChanceLeaf>| {
                    let mut map=HashMap::<u128,f64>::new();
                    for leaf in leaves {*map.entry(leaf.battle.state_key128()).or_default()+=leaf.prob;}
                    map
                };
                let found=aggregate(found.leaves);let expected=aggregate(expected.leaves);
                assert_eq!(found.len(),expected.len());
                for (key,mass) in found {assert!((mass-expected[&key]).abs()<1e-10);}
            }
            let stats=cache.stats();assert_eq!(stats.switch_hits,stats.hits);
            assert_eq!(stats.switch_eligible,stats.eligible);assert_eq!(stats.observations,stats.runs+stats.hits);
        }}}
        assert!(hits>1000 && replacement_hits>100);
    }

    #[test]
    fn switching_keys_preserve_incoming_hp_and_reject_hp_sensitive_paths() {
        let dex=conformance::load_dex();let (base,choices)=switching_battle(&dex,1,false);
        let original=HpTransitionKey::new_switching(&dex,&base,choices,&[]).unwrap();
        let mut b=base.clone();for side in 0..2 {let id=b.active_id(side).unwrap();b.poke_mut(id).hp=1;}
        assert_eq!(HpTransitionKey::new_switching(&dex,&b,choices,&[]),Some(original.clone()));
        let incoming=b.sides[1].party[1] as usize;b.sides[1].roster[incoming].hp-=1;
        assert_ne!(HpTransitionKey::new_switching(&dex,&b,choices,&[]),Some(original));
        let outgoing=base.active_id(1).unwrap();let attacker=base.active_id(0).unwrap();
        let mut b=base.clone();b.poke_mut(outgoing).trapped=true;assert!(HpTransitionKey::new_switching(&dex,&b,choices,&[]).is_none());
        for pos in [0,1,99] {let mut choices=choices;choices[1]=Some(SearchChoice::Switch(pos));assert!(HpTransitionKey::new_switching(&dex,&base,choices,&[]).is_none());}
        for name in ["pursuit","doubleedge","reversal"] {
            let mut b=base.clone();let mv=dex.moves.id(name).unwrap();b.poke_mut(attacker).move_slots[0].id=mv;
            let mut choices=choices;choices[0]=Some(SearchChoice::Move(mv));assert!(HpTransitionKey::new_switching(&dex,&b,choices,&[]).is_none());
        }
        let mut b=base.clone();b.poke_mut(attacker).item=dex.items.id("leftovers");assert!(HpTransitionKey::new_switching(&dex,&b,choices,&[]).is_none());
        let mut b=base.clone();b.sides[1].roster[incoming].item=dex.items.id("focusband");assert!(HpTransitionKey::new_switching(&dex,&b,choices,&[]).is_none());
    }

}
