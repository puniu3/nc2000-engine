use super::*;
use nc2000_engine::battle::enumerate::{enumerate_step_with_runner,run_scripted,StepEnum};
use nc2000_engine::state::RequestState;

pub(super) fn admitted(dex:&Dex,b:&Battle,choices:[Option<SearchChoice>;2])->bool {
    let names=choices.map(|choice|match choice {Some(SearchChoice::Move(id))=>dex.moves.key(id),_=>""});
    if !(names==["bodyslam","selfdestruct"] || names==["selfdestruct","bodyslam"])
        || !b.started || b.ended || b.turn==0 || b.mid_turn || b.request_state!=RequestState::Move
        || b.needs_choice()!=[true,true] || !b.queue.is_empty() || !b.faint_queue.is_empty()
        || !b.event_stack.is_empty() || !b.effect_stack.is_empty() || b.active_move.is_some()
        || b.active_pokemon.is_some() || b.active_target.is_some() || b.pending_boosts.is_some()
        || b.field.weather.is_some()
        || b.field.pseudo_weather.iter().any(|(id,_)|!matches!(dex.conds.key(*id),"maxtotallevel"|"stadiumsleepclause"|"freezeclausemod")) {return false;}
    for side in 0..2 {
        let team=&b.sides[side];
        if !team.side_conditions.is_empty() || !team.slot_conditions.is_empty()
            || !team.choice.actions.is_empty() || !team.choice.switch_ins.is_empty()
            || team.choice.forced_switches_left!=0 || team.choice.forced_passes_left!=0 {return false;}
        let Some(id)=b.active_id(side) else {return false;};let p=b.poke(id);
        if p.hp<=0 || p.hp>p.maxhp || p.fainted || p.transformed
            || !matches!(p.status,Status::None|Status::Par|Status::Slp|Status::Frz)
            || p.volatiles.iter().any(|(id,_)|dex.conds.key(*id)!="parspeeddrop")
            || p.item.is_some_and(|id|!matches!(dex.items.key(id),"mintberry"|"miracleberry"|"brightpowder"|"quickclaw"|"leftovers")) {return false;}
        let Some(SearchChoice::Move(mv))=choices[side] else {return false;};
        if !p.move_slots.iter().any(|slot|slot.id==mv && slot.pp>0 && !slot.disabled) {return false;}
    }
    true
}

/// Only terminal state identity is weakened. In the admitted two-move domain,
/// HP-sensitive hooks are absent; matching endpoint winners and downstream
/// traces therefore preserve damage-range monotonicity. Live states stay exact.
pub(super) fn enumerate(dex:&Dex,b:&Battle,choices:[Option<SearchChoice>;2],cap:usize,quotient:bool)->(Option<StepEnum>,usize,usize) {
    assert!(admitted(dex,b,choices));
    let mut terminals:[Option<Battle>;3]=[None,None,None];
    let mut runs=0;let mut replacements=0;
    let result=enumerate_step_with_runner(dex,b,choices,cap,&mut |script| {
        let (mut battle,trace)=run_scripted(dex,b,choices,script);runs+=1;
        if quotient {
            if let Some(outcome)=battle.outcome() {
                let index=match outcome {Outcome::P1Win=>0,Outcome::Tie=>1,Outcome::P2Win=>2};
                if let Some(first)=&terminals[index] {battle=first.clone();replacements+=1;}
                else {terminals[index]=Some(battle.clone());}
            }
        }
        (battle,trace,true)
    });
    assert!(runs<=cap);if let Some(step)=&result {assert_eq!(step.runs,runs);}
    (result,runs,replacements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    fn fixture(dex:&Dex,bench:bool)->Battle {
        let mon=|species:&str,moves:&[&str],item:&str|PokemonSet {
            name:species.into(),species:species.into(),item:item.into(),ability:String::new(),
            moves:moves.iter().map(|m|(*m).into()).collect(),level:50,evs:None,ivs:None,gender:None,happiness:None};
        let own=mon("Snorlax",&["Body Slam","Self-Destruct"],"Mint Berry");
        let mut foe=vec![mon("Snorlax",&["Body Slam","Self-Destruct"],"Leftovers")];
        if bench {foe.push(mon("Vaporeon",&["Hydro Pump","Rest"],"Bright Powder"));}
        let mut b=Battle::from_fixture(dex,"1,2,3,4",&[own],&foe).unwrap();
        let choices=[0,1].map(|s|b.legal_choices(dex,s)[0]);b.apply_choices(dex,choices.map(Some)).unwrap();b
    }
    fn outcome_key(b:&Battle)->(u8,u128) {
        match b.outcome() {Some(Outcome::P1Win)=>(1,0),Some(Outcome::Tie)=>(2,0),Some(Outcome::P2Win)=>(3,0),None=>(0,b.state_key128())}
    }
    #[test]
    fn terminal_ranges_match_unmerged_scripts_in_both_orientations() {
        let dex=conformance::load_dex();let mut saved=0;let mut total=0;
        for bench in [false,true] {for flipped in [false,true] {for hp in [[20,20],[100,100],[171,133],[250,250]] {
            let mut b=fixture(&dex,bench);
            for side in 0..2 {let id=b.active_id(side).unwrap();b.poke_mut(id).hp=hp[side].min(b.poke(id).maxhp);}
            let choices=if flipped {["selfdestruct","bodyslam"]} else {["bodyslam","selfdestruct"]};
            let choices=choices.map(|name|Some(SearchChoice::Move(dex.moves.id(name).unwrap())));
            let mut cache=nc2000_bot::hp_transition::HpHealingCache::new(0);
            let (raw,unknown,runs)=super::super::pair_bound::partial_step(&dex,&b,choices,200_000,&mut cache);
            assert_eq!(unknown,0.0);
            let mut expected=BTreeMap::<(u8,u128),f64>::new();
            for (_, (leaf,mass)) in raw {*expected.entry(outcome_key(&leaf)).or_default()+=mass;}
            let (found,work,_)=enumerate(&dex,&b,choices,200_000,true);let found=found.unwrap();
            let mut masses=BTreeMap::<(u8,u128),f64>::new();
            for leaf in found.leaves {*masses.entry(outcome_key(&leaf.battle)).or_default()+=leaf.prob;}
            assert_eq!(masses.len(),expected.len());
            for (key,mass) in masses {assert!((mass-expected[&key]).abs()<1e-10,"{key:?}: {mass} != {}",expected[&key]);}
            total+=runs;saved+=runs.saturating_sub(work);
        }}}
        assert!(saved>total/2);
    }
    #[test]
    fn terminal_quotient_rejects_hp_dependent_hooks_and_honors_caps() {
        let dex=conformance::load_dex();let base=fixture(&dex,true);
        let choices=["bodyslam","selfdestruct"].map(|name|Some(SearchChoice::Move(dex.moves.id(name).unwrap())));
        assert!(admitted(&dex,&base,choices));
        for item in ["berry","goldberry","focusband"] {
            let mut b=base.clone();let id=b.active_id(0).unwrap();b.poke_mut(id).item=dex.items.id(item);
            assert!(!admitted(&dex,&b,choices));
        }
        let mut b=base.clone();let id=b.active_id(0).unwrap();b.add_volatile(&dex,id,"destinybond",None,nc2000_engine::battle::EffectHandle::None);
        assert!(!admitted(&dex,&b,choices));
        let mut b=base.clone();b.poke_mut(id).status=Status::Brn;assert!(!admitted(&dex,&b,choices));
        for cap in [0,1,7] {let (step,runs,_)=enumerate(&dex,&base,choices,cap,true);assert!(step.is_none());assert_eq!(runs,cap);}
    }
}
