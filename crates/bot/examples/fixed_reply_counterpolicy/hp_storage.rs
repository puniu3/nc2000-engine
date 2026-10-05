use super::*;
use std::collections::{BTreeMap,HashSet};

#[derive(Default)]
struct Group {
    states: BTreeMap<u128,(Vec<Vec<i32>>,f64)>,
    template: Option<Battle>,
    roundtrips: usize,
    actions: HashSet<[Option<SearchChoice>;2]>,
    eligible: HashSet<u128>,
}

#[derive(Default)]
pub(super) struct Census { groups:BTreeMap<(bool,bool,u128),Group> }

fn key(dex:&Dex,b:&Battle,damage:bool,claw:bool)->u128 {
    if claw {b.state_key128_without_quick_claw_roll(dex,damage)}
    else if damage {b.state_key128_without_damage_bookkeeping()}
    else {b.state_key128()}
}

impl Census {
    pub(super) fn observe(&mut self,dex:&Dex,b:&Battle,mass:f64,response:CounterResponse) {
        let damage=!b.damage_bookkeeping_observable(dex);
        let claw=!b.quick_claw_roll_observable(dex);
        let state=key(dex,b,damage,claw);
        let mut shape=b.clone();
        let hp=shape.sides.iter().map(|side|side.roster.iter().map(|p|p.hp).collect::<Vec<_>>()).collect::<Vec<_>>();
        for side in &mut shape.sides {for p in &mut side.roster {if p.hp>0 {p.hp=p.maxhp;}}}
        let shape_key=key(dex,&shape,damage,claw);
        let group=self.groups.entry((damage,claw,shape_key)).or_default();
        let template=group.template.get_or_insert(shape);
        let mut restored=template.clone();
        for (side,hps) in restored.sides.iter_mut().zip(&hp) {for (p,&hp) in side.roster.iter_mut().zip(hps) {p.hp=hp;}}
        assert_eq!(key(dex,&restored,damage,claw),state);
        let choices=[0,1].map(|side|choose_response(dex,&mut restored,side,false,OwnPolicy::AfterBoom75,response));
        let mut original=b.clone();
        let expected=[0,1].map(|side|choose_response(dex,&mut original,side,false,OwnPolicy::AfterBoom75,response));
        assert_eq!(choices,expected);
        group.roundtrips+=1;
        let eligible=nc2000_bot::hp_transition::HpTransitionKey::new(dex,b,choices,&[]).is_some();
        let entry=group.states.entry(state).or_insert((hp.clone(),0.0));
        assert_eq!(entry.0,hp);entry.1+=mass;
        group.actions.insert(choices);
        if eligible {group.eligible.insert(state);}
    }

    pub(super) fn report(&self,width:usize)->Value {
        let mut states=self.groups.values().flat_map(|g|g.states.values().map(|(_,mass)|*mass)).collect::<Vec<_>>();
        let mut shapes=self.groups.values().map(|g|g.states.values().map(|(_,mass)|mass).sum::<f64>()).collect::<Vec<_>>();
        states.sort_by(|a,b|b.total_cmp(a));shapes.sort_by(|a,b|b.total_cmp(a));
        let eligible_states:usize=self.groups.values().map(|g|g.eligible.len()).sum();
        let eligible_mass:f64=self.groups.values().map(|g|g.eligible.iter().map(|key|g.states[key].1).sum::<f64>()).sum();
        json!({"candidate_states":states.len(),"hp_shapes":shapes.len(),"candidate_mass":states.iter().sum::<f64>(),
            "top_state_mass":states.iter().take(width).sum::<f64>(),"top_shape_mass":shapes.iter().take(width).sum::<f64>(),
            "largest_shape":self.groups.values().map(|g|g.states.len()).max().unwrap_or(0),
            "multi_action_shapes":self.groups.values().filter(|g|g.actions.len()>1).count(),
            "affine_eligible_states":eligible_states,"affine_eligible_mass":eligible_mass,
            "roundtrip_checks":self.groups.values().map(|g|g.roundtrips).sum::<usize>(),
            "stored_hp_scalars":self.groups.values().flat_map(|g|g.states.values()).map(|(hp,_)|hp.iter().map(Vec::len).sum::<usize>()).sum::<usize>(),
            "width":width})
    }
}
