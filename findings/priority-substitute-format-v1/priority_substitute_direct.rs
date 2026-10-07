use conformance::load_dex;
use nc2000_engine::{battle::{enumerate::enumerate_step_partial, Outcome, SearchChoice}, dex::Dex, state::Battle};
use serde_json::json;

#[allow(dead_code)]
mod fixture {
    include!("priority_substitute_choice.rs");
    pub fn root(dex: &Dex, side: usize) -> Battle { position(dex, side).1 }
}

struct Audit { runs: usize, cells: usize, leaves: usize }

fn outcome(b: &Battle, side: usize) -> Option<f64> {
    b.outcome().map(|o| match o {
        Outcome::P1Win => f64::from(side == 0),
        Outcome::P2Win => f64::from(side == 1),
        Outcome::Tie => 0.5,
    })
}

fn evaluate(dex: &Dex, b: &Battle, side: usize, stage: usize, attack_first: bool, audit: &mut Audit) -> (f64,f64) {
    if let Some(value)=outcome(b,side) {return (value,value);}
    let limit=if attack_first {2} else {3};
    if stage==limit {return (0.0,1.0);}
    let mut legal_b=b.clone();
    let legal=[0,1].map(|s|legal_b.legal_choices(dex,s));
    assert_eq!(legal[1-side].len(),1);
    let own:Vec<_>=if attack_first && stage>0 {legal[side].clone()} else {
        let key=if stage==0 {if attack_first {"agility"} else {"substitute"}} else {"thunderbolt"};
        let choice=SearchChoice::Move(dex.moves.id(key).unwrap());
        assert!(legal[side].contains(&choice));vec![choice]
    };
    let mut best=(0.0f64,0.0f64);
    for action in own {
        let mut joint=[None;2];joint[side]=Some(action);joint[1-side]=Some(legal[1-side][0]);
        let step=enumerate_step_partial(dex,b,joint,4096);
        assert!(!step.exhausted && step.pending.is_empty());
        audit.runs+=step.runs;audit.cells+=1;audit.leaves+=step.leaves.len();
        assert!(audit.runs<=500000);
        let mut value=(0.0,0.0);let mut mass=0.0;
        for leaf in step.leaves {
            let child=evaluate(dex,&leaf.battle,side,stage+1,attack_first,audit);
            value.0+=leaf.prob*child.0;value.1+=leaf.prob*child.1;mass+=leaf.prob;
        }
        assert!((mass-1.0).abs()<1e-9);
        best.0=best.0.max(value.0);best.1=best.1.max(value.1);
    }
    best
}

fn main() {
    let dex=load_dex();
    for side in 0..2 {
        let b=fixture::root(&dex,side);
        for attack_first in [false,true] {
            let mut audit=Audit {runs:0,cells:0,leaves:0};
            let value=evaluate(&dex,&b,side,0,attack_first,&mut audit);
            println!("{}",json!({"side":side,"first":if attack_first {"move agility"} else {"move substitute"},
                "continuation":if attack_first {"all legal own actions"} else {"Thunderbolt at each later turn"},
                "opponent":"all legal actions, asserted singleton at every nonterminal node", "value":value,
                "runs":audit.runs,"cells":audit.cells,"leaves":audit.leaves}));
        }
    }
}
