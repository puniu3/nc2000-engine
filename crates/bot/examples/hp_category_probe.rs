#[path = "../tests/support/hidden_power_damage.rs"]
mod support;

use serde_json::json;
use support::*;

fn main() {
    let dex = conformance::load_dex();
    let mut rows = Vec::new();
    for index in 0..16 {
        for high in [false, true] {
            let b = fixture(&dex, index, high);
            let (native, runs) = damage(&dex, &b);
            let controls = ["tackle", "psychic", "icebeam", "earthquake", "flamethrower"].map(|key| {
                let att = b.active_id(0).unwrap();
                let def = b.active_id(1).unwrap();
                json!({"move":key,"estimate":nc2000_bot::eval::expected_hit_fraction(&b,&dex,att,def,dex.moves.id(key).unwrap(),true)*f64::from(b.poke(def).hp)})
            });
            let perturbations=[true,false].map(|attacker| {
                [false,true].map(|relevant|{
                    let changed=perturb(&b,index,attacker,relevant);
                    let (histogram,runs)=damage(&dex,&changed);
                    json!({"attacker":attacker,"relevant":relevant,"estimate":estimate(&dex,&changed),"native":histogram,"runs":runs})
                })
            });
            rows.push(json!({"type":TYPES[index],"power":if high {70}else{35},"estimate":estimate(&dex,&b),"native":native,"native_mean":native.iter().map(|(d,p)|f64::from(*d)*p).sum::<f64>(),"runs":runs,"perturbations":perturbations,"controls":controls}));
        }
    }
    println!("{}", json!(rows));
}
