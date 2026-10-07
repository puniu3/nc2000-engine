use nc2000_engine::battle::{enumerate::enumerate_step, PokemonSet, SearchChoice};
use nc2000_engine::state::Battle;
use serde_json::json;
use std::collections::BTreeMap;

fn main() {
    let dex = conformance::load_dex();
    let mut records = Vec::new();
    for low in [false, true] {
        for typed in [false, true] {
            let name = if typed { "hiddenpowerice" } else { "hiddenpower" };
            let mon = |species: &str, moves: Vec<&str>| {
                serde_json::from_value::<PokemonSet>(json!({"species":species,"name":species,"level":50,"moves":moves,
                    "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                    "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}})).unwrap()
            };
            let mut own = vec![mon("Zapdos", vec![name]), mon("Snorlax", vec!["Tackle"]), mon("Tauros", vec!["Tackle"])];
            let mut ivs = json!({"hp":30,"atk":30,"def":26,"spa":30,"spd":30,"spe":30});
            if low {
                ivs = json!({"hp":30,"atk":6,"def":2,"spa":30,"spd":30,"spe":2});
            }
            own[0].ivs = Some(serde_json::from_value(ivs).unwrap());
            let foe = vec![mon("Lapras", vec!["Growl"]), mon("Miltank", vec!["Tackle"]), mon("Starmie", vec!["Tackle"])];
            let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &own, &foe).unwrap();
            let choices = [0, 1].map(|s| Some(battle.legal_choices(&dex, s)[0]));
            battle.apply_choices(&dex, choices).unwrap();
            let attacker = battle.active_id(0).unwrap();
            let defender = battle.active_id(1).unwrap();
            let hp = battle.poke(defender).hp;
            let stats = battle.poke(attacker);
            let mut record = json!({"typed":typed,"low":low,"move":name,"hp_type":dex.type_name(stats.hp_type),"hp_power":stats.hp_power,"stats":stats.base_stored_stats,"ivs":stats.set_ivs});
            let choices = [Some(SearchChoice::Move(dex.moves.id(name).unwrap())), Some(SearchChoice::Move(dex.moves.id("growl").unwrap()))];
            let step = enumerate_step(&dex, &battle, choices, 200_000).unwrap();
            let mut histogram = BTreeMap::<i32, f64>::new();
            for leaf in step.leaves {
                *histogram.entry(hp - leaf.battle.poke(defender).hp).or_default() += leaf.prob;
            }
            record["damage"] = json!(histogram);
            record["mean_damage"] = json!(histogram.iter().map(|(damage, probability)| *damage as f64 * probability).sum::<f64>());
            record["runs"] = json!(step.runs);
            records.push(record);
        }
    }
    println!("{}", json!(records));
}
