use nc2000_bot::{smmcts::dominated_actions, Observer};
use nc2000_engine::{battle::PokemonSet, dex::Category, state::Battle, validate::Learnsets};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::BufRead;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "corpus output");
    let dex = conformance::load_dex();
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut rows = Vec::new();
    let mut total = 0;
    let mut games = 0;
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        games += 1;
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        b.set_log_enabled(true);
        let mut obs = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
            for side in [0, 1] {
                if !b.needs_choice()[side] || legal[side].len() < 2 {
                    continue;
                }
                total += 1;
                let (Some(me), Some(foe)) = (b.active_id(side), b.active_id(1 - side)) else {
                    continue;
                };
                let enemy = b.poke(foe);
                let knowledge = &obs[side].mons()[foe.slot as usize];
                let priorities = learnsets
                    .species(dex.species.key(enemy.species))
                    .map(|ls| {
                        ls.moves
                            .iter()
                            .filter_map(|m| dex.moves.id(m))
                            .filter(|&m| {
                                let ms = dex.move_static(m);
                                ms.priority > 0 && ms.category != Category::Status
                            })
                            .map(|m| dex.moves.key(m).to_string())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let masked = dominated_actions(&b, &dex, side);
                for (_, why) in &masked {
                    *counts.entry(why.to_string()).or_default() += 1;
                }
                let sub = b
                    .poke(me)
                    .volatile(dex.conds_id("substitute").unwrap())
                    .map(|v| v.get_int(nc2000_engine::state::DK::Hp));
                let sub_masked = masked
                    .iter()
                    .any(|(a, _)| a.to_input(&dex) == "move substitute");
                if sub_masked && sub.is_some() && !priorities.is_empty() {
                    rows.push(json!({"kind":"priority_substitute","game":game["game"],"decision":decision,"side":side,
                        "state_key":d["key"],"own":dex.species.key(b.poke(me).species),"foe":dex.species.key(enemy.species),
                        "priority_candidates":priorities,"revealed_moves":knowledge.revealed_moves.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>(),
                        "substitute_hp":sub,"own_hp":b.poke(me).hp,"own_maxhp":b.poke(me).maxhp}));
                }
                if b.sides[0].pokemon_left == 1
                    && b.sides[1].pokemon_left == 1
                    && knowledge.revealed_moves.len() == 4
                {
                    rows.push(json!({"kind":"revealed_duel","game":game["game"],"decision":decision,"side":side,
                        "state_key":d["key"],"own":dex.species.key(b.poke(me).species),"foe":dex.species.key(enemy.species),
                        "item_known":knowledge.item.current.is_some(),"legal":legal[side].iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()}));
                }
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| legal[s][i as usize])),
            )
            .unwrap();
            for o in &mut obs {
                o.observe(&b, &dex);
            }
        }
    }
    let output = json!({"games":games,"decisions":total,"mask_counts":counts,"opportunities":rows});
    eprintln!(
        "games={games} decisions={total} opportunities={}",
        rows.len()
    );
    std::fs::write(&args[2], serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
