use nc2000_bot::{
    preview::load_meta_pool,
    smmcts::{SearchTrace, SelRule},
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::BufRead, path::Path};

#[derive(Default)]
struct Group {
    actions: Vec<String>,
    counts: BTreeMap<u8, Vec<u32>>,
    loadouts: BTreeMap<String, BTreeMap<u8, Vec<u32>>>,
    nodes: BTreeMap<usize, (u8, Vec<u32>, Vec<f64>)>,
}

fn snapshot(s: &BlindSearch, dex: &Dex) -> Value {
    json!({"best":s.best().unwrap().to_input(dex),"visits":s.visits(),"means":s.means(),"rng":s.search_rng_state(),"nodes":s.node_count()})
}

fn information(
    b: &Battle,
    side: usize,
    own_choice: SearchChoice,
    actions: &[SearchChoice],
    dex: &Dex,
) -> String {
    let foe = &b.sides[1 - side];
    serde_json::to_string(&json!({"turn":b.turn,"request":format!("{:?}",b.request_state),
        "own":format!("{:?}",b.sides[side]),"own_choice":own_choice.to_input(dex),
        "actions":actions.iter().map(|a|a.to_input(dex)).collect::<Vec<_>>(),
        "field":format!("{:?}",b.field),"coin":b.quick_claw_roll,"log":b.log,
        "foe_active":foe.active,"foe_left":foe.pokemon_left,"foe_fainted":foe.total_fainted,
        "foe_side_conditions":format!("{:?}",foe.side_conditions),"foe_slot_conditions":format!("{:?}",foe.slot_conditions),
        "foe_counters":foe.roster.iter().map(|p|format!("{:?}",(p.hp,p.maxhp,p.status,&p.status_state,&p.volatiles,p.boosts))).collect::<Vec<_>>()
    })).unwrap()
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 3, "corpus output");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut cases = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        b.set_log_enabled(true);
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        let mut selected = None;
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
            if b.is_idle_move_request() {
                for side in 0..2 {
                    let foe = &b.sides[1 - side];
                    let revealed = foe
                        .roster
                        .iter()
                        .filter(|p| p.previously_switched_in > 0 || p.is_active)
                        .count();
                    if acts[side].len() < 2 || revealed >= foe.party.len() {
                        continue;
                    }
                    let rank = b
                        .state_key128()
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add(side as u128);
                    if selected
                        .as_ref()
                        .is_none_or(|(old, _, _, _, _)| rank < *old)
                    {
                        selected = Some((rank, decision, side, b.clone(), observers[side].clone()));
                    }
                }
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| acts[s][i as usize])),
            )
            .unwrap();
            for obs in &mut observers {
                obs.observe(&b, &dex);
            }
        }
        if let Some((_, decision, side, b, obs)) = selected {
            cases.push((game["game"].clone(), decision, side, b, obs));
        }
    }
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut rows = Vec::new();
    for (game, decision, side, b, obs) in cases {
        let belief = Belief::new(&dex, &pool, &obs);
        let seed = 0x2026103000 ^ b.state_key128() as u64 ^ ((side as u64) << 32);
        let mut plain = BlindSearch::new(&b, &dex, cfg.clone(), side, seed);
        plain.step(&dex, &belief, &obs, 27000);
        let expected = snapshot(&plain, &dex);
        drop(plain);
        let mut search = BlindSearch::new(&b, &dex, cfg.clone(), side, seed);
        let mut groups = BTreeMap::<String, Group>::new();
        let mut shadow_checks = 0;
        let mut continuation_choices = 0;
        for _ in 0..27000 {
            let mut depth = 0;
            let mut shadow = None;
            let mut own_choice = None;
            let mut pending = None;
            search.step_observed(&dex, &belief, &obs, 1, &mut |event| match event {
                SearchTrace::Choice {
                    battle,
                    node,
                    actions,
                    visits,
                    rewards,
                    chosen,
                } => {
                    if depth == 0 {
                        let mut copy = battle.clone();
                        copy.set_log_enabled(true);
                        copy.apply_choices(&dex, chosen).unwrap();
                        shadow = Some(copy);
                        own_choice = chosen[side];
                    } else if depth == 1 {
                        continuation_choices += 1;
                        let copy = shadow.as_ref().unwrap();
                        assert_eq!(copy.state_key128(), battle.state_key128());
                        shadow_checks += 1;
                        if battle.is_idle_move_request() && actions[side].len() > 1 {
                            let info =
                                information(copy, side, own_choice.unwrap(), &actions[side], &dex);
                            let mask = battle.sides[1 - side]
                                .party
                                .iter()
                                .fold(0u8, |m, &slot| m | (1 << slot));
                            let group = groups.entry(info.clone()).or_default();
                            let names = actions[side]
                                .iter()
                                .map(|a| a.to_input(&dex))
                                .collect::<Vec<_>>();
                            if group.actions.is_empty() {
                                group.actions = names;
                            } else {
                                assert_eq!(group.actions, names);
                            }
                            let action = actions[side]
                                .iter()
                                .position(|&a| Some(a) == chosen[side])
                                .unwrap();
                            group
                                .counts
                                .entry(mask)
                                .or_insert_with(|| vec![0; actions[side].len()])[action] += 1;
                            let loadout = format!(
                                "{:?}",
                                battle.sides[1 - side]
                                    .roster
                                    .iter()
                                    .map(|p| (
                                        (p.species, p.base_species, p.level, p.gender, p.happiness),
                                        (p.set_ivs, p.set_evs, p.base_stored_stats, p.stored_stats),
                                        p.item,
                                        p.hp_type,
                                        p.hp_power,
                                        p.base_move_slots,
                                        p.move_slots,
                                    ))
                                    .collect::<Vec<_>>()
                            );
                            let signature = format!("{:x}", Sha256::digest(loadout.as_bytes()));
                            group
                                .loadouts
                                .entry(signature)
                                .or_default()
                                .entry(mask)
                                .or_insert_with(|| vec![0; actions[side].len()])[action] += 1;
                            group
                                .nodes
                                .insert(node, (mask, visits[side].clone(), rewards[side].clone()));
                            pending = Some((info, node, action));
                        }
                    }
                    depth += 1;
                }
                SearchTrace::Result { reward0, .. } => {
                    if let Some((ref info, node, action)) = pending {
                        groups
                            .get_mut(info)
                            .unwrap()
                            .nodes
                            .get_mut(&node)
                            .unwrap()
                            .2[action] += if side == 0 { reward0 } else { 1. - reward0 };
                    }
                }
                _ => {}
            });
        }
        assert_eq!(snapshot(&search, &dex), expected);
        let total_groups = groups.len();
        let grouped=groups.into_iter().filter(|(_,g)|g.counts.len()>1).map(|(info,g)| {
            let digest=format!("{:x}",Sha256::digest(info.as_bytes()));
            json!({"view_hash":digest,"view":serde_json::from_str::<Value>(&info).unwrap(),"actions":g.actions,
                "types":g.counts.into_iter().map(|(mask,counts)|json!({"selection_mask":mask,"counts":counts})).collect::<Vec<_>>(),
                "loadouts":g.loadouts.into_iter().map(|(signature,types)|json!({"signature":signature,"types":types.into_iter().map(|(mask,counts)|json!({"selection_mask":mask,"counts":counts})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
                "nodes":g.nodes.into_iter().map(|(node,(mask,n,w))|json!({"node":node,"selection_mask":mask,"visits":n,"rewards":w})).collect::<Vec<_>>()})
        }).collect::<Vec<_>>();
        eprintln!(
            "game={game} groups={total_groups} hidden-selection-groups={}",
            grouped.len()
        );
        rows.push(json!({"game":game,"decision":decision,"side":side,"key":b.state_key128().to_string(),"seed":seed,
            "iterations":27000,"baseline_identical":true,"snapshot":expected,"continuation_choices":continuation_choices,
            "shadow_checks":shadow_checks,"total_groups":total_groups,"groups":grouped}));
    }
    std::fs::write(&args[2], serde_json::to_vec(&json!({"rows":rows})).unwrap()).unwrap();
}
