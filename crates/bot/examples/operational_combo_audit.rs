use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    corpus::{load_battle, load_sources, reconstruct_context_with_cfg},
    import::ProtocolAgent,
    preview::load_meta_pool,
    smmcts::{RmConfig, SelRule},
};
use nc2000_engine::dex::toid;
use serde_json::{json, Value};
use std::{
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| {
        args.iter()
            .position(|s| s == k)
            .map(|i| args[i + 1].clone())
            .unwrap_or_else(|| d.into())
    };
    let root = repo_root();
    let dex = load_dex();
    let sources = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let cases: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(arg("--cases", "tmp/operational-audit/candidates.json")).unwrap(),
    )
    .unwrap();
    let iters = arg("--iters", "27000").parse::<u32>().unwrap();
    let seeds = arg("--seeds", "1").parse::<u64>().unwrap();
    let cursor = AtomicUsize::new(0);
    let out = std::sync::Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(arg("--out", "tmp/operational-audit/search.jsonl")).unwrap(),
    ));
    std::thread::scope(|scope| {
        for _ in 0..6 {
            scope.spawn(|| loop {
                let index = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(case) = cases.get(index) else { break };
                let log = load_battle(&root.join(case["log"].as_str().unwrap()));
                let side = case["side"].as_u64().unwrap() as usize;
                let turn = case["turn"].as_u64().unwrap() as u16;
                let Some(d) = log.decisions.iter().find(|d| d.side == side && d.turn == turn) else { continue };
                let cfg = RmConfig { iterations: iters, c: 0.4, rule: SelRule::Ucb, ..Default::default() };
                let Some(rec) = reconstruct_context_with_cfg(&dex, &sources, pool.clone(), &log.lines, &log.evidence, d, 1, cfg.clone()) else { continue };
                let mut spec = rec.agent.to_position_spec(&dex).unwrap();
                let mut own_picks = Vec::new();
                for ln in &log.lines {
                    let p: Vec<_> = ln.split('|').collect();
                    if p.len() >= 4 && matches!(p[1], "switch" | "drag") && p[2].starts_with(&format!("p{}", side+1)) {
                        let species = toid(p[3].split(',').next().unwrap());
                        if !own_picks.contains(&species) { own_picks.push(species); }
                    }
                }
                if own_picks.len() != 3 {
                    writeln!(out.lock().unwrap(), "{}", json!({"case":case,"skip":"own selection not fully revealed"})).unwrap();
                    continue;
                }
                let me = &mut spec.sides[side];
                let mut party: Vec<_> = me.active.into_iter().collect();
                for species in &own_picks {
                    let slot = me.mons.iter().position(|m| &m.species == species).unwrap();
                    if !party.contains(&slot) { party.push(slot); }
                }
                me.party = party;
                let mut probe_pool = pool.clone();
                let mut conditional_loadout = Value::Null;
                if args.iter().any(|a| a == "--conditioned-probe") {
                    let foe = &spec.sides[1-side];
                    let mon = &foe.mons[foe.active.unwrap()];
                    let mut groups = Vec::<((String, Vec<String>), f64, nc2000_engine::battle::PokemonSet)>::new();
                    for team in &pool.teams {
                        for set in &team.sets {
                            let mut moves: Vec<_> = set.moves.iter().map(|m| toid(m)).collect();
                            if toid(&set.species) != mon.species || !mon.revealed_moves.iter().all(|m| moves.contains(m)) { continue; }
                            moves.sort();
                            let key = (toid(&set.item), moves);
                            if let Some(g) = groups.iter_mut().find(|g| g.0 == key) { g.1 += team.weight.unwrap_or(1.0); }
                            else { groups.push((key, team.weight.unwrap_or(1.0), set.clone())); }
                        }
                    }
                    if let Some((_, _, replacement)) = groups.iter().max_by(|a,b| a.1.total_cmp(&b.1)) {
                        conditional_loadout = json!(replacement);
                        for team in &mut probe_pool.teams {
                            for set in &mut team.sets {
                                if toid(&set.species) == mon.species { *set = replacement.clone(); }
                            }
                        }
                    }
                }
                for seed in 120001..120001+seeds {
                    let mut agent = ProtocolAgent::new(&dex, side, probe_pool.clone(), cfg.clone(), seed);
                    agent.set_position(&dex, &spec).unwrap();
                    agent.step(&dex, iters).unwrap();
                    let search = agent.search().unwrap();
                    let actions: Vec<_> = search.actions().iter().enumerate().map(|(i,a)| json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i],"dominated":search.dominated()[i]})).collect();
                    let best = agent.best(&dex);
                    let belief_info: Value = serde_json::from_str(&agent.belief_info()).unwrap();
                    let mut model_kits = std::collections::BTreeMap::<Vec<String>, usize>::new();
                    let mut rng = nc2000_bot::SplitMix64::new(771);
                    for _ in 0..64 {
                        let b = agent.belief().unwrap().determinize(&dex, agent.battle().unwrap(), agent.observer().unwrap(), &mut rng);
                        let mon = b.poke(b.active_id(1-side).unwrap());
                        let kit = mon.move_slots.iter().map(|m| dex.moves.key(m.id).to_string()).collect();
                        *model_kits.entry(kit).or_default() += 1;
                    }
                    let model_kits: Vec<_> = model_kits.into_iter().map(|(moves,n)| json!({"moves":moves,"n":n})).collect();
                    let row = json!({"case":case,"seed":seed,"iterations":iters,"c":0.4,"pool":"belief-pool-v3","conditional_loadout":conditional_loadout,"belief":belief_info,"model_kits":model_kits,"old_action":format!("{:?}",d.action),"own_set_provenance":rec.provenance,"own_picks":own_picks,"own_pick_source":"full log, own side only","opponent_information":"prefix only","best":best,"actions":actions,"position":spec});
                    let mut out = out.lock().unwrap();
                    writeln!(out,"{row}").unwrap();out.flush().unwrap();
                }
                eprintln!("finished {}/{}", index+1,cases.len());
            });
        }
    });
}
