use std::io::Write;

use nc2000_bot::corpus::{load_battle, load_sources, reconstruct_context_with_cfg};
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{RmConfig, SplitMix64};
use nc2000_engine::dex::{toid, Dex};
use serde_json::{json, Value};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "hp_evidence_probe PROTOCOL.json OUTPUT.jsonl [--public-constraint]"
    );
    let public_constraint = args.get(3).is_some_and(|s| s == "--public-constraint");
    assert!(args.len() == 3 || public_constraint);
    assert!(!public_constraint || cfg!(feature = "hp-type-evidence"), "build with --features hp-type-evidence");
    let root = std::env::current_dir().unwrap();
    let dex =
        Dex::from_json(&std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap())
            .unwrap();
    let sources = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let protocol: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let iterations = protocol["iterations"].as_u64().unwrap() as u32;
    let cfg = RmConfig {
        iterations,
        c: protocol["c"].as_f64().unwrap(),
        ..Default::default()
    };
    let mut out = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])
            .unwrap(),
    );
    for case in protocol["cases"].as_array().unwrap() {
        let log_path = root.join(case["log"].as_str().unwrap());
        assert!(log_path.is_file(), "missing corpus log");
        let cb = load_battle(&log_path);
        let side = case["side"].as_u64().unwrap() as usize;
        let turn = case["turn"].as_u64().unwrap() as u16;
        let d = cb
            .decisions
            .iter()
            .find(|d| d.side == side && d.turn == turn)
            .unwrap();
        let rec = reconstruct_context_with_cfg(
            &dex,
            &sources,
            pool.clone(),
            &cb.lines,
            &cb.evidence,
            d,
            1,
            cfg.clone(),
        )
        .unwrap();
        let mut spec = rec.agent.to_position_spec(&dex).unwrap();
        let mut own_picks = Vec::new();
        for line in &cb.lines {
            let parts: Vec<_> = line.split('|').collect();
            if parts.len() > 3
                && matches!(parts[1], "switch" | "drag")
                && parts[2].starts_with(&format!("p{}", side + 1))
            {
                let species = toid(parts[3].split(',').next().unwrap());
                if !own_picks.contains(&species) {
                    own_picks.push(species);
                }
            }
        }
        if own_picks.len() != 3 {
            writeln!(
                out,
                "{}",
                json!({"battle":case["battle"],"skip":"own picks incomplete"})
            )
            .unwrap();
            continue;
        }
        let own = &mut spec.sides[side];
        let mut party: Vec<_> = own.active.into_iter().collect();
        for species in own_picks {
            let slot = own.mons.iter().position(|m| m.species == species).unwrap();
            if !party.contains(&slot) {
                party.push(slot);
            }
        }
        own.party = party;
        let inferred = spec.clone();
        for mon in &mut spec.sides[1 - side].mons {
            mon.hp_type_mask = u16::MAX;
        }
        let mut typed = spec.clone();
        let foe = &mut typed.sides[1 - side];
        let mon = &mut foe.mons[foe.active.unwrap()];
        assert_eq!(mon.species, case["foe"].as_str().unwrap());
        assert_eq!(case["allowed_types"].as_array().unwrap().len(), 1);
        let typed_move = format!(
            "hiddenpower{}",
            toid(case["allowed_types"][0].as_str().unwrap())
        );
        let revealed = mon
            .revealed_moves
            .iter_mut()
            .find(|m| m.as_str() == "hiddenpower")
            .unwrap();
        *revealed = typed_move.clone();
        if public_constraint {
            typed = inferred;
        }
        let corrected_arm = if public_constraint {
            "inferred"
        } else {
            "known_type"
        };
        for seed in protocol["seeds"].as_array().unwrap() {
            for (arm, position) in [("baseline", &spec), (corrected_arm, &typed)] {
                let mut agent = ProtocolAgent::new(
                    &dex,
                    side,
                    pool.clone(),
                    cfg.clone(),
                    seed.as_u64().unwrap(),
                );
                agent.set_position(&dex, position).unwrap();
                let model = agent.belief().unwrap().determinize(
                    &dex,
                    agent.battle().unwrap(),
                    agent.observer().unwrap(),
                    &mut SplitMix64::new(1),
                );
                let foe = model.poke(model.active_id(1 - side).unwrap());
                if arm != "inferred" {
                    assert_eq!(foe.hp_power, 70);
                }
                let moves: Vec<_> = foe.move_slots.iter().map(|m| dex.moves.key(m.id)).collect();
                if arm == "known_type" {
                    assert!(moves.contains(&typed_move.as_str()));
                }
                if arm == "inferred" {
                    assert_eq!(
                        dex.type_name(foe.hp_type),
                        case["allowed_types"][0].as_str().unwrap()
                    );
                }
                agent.step(&dex, iterations).unwrap();
                let search = agent.search().unwrap();
                let actions:Vec<_>=search.actions().iter().enumerate().map(|(i,a)|json!({"action":a.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i]})).collect();
                writeln!(out,"{}",json!({"battle":case["battle"],"side":side,"turn":turn,"arm":arm,"seed":seed,"best":agent.best(&dex),"actions":actions,"modeled_moves":moves,"hp_power":foe.hp_power,"own_set_provenance":rec.provenance})).unwrap();
                out.flush().unwrap();
            }
        }
        eprintln!("finished battle {}", case["battle"]);
    }
}
