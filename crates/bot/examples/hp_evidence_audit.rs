use std::collections::BTreeMap;
use std::io::Write;

use nc2000_bot::corpus::{load_battle, load_sources, reconstruct_context_with_cfg};
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{RmConfig, SplitMix64};
use serde_json::{json, Value};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "hp_evidence_audit CASES.json OUTPUT.jsonl");
    let root = std::env::current_dir().unwrap();
    let dex = nc2000_engine::dex::Dex::from_json(
        &std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap(),
    )
    .unwrap();
    let sources = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json"));
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let mut out = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])
            .unwrap(),
    );
    for case in cases {
        let log_path = root.join(case["log"].as_str().unwrap());
        assert!(log_path.is_file(), "missing corpus log");
        let cb = load_battle(&log_path);
        let mut coordinate = case.clone();
        coordinate.as_object_mut().unwrap().remove("log");
        let side = case["side"].as_u64().unwrap() as usize;
        let turn = case["turn"].as_u64().unwrap() as u16;
        let result = (|| -> Result<Value, &'static str> {
            let d = cb
                .decisions
                .iter()
                .find(|d| d.side == side && d.turn == turn)
                .ok_or("no ordinary decision")?;
            let cfg = RmConfig {
                iterations: 1,
                c: 0.4,
                ..Default::default()
            };
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
            .ok_or("reconstruction failed")?;
            let agent = &rec.agent;
            let battle = agent.battle().ok_or("no battle")?;
            let belief = agent.belief().ok_or("no belief")?;
            let observer = agent.observer().ok_or("no observer")?;
            let id = battle.active_id(1 - side).ok_or("no active foe")?;
            let foe = battle.poke(id);
            if dex.species.key(foe.species) != case["foe"].as_str().unwrap() {
                return Err("active foe changed");
            }
            let observed = observer
                .mons()
                .iter()
                .find(|m| m.species == foe.species)
                .unwrap();
            let reveals: Vec<_> = observed
                .revealed_moves
                .iter()
                .map(|m| dex.moves.key(*m))
                .collect();
            if !reveals.iter().any(|m| m.starts_with("hiddenpower")) {
                return Err("Hidden Power reveal missing");
            }
            let allowed: Vec<_> = case["allowed_types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            let mut rng = SplitMix64::new(2026100523);
            let mut hp_types = BTreeMap::<String, usize>::new();
            let mut kits = BTreeMap::<Vec<String>, usize>::new();
            let mut contradictions = 0;
            for _ in 0..64 {
                let model = belief.determinize(&dex, battle, observer, &mut rng);
                let p = model.poke(model.active_id(1 - side).unwrap());
                let slots: Vec<_> = p
                    .move_slots
                    .iter()
                    .filter(|m| dex.moves.key(m.id).starts_with("hiddenpower"))
                    .collect();
                assert_eq!(slots.len(), 1);
                let slot = slots[0];
                let name = if dex.moves.key(slot.id) == "hiddenpower" {
                    dex.type_name(p.hp_type)
                } else {
                    dex.moves.get(slot.id).move_type.as_str()
                };
                if !allowed.contains(&name) {
                    contradictions += 1;
                }
                *hp_types.entry(name.to_string()).or_default() += 1;
                *kits
                    .entry(
                        p.move_slots
                            .iter()
                            .map(|m| dex.moves.key(m.id).to_string())
                            .collect(),
                    )
                    .or_default() += 1;
            }
            let mut erased_lines = cb.lines.clone();
            for line in &mut erased_lines[..=d.cut] {
                let parts: Vec<_> = line.split('|').collect();
                if parts.len() > 3
                    && parts[1] == "move"
                    && parts[2].starts_with(&format!("p{}", 2 - side))
                    && parts[3] == "Hidden Power"
                {
                    line.clear();
                }
            }
            let erased = reconstruct_context_with_cfg(
                &dex,
                &sources,
                pool.clone(),
                &erased_lines,
                &cb.evidence,
                d,
                1,
                cfg,
            )
            .ok_or("ablation reconstruction failed")?;
            let ea = &erased.agent;
            let mut erased_types = BTreeMap::<String, usize>::new();
            let mut erased_rng = SplitMix64::new(2026100523);
            for _ in 0..64 {
                let model = ea.belief().unwrap().determinize(
                    &dex,
                    ea.battle().unwrap(),
                    ea.observer().unwrap(),
                    &mut erased_rng,
                );
                let p = model.poke(model.active_id(1 - side).unwrap());
                let hp = p
                    .move_slots
                    .iter()
                    .find(|m| dex.moves.key(m.id).starts_with("hiddenpower"));
                let name = hp
                    .map(|m| {
                        if dex.moves.key(m.id) == "hiddenpower" {
                            dex.type_name(p.hp_type)
                        } else {
                            dex.moves.get(m.id).move_type.as_str()
                        }
                    })
                    .unwrap_or("absent");
                *erased_types.entry(name.to_string()).or_default() += 1;
            }
            let kits: Vec<_> = kits
                .into_iter()
                .map(|(moves, n)| json!({"moves":moves,"n":n}))
                .collect();
            Ok(
                json!({"case":coordinate,"fallback":belief.is_fallback(),"revealed_moves":reveals,"model_types":hp_types,"contradictions":contradictions,"samples":64,"kits":kits,"own_set_provenance":rec.provenance,"own_pick_imputed":rec.imputed_pick,"hp_move_line_ablation_types":erased_types}),
            )
        })();
        let row = result.unwrap_or_else(|reason| json!({"case":coordinate,"skip":reason}));
        writeln!(out, "{row}").unwrap();
    }
    out.flush().unwrap();
}
