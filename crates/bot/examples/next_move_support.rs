use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::corpus::{
    corpus_files, corpus_fingerprint, load_battle, load_sources, reconstruct_context_with_cfg,
    HumanAction,
};
use nc2000_bot::import::ProtocolTracker;
use nc2000_bot::observe::move_matches;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{Belief, FallbackPolicy, RmConfig, SplitMix64};
use nc2000_engine::dex::{toid, Dex};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "next_move_support PROTOCOL OUTPUT_DIRECTORY [SMOKE_FILE_LIMIT]"
    );
    assert!(
        !cfg!(feature = "hp-type-evidence"),
        "measure the default product without experimental HP inference"
    );
    let protocol: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    assert_eq!(protocol["schema"], "nc2000-next-move-support-v1");
    for (path, expected) in protocol["inputs_sha256"].as_object().unwrap() {
        assert_eq!(*expected, hash(Path::new(path)), "input changed: {path}");
    }
    let root = std::env::current_dir().unwrap();
    let dex = Dex::from_json(&std::fs::read_to_string("data/gen2stadium2.json").unwrap()).unwrap();
    let src = load_sources(&dex, &root);
    let pool_path = Path::new(protocol["pool"].as_str().unwrap());
    assert_eq!(protocol["pool_sha256"], hash(pool_path));
    let pool = load_meta_pool(pool_path);
    let mut files = corpus_files(Path::new(protocol["corpus"].as_str().unwrap()));
    assert_eq!(files.len(), 570);
    if let Some(limit) = args.get(3) {
        files.truncate(limit.parse().unwrap());
    }
    let dir = Path::new(&args[2]);
    std::fs::create_dir(dir).unwrap();
    let metadata = json!({"protocol":protocol,"corpus_fingerprint":corpus_fingerprint(&files),"files":files.len(),"smoke":args.len()==4,"source_sha256":hash(Path::new("crates/bot/examples/next_move_support.rs")),"binary_sha256":hash(&std::env::current_exe().unwrap())});
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let policies = [
        FallbackPolicy::default(),
        FallbackPolicy::Layered,
        FallbackPolicy::RevealedLoadoutDraw,
    ];
    let next = AtomicUsize::new(0);
    let outputs = Mutex::new(Vec::<(usize, Vec<Value>, BTreeMap<&str, usize>)>::new());
    let samples = protocol["samples"].as_u64().unwrap() as usize;
    let seed = protocol["seed"].as_u64().unwrap();
    std::thread::scope(|scope| {
        for _ in 0..protocol["threads"].as_u64().unwrap() {
            scope.spawn(|| loop {
            let bi = next.fetch_add(1,Ordering::Relaxed);
            let Some(path) = files.get(bi) else { break };
            let cb = load_battle(path);
            let mut counts = BTreeMap::new();
            let mut rows = Vec::new();
            if cb.lines.iter().any(|l| {
                let p: Vec<_> = l.split('|').collect();
                p.get(1)==Some(&"-transform") || p.get(1)==Some(&"move") && p.get(3).is_some_and(|m|matches!(toid(m).as_str(),"transform"|"mimic"|"metronome"|"mirrormove"))
            }) {
                counts.insert("excluded_copy_game",1);
                outputs.lock().unwrap().push((bi,rows,counts));
                continue;
            }
            counts.insert("included_game",1);
            let mut tracker = ProtocolTracker::new(0);
            let mut consumed = 0;
            for d in &cb.decisions {
                let HumanAction::Move(key) = &d.action else {continue};
                *counts.entry("direct_move_decisions").or_default() += 1;
                if matches!(key.as_str(),"struggle"|"recharge") { *counts.entry("synthetic_move").or_default()+=1; continue; }
                for line in &cb.lines[consumed..=d.cut] {tracker.push_line(&dex,line);}
                consumed=d.cut+1;
                let (mons,active) = tracker.snapshot(d.side);
                let Some(slot) = active else { *counts.entry("no_active").or_default()+=1; continue; };
                let target = dex.moves.id(key).unwrap();
                if mons[slot].uses.iter().any(|(m,_)|move_matches(&dex,*m,target)) { *counts.entry("already_used").or_default()+=1; continue; }
                let mut view=d.clone();
                view.side=1-d.side;
                let cfg=RmConfig {iterations:1,c:0.4,..Default::default()};
                let rec=reconstruct_context_with_cfg(&dex,&src,pool.clone(),&cb.lines,&cb.evidence,&view,1,cfg);
                let Some(rec)=rec else {
                    *counts.entry("reconstruction_failed").or_default()+=1;
                    rows.push(json!({"battle":bi,"side":d.side,"turn":d.turn,"move":key,"species":dex.species.key(mons[slot].species),"skip":"reconstruction_failed"}));
                    continue;
                };
                let obs=rec.agent.observer().unwrap();
                let b=rec.agent.battle().unwrap();
                let id=b.active_id(d.side).unwrap();
                assert_eq!(b.poke(id).species,mons[slot].species);
                let mo=&obs.mons()[id.slot as usize];
                if mo.revealed_moves.iter().any(|m|move_matches(&dex,*m,target)) { *counts.entry("already_revealed").or_default()+=1; continue; }
                assert_eq!(mo.hp_type_mask,u16::MAX);
                let mut arms=Vec::new();
                for policy in policies {
                    let belief=Belief::with_fallback_policy(&dex,&pool,obs,policy);
                    let mut rng=SplitMix64::new(seed ^ bi as u64 ^ ((d.side as u64)<<32) ^ ((d.turn as u64)<<40));
                    let mut hits=0;
                    let mut kits=BTreeMap::new();
                    for _ in 0..samples {
                        let sampled=belief.determinize(&dex,b,obs,&mut rng);
                        let mon=&sampled.sides[d.side].roster[id.slot as usize];
                        assert!(mo.revealed_moves.iter().all(|r|mon.base_move_slots.iter().any(|m|move_matches(&dex,m.id,*r))));
                        hits+=usize::from(mon.base_move_slots.iter().any(|m|move_matches(&dex,m.id,target)));
                        let kit: Vec<_>=mon.base_move_slots.iter().map(|m|dex.moves.key(m.id).to_string()).collect();
                        *kits.entry(kit).or_insert(0)+=1;
                    }
                    arms.push(json!({"policy":policy.id(),"hits":hits,"samples":samples,"fallback":belief.is_fallback(),"source":format!("{:?}",belief.fallback_sources(&dex,obs)[id.slot as usize]),"kits":kits.into_iter().map(|(moves,count)|json!({"moves":moves,"count":count})).collect::<Vec<_>>() }));
                }
                *counts.entry("novel_labels").or_default()+=1;
                let revealed: Vec<_>=mo.revealed_moves.iter().map(|m|dex.moves.key(*m)).collect();
                let ceiling: BTreeSet<_>=revealed.iter().copied().chain(std::iter::once(key.as_str())).collect();
                assert!(ceiling.contains(key.as_str()));
                rows.push(json!({"battle":bi,"side":d.side,"turn":d.turn,"cut":d.cut,"species":dex.species.key(mo.species),"level":mo.level,"gender":format!("{:?}",mo.gender),"move":key,"revealed":revealed,"arms":arms,"revealed_only_control":0,"label_added_control":samples}));
            }
            outputs.lock().unwrap().push((bi,rows,counts));
            if bi%50==0 {eprintln!("processed {bi}/{}",files.len());}
        });
        }
    });
    let mut output = outputs.into_inner().unwrap();
    output.sort_by_key(|(bi, _, _)| *bi);
    let mut writer =
        std::io::BufWriter::new(std::fs::File::create(dir.join("rows.jsonl")).unwrap());
    let mut counts = BTreeMap::new();
    for (_, rows, cs) in output {
        for row in rows {
            writeln!(writer, "{row}").unwrap();
        }
        for (k, v) in cs {
            *counts.entry(k).or_insert(0) += v;
        }
    }
    writer.flush().unwrap();
    std::fs::write(
        dir.join("counts.json"),
        serde_json::to_vec_pretty(&counts).unwrap(),
    )
    .unwrap();
    println!("{}", json!(counts));
}
