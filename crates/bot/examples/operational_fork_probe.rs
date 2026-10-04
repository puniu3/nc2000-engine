use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    blind::BlindSearch,
    preview::{load_meta_pool, MetaPool},
    smmcts::SelRule,
    Belief, FallbackPolicy, Observer, RmConfig, SplitMix64,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, PokemonSet, SearchChoice},
    state::Battle,
};
use serde_json::{json, Value};
use std::{io::Write, path::Path};

fn read(path: impl AsRef<Path>) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}
fn lines(path: impl AsRef<Path>) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let job: usize = args[1].parse().unwrap();
    let dir = Path::new(&args[2]);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3]).unwrap());
    let a = lines(dir.join(format!("{job}-layered-meta-rentals-learnset-v1.jsonl")));
    let z = lines(dir.join(format!("{job}-revealed-loadout-mode-v1.jsonl")));
    let (old, new) = a
        .iter()
        .zip(&z)
        .find(|(a, b)| a["choice"] != b["choice"])
        .unwrap();
    assert_eq!(old["state_key"], new["state_key"]);
    let replay = read(dir.join(format!(
        "{job}-layered-meta-rentals-learnset-v1-replay.json"
    )));
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(replay["teams"].clone()).unwrap();
    let dex = load_dex();
    let pool = load_meta_pool(&repo_root().join("data/belief-pool-v3/belief-pool.json"));
    let side = old["side"].as_u64().unwrap() as usize;
    let mut b = Battle::from_fixture(
        &dex,
        replay["battle_seed"].as_str().unwrap(),
        &teams[0],
        &teams[1],
    )
    .unwrap();
    b.set_log_enabled(true);
    let mut obs = Observer::new(&b, side);
    let inputs = replay["inputs"].as_array().unwrap();
    let mut i = 0;
    loop {
        let legal: [Vec<SearchChoice>; 2] = std::array::from_fn(|s| b.legal_choices(&dex, s));
        let key = format!("{:032x}", b.state_key128());
        assert_eq!(inputs[i]["key"], key);
        if !legal[side].is_empty() {
            obs.observe(&b, &dex);
        }
        if key == old["state_key"].as_str().unwrap() {
            break;
        }
        let mut picks = [None, None];
        while i < inputs.len() && inputs[i]["key"] == key {
            let s = inputs[i]["side"].as_u64().unwrap() as usize;
            picks[s] = Some(
                *legal[s]
                    .iter()
                    .find(|c| c.to_input(&dex) == inputs[i]["choice"])
                    .unwrap(),
            );
            i += 1;
        }
        b.apply_choices(&dex, picks).unwrap();
    }
    let cfg = RmConfig {
        iterations: 27000,
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let model = |pool: &MetaPool, policy| {
        let belief = Belief::with_fallback_policy(&dex, pool, &obs, policy);
        assert!(belief.is_fallback());
        belief
    };
    let mut variants = vec![
        (
            "baseline".to_string(),
            pool.clone(),
            FallbackPolicy::Layered,
        ),
        (
            "conditioned".to_string(),
            pool.clone(),
            FallbackPolicy::RevealedLoadout,
        ),
    ];
    if b.turn == 2 {
        let base = pool
            .teams
            .iter()
            .flat_map(|t| &t.sets)
            .find(|s| {
                norm(&s.species) == "misdreavus"
                    && norm(&s.item) == "miracleberry"
                    && s.moves.iter().map(|m| norm(m)).collect::<Vec<_>>()
                        == ["toxic", "confuseray", "painsplit", "perishsong"]
            })
            .unwrap()
            .clone();
        for dropped in ["toxic", "confuseray", "painsplit"] {
            let mut source = base.clone();
            source.moves = vec!["perishsong".into()];
            source.moves.extend(
                base.moves
                    .iter()
                    .filter(|m| norm(m) != "perishsong" && norm(m) != dropped)
                    .cloned(),
            );
            source.moves.push(dropped.into());
            let mut p = pool.clone();
            for t in &mut p.teams {
                for s in &mut t.sets {
                    if norm(&s.species) == "misdreavus" {
                        *s = source.clone();
                    }
                }
            }
            variants.push((format!("perish-drop-{dropped}"), p, FallbackPolicy::Layered));
        }
        let mut p = pool.clone();
        for t in &mut p.teams {
            for s in &mut t.sets {
                if norm(&s.species) == "misdreavus" {
                    for m in &mut s.moves {
                        if norm(m) == "destinybond" {
                            *m = "toxic".into();
                        }
                    }
                }
            }
        }
        variants.push((
            "conditioned-no-bond".into(),
            p,
            FallbackPolicy::RevealedLoadout,
        ));
    }
    if args.get(4).is_some_and(|s| s == "--item-probe") {
        assert_eq!(job, 25);
        variants.retain(|(name, _, _)| name == "conditioned" || name == "conditioned-no-bond");
        for (name, p, _) in &mut variants {
            name.push_str("-miracle");
            for t in &mut p.teams {
                for s in &mut t.sets {
                    if norm(&s.species) == "misdreavus" && norm(&s.item) == "goldberry" {
                        s.item = "Miracle Berry".into();
                    }
                }
            }
        }
    }
    if args.get(4).is_some_and(|s| s == "--draw-probe") {
        variants.retain(|(name, _, _)| name == "baseline" || name == "conditioned");
        variants.push((
            "draw".into(),
            pool.clone(),
            FallbackPolicy::RevealedLoadoutDraw,
        ));
    }
    if args.get(4).is_some_and(|s| s == "--retain-perish-probe") {
        variants.retain(|(name, _, _)| name == "baseline");
        variants.push(("retain-perish".into(), pool.clone(), FallbackPolicy::RetainPerish));
    }
    let mut seeds = vec![old["seed"].as_u64().unwrap()];
    if new["seed"] != old["seed"] {
        seeds.push(new["seed"].as_u64().unwrap());
    }
    for k in 0..7 {
        seeds.push(202610050000 + job as u64 * 100 + k);
    }
    for (name, p, policy) in variants {
        let belief = model(&p, policy);
        let sim = belief.determinize(&dex, &b, &obs, &mut SplitMix64::new(771));
        let foe = sim.poke(sim.active_id(1 - side).unwrap());
        if name.starts_with("perish-drop-") || name.starts_with("conditioned-") || name == "retain-perish" {
            let reference = model(
                &pool,
                if name.starts_with("conditioned-") {
                    FallbackPolicy::RevealedLoadout
                } else {
                    FallbackPolicy::Layered
                },
            )
            .determinize(&dex, &b, &obs, &mut SplitMix64::new(771));
            let mut patched = sim.clone();
            let id = patched.active_id(1 - side).unwrap();
            patched.poke_mut(id).move_slots = reference.poke(id).move_slots.clone();
            patched.poke_mut(id).base_move_slots = reference.poke(id).base_move_slots.clone();
            if name.ends_with("-miracle") {
                patched.poke_mut(id).item = reference.poke(id).item;
                patched.poke_mut(id).item_state = reference.poke(id).item_state.clone();
                patched.poke_mut(id).handler_mask = reference.poke(id).handler_mask;
            }
            assert_eq!(
                patched.state_key128(),
                reference.state_key128(),
                "intervention must only alter the active foe's moves and explicitly probed item: {name}"
            );
        }
        for &seed in &seeds {
            let started = std::time::Instant::now();
            let mut search = BlindSearch::new(&b, &dex, cfg.clone(), side, seed);
            search.step(&dex, &belief, &obs, cfg.iterations);
            let choice = search.best().unwrap().to_input(&dex);
            let root:Vec<Value>=search.actions().iter().enumerate().map(|(i,c)|json!({"action":c.to_input(&dex),"visits":search.visits()[i],"mean":search.means()[i],"dominated":search.dominated()[i]})).collect();
            if (name == "baseline" && seed == old["seed"].as_u64().unwrap())
                || (name == "conditioned" && seed == new["seed"].as_u64().unwrap())
            {
                let original = if name == "baseline" { old } else { new };
                assert_eq!(choice, original["choice"]);
                assert_eq!(json!(root), original["root"], "root must exactly replay");
            }
            writeln!(out,"{}",json!({"kind":"root","job":job,"turn":b.turn,"side":side,"key":old["state_key"],"variant":name,"seed":seed,"choice":choice,"root":root,"model_moves":foe.move_slots.iter().map(|m|dex.moves.key(m.id)).collect::<Vec<_>>(),"model_item":foe.item.map(|i|dex.items.key(i)),"elapsed_seconds":started.elapsed().as_secs_f64(),"reply_availability":search.root_replies().iter().map(|(a,n)|json!({"action":a.to_input(&dex),"iterations":n})).collect::<Vec<_>>()})).unwrap();
            out.flush().unwrap();
        }
    }
    let legal: [Vec<SearchChoice>; 2] = std::array::from_fn(|s| b.legal_choices(&dex, s));
    let mine: Vec<_> = legal[side]
        .iter()
        .filter(|c| c.to_input(&dex) == old["choice"] || c.to_input(&dex) == new["choice"])
        .copied()
        .collect();
    let ids = [b.active_id(side).unwrap(), b.active_id(1 - side).unwrap()];
    for my in mine {
        for &opp in &legal[1 - side] {
            let mut joint = [None, None];
            joint[side] = Some(my);
            joint[1 - side] = Some(opp);
            let result = enumerate_step(&dex, &b, joint, 10000);
            let stats = result.as_ref().map(|e| {
                let mut faint = [0.0; 2];
                let mut hp = [0.0; 2];
                let mass: f64 = e.leaves.iter().map(|l| l.prob).sum();
                assert!((mass - 1.0).abs() < 1e-9);
                for l in &e.leaves {
                    for k in 0..2 {
                        let p = l.battle.poke(ids[k]);
                        faint[k] += l.prob * f64::from(p.fainted);
                        hp[k] += l.prob * p.hp as f64;
                    }
                }
                json!({"runs":e.runs,"leaves":e.leaves.len(),"mass":mass,"faint":faint,"hp":hp})
            });
            writeln!(out,"{}",json!({"kind":"exact_step","job":job,"mine":my.to_input(&dex),"opponent":opp.to_input(&dex),"stats":stats})).unwrap();
            out.flush().unwrap();
        }
    }
}
