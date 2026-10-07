use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nc2000_bot::blind::BlindSearch;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{Belief, Observer, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{validate_team, Learnsets};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    teams: [Vec<PokemonSet>; 2],
    prefix: Vec<[String; 2]>,
    pairs: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    cases: Vec<Case>,
    iterations: u32,
    c: f64,
    seed: u64,
    max_turns: u16,
    threads: usize,
    prior: String,
    prior_sha256: String,
    interpretation: String,
}

fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn sources() -> Value {
    fn visit(dir: &Path, result: &mut std::collections::BTreeMap<String, String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, result);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                result.insert(path.to_str().unwrap().to_string(), hash(&path));
            }
        }
    }
    let mut result = std::collections::BTreeMap::new();
    for dir in ["crates/engine/src", "crates/bot/src"] {
        visit(Path::new(dir), &mut result);
    }
    for path in [
        "data/gen2stadium2.json",
        "data/learnsets-gen2.json",
        "Cargo.lock",
        "Cargo.toml",
        "crates/bot/Cargo.toml",
        "crates/engine/Cargo.toml",
        "crates/bot/examples/hp_information_duel.rs",
    ] {
        result.insert(path.to_string(), hash(Path::new(path)));
    }
    json!(result)
}

fn score(b: &Battle, side: usize) -> Option<f64> {
    b.outcome().map(|o| match o {
        Outcome::Tie => 0.5,
        Outcome::P1Win => f64::from(side == 0),
        Outcome::P2Win => f64::from(side == 1),
    })
}

fn initial(dex: &Dex, p: &Protocol, ci: usize, trial: usize) -> (Battle, String, [u64; 2]) {
    let case = &p.cases[ci];
    let side = trial % 2;
    let mut rng = SplitMix64::new(
        p.seed
            ^ (trial as u64).wrapping_mul(0x9e3779b97f4a7c15)
            ^ (ci as u64).wrapping_mul(0xbf58476d1ce4e5b9),
    );
    let seed = rng.battle_seed();
    let agents = [rng.next(), rng.next()];
    let b = Battle::from_fixture(dex, &seed, &case.teams[side], &case.teams[1 - side]).unwrap();
    (b, seed, agents)
}

fn observation(obs: &Observer, dex: &Dex) -> Value {
    json!(obs
        .mons()
        .iter()
        .map(|m| json!({
            "moves": m.revealed_moves.iter().map(|id| dex.moves.key(*id)).collect::<Vec<_>>(),
            "hp_mask": m.hp_type_mask,
            "other": format!("{:?}", {let mut x=m.clone(); x.revealed_moves.clear(); x})
        }))
        .collect::<Vec<_>>())
}

fn study_view(
    raw: &Observer,
    oracle: &mut Observer,
    b: &Battle,
    dex: &Dex,
    arm: usize,
) -> Observer {
    assert!(arm < 4);
    let view = if arm >= 2 {
        let mut input = b.clone();
        for mon in &mut input.sides[raw.opp()].roster {
            for slot in &mut mon.base_move_slots {
                slot.used = true;
            }
        }
        oracle.observe(&input, dex);
        for (known, public) in oracle.mons().iter().zip(raw.mons()) {
            let mut stripped = known.clone();
            stripped.revealed_moves = public.revealed_moves.clone();
            assert_eq!(format!("{stripped:?}"), format!("{public:?}"));
        }
        oracle.clone()
    } else {
        raw.clone()
    };
    if arm % 2 == 0 {
        view.without_hp_type_evidence()
    } else {
        view
    }
}

fn move_keys(dex: &Dex, mon: &nc2000_engine::state::Pokemon) -> Vec<String> {
    let mut keys = mon
        .base_move_slots
        .iter()
        .map(|s| dex.moves.key(s.id).to_string())
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

fn model_audit(
    dex: &Dex,
    b: &Battle,
    belief: &Belief,
    view: &Observer,
    raw: &Observer,
    arm: usize,
    case: &str,
    trial: usize,
    step: usize,
) -> Value {
    let mut active_missing = 0;
    let mut type_contradictions = 0;
    let mut roster_mismatches = 0;
    let mut active_types = std::collections::BTreeMap::new();
    let mut active_powers = std::collections::BTreeMap::new();
    let active = b.active_id(view.opp());
    for seed in 0..64 {
        let sample = belief.determinize(dex, b, view, &mut SplitMix64::new(seed));
        for (actual, model) in b.sides[view.opp()]
            .roster
            .iter()
            .zip(&sample.sides[view.opp()].roster)
        {
            roster_mismatches += usize::from(move_keys(dex, actual) != move_keys(dex, model));
        }
        if let Some(id) = active {
            let actual = move_keys(dex, b.poke(id));
            let model = sample.poke(id);
            let modeled = move_keys(dex, model);
            active_missing += usize::from(actual.iter().any(|m| !modeled.contains(m)));
            let types = [
                "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel", "Fire",
                "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
            ];
            let slot = b.sides[view.opp()].active.unwrap() as usize;
            let mask = raw.mons()[slot].hp_type_mask;
            if mask != u16::MAX {
                let valid = modeled.iter().any(|m| m == "hiddenpower")
                    && types
                        .iter()
                        .position(|t| *t == dex.type_name(model.base_hp_type))
                        .is_some_and(|i| mask & (1 << i) != 0);
                type_contradictions += usize::from(!valid);
            }
            *active_types
                .entry(dex.type_name(model.base_hp_type).to_string())
                .or_insert(0) += 1;
            *active_powers.entry(model.base_hp_power).or_insert(0) += 1;
        }
    }
    if arm >= 2 {
        assert_eq!(roster_mismatches, 0, "{case} {trial} {step}");
    }
    if arm % 2 == 1 {
        assert_eq!(type_contradictions, 0);
    }
    json!({"case":case,"trial":trial,"arm":arm,"step":step,"turn":b.turn,"samples":64,"roster_mismatches":roster_mismatches,"active_missing":active_missing,"type_contradictions":type_contradictions,"active_types":active_types,"active_powers":active_powers,"fallback":belief.is_fallback()})
}

fn verify(dex: &Dex, p: &Protocol, dir: &Path, audit: bool) {
    let pool = load_meta_pool(Path::new(&p.prior));
    let mut diagnostics = Vec::new();
    let mut games = 0;
    let mut decisions = 0;
    for (ci, case) in p.cases.iter().enumerate() {
        for trial in 0..case.pairs {
            for arm in 0..4 {
                let row: Value = serde_json::from_slice(
                    &std::fs::read(dir.join(format!("{}-t{trial:03}-a{arm}.json", case.id)))
                        .unwrap(),
                )
                .unwrap();
                let (mut b, seed, agents) = initial(dex, p, ci, trial);
                assert_eq!(row["battle_seed"], seed);
                assert_eq!(row["agent_seeds"], json!(agents));
                assert_eq!(row["case"], case.id);
                assert_eq!(row["trial"], trial);
                assert_eq!(row["arm"], arm);
                assert_eq!(row["side"], trial % 2);
                let mut rngs = agents.map(SplitMix64::new);
                let mut observers = [0, 1].map(|s| Observer::new(&b, s));
                let mut oracle = Observer::new(&b, trial % 2);
                let mut belief = Belief::new(dex, &pool, &oracle);
                for (step, r) in row["decisions"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(r["key"], b.state_key128().to_string());
                    assert_eq!(r["turn"], b.turn);
                    let legal = [0, 1].map(|s| b.legal_choices(dex, s));
                    let picks = [0, 1].map(|s| {
                        assert_eq!(legal[s].is_empty(), r["choices"][s].is_null());
                        if legal[s].is_empty() {
                            return None;
                        }
                        assert_eq!(r["search_seeds"][s], rngs[s].next());
                        observers[s].observe(&b, dex);
                        if s == trial % 2 {
                            assert_eq!(
                                r["hp_masks"],
                                json!(observers[s]
                                    .mons()
                                    .iter()
                                    .map(|m| m.hp_type_mask)
                                    .collect::<Vec<_>>())
                            );
                        }
                        if s == trial % 2 {
                            let view = study_view(&observers[s], &mut oracle, &b, dex, arm);
                            assert_eq!(r["view"], observation(&view, dex));
                            belief.sync(dex, &view);
                            if audit && step >= case.prefix.len() && legal[s].len() > 1 {
                                diagnostics.push(model_audit(
                                    dex,
                                    &b,
                                    &belief,
                                    &view,
                                    &observers[s],
                                    arm,
                                    &case.id,
                                    trial,
                                    step,
                                ));
                            }
                        }
                        let input = r["choices"][s].as_str().unwrap();
                        if let Some(prefix) = case.prefix.get(step) {
                            assert_eq!(input, prefix[s ^ (trial % 2)]);
                        }
                        Some(*legal[s].iter().find(|a| a.to_input(dex) == input).unwrap())
                    });
                    b.apply_choices(dex, picks).unwrap();
                    decisions += 1;
                }
                assert_eq!(row["score"], json!(score(&b, trial % 2)));
                assert_eq!(row["terminal"], b.outcome().is_some());
                assert_eq!(row["turns"], b.turn);
                assert_eq!(row["final_key"], b.state_key128().to_string());
                assert!(b.outcome().is_some() || b.turn > p.max_turns);
                games += 1;
            }
        }
    }
    println!(
        "{}",
        json!({"games":games,"decisions":decisions,"legal_replay":true,"state_keys":true,"public_masks":true,"seeds":true,"outcomes":true,"diagnostics":diagnostics})
    );
}

fn main() {
    assert!(
        cfg!(feature = "hp-type-evidence"),
        "build with --features hp-type-evidence"
    );
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "hp_information_duel PROTOCOL.json OUTPUT_DIRECTORY [--verify|--audit]"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let p: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(p.schema, "nc2000-hp-information-duel-v1");
    assert!(p.threads > 0 && p.iterations > 0 && !p.interpretation.is_empty());
    assert_eq!(hash(Path::new(&p.prior)), p.prior_sha256);
    let dex = Dex::from_json(&std::fs::read_to_string("data/gen2stadium2.json").unwrap()).unwrap();
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    for (ci, case) in p.cases.iter().enumerate() {
        assert!(case.pairs > 0 && !case.prefix.is_empty());
        for team in &case.teams {
            let result = validate_team(&dex, &learnsets, &serde_json::to_string(team).unwrap());
            assert_eq!(result["ok"], true, "{}: {result}", case.id);
        }
        let (b, _, _) = initial(&dex, &p, ci, 0);
        for s in &b.sides {
            for mon in &s.roster {
                assert_eq!(mon.base_move_slots.len(), 4);
                for m in &mon.base_move_slots {
                    let key = dex.moves.key(m.id);
                    assert!(!key.starts_with("hiddenpower") || key == "hiddenpower");
                }
            }
        }
        for trial in 0..case.pairs {
            let (mut b, _, _) = initial(&dex, &p, ci, trial);
            for prefix in &case.prefix {
                let picks = [0, 1].map(|s| {
                    let input = &prefix[s ^ (trial % 2)];
                    Some(
                        *b.legal_choices(&dex, s)
                            .iter()
                            .find(|a| a.to_input(&dex) == *input)
                            .unwrap_or_else(|| panic!("{} {trial} illegal {input}", case.id)),
                    )
                });
                b.apply_choices(&dex, picks).unwrap();
            }
        }
    }
    let dir = Path::new(&args[2]);
    if args.len() == 4 {
        assert!(matches!(args[3].as_str(), "--verify" | "--audit"));
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["protocol_sha256"], hash(Path::new(&args[1])));
        assert_eq!(manifest["sources"], sources());
        verify(&dex, &p, dir, args[3] == "--audit");
        return;
    }
    let pool = load_meta_pool(Path::new(&p.prior));
    std::fs::create_dir(dir).unwrap();
    let manifest = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),"protocol_sha256":hash(Path::new(&args[1])),"sources":sources(),"source_sha256":hash(Path::new("crates/bot/examples/hp_information_duel.rs")),"binary_sha256":hash(&std::env::current_exe().unwrap())});
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let cfg = RmConfig {
        iterations: p.iterations,
        c: p.c,
        ..Default::default()
    };
    let jobs: Vec<_> = p
        .cases
        .iter()
        .enumerate()
        .flat_map(|(ci, case)| {
            (0..case.pairs).flat_map(move |trial| (0..4).map(move |arm| (ci, trial, arm)))
        })
        .collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..p.threads {
            scope.spawn(|| loop {
            let job=next.fetch_add(1,Ordering::Relaxed);
            let Some(&(ci,trial,arm))=jobs.get(job) else {break};
            let case=&p.cases[ci];
            let side=trial%2;
            let (mut b,battle_seed,agent_seeds)=initial(&dex,&p,ci,trial);
            let mut rngs=agent_seeds.map(SplitMix64::new);
            let mut observers=[0,1].map(|s|Observer::new(&b,s));
            let mut oracle=Observer::new(&b,side);
            let mut beliefs=[0,1].map(|s|Belief::new(&dex,&pool,&observers[s]));
            let mut records=Vec::new();
            let started=Instant::now();
            while b.outcome().is_none() && b.turn<=p.max_turns {
                let legal=[0,1].map(|s|b.legal_choices(&dex,s));
                let mut seeds=[None,None];
                let mut masks=Vec::new();
                let mut view_record=Value::Null;
                let picks=[0,1].map(|s| {
                    if legal[s].is_empty() {return None}
                    observers[s].observe(&b,&dex);
                    if s==side {masks=observers[s].mons().iter().map(|m|m.hp_type_mask).collect();}
                    let view=if s==side {study_view(&observers[s],&mut oracle,&b,&dex,arm)}else{observers[s].clone()};
                    if s==side {view_record=observation(&view,&dex);}
                    beliefs[s].sync(&dex,&view);
                    let seed=rngs[s].next();
                    seeds[s]=Some(seed);
                    if let Some(prefix)=case.prefix.get(records.len()) {
                        return Some(*legal[s].iter().find(|a|a.to_input(&dex)==prefix[s^side]).unwrap());
                    }
                    if legal[s].len()==1 {return Some(legal[s][0])}
                    let mut search=BlindSearch::new(&b,&dex,cfg.clone(),s,seed);
                    search.step(&dex,&beliefs[s],&view,p.iterations);
                    search.best()
                });
                assert!(picks.iter().any(Option::is_some));
                records.push(json!({"key":b.state_key128().to_string(),"turn":b.turn,"choices":picks.map(|a|a.map(|a|a.to_input(&dex))),"search_seeds":seeds,"hp_masks":masks,"view":view_record}));
                b.apply_choices(&dex,picks).unwrap();
            }
            let row=json!({"case":case.id,"trial":trial,"arm":arm,"side":side,"battle_seed":battle_seed,"agent_seeds":agent_seeds,"score":score(&b,side),"terminal":b.outcome().is_some(),"turns":b.turn,"final_key":b.state_key128().to_string(),"seconds":started.elapsed().as_secs_f64(),"decisions":records});
            std::fs::write(dir.join(format!("{}-t{trial:03}-a{arm}.json",case.id)),serde_json::to_vec(&row).unwrap()).unwrap();
            eprintln!("{}/{} {} trial={trial} arm={arm} score={} turns={} seconds={:.1}",job+1,jobs.len(),case.id,row["score"],row["turns"],started.elapsed().as_secs_f64());
        });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_changes_only_move_knowledge_and_keeps_revision_stable() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dex =
            Dex::from_json(&std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap())
                .unwrap();
        let p: Protocol = serde_json::from_slice(
            &std::fs::read(root.join("findings/hp-information-v1/protocol.json")).unwrap(),
        )
        .unwrap();
        for ci in 0..p.cases.len() {
            for side in 0..2 {
                let (mut b, _, _) = initial(&dex, &p, ci, side);
                let mut raw = Observer::new(&b, side);
                let mut oracle = raw.clone();
                for step in 0..=p.cases[ci].prefix.len() {
                    raw.observe(&b, &dex);
                    let before_key = b.state_key128();
                    let before_moves = format!(
                        "{:?}",
                        b.sides[1 - side]
                            .roster
                            .iter()
                            .map(|m| m.base_move_slots)
                            .collect::<Vec<_>>()
                    );
                    let view = study_view(&raw, &mut oracle, &b, &dex, 3);
                    let revision = oracle.revision();
                    assert_eq!(
                        observation(&view, &dex),
                        observation(&study_view(&raw, &mut oracle, &b, &dex, 3), &dex)
                    );
                    assert_eq!(revision, oracle.revision());
                    assert_eq!(before_key, b.state_key128());
                    assert_eq!(
                        before_moves,
                        format!(
                            "{:?}",
                            b.sides[1 - side]
                                .roster
                                .iter()
                                .map(|m| m.base_move_slots)
                                .collect::<Vec<_>>()
                        )
                    );
                    for (mo, actual) in view.mons().iter().zip(&b.sides[1 - side].roster) {
                        let mut keys = mo
                            .revealed_moves
                            .iter()
                            .map(|id| dex.moves.key(*id).to_string())
                            .collect::<Vec<_>>();
                        keys.sort();
                        assert_eq!(keys, move_keys(&dex, actual));
                        if step == 0 {
                            assert_eq!(mo.hp_type_mask, u16::MAX);
                        }
                    }
                    assert_eq!(
                        observation(&study_view(&raw, &mut oracle, &b, &dex, 1), &dex),
                        observation(&raw, &dex)
                    );
                    assert!(study_view(&raw, &mut oracle, &b, &dex, 2)
                        .mons()
                        .iter()
                        .all(|m| m.hp_type_mask == u16::MAX));
                    if let Some(prefix) = p.cases[ci].prefix.get(step) {
                        let picks = [0, 1].map(|s| {
                            Some(
                                *b.legal_choices(&dex, s)
                                    .iter()
                                    .find(|a| a.to_input(&dex) == prefix[s ^ side])
                                    .unwrap(),
                            )
                        });
                        b.apply_choices(&dex, picks).unwrap();
                    }
                }
            }
        }
    }
}
