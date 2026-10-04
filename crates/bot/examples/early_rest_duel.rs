use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nc2000_bot::blind::BlindSearch;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{Belief, FallbackPolicy, Observer, RmConfig, SplitMix64};
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
    family: String,
    actual_rest: Option<bool>,
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
    candidate: String,
}

fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
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

fn verify(dex: &Dex, p: &Protocol, dir: &Path) {
    let mut games = 0;
    let mut decisions = 0;
    for (ci, case) in p.cases.iter().enumerate() {
        for trial in 0..case.pairs {
            for arm in 0..2 {
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
        json!({"games":games,"decisions":decisions,"legal_replay":true,"state_keys":true,"public_masks":true,"seeds":true,"outcomes":true})
    );
}

fn audit(dex: &Dex, p: &Protocol, dir: &Path) {
    let pool = load_meta_pool(Path::new(&p.prior));
    let mut result = Vec::new();
    for (ci, case) in p.cases.iter().enumerate() {
        for trial in 0..case.pairs {
            let arms: [Value; 2] = [0, 1].map(|arm| {
                serde_json::from_slice(
                    &std::fs::read(dir.join(format!("{}-t{trial:03}-a{arm}.json", case.id)))
                        .unwrap(),
                )
                .unwrap()
            });
            let first = arms[0]["decisions"]
                .as_array()
                .unwrap()
                .iter()
                .zip(arms[1]["decisions"].as_array().unwrap())
                .position(|(a, b)| a["choices"] != b["choices"]);
            let Some(first) = first else { continue };
            let side = trial % 2;
            let (mut b, _, _) = initial(dex, p, ci, trial);
            let mut obs = Observer::new(&b, side);
            let mut beliefs = [FallbackPolicy::EarlyPerish, FallbackPolicy::EarlyRest]
                .map(|policy| Belief::with_fallback_policy(dex, &pool, &obs, policy));
            for (step, row) in arms[0]["decisions"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .take(first + 1)
            {
                assert_eq!(row["key"], b.state_key128().to_string());
                obs.observe(&b, dex);
                let views = [obs.clone(), obs.clone()];
                for arm in 0..2 {
                    beliefs[arm].sync(dex, &views[arm]);
                }
                if step == first {
                    let foe_id = b.active_id(1 - side).unwrap();
                    let own = b.poke(b.active_id(side).unwrap());
                    let foe = b.poke(foe_id);
                    let models = [0, 1].map(|arm| {
                        let mut search = BlindSearch::new(&b, dex, RmConfig { iterations: p.iterations, c: p.c, ..Default::default() }, side, arms[arm]["decisions"][first]["search_seeds"][side].as_u64().unwrap());
                        search.step(dex, &beliefs[arm], &views[arm], p.iterations);
                        assert_eq!(search.best().unwrap().to_input(dex), arms[arm]["decisions"][first]["choices"][side]);
                        let mut kits = std::collections::BTreeMap::new();
                        for seed in 0..64 {
                            let sample = beliefs[arm].determinize(
                                dex,
                                &b,
                                &views[arm],
                                &mut SplitMix64::new(seed),
                            );
                            let mon = sample.poke(foe_id);
                            let key = format!(
                                "{}:{}:{}",
                                mon.base_move_slots
                                    .iter()
                                    .map(|m| dex.moves.key(m.id))
                                    .collect::<Vec<_>>()
                                    .join(","),
                                dex.type_name(mon.base_hp_type),
                                mon.base_hp_power
                            );
                            *kits.entry(key).or_insert(0) += 1;
                        }
                        json!({"fallback":beliefs[arm].is_fallback(),"kits":kits,"choice_reproduced":true})
                    });
                    result.push(json!({"case":case.id,"trial":trial,"decision":first,"turn":b.turn,
                        "scores":[arms[0]["score"],arms[1]["score"]],
                        "choices":[arms[0]["decisions"][first]["choices"][side],arms[1]["decisions"][first]["choices"][side]],
                        "own":dex.species.key(own.species),"own_hp":[own.hp,own.maxhp],
                        "foe":dex.species.key(foe.species),"foe_hp":[foe.hp,foe.maxhp],
                        "true_foe_moves":foe.base_move_slots.iter().map(|m|dex.moves.key(m.id)).collect::<Vec<_>>(),
                        "true_hp_type":dex.type_name(foe.base_hp_type),"models":models}));
                } else {
                    let picks = [0, 1].map(|s| {
                        row["choices"][s].as_str().map(|input| {
                            *b.legal_choices(dex, s)
                                .iter()
                                .find(|a| a.to_input(dex) == input)
                                .unwrap()
                        })
                    });
                    b.apply_choices(dex, picks).unwrap();
                }
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn main() {
    assert!(
        !cfg!(feature = "hp-type-evidence"),
        "HP inference must remain disabled"
    );
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "early_rest_duel PROTOCOL.json OUTPUT_DIRECTORY [--verify|--audit]"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let p: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(p.schema, "nc2000-early-rest-duel-v1");
    assert!(
        p.threads > 0
            && p.iterations > 0
            && !p.interpretation.is_empty()
            && !p.candidate.is_empty()
    );
    assert_eq!(hash(Path::new(&p.prior)), p.prior_sha256);
    let dex = Dex::from_json(&std::fs::read_to_string("data/gen2stadium2.json").unwrap()).unwrap();
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    for (ci, case) in p.cases.iter().enumerate() {
        assert!(case.pairs > 0);
        for team in &case.teams {
            let result = validate_team(&dex, &learnsets, &serde_json::to_string(team).unwrap());
            assert_eq!(result["ok"], true, "{}: {result}", case.id);
        }
        let (b, _, _) = initial(&dex, &p, ci, 0);
        for s in &b.sides {
            for mon in &s.roster {
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
        if args[3] == "--audit" {
            audit(&dex, &p, dir);
        } else {
            verify(&dex, &p, dir);
        }
        return;
    }
    let pool = load_meta_pool(Path::new(&p.prior));
    std::fs::create_dir(dir).unwrap();
    let manifest = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),"protocol_sha256":hash(Path::new(&args[1])),"source_sha256":hash(Path::new("crates/bot/examples/early_rest_duel.rs")),"binary_sha256":hash(&std::env::current_exe().unwrap())});
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
            (0..case.pairs).flat_map(move |trial| (0..2).map(move |arm| (ci, trial, arm)))
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
            let mut beliefs=[0,1].map(|s|Belief::with_fallback_policy(&dex,&pool,&observers[s],if s==side && arm==1 {FallbackPolicy::EarlyRest}else{FallbackPolicy::EarlyPerish}));
            let mut records=Vec::new();
            let started=Instant::now();
            while b.outcome().is_none() && b.turn<=p.max_turns {
                let legal=[0,1].map(|s|b.legal_choices(&dex,s));
                let mut seeds=[None,None];
                let mut masks=Vec::new();
                let picks=[0,1].map(|s| {
                    if legal[s].is_empty() {return None}
                    observers[s].observe(&b,&dex);
                    if s==side {masks=observers[s].mons().iter().map(|m|m.hp_type_mask).collect();}
                    let view=observers[s].clone();
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
                records.push(json!({"key":b.state_key128().to_string(),"turn":b.turn,"choices":picks.map(|a|a.map(|a|a.to_input(&dex))),"search_seeds":seeds,"hp_masks":masks}));
                b.apply_choices(&dex,picks).unwrap();
            }
            let row=json!({"case":case.id,"family":case.family,"actual_rest":case.actual_rest,"trial":trial,"arm":arm,"side":side,"battle_seed":battle_seed,"agent_seeds":agent_seeds,"score":score(&b,side),"terminal":b.outcome().is_some(),"turns":b.turn,"final_key":b.state_key128().to_string(),"seconds":started.elapsed().as_secs_f64(),"decisions":records});
            std::fs::write(dir.join(format!("{}-t{trial:03}-a{arm}.json",case.id)),serde_json::to_vec(&row).unwrap()).unwrap();
            eprintln!("{}/{} {} trial={trial} arm={arm} score={} turns={} seconds={:.1}",job+1,jobs.len(),case.id,row["score"],row["turns"],started.elapsed().as_secs_f64());
        });
        }
    });
}
