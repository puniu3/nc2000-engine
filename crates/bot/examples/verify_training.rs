//! Independent adversarial stat verification; does not call the JS optimizer.
//! cargo run --release -p nc2000-bot --example verify_training -- OUTPUT.json [POOL.json ...]
use nc2000_bot::{
    preview::{load_meta_pool, MetaPool},
    Belief, Observer, SplitMix64,
};
use nc2000_engine::{
    dex::{toid, Dex},
    state::{Battle, Pokemon},
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, path::Path};

const INPUTS: [&str; 12] = [
    "data/team-pool-v4/team-pool.json",
    "data/belief-pool-v3/belief-pool.json",
    "data/team-pool-v4/flagship-mixture.json",
    "data/team-pool-v4/environment-pool-32.json",
    "data/team-pool-v3/team-pool.json",
    "data/team-pool-v3/flagship-mixture.json",
    "data/team-pool-v2/team-pool.json",
    "data/kodoku-r3/conclusion-equilibrium-deme.json",
    "data/kodoku-r3/conclusion-equilibrium-r1.json",
    "data/meta-nash-v3/pool-artifact.json",
    "data/training-v3/supplemental.json",
    "data/training-v3/rentals.json",
];
fn stats(p: &Pokemon) -> [i32; 6] {
    let [a, d, c, z, s] = p.base_stored_stats;
    [p.base_maxhp, a, d, c, z, s]
}
fn shiny(a: i32, d: i32, s: i32, c: i32) -> bool {
    [d, s, c] == [10; 3] && [2, 3].contains(&(a % 4))
}
fn attacks(set: &nc2000_engine::battle::PokemonSet, dex: &Dex) -> bool {
    set.moves.iter().any(|m| {
        let k = toid(m);
        ["metronome", "mimic", "mirrormove"].contains(&k.as_str())
            || (dex.moves.get(dex.moves.id(&k).unwrap()).category == "Physical"
                && !["counter", "seismictoss", "superfang", "bide"].contains(&k.as_str()))
    })
}

// Enumerate every Gen 2 DV tuple. The adversary may change any/all DVs and
// training at once, under public gender, shiny, HP type and Unown constraints.
// It wins if a protected stat improves without sacrificing another, or if
// Attack improves in its required direction at unchanged protected stats.
fn adversary(dex: &Dex, p: &Pokemon, attack: bool, has_hp: bool) -> Option<Value> {
    let sp = dex.species.get(p.species);
    let old = p.set_ivs.map(|v| v / 2);
    let is_shiny = shiny(old[1], old[2], old[5], old[3]);
    let hp_type = 4 * (old[1] % 4) + old[2] % 4;
    let unown = |a: i32, d: i32, s: i32, c: i32| {
        (((a & 6) << 5) | ((d & 6) << 3) | ((s & 6) << 1) | ((c & 6) >> 1)) / 10
    };
    let old_form = unown(old[1], old[2], old[5], old[3]);
    let bases = [
        sp.base_stats.hp,
        sp.base_stats.atk,
        sp.base_stats.def,
        sp.base_stats.spa,
        sp.base_stats.spd,
        sp.base_stats.spe,
    ];
    let truth = stats(p);
    for a in 0..16 {
        for d in 0..16 {
            for s in 0..16 {
                for c in 0..16 {
                    let female = sp
                        .extra
                        .get("genderRatio")
                        .and_then(|g| g["F"].as_f64())
                        .unwrap_or(0.5);
                    let gender = sp
                        .gender
                        .as_deref()
                        .unwrap_or(if (a as f64) < female * 16.0 { "F" } else { "M" });
                    if gender != p.gender.as_str()
                        && !(gender == "N" && p.gender.as_str().is_empty())
                    {
                        continue;
                    }
                    if shiny(a, d, s, c) != is_shiny || (has_hp && 4 * (a % 4) + d % 4 != hp_type) {
                        continue;
                    }
                    if dex.species.key(p.species) == "unown" && unown(a, d, s, c) != old_form {
                        continue;
                    }
                    let h = (a % 2) * 8 + (d % 2) * 4 + (s % 2) * 2 + c % 2;
                    let dvs = [h, a, d, c, c, s];
                    let mut candidate = [0; 6];
                    for k in 0..6 {
                        let ev = if k == 1 && !attack { 0 } else { 63 };
                        candidate[k] =
                            (2 * bases[k] as i32 + 2 * dvs[k] + ev + if k == 0 { 100 } else { 0 })
                                * i32::from(p.level)
                                / 100
                                + if k == 0 { 10 } else { 5 };
                    }
                    let power =
                        (5 * ((a / 8) * 8 + (d / 8) * 4 + (s / 8) * 2 + c / 8) + c % 4) / 2 + 31;
                    if [0, 2, 3, 4, 5].iter().any(|&k| candidate[k] < truth[k])
                        || (has_hp && power < p.base_hp_power)
                    {
                        continue;
                    }
                    let protected_improvement =
                        [0, 2, 3, 4, 5].iter().any(|&k| candidate[k] > truth[k])
                            || (has_hp && power > p.base_hp_power);
                    let attack_improvement = if attack {
                        candidate[1] > truth[1]
                    } else {
                        candidate[1] < truth[1]
                    };
                    if protected_improvement || attack_improvement {
                        return Some(json!({"ivs_dv":dvs,"stats":candidate,"hp_power":power}));
                    }
                }
            }
        }
    }
    None
}

fn same_set(a: &Pokemon, b: &Pokemon) -> bool {
    a.species == b.species
        && a.level == b.level
        && a.gender == b.gender
        && stats(a) == stats(b)
        && a.base_hp_type == b.base_hp_type
        && a.base_hp_power == b.base_hp_power
        && a.item == b.item
        && a.happiness == b.happiness
        && a.base_move_slots.iter().map(|m| m.id).collect::<Vec<_>>()
            == b.base_move_slots.iter().map(|m| m.id).collect::<Vec<_>>()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let output = args.get(1).expect("OUTPUT.json required");
    let paths: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(String::as_str).collect()
    } else {
        INPUTS.to_vec()
    };
    let dex = Dex::from_json(&std::fs::read_to_string("data/gen2stadium2.json").unwrap()).unwrap();
    let ls = Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
        .unwrap();
    let mut issues = vec![];
    let mut hashes = serde_json::Map::new();
    let mut checks = 0usize;
    let mut excluded_sets = 0usize;
    let mut excluded_rentals = vec![];
    let mut classes = HashMap::new();
    for path in paths {
        let bytes = std::fs::read(path).unwrap();
        hashes.insert(path.into(), json!(format!("{:x}", Sha256::digest(&bytes))));
        let pool: MetaPool = serde_json::from_slice(&bytes).unwrap();
        let metadata: Value = serde_json::from_slice(&bytes).unwrap();
        for (index, t) in pool.teams.iter().enumerate() {
            let validation = validate_team(&dex, &ls, &serde_json::to_string(&t.sets).unwrap());
            if path == "data/training-v3/rentals.json"
                && (validation["ok"] != true
                    || metadata["teams"][index]["archetype"]
                        .as_str()
                        .unwrap_or("")
                        .contains("専用"))
            {
                excluded_rentals.push(t.id.clone());
                excluded_sets += t.sets.len();
                continue;
            }
            if validation["ok"] != true {
                issues.push(json!({"path":path,"team":t.id,"legality":validation}));
                continue;
            }
            let b = Battle::from_fixture(&dex, "1,2,3,4", &t.sets, &t.sets).unwrap();
            for (s, p) in t.sets.iter().zip(&b.sides[0].roster) {
                checks += 1;
                let attack = attacks(s, &dex);
                let has_hp = s.moves.iter().any(|m| toid(m).starts_with("hiddenpower"));
                let key = format!(
                    "{:?}:{}:{}:{}:{}:{:?}:{}",
                    p.species.0,
                    p.level,
                    p.gender.as_str(),
                    attack,
                    has_hp,
                    p.set_ivs,
                    p.set_evs[1]
                );
                let witness = classes
                    .entry(key)
                    .or_insert_with(|| adversary(&dex, p, attack, has_hp));
                let expected = [255, if attack { 255 } else { 0 }, 255, 255, 255, 255];
                let happiness_ok = (!s.moves.iter().any(|m| toid(m) == "return")
                    || p.happiness == 255)
                    && (!s.moves.iter().any(|m| toid(m) == "frustration") || p.happiness == 0);
                if witness.is_some() || p.set_evs != expected || !happiness_ok {
                    issues.push(json!({"path":path,"team":t.id,"species":s.species,"actual_stats":stats(p),"evs":p.set_evs,"expected_evs":expected,"happiness":p.happiness,"happiness_optimal":happiness_ok,"better_feasible_spread":witness}));
                }
            }
        }
    }
    let mut supported = 0;
    if args.len() == 2 {
        let prior = load_meta_pool(Path::new(INPUTS[1]));
        let catalog = load_meta_pool(Path::new(INPUTS[0]));
        for t in &catalog.teams {
            let b = Battle::from_fixture(&dex, "1,2,3,4", &t.sets, &t.sets).unwrap();
            let obs = Observer::new(&b, 0);
            let belief = Belief::new(&dex, &prior, &obs);
            let exact = belief.alive().iter().any(|&i| {
                let d = belief.determinize_with(&dex, &b, &obs, Some(i), &mut SplitMix64::new(17));
                b.sides[1]
                    .roster
                    .iter()
                    .zip(&d.sides[1].roster)
                    .all(|(a, b)| same_set(a, b))
            });
            if exact {
                supported += 1;
            } else {
                issues.push(json!({"team":t.id,"missing_exact_prior_support":true}));
            }
        }
        // Keep the historical lineage stat regression even when the deployed
        // known-party policy excludes this off-catalog generated team.
        let prior = load_meta_pool(Path::new("data/belief-pool-v3/bake-prior.json"));
        let t = prior
            .teams
            .iter()
            .find(|t| t.id == "lineage-fine-s07")
            .unwrap();
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t.sets, &t.sets).unwrap();
        let mut obs = Observer::new(&b, 0);
        for side in 0..2 {
            b.choose(&dex, side, "team 1,3,6").unwrap();
        }
        obs.observe(&b, &dex);
        let mut belief = Belief::new(&dex, &prior, &obs);
        for side in 0..2 {
            b.choose(&dex, side, "move raindance").unwrap();
        }
        obs.observe(&b, &dex);
        belief.sync(&dex, &obs);
        assert_eq!(belief.candidate_count(), 1);
        let d = belief.determinize(&dex, &b, &obs, &mut SplitMix64::new(17));
        assert_eq!(d.sides[1].roster[5].base_maxhp, 215);
        assert_eq!(d.sides[1].roster[2].base_stored_stats[4], 121);
    }
    let mut code_and_rules = serde_json::Map::new();
    for path in [
        "crates/bot/examples/verify_training.rs",
        "tools/normalize-training.js",
        "data/gen2stadium2.json",
        "data/learnsets-gen2.json",
    ] {
        code_and_rules.insert(
            path.into(),
            json!(format!(
                "{:x}",
                Sha256::digest(std::fs::read(path).unwrap())
            )),
        );
    }
    let report = json!({"inputs":hashes,"code_and_rules":code_and_rules,"checked_sets":checks,"unique_adversarial_classes":classes.len(),
        "dv_tuples_per_class":65536,"catalog_teams_with_exact_prior_support":supported,
        "excluded_sets":excluded_sets,"excluded_fallback_rentals":excluded_rentals,"issues":issues});
    std::fs::write(
        output,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
    println!(
        "sets={checks} classes={} exact_catalog_support={supported} issues={}",
        classes.len(),
        issues.len()
    );
    if !issues.is_empty() {
        std::process::exit(1);
    }
}
