use nc2000_bot::Observer;
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::time::Instant;

mod threshold_witness;

fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        [4, 6].contains(&args.len()),
        "corpus screen output [start count]"
    );
    let screen: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let previous: Value = serde_json::from_slice(
        &std::fs::read("findings/hidden-parameter-family-v1/summary.json").unwrap(),
    )
    .unwrap();
    let p = previous["finite_family"]["rockslide_probability"]
        .as_f64()
        .unwrap();
    let target = previous["finite_family"]["lower"].as_f64().unwrap() - 1e-6;
    let first: usize = if args.len() == 6 {
        args[4].parse().unwrap()
    } else {
        0
    };
    let count: usize = if args.len() == 6 {
        args[5].parse().unwrap()
    } else {
        screen["rows"].as_array().unwrap().len()
    };
    let case = &screen["case"];
    let side = case["side"].as_u64().unwrap() as usize;
    let slot = case["foe_slot"].as_u64().unwrap() as usize;
    let decision = case["decision"].as_u64().unwrap() as usize;
    let game: Value = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .find(|g| g["game"] == case["game"])
        .unwrap();
    let dex = conformance::load_dex();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let seed = game["seed"].as_str().unwrap();
    let mut base = Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
    let mut actions = Vec::new();
    for record in game["decisions"].as_array().unwrap().iter().take(decision) {
        assert_eq!(
            base.state_key128().to_string(),
            record["key"].as_str().unwrap()
        );
        let legal = [0, 1].map(|s| base.legal_choices(&dex, s));
        let choices = [0, 1].map(|s| record["choices"][s].as_u64().map(|i| legal[s][i as usize]));
        actions.push(choices);
        base.apply_choices(&dex, choices).unwrap();
    }
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[3]).unwrap());
    writeln!(
        output,
        "{}",
        json!({"kind":"config","case":case,"p":p,"target":target,"first":first,"count":count})
    )
    .unwrap();
    let started = Instant::now();
    for (index, row) in screen["rows"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .skip(first)
        .take(count)
    {
        assert_eq!(row["validity"]["ok"], true);
        assert_eq!(row["prefix_checks"], decision + 1);
        assert!(row.get("mismatch").is_none());
        let request = &row["request"];
        let mut variant = teams.clone();
        let mut mon = serde_json::to_value(&variant[1 - side][slot]).unwrap();
        for (field, value) in request["patch"].as_object().unwrap() {
            mon[field] = value.clone();
        }
        variant[1 - side][slot] = serde_json::from_value(mon).unwrap();
        if let Some(other) = request["swapped_slot"].as_u64() {
            variant[1 - side][other as usize].item = teams[1 - side][slot].item.clone();
        }
        let mut battle = Battle::from_fixture(&dex, seed, &variant[0], &variant[1]).unwrap();
        battle.set_log_enabled(true);
        let mut obs = Observer::new(&battle, side);
        for &choices in &actions {
            battle.apply_choices(&dex, choices).unwrap();
            obs.observe(&battle, &dex);
        }
        assert_eq!(
            battle.state_key128().to_string(),
            row["root"]["key"].as_str().unwrap()
        );
        assert_eq!(
            digest(&serde_json::to_string(&battle.log).unwrap()),
            case["log_hash"].as_str().unwrap()
        );
        assert_eq!(
            digest(&format!("{:?}", obs.mons())),
            case["observer_hash"].as_str().unwrap()
        );
        let begin = Instant::now();
        let config = |first: &str| {
            json!({"game":case["game"],"decision":case["decision"],"target_side":side,"policy_side":side,
            "first_choice":format!("move {first}"),"priority":["move rockslide"],"rest_at_percent":null,
            "horizon":2,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true})
        };
        let swords =
            threshold_witness::solve(&dex, battle.clone(), config("swordsdance"), 1., "native");
        let swords_lower = swords["bounds"][0].as_f64().unwrap();
        let needed = ((target + 1e-9 - (1. - p) * swords_lower) / p).clamp(0., 1.);
        let rock = if (1. - p) * swords_lower > target + 1e-9 {
            None
        } else {
            Some(threshold_witness::solve(
                &dex,
                battle,
                config("rockslide"),
                needed,
                "eval",
            ))
        };
        let rock_lower = rock
            .as_ref()
            .map_or(0., |l| l["bounds"][0].as_f64().unwrap());
        let lower = p * rock_lower + (1. - p) * swords_lower;
        let seconds = begin.elapsed().as_secs_f64();
        writeln!(
            output,
            "{}",
            json!({"index":index,"id":request["id"],"seconds":seconds,"swordsdance":swords,
            "rockslide":rock,"rockslide_threshold":needed,"mixed_lower":lower})
        )
        .unwrap();
        if index % 40 == 39 || lower < target {
            output.flush().unwrap();
            eprintln!(
                "{index} {} lower={lower:.12} target={target:.12} elapsed={:.3}",
                request["id"],
                started.elapsed().as_secs_f64()
            );
        }
    }
    output.flush().unwrap();
    eprintln!(
        "finished {count} rows in {:.3}s",
        started.elapsed().as_secs_f64()
    );
}
