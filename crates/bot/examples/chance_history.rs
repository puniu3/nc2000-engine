use nc2000_bot::player::{action_input, PlayerChannel, PlayerFrame};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    dex::Dex,
    prng::{BattleRng, Draw},
    state::{Battle, PokeId},
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use std::io::Write;
use std::time::Instant;

mod threshold_witness;

fn frame(battle: &Battle, dex: &Dex, side: usize) -> PlayerFrame {
    PlayerChannel::new(side)
        .frame(&mut battle.clone(), dex)
        .unwrap()
}

fn choices(battle: &mut Battle, dex: &Dex, tape: &Value) -> [Option<SearchChoice>; 2] {
    [0, 1].map(|side| {
        tape["chosen"][side].as_str().map(|input| {
            *battle
                .legal_choices(dex, side)
                .iter()
                .find(|&&c| action_input(dex, c) == input)
                .unwrap()
        })
    })
}

fn guided_key(dex: &Dex, battle: &Battle, reference: &Battle, id: PokeId) -> u128 {
    let mut comparison = battle.clone();
    let original = reference.poke(id);
    let pokemon = comparison.poke_mut(id);
    pokemon.item = original.item;
    pokemon.last_item = original.last_item;
    pokemon.item_state = original.item_state.clone();
    comparison.refresh_poke_mask(dex, id);
    comparison.state_key128()
}

fn run(
    dex: &Dex,
    base: &Battle,
    selected: [Option<SearchChoice>; 2],
    script: &[usize],
) -> (Battle, Vec<Draw>) {
    let mut battle = base.clone();
    battle.prng = BattleRng::enumerating(script.to_vec());
    battle.apply_choices(dex, selected).unwrap();
    let trace = battle.prng.oracle.take().unwrap().trace;
    assert!(trace
        .iter()
        .all(|d| d.counts.get(d.chosen).is_some_and(|&n| n > 0)));
    (battle, trace)
}

fn find(
    dex: &Dex,
    base: &Battle,
    selected: [Option<SearchChoice>; 2],
    reference: &Battle,
    side: usize,
    id: PokeId,
    cap: usize,
) -> Option<(Battle, Vec<Draw>, usize)> {
    let expected = frame(reference, dex, side);
    let target = reference.state_key128();
    let mut pending = vec![Vec::new()];
    let mut runs = 0;
    while let Some(script) = pending.pop() {
        if runs == cap {
            return None;
        }
        runs += 1;
        let (battle, trace) = run(dex, base, selected, &script);
        if guided_key(dex, &battle, reference, id) == target
            && frame(&battle, dex, side) == expected
        {
            return Some((battle, trace, runs));
        }
        if let Some(draw) = trace.get(script.len()) {
            let mut children: Vec<_> = draw
                .counts
                .iter()
                .enumerate()
                .filter(|(_, n)| **n > 0)
                .map(|(i, &n)| (n, i))
                .collect();
            children.sort();
            for (_, class) in children {
                let mut child = script.clone();
                child.push(class);
                pending.push(child);
            }
        }
    }
    None
}

fn trace_json(trace: &[Draw]) -> Value {
    json!(trace
        .iter()
        .map(|d| json!({"label":d.label,"counts":d.counts,"chosen":d.chosen}))
        .collect::<Vec<_>>())
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus screen output");
    let screen: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
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
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let dex = conformance::load_dex();
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let seed = game["seed"].as_str().unwrap();
    let mut original = Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
    original.set_log_enabled(true);
    let mut references = vec![original.clone()];
    for step in 0..decision {
        let selected = choices(&mut original, &dex, &screen["tape"][step]);
        original.apply_choices(&dex, selected).unwrap();
        assert_eq!(
            original.state_key128().to_string(),
            screen["tape"][step + 1]["key"].as_str().unwrap()
        );
        references.push(original.clone());
    }
    let id = PokeId {
        side: (1 - side) as u8,
        slot: slot as u8,
    };
    let mut patched = teams.clone();
    patched[1 - side][slot].item = "kingsrock".into();
    let mut alternate = Battle::from_fixture(&dex, seed, &patched[0], &patched[1]).unwrap();
    alternate.set_log_enabled(true);
    let mut witness = Vec::new();
    let mut scripts = Vec::new();
    for step in 0..decision {
        assert_eq!(
            frame(&alternate, &dex, side),
            frame(&references[step], &dex, side)
        );
        let selected = choices(&mut alternate, &dex, &screen["tape"][step]);
        let Some((next, trace, runs)) = find(
            &dex,
            &alternate,
            selected,
            &references[step + 1],
            side,
            id,
            200_000,
        ) else {
            panic!("no guided witness within cap at step {step}")
        };
        let script: Vec<_> = trace.iter().map(|d| d.chosen).collect();
        witness.push(json!({"step":step,"runs":runs,"script":script,"trace":trace_json(&trace),"key":next.state_key128().to_string(),"guided_key":guided_key(&dex,&next,&references[step+1],id).to_string(),"frame":frame(&next,&dex,side)}));
        eprintln!("step {step}: {runs} runs, {} draws", trace.len());
        scripts.push(script);
        alternate = next;
    }
    let previous: Value = serde_json::from_slice(
        &std::fs::read("findings/hidden-parameter-family-v1/summary.json").unwrap(),
    )
    .unwrap();
    let p = previous["finite_family"]["rockslide_probability"]
        .as_f64()
        .unwrap();
    let target = previous["finite_family"]["lower"].as_f64().unwrap() - 1e-6;
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[3]).unwrap());
    writeln!(output,"{}",json!({"kind":"config","case":case,"p":p,"target":target,"first":0,"count":screen["rows"].as_array().unwrap().len(),"witness":witness})).unwrap();
    for (index, row) in screen["rows"].as_array().unwrap().iter().enumerate() {
        let request = &row["request"];
        let mut variant = teams.clone();
        let mut set = serde_json::to_value(&variant[1 - side][slot]).unwrap();
        for (k, v) in request["patch"].as_object().unwrap() {
            set[k] = v.clone();
        }
        variant[1 - side][slot] = serde_json::from_value(set).unwrap();
        if let Some(other) = request["swapped_slot"].as_u64() {
            variant[1 - side][other as usize].item = teams[1 - side][slot].item.clone();
        }
        assert_eq!(variant[1 - side][slot].item, "kingsrock");
        let validity = validate_team(
            &dex,
            &learnsets,
            &serde_json::to_string(&variant[1 - side]).unwrap(),
        );
        assert_eq!(validity["ok"], true);
        let mut battle = Battle::from_fixture(&dex, seed, &variant[0], &variant[1]).unwrap();
        battle.set_log_enabled(true);
        let mut steps = Vec::new();
        for step in 0..decision {
            assert_eq!(
                frame(&battle, &dex, side),
                frame(&references[step], &dex, side)
            );
            let selected = choices(&mut battle, &dex, &screen["tape"][step]);
            let (next, trace) = run(&dex, &battle, selected, &scripts[step]);
            assert_eq!(
                frame(&next, &dex, side),
                frame(&references[step + 1], &dex, side)
            );
            assert_eq!(
                trace.iter().map(|d| d.chosen).collect::<Vec<_>>(),
                scripts[step]
            );
            steps.push(json!({"step":step,"trace":trace_json(&trace),"key":next.state_key128().to_string(),"frame":frame(&next,&dex,side)}));
            battle = next;
        }
        let root = json!({"key":battle.state_key128().to_string(),"full_log":battle.log,"state":battle.essence(&dex)});
        battle.set_log_enabled(false);
        let started = Instant::now();
        let config = |first: &str| json!({"game":case["game"],"decision":case["decision"],"target_side":side,"policy_side":side,"first_choice":format!("move {first}"),"priority":["move rockslide"],"rest_at_percent":null,"horizon":2,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true});
        let swords =
            threshold_witness::solve(&dex, battle.clone(), config("swordsdance"), 1., "native");
        let sword = swords["bounds"][0].as_f64().unwrap();
        let needed = ((target + 1e-9 - (1. - p) * sword) / p).clamp(0., 1.);
        let rock = if (1. - p) * sword > target + 1e-9 {
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
        let lower = p * rock
            .as_ref()
            .map_or(0., |l| l["bounds"][0].as_f64().unwrap())
            + (1. - p) * sword;
        writeln!(output,"{}",json!({"index":index,"id":request["id"],"request":request,"seconds":started.elapsed().as_secs_f64(),"root":root,"steps":steps,"swordsdance":swords,"rockslide":rock,"rockslide_threshold":needed,"mixed_lower":lower})).unwrap();
        eprintln!("{} lower={lower:.12}", request["id"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ready(dex: &Dex, item: &str) -> Battle {
        let mon: PokemonSet = serde_json::from_value(
            json!({"name":"Zapdos","species":"Zapdos","level":50,"moves":["Agility"],"item":item}),
        )
        .unwrap();
        let foe: PokemonSet = serde_json::from_value(
            json!({"name":"Miltank","species":"Miltank","level":50,"moves":["Growl"]}),
        )
        .unwrap();
        let mut battle = Battle::from_fixture(dex, "1,2,3,4", &[mon], &[foe]).unwrap();
        battle.set_log_enabled(true);
        let selected = [0, 1].map(|side| Some(battle.legal_choices(dex, side)[0]));
        battle.apply_choices(dex, selected).unwrap();
        battle
    }
    #[test]
    fn guide_finds_a_positive_path_without_changing_the_real_item() {
        let dex = conformance::load_dex();
        let mut base = ready(&dex, "przcureberry");
        let alternate = ready(&dex, "kingsrock");
        let id = base.active_id(0).unwrap();
        let selected = [0, 1].map(|side| Some(base.legal_choices(&dex, side)[0]));
        let (reference, _) = run(&dex, &base, selected, &[]);
        let (actual, trace, _) = find(&dex, &alternate, selected, &reference, 1, id, 100).unwrap();
        assert_eq!(dex.items.key(actual.poke(id).item.unwrap()), "kingsrock");
        assert_ne!(actual.state_key128(), reference.state_key128());
        assert_eq!(
            guided_key(&dex, &actual, &reference, id),
            reference.state_key128()
        );
        assert!(trace.iter().all(|d| d.counts[d.chosen] > 0));
        let script: Vec<_> = trace.iter().map(|d| d.chosen).collect();
        let (replayed, repeated) = run(&dex, &alternate, selected, &script);
        assert_eq!(replayed.state_key128(), actual.state_key128());
        assert_eq!(trace_json(&repeated), trace_json(&trace));
    }
    #[test]
    fn visible_mismatch_and_zero_work_cannot_be_reported_as_witnesses() {
        let dex = conformance::load_dex();
        let mut base = ready(&dex, "przcureberry");
        let alternate = ready(&dex, "kingsrock");
        let id = base.active_id(0).unwrap();
        let selected = [0, 1].map(|side| Some(base.legal_choices(&dex, side)[0]));
        let (mut reference, _) = run(&dex, &base, selected, &[]);
        assert!(find(&dex, &alternate, selected, &reference, 1, id, 0).is_none());
        reference.log.push("|turn|999".into());
        assert!(find(&dex, &alternate, selected, &reference, 1, id, 100).is_none());
    }
}
