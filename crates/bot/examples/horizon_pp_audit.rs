use nc2000_bot::horizon_pp::{HorizonPpKey, HorizonPpState};
use nc2000_bot::player::action_input;
use nc2000_engine::battle::{enumerate::enumerate_step, Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    corpus: String,
    corpus_sha256: String,
    game: usize,
    decision: usize,
    horizons: Vec<u16>,
    cell_cap: usize,
    sample_paths: usize,
    sample_turns: u16,
    sample_seed: u64,
}

#[derive(Deserialize)]
struct Decision {
    key: String,
    choices: [Option<usize>; 2],
}

#[derive(Deserialize)]
struct Game {
    game: usize,
    seed: String,
    teams: [Vec<PokemonSet>; 2],
    decisions: Vec<Decision>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum Successor {
    Win,
    Loss,
    Tie,
    Frontier,
    State(HorizonPpKey),
}

fn successor(b: &Battle, dex: &Dex, left: u16) -> Successor {
    match b.outcome() {
        Some(Outcome::P1Win) => Successor::Win,
        Some(Outcome::P2Win) => Successor::Loss,
        Some(Outcome::Tie) => Successor::Tie,
        None if left == 0 => Successor::Frontier,
        None => Successor::State(HorizonPpState::new(b, dex, left).unwrap().key()),
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "horizon_pp_audit PROTOCOL OUTPUT");
    let bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-horizon-pp-audit-v1");
    assert!(!protocol.horizons.is_empty() && protocol.horizons.iter().all(|&h| h > 0));
    let corpus = std::fs::read(&protocol.corpus).unwrap();
    assert_eq!(sha(&corpus), protocol.corpus_sha256);
    let games: Vec<Game> = String::from_utf8(corpus)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let game = games.iter().find(|g| g.game == protocol.game).unwrap();
    let dex = conformance::load_dex();
    let mut root = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
    for record in &game.decisions[..protocol.decision] {
        assert_eq!(root.state_key128().to_string(), record.key);
        let choices = [0, 1].map(|side| root.legal_choices(&dex, side));
        root.apply_choices(
            &dex,
            [0, 1].map(|side| record.choices[side].map(|i| choices[side][i])),
        )
        .unwrap();
    }
    assert_eq!(
        root.state_key128().to_string(),
        game.decisions[protocol.decision].key
    );
    let started = Instant::now();
    let actions = [0, 1].map(|side| root.legal_choices(&dex, side));
    let mut original = Vec::new();
    let mut original_runs = 0;
    let mut original_states = HashSet::new();
    for &a in &actions[0] {
        for &b in &actions[1] {
            let step = enumerate_step(&dex, &root, [Some(a), Some(b)], protocol.cell_cap).unwrap();
            original_runs += step.runs;
            for leaf in &step.leaves {
                assert!(leaf.battle.outcome().is_some() || leaf.battle.turn == root.turn + 1);
                if leaf.battle.outcome().is_none() {
                    original_states.insert(leaf.battle.state_key128_without_damage_bookkeeping());
                }
            }
            original.push(step);
        }
    }
    let mut horizons = Vec::new();
    let mut capped_runs = 0;
    for &horizon in &protocol.horizons {
        let canonical = HorizonPpState::new(&root, &dex, horizon).unwrap();
        let mut representative = canonical.battle().clone();
        for side in 0..2 {
            assert_eq!(actions[side], representative.legal_choices(&dex, side));
        }
        let mut quotient_states = HashSet::new();
        let mut cells = Vec::new();
        for (i, &a) in actions[0].iter().enumerate() {
            for (j, &b) in actions[1].iter().enumerate() {
                let index = i * actions[1].len() + j;
                let capped =
                    enumerate_step(&dex, &representative, [Some(a), Some(b)], protocol.cell_cap)
                        .unwrap();
                capped_runs += capped.runs;
                let mut distributions = Vec::new();
                for step in [&original[index], &capped] {
                    let mut distribution = HashMap::<_, f64>::new();
                    for leaf in &step.leaves {
                        let key = successor(&leaf.battle, &dex, horizon - 1);
                        if let Successor::State(key) = &key {
                            quotient_states.insert(*key);
                        }
                        *distribution.entry(key).or_default() += leaf.prob;
                    }
                    assert!((distribution.values().sum::<f64>() - 1.0).abs() < 1e-10);
                    distributions.push(distribution);
                }
                assert_eq!(distributions[0].len(), distributions[1].len());
                let maximum_error = distributions[0]
                    .iter()
                    .map(|(key, mass)| (mass - distributions[1][key]).abs())
                    .fold(0.0, f64::max);
                assert!(maximum_error < 1e-10);
                cells.push(json!({"actions":[action_input(&dex,a),action_input(&dex,b)],
                "original_runs":original[index].runs,"capped_runs":capped.runs,
                "successor_classes":distributions[0].len(),"maximum_probability_error":maximum_error,
                "probability_mass":distributions.iter().map(|d|d.values().sum::<f64>()).collect::<Vec<_>>()}));
            }
        }
        horizons.push(json!({"turns":horizon,"cells":cells,"nonterminal_quotient_states":quotient_states.len(),
            "cutoff_at_successors":horizon==1}));
    }
    assert!(protocol.sample_turns > 0);
    let mut rng = nc2000_engine::prng::Prng::new(protocol.sample_seed);
    let mut samples: Vec<_> = (0..protocol.sample_turns)
        .map(|_| (HashSet::new(), HashSet::new()))
        .collect();
    let mut sampled_runs = 0;
    for _ in 0..protocol.sample_paths {
        let mut actual = root.clone();
        actual.reseed((u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32()));
        let mut representative = HorizonPpState::new(&actual, &dex, protocol.sample_turns)
            .unwrap()
            .battle()
            .clone();
        for depth in 0..protocol.sample_turns {
            assert_eq!(actual.outcome(), representative.outcome());
            if actual.outcome().is_some() {
                break;
            }
            let left = protocol.sample_turns - depth;
            let key = HorizonPpState::new(&actual, &dex, left).unwrap().key();
            assert_eq!(
                key,
                HorizonPpState::new(&representative, &dex, left)
                    .unwrap()
                    .key()
            );
            samples[depth as usize]
                .0
                .insert(actual.state_key128_without_damage_bookkeeping());
            samples[depth as usize].1.insert(key);
            let actions = [0, 1].map(|side| actual.legal_choices(&dex, side));
            for side in 0..2 {
                assert_eq!(actions[side], representative.legal_choices(&dex, side));
            }
            let choices = [0, 1]
                .map(|side| Some(actions[side][rng.random(actions[side].len() as u32) as usize]));
            actual.apply_choices(&dex, choices).unwrap();
            representative.apply_choices(&dex, choices).unwrap();
            assert!(actual.outcome().is_some() || actual.turn == root.turn + depth + 1);
            sampled_runs += 2;
        }
        assert_eq!(actual.outcome(), representative.outcome());
    }
    let samples: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(depth, (exact, quotient))| {
            json!({
        "depth":depth,"turns_left":protocol.sample_turns-depth as u16,
        "exact_states":exact.len(),"quotient_states":quotient.len()})
        })
        .collect();
    let git = |args: &[&str]| {
        String::from_utf8(
            std::process::Command::new("git")
                .args(args)
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string()
    };
    let paths = [
        "crates/bot/src/horizon_pp.rs",
        "crates/bot/src/stall.rs",
        "crates/bot/examples/horizon_pp_audit.rs",
        "crates/engine/src/state.rs",
        "crates/engine/src/battle/search.rs",
        "crates/engine/src/battle/pokemon.rs",
        "crates/engine/src/battle/moveexec.rs",
        "crates/engine/src/battle/conditions.rs",
        "crates/engine/src/battle/items.rs",
    ];
    let report = json!({"schema":protocol.schema,"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),
        "protocol_sha256":sha(&bytes),"revision":git(&["rev-parse","HEAD"]),"dirty":git(&["status","--porcelain"]),
        "binary_sha256":sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
        "source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),"state_key":root.state_key128().to_string(),
        "original_nonterminal_states":original_states.len(),"original_runs":original_runs,"capped_runs":capped_runs,
        "sampled_runs":sampled_runs,"sampled_states":samples,
        "total_runs":original_runs+capped_runs+sampled_runs,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"horizons":horizons,
        "interpretation":"One-step transition-equivalence and seeded uniform-legal paired-path audits with a remaining-turn-dependent PP key. Sample counts describe only the sampled reachable frontier. H=1 collapses nonterminal successors solely because they are at the unknown boundary; its state count is not a PP-merging gain. This is not a full-game value certificate or a strength claim."});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!(
        "runs={} states={} horizons={}",
        original_runs + capped_runs + sampled_runs,
        original_states.len(),
        protocol.horizons.len()
    );
}
