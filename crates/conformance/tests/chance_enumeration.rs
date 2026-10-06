use std::collections::BTreeMap;

use nc2000_engine::battle::enumerate::{enumerate_step, enumerate_step_partial, run_scripted};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn mon(species: &str, moves: &[&str]) -> PokemonSet {
    PokemonSet {
        name: species.into(), species: species.into(), item: String::new(), ability: String::new(),
        moves: moves.iter().map(|s| (*s).into()).collect(), level: 50,
        evs: None, ivs: None, happiness: None, gender: None,
    }
}

fn reference(dex: &Dex, root: &Battle, actions: [Option<SearchChoice>; 2]) -> (BTreeMap<u128, f64>, usize) {
    let mut pending = vec![vec![]];
    let mut out = BTreeMap::new();
    let mut runs = 0;
    while let Some(script) = pending.pop() {
        let (leaf, trace) = run_scripted(dex, root, actions, &script);
        runs += 1;
        assert!(runs < 1_000_000);
        if let Some(draw) = trace.get(script.len()) {
            for (class, &mass) in draw.counts.iter().enumerate() {
                if mass != 0 {
                    let mut next = script.clone();
                    next.push(class);
                    pending.push(next);
                }
            }
        } else {
            *out.entry(leaf.state_key128()).or_insert(0.0) += trace.iter().map(|d| d.prob()).product::<f64>();
        }
    }
    (out, runs)
}

#[test]
fn reused_traces_preserve_exhaustive_successor_distributions_and_run_caps() {
    let dex = conformance::load_dex();
    for (species, first, second, hp) in [
        ("Pikachu", "Quick Attack", "Quick Attack", 1),
        ("Pikachu", "Thunderbolt", "Substitute", 100),
        ("Pikachu", "Double Kick", "Counter", 100),
        ("Snorlax", "Double-Edge", "Rest", 80),
        ("Exeggutor", "Mega Drain", "Future Sight", 80),
        ("Snorlax", "Body Slam", "Earthquake", 120),
    ] {
        let mut root = Battle::from_fixture(&dex, "1,2,3,4", &[mon(species, &[first])], &[mon(species, &[second])]).unwrap();
        let picks = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, picks).unwrap();
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = hp.min(root.poke(id).maxhp);
        }
        let actions = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        let (expected, reference_runs) = reference(&dex, &root, actions);
        let result = enumerate_step(&dex, &root, actions, reference_runs).unwrap();
        assert!(result.runs < reference_runs, "{first} vs {second}");
        let mut actual = BTreeMap::new();
        for leaf in result.leaves {
            *actual.entry(leaf.battle.state_key128()).or_insert(0.0) += leaf.prob;
        }
        assert_eq!(actual.len(), expected.len(), "{first} vs {second}");
        assert!((actual.values().sum::<f64>() - 1.0).abs() < 1e-10);
        for (state, probability) in &expected {
            assert!((actual[state] - probability).abs() < 1e-10, "{first} vs {second}");
        }
        let exact_cap = enumerate_step(&dex, &root, actions, result.runs).unwrap();
        assert_eq!(exact_cap.runs, result.runs);
        assert!(enumerate_step(&dex, &root, actions, result.runs - 1).is_none());
        assert!(enumerate_step(&dex, &root, actions, 0).is_none());
        for cap in [0, 1, 7, result.runs - 1, result.runs] {
            let partial = enumerate_step_partial(&dex, &root, actions, cap);
            assert_eq!(partial.leaf_scripts.is_some(),partial.exhausted);
            if let Some(scripts) = &partial.leaf_scripts {
                assert_eq!(scripts.len(),partial.leaves.len());
                for (script,leaf) in scripts.iter().zip(&partial.leaves) {
                    let (replayed,trace) = run_scripted(&dex,&root,actions,script);
                    assert_eq!(script.len(),trace.len());
                    assert_eq!(replayed.state_key128(),leaf.battle.state_key128());
                    assert!((trace.iter().map(|d|d.prob()).product::<f64>()-leaf.prob).abs()<1e-12);
                }
            }
            assert!(partial.runs <= cap);
            if !partial.pending.is_empty() { assert_eq!(partial.runs, cap); }
            assert!((partial.leaves.iter().map(|l|l.prob).sum::<f64>()
                + partial.pending.iter().map(|p|p.mass).sum::<f64>() - 1.).abs()<1e-10);
            let mut actual = BTreeMap::new();
            for leaf in partial.leaves {
                *actual.entry(leaf.battle.state_key128()).or_insert(0.) += leaf.prob;
            }
            let mut pending: Vec<_> = partial.pending.into_iter().map(|p| {
                let (_, trace) = run_scripted(&dex, &root, actions, &p.script);
                assert!((trace.iter().take(p.script.len()).map(|d|d.prob()).product::<f64>()-p.mass).abs()<1e-10);
                p.script
            }).collect();
            while let Some(script) = pending.pop() {
                let (leaf, trace) = run_scripted(&dex, &root, actions, &script);
                if let Some(draw) = trace.get(script.len()) {
                    for (class, &count) in draw.counts.iter().enumerate() {
                        if count>0 { let mut next=script.clone();next.push(class);pending.push(next); }
                    }
                } else {
                    *actual.entry(leaf.state_key128()).or_insert(0.) += trace.iter().map(|d|d.prob()).product::<f64>();
                }
            }
            assert_eq!(actual.len(),expected.len(),"cap={cap}, {first} vs {second}");
            for (state, probability) in &expected {
                assert!((actual[state]-probability).abs()<1e-10,"cap={cap}, {first} vs {second}");
            }
        }
    }
}
