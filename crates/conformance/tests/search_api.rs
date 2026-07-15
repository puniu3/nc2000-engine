//! M3 search-API conformance:
//! 1. Legality: at every decision point of every golden fixture, the choice
//!    PS actually took is inside `legal_choices`, and (sampled) every
//!    enumerated choice is accepted by the engine on a clone.
//! 2. No-log parity: replaying with the protocol log disabled reaches a
//!    bit-identical final state + PRNG seed.
//! 3. Random playouts: driving battles with uniformly random legal choices
//!    terminates cleanly (engine's own turn-1000 tie is the backstop).

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_engine::state::Battle;

fn all_fixtures() -> Vec<Fixture> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut out = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            out.push(Fixture::load(&path).unwrap());
        }
    }
    assert!(!out.is_empty());
    out
}

fn side_index(side: &str) -> usize {
    if side == "p1" {
        0
    } else {
        1
    }
}

/// 1. Fixture choices ⊆ legal_choices; enumerated choices are all playable.
#[test]
fn legal_choices_cover_corpus() {
    let dex = load_dex();
    for fx in all_fixtures() {
        let mut battle = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        for (i, line) in fx.choices.iter().enumerate() {
            let side_n = side_index(&line.side);
            let legal = battle.legal_choices(&dex, side_n);
            let strings: Vec<String> = legal.iter().map(|c| c.to_input(&dex)).collect();
            assert!(
                strings.iter().any(|s| s == &line.choice),
                "fixture {}-{:03}: choice {:?} (line {i}) not in legal set {strings:?}",
                fx.meta.pool,
                fx.meta.index,
                line.choice,
            );
            // Sampled exhaustive check: every enumerated choice must be
            // accepted by choose() — on a clone so the replay stays on the
            // fixture's path. (Every 5th decision point; committing clones
            // run full off-fixture turns, which is the point.)
            if i % 5 == 0 {
                for choice in &legal {
                    let mut probe = battle.clone();
                    probe.set_log_enabled(false);
                    if let Err(e) = probe.apply_choice(&dex, side_n, *choice) {
                        panic!(
                            "fixture {}-{:03} line {i}: enumerated choice {:?} rejected: {e:?}",
                            fx.meta.pool,
                            fx.meta.index,
                            choice.to_input(&dex),
                        );
                    }
                }
            }
            battle.choose(&dex, side_n, &line.choice).unwrap();
        }
    }
}

/// 2. Log-off replay is state-identical to log-on replay.
#[test]
fn nolog_replay_reaches_identical_state() {
    let dex = load_dex();
    for fx in all_fixtures() {
        let mut on = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        let mut off = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        off.set_log_enabled(false);
        let frozen_log_len = off.log.len();
        for line in &fx.choices {
            let side_n = side_index(&line.side);
            on.choose(&dex, side_n, &line.choice).unwrap();
            off.choose(&dex, side_n, &line.choice).unwrap();
        }
        let tag = format!("{}-{:03}", fx.meta.pool, fx.meta.index);
        assert_eq!(off.log.len(), frozen_log_len, "{tag}: log grew while disabled");
        assert_eq!(on.prng.seed_str(), off.prng.seed_str(), "{tag}: PRNG diverged");
        assert_eq!(on.turn, off.turn, "{tag}: turn diverged");
        assert_eq!(on.winner, off.winner, "{tag}: winner diverged");
        assert_eq!(on.essence(&dex), off.essence(&dex), "{tag}: state essence diverged");
    }
}

/// Cheap deterministic test RNG (splitmix64) — NOT the battle PRNG.
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick(&mut self, len: usize) -> usize {
        (self.next() % len as u64) as usize
    }
}

/// 3. Random playouts terminate and produce an outcome.
#[test]
fn random_playouts_terminate() {
    let dex = load_dex();
    for (fi, fx) in all_fixtures().into_iter().enumerate() {
        for playout in 0..3u64 {
            let mut battle =
                Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            battle.set_log_enabled(false);
            battle.reseed(0xC0FFEE ^ (fi as u64) << 8 ^ playout);
            let mut rng = TestRng(0xDEAD_BEEF ^ (fi as u64) << 32 ^ playout);
            let mut steps = 0u32;
            while battle.outcome().is_none() {
                steps += 1;
                assert!(
                    steps < 10_000,
                    "{}-{:03} playout {playout}: no termination after {steps} decision points",
                    fx.meta.pool,
                    fx.meta.index,
                );
                let picks = [0usize, 1].map(|side_n| {
                    let legal = battle.legal_choices(&dex, side_n);
                    if legal.is_empty() {
                        None
                    } else {
                        Some(legal[rng.pick(legal.len())])
                    }
                });
                assert!(
                    picks.iter().any(|p| p.is_some()),
                    "{}-{:03} playout {playout}: battle running but nobody owes a choice",
                    fx.meta.pool,
                    fx.meta.index,
                );
                battle.apply_choices(&dex, picks).unwrap();
            }
        }
    }
}
