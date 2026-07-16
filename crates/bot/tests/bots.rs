//! Bot smoke tests: every agent drives real battles to completion through
//! the search API without ever producing an illegal choice. (Strength
//! ordering is measured by the arena example, not asserted here beyond a
//! coarse sanity floor.)

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::mcts::Playout;
use nc2000_bot::{
    eval, play_game, Agent, EvalWeights, GameResult, MaxDamageAgent, MctsAgent, MctsConfig,
    RandomAgent, SplitMix64,
};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::state::Battle;

fn team_pool() -> Vec<Vec<PokemonSet>> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut teams = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            let fx = Fixture::load(&path).unwrap();
            teams.push(fx.p1team);
            teams.push(fx.p2team);
        }
    }
    teams
}

/// Play `games` seed-paired games, return agent A's mean score.
fn duel(
    build_a: &dyn Fn(u64) -> Box<dyn Agent>,
    build_b: &dyn Fn(u64) -> Box<dyn Agent>,
    games: usize,
    base_seed: u64,
) -> f64 {
    let dex = load_dex();
    let teams = team_pool();
    let mut rng = SplitMix64::new(base_seed);
    let mut score = 0.0;
    for k in 0..games / 2 {
        let t1 = rng.below(teams.len());
        let t2 = rng.below(teams.len());
        let seed = rng.battle_seed();
        for a_is_p1 in [true, false] {
            let g = (k * 2 + a_is_p1 as usize) as u64;
            let mut a = build_a(base_seed ^ g.wrapping_mul(0xA24B_AED4_963E_E407));
            let mut b = build_b(base_seed ^ g.wrapping_mul(0x9FB2_1C65_1E98_DF25));
            let mut battle =
                Battle::from_fixture(&dex, &seed, &teams[t1], &teams[t2]).unwrap();
            battle.set_log_enabled(false);
            let (p1, p2): (&mut dyn Agent, &mut dyn Agent) =
                if a_is_p1 { (a.as_mut(), b.as_mut()) } else { (b.as_mut(), a.as_mut()) };
            let res = play_game(&dex, &mut battle, &mut [p1, p2], 500).unwrap();
            let p1_score = match res {
                GameResult::Outcome(Outcome::P1Win) => 1.0,
                GameResult::Outcome(Outcome::P2Win) => 0.0,
                GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
            };
            score += if a_is_p1 { p1_score } else { 1.0 - p1_score };
        }
    }
    score / games as f64
}

#[test]
fn random_vs_random_completes() {
    let s = duel(
        &|seed| Box::new(RandomAgent::new(seed)),
        &|seed| Box::new(RandomAgent::new(seed ^ 0xDEAD)),
        10,
        42,
    );
    assert!((0.0..=1.0).contains(&s));
}

#[test]
fn maxdamage_beats_random() {
    let s = duel(
        &|_| Box::new(MaxDamageAgent::new()),
        &|seed| Box::new(RandomAgent::new(seed)),
        10,
        7,
    );
    assert!(s >= 0.6, "maxdamage vs random score {s}");
}

#[test]
fn mcts_uniform_smoke() {
    let s = duel(
        &|seed| Box::new(MctsAgent::new(MctsConfig::uniform(32, 1.0), seed)),
        &|seed| Box::new(RandomAgent::new(seed)),
        2,
        3,
    );
    assert!((0.0..=1.0).contains(&s));
}

#[test]
fn mcts_heavy_smoke() {
    // default config = M6 heavy playout
    let s = duel(
        &|seed| {
            Box::new(MctsAgent::new(
                MctsConfig { iterations: 32, ..Default::default() },
                seed,
            ))
        },
        &|seed| Box::new(RandomAgent::new(seed)),
        2,
        3,
    );
    assert!((0.0..=1.0).contains(&s));
    assert!(matches!(MctsConfig::default().playout, Playout::Heavy { .. }));
}

#[test]
fn eval_symmetry_and_direction() {
    let dex = load_dex();
    let teams = team_pool();
    let w = EvalWeights::default();

    // mirror match: exactly symmetric state -> 0.5
    let b = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[0]).unwrap();
    let e = eval::eval01(&b, &dex, &w);
    assert!((e - 0.5).abs() < 1e-12, "mirror eval {e}");

    // damage side 1's whole team -> side 0 favored, leaf stays in (0.25, 0.75)
    let mut b2 = b.clone();
    for p in b2.sides[1].roster.iter_mut() {
        p.hp = (p.maxhp / 10).max(1);
    }
    let e2 = eval::eval01(&b2, &dex, &w);
    assert!(e2 > 0.6, "damaged-foe eval {e2}");
    let leaf = eval::eval_leaf(&b2, &dex, &w);
    assert!((0.25..0.75).contains(&leaf), "leaf {leaf}");
}

#[test]
fn eval_weights_roundtrip() {
    let w = EvalWeights::default();
    let rt = EvalWeights::from_vec(&w.to_vec(), w.scale);
    assert_eq!(w, rt);
}
