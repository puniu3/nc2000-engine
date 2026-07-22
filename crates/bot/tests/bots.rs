//! Bot smoke tests: every agent drives real battles to completion through
//! the search API without ever producing an illegal choice. (Strength
//! ordering is measured by the arena example, not asserted here beyond a
//! coarse sanity floor.)

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::mcts::Playout;
use nc2000_bot::{
    eval, play_game, Agent, BrAgent, EvalWeights, GameResult, MaxDamageAgent, MctsAgent,
    MctsConfig, RandomAgent, RmAgent, RmConfig, SplitMix64,
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
fn rm_smoke_and_determinism() {
    let build = |seed: u64| -> Box<dyn Agent> {
        Box::new(RmAgent::new(RmConfig { iterations: 32, ..Default::default() }, seed))
    };
    let s1 = duel(&build, &|seed| Box::new(RandomAgent::new(seed)), 2, 3);
    let s2 = duel(&build, &|seed| Box::new(RandomAgent::new(seed)), 2, 3);
    assert!((0.0..=1.0).contains(&s1));
    assert_eq!(s1, s2, "rm agent must be deterministic per seed");
}

#[test]
fn exploit_smoke() {
    let s = duel(
        &|seed| {
            let model: Box<dyn Agent> = Box::new(MaxDamageAgent::new());
            Box::new(BrAgent::new(
                model,
                2,
                MctsConfig { iterations: 32, ..Default::default() },
                seed,
            ))
        },
        &|_| Box::new(MaxDamageAgent::new()),
        2,
        5,
    );
    assert!((0.0..=1.0).contains(&s));
}

#[test]
fn rm_root_policy_is_a_distribution() {
    let dex = load_dex();
    let teams = team_pool();
    let mut b = Battle::from_fixture(&dex, "11,22,33,44", &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(false);
    let choices = b.legal_choices(&dex, 0);
    assert!(choices.len() >= 2);

    let mut rm = RmAgent::new(RmConfig { iterations: 64, ..Default::default() }, 9);
    let probs = rm.root_policy(&b, &dex, 0, &choices);
    assert_eq!(probs.len(), choices.len());
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "policy sums to {total}");
    assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));

    // default root_policy (argmax agents): a point mass on choose()
    let mut md = MaxDamageAgent::new();
    let probs = md.root_policy(&b, &dex, 0, &choices);
    assert_eq!(probs.iter().filter(|&&p| p == 1.0).count(), 1);
    assert_eq!(probs.iter().sum::<f64>(), 1.0);
}

#[test]
fn state_key_contract() {
    let dex = load_dex();
    let teams = team_pool();
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(false);

    // clone: same key; reseed: PRNG is excluded from the key
    let mut c = b.clone();
    assert_eq!(b.state_key(), c.state_key());
    c.reseed(0xDEAD_BEEF);
    assert_eq!(b.state_key(), c.state_key());

    // log recording must not affect the key
    let mut logged = b.clone();
    logged.set_log_enabled(true);
    assert_eq!(b.state_key(), logged.state_key());

    // advancing the battle changes the key
    let picks = [0, 1].map(|s| {
        let cs = c.legal_choices(&dex, s);
        cs.first().copied()
    });
    c.apply_choices(&dex, picks).unwrap();
    assert_ne!(b.state_key(), c.state_key());

    // same choices from the same state + same seed: identical key
    let mut d = b.clone();
    d.reseed(0xDEAD_BEEF); // match c's seed so chance rolls identically
    let picks = [0, 1].map(|s| {
        let cs = d.legal_choices(&dex, s);
        cs.first().copied()
    });
    d.apply_choices(&dex, picks).unwrap();
    assert_eq!(c.state_key(), d.state_key());
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

    // damage side 1's whole team -> side 0 favored; the shipped cutoff
    // keeps eval01's probability-shaped value.
    let mut b2 = b.clone();
    for p in b2.sides[1].roster.iter_mut() {
        p.hp = (p.maxhp / 10).max(1);
    }
    let e2 = eval::eval01(&b2, &dex, &w);
    assert!(e2 > 0.6, "damaged-foe eval {e2}");
    let leaf = eval::eval_leaf(&b2, &dex, &w);
    assert!((leaf - e2).abs() < 1e-12, "leaf {leaf} eval {e2}");

    let legacy = EvalWeights { leaf_alpha: 0.5, ..EvalWeights::default() };
    let legacy_leaf = eval::eval_leaf(&b2, &dex, &legacy);
    assert!((legacy_leaf - (0.25 + 0.5 * e2)).abs() < 1e-12);
}

#[test]
fn eval_weights_roundtrip() {
    let w = EvalWeights::default();
    let rt = EvalWeights::from_vec(&w.to_vec(), w.scale);
    assert_eq!(w, rt);
}
