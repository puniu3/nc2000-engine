//! M11a mutation-operator + gauntlet-fitness tests: every proposal is
//! validator-clean (zero findings), the format's level-sum landmines hold,
//! proposals are diverse and deterministic per seed, and fitness evaluation
//! is bit-identical at any thread count.

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::teamgen::{gauntlet_eval, team_key, to_sets, EvalCfg, MutOp, TeamGen};
use nc2000_bot::SplitMix64;
use nc2000_engine::dex::Dex;
use nc2000_engine::validate::{validate_team, Learnsets};
use serde_json::Value;

fn load() -> (Dex, TeamGen, Learnsets) {
    let root = repo_root();
    let dex = load_dex();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text =
        std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();
    let ls = Learnsets::from_json(&ls_text).unwrap();
    (dex, gen, ls)
}

/// The format's level rules, asserted independently of the validator:
/// levels in 50..=55, lowest-3 sum <= 155, highest + lowest-2 <= 155.
fn assert_level_rules(team: &[Value]) {
    let mut levels: Vec<i64> =
        team.iter().map(|s| s["level"].as_i64().expect("canonical level")).collect();
    levels.sort_unstable();
    assert_eq!(levels.len(), 6);
    for &l in &levels {
        assert!((50..=55).contains(&l), "level {l} out of range");
    }
    assert!(levels[0] + levels[1] + levels[2] <= 155, "lowest-3 sum: {levels:?}");
    assert!(levels[5] + levels[0] + levels[1] <= 155, "highest unusable: {levels:?}");
}

fn assert_clean(dex: &Dex, ls: &Learnsets, team: &[Value], ctx: &str) {
    let v = validate_team(dex, ls, &serde_json::to_string(team).unwrap());
    assert_eq!(v["ok"], true, "{ctx}: {v}");
    assert!(
        v["findings"].as_array().unwrap().is_empty(),
        "{ctx}: non-canonical proposal: {v}"
    );
}

#[test]
fn mutations_are_validator_clean_and_diverse() {
    let (dex, gen, ls) = load();
    // parents: a T1 team, a mid sample team, and a deliberately weakened team
    let mut wrng = SplitMix64::new(7);
    let parents: Vec<Vec<Value>> = vec![
        gen.canonize(&dex, &gen.team_json(0)).unwrap(),
        gen.canonize(&dex, &gen.team_json(20)).unwrap(),
        gen.weaken(&dex, &gen.canonize(&dex, &gen.team_json(3)).unwrap(), &mut wrng, 4).unwrap(),
    ];

    let mut ops_seen = std::collections::HashSet::new();
    let mut distinct = std::collections::HashSet::new();
    let mut total = 0usize;
    for (i, seed) in [11u64, 222, 3333, 44444].into_iter().enumerate() {
        let parent = &parents[i % parents.len()];
        let mut rng = SplitMix64::new(seed);
        for _ in 0..250 {
            let p = gen
                .propose_valid(&dex, parent, &mut rng, 60)
                .expect("a valid proposal within 60 draws");
            total += 1;
            ops_seen.insert(p.op.name());
            distinct.insert(team_key(&p.team));
            assert_clean(&dex, &ls, &p.team, &format!("seed {seed} op {}", p.op.name()));
            assert_level_rules(&p.team);
            assert!(team_key(&p.team) != team_key(parent), "no-op proposal emitted");
            // proposals must construct engine sets (battle-ready)
            to_sets(&p.team).unwrap();
        }
    }
    assert_eq!(total, 1000);
    assert_eq!(
        ops_seen.len(),
        MutOp::ALL.len(),
        "all operators should fire across 1000 proposals: {ops_seen:?}"
    );
    assert!(
        distinct.len() >= total / 2,
        "diversity: only {} distinct teams in {total} proposals",
        distinct.len()
    );
}

#[test]
fn proposals_are_deterministic_per_seed() {
    let (dex, gen, _) = load();
    let parent = gen.canonize(&dex, &gen.team_json(1)).unwrap();
    let run = |seed: u64| -> Vec<String> {
        let mut rng = SplitMix64::new(seed);
        (0..40)
            .map(|_| team_key(&gen.propose_valid(&dex, &parent, &mut rng, 60).unwrap().team))
            .collect()
    };
    assert_eq!(run(5), run(5), "same seed must reproduce the proposal stream");
    assert_ne!(run(5), run(6), "different seeds should diverge");
}

#[test]
fn weaken_strips_items_and_stays_legal() {
    let (dex, gen, ls) = load();
    let parent = gen.canonize(&dex, &gen.team_json(0)).unwrap();
    let mut rng = SplitMix64::new(42);
    let weak = gen.weaken(&dex, &parent, &mut rng, 3).unwrap();
    assert_clean(&dex, &ls, &weak, "weakened team");
    assert_level_rules(&weak);
    assert_ne!(team_key(&weak), team_key(&parent));
    for set in &weak {
        assert_eq!(set["item"].as_str().unwrap(), "", "weaken must strip items");
    }
}

#[test]
fn random_teams_are_legal() {
    let (dex, gen, ls) = load();
    let mut rng = SplitMix64::new(9);
    for _ in 0..5 {
        let t = gen.random_team_valid(&dex, &mut rng, 40).expect("a random legal team");
        assert_clean(&dex, &ls, &t, "random team");
        assert_level_rules(&t);
    }
}

#[test]
fn gauntlet_eval_is_thread_count_invariant() {
    let (dex, gen, _) = load();
    let cand = to_sets(&gen.canonize(&dex, &gen.team_json(0)).unwrap()).unwrap();
    let opp = to_sets(&gen.canonize(&dex, &gen.team_json(1)).unwrap()).unwrap();
    let gauntlet = vec![opp];
    let cfg = |threads: usize| EvalCfg {
        games_per_opponent: 2,
        agent_iters: 12,
        max_turns: 100,
        threads,
        seed: 21,
    };
    let a = gauntlet_eval(&dex, &cand, &gauntlet, &cfg(1));
    let b = gauntlet_eval(&dex, &cand, &gauntlet, &cfg(4));
    assert_eq!(a.games, 2);
    assert_eq!(a.score.to_bits(), b.score.to_bits(), "{} vs {}", a.score, b.score);
    assert_eq!(a.per_opponent.len(), 1);
}
