//! M10b Gate B — no psychic tells at parity budget — plus blind-agent
//! lifecycle smoke.
//!
//! # Gate B (`no_psychic_tells_on_the_collision_pair`)
//!
//! Behavioral invariance: run the blind agent (fixed agent seed, fixed
//! budget) as P1 against BOTH collision-pair teams as the hidden truth,
//! with the opponent driven by a shared script (choices legal under both
//! truths) so the two games' *public* trajectories agree as long as
//! possible. While they agree, the blind agent's decisions must be
//! IDENTICAL — any divergence is a leak of set-level hidden information.
//!
//! "Public trajectories agree" is checked by the strongest available
//! projection: determinize both true battles with the SAME candidate and
//! the SAME rng — the determinizer overwrites exactly the hidden fields, so
//! Debug-equality of the outputs means everything the search can see is
//! bit-identical. This is conservative: it also pins the declared non-goal
//! fields the determinizer keeps (hidden status counters, quick_claw_roll),
//! which can legitimately drift once the two truths' PRNG streams diverge —
//! the loop then just stops, it does not fail. The gate requires at least
//! the preview decision plus one in-battle decision proven identical, with
//! the belief still holding both candidates (the ambiguity actually
//! exercised).
//!
//! The two truth battles align the collision teams' display order (bench
//! order is the builder's free choice, not team identity — the belief's
//! preview filter is order-insensitive for the same reason), so the
//! species-by-display-position public view matches.

use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::{Agent, Belief, BlindAgent, Observer, RmConfig, SplitMix64};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn pool() -> Arc<MetaPool> {
    Arc::new(load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json")))
}

/// The pool's known species+level collision (README: exactly one).
fn collision_pair(pool: &MetaPool) -> (usize, usize) {
    let sig = |sets: &[PokemonSet]| {
        let mut v: Vec<(String, u8)> =
            sets.iter().map(|s| (s.species.clone(), s.level)).collect();
        v.sort();
        v
    };
    for i in 0..pool.teams.len() {
        for j in i + 1..pool.teams.len() {
            if sig(&pool.teams[i].sets) == sig(&pool.teams[j].sets) {
                return (i, j);
            }
        }
    }
    panic!("no species+level collision in the pool (README says there is one)");
}

/// Permute `other` to `reference`'s (species, level) display order.
fn align_order(reference: &[PokemonSet], other: &[PokemonSet]) -> Vec<PokemonSet> {
    let mut remaining: Vec<PokemonSet> = other.to_vec();
    reference
        .iter()
        .map(|r| {
            let k = remaining
                .iter()
                .position(|t| t.species == r.species && t.level == r.level)
                .expect("collision pair must share the species+level multiset");
            remaining.remove(k)
        })
        .collect()
}

/// Debug projection of everything the blind search can see of `battle`:
/// determinized with a fixed candidate + fixed rng, hidden fields are
/// overwritten identically, so equality across the two truths ⇔ the
/// search-visible state is bit-identical.
fn projection(
    dex: &Dex,
    bel: &Belief,
    battle: &Battle,
    obs: &Observer,
    cand: usize,
) -> String {
    let mut rng = SplitMix64::new(0x9E37);
    format!("{:?}", bel.determinize_with(dex, battle, obs, Some(cand), &mut rng))
}

#[test]
fn no_psychic_tells_on_the_collision_pair() {
    let dex = load_dex();
    let meta = pool();
    let (ci, cj) = collision_pair(&meta);
    let our = meta.teams[0].sets.clone();
    let truth_a = meta.teams[ci].sets.clone();
    let truth_b = align_order(&truth_a, &meta.teams[cj].sets);

    let mut total_checkpoints = 0usize;
    let mut ambiguous_battle_checkpoints = 0usize;

    for (run, battle_seed) in ["11,22,33,44", "1,3,5,7"].iter().enumerate() {
        let mut battle_a = Battle::from_fixture(&dex, battle_seed, &our, &truth_a).unwrap();
        let mut battle_b = Battle::from_fixture(&dex, battle_seed, &our, &truth_b).unwrap();
        // outer battles log-ON: the observer's log channel is part of the
        // surface under test

        let cfg = RmConfig { iterations: 96, ..Default::default() };
        let agent_seed = 0xB11D + run as u64;
        let mut agent_a = BlindAgent::new(cfg.clone(), meta.clone(), None, agent_seed);
        let mut agent_b = BlindAgent::new(cfg.clone(), meta.clone(), None, agent_seed);

        // test-owned observers/beliefs (independent of the agents') define
        // the public-agreement region
        let mut obs = [Observer::new(&battle_a, 0), Observer::new(&battle_b, 0)];
        let mut bel =
            [Belief::new(&dex, &meta, &obs[0]), Belief::new(&dex, &meta, &obs[1])];

        let mut points = 0usize;
        loop {
            if battle_a.outcome().is_some()
                || battle_b.outcome().is_some()
                || battle_a.turn > 8
            {
                break;
            }
            obs[0].observe(&battle_a, &dex);
            bel[0].sync(&dex, &obs[0]);
            obs[1].observe(&battle_b, &dex);
            bel[1].sync(&dex, &obs[1]);

            if points == 0 {
                // premise: the collision keeps both candidates alive at preview
                for b in &bel {
                    assert!(
                        b.alive().contains(&ci) && b.alive().contains(&cj),
                        "premise broken: collision candidates not both alive at preview \
                         (alive {:?})",
                        b.alive()
                    );
                }
            }

            // ---- public-agreement region check (see module doc)
            if bel[0].alive() != bel[1].alive() {
                break;
            }
            let agree = bel[0].alive().iter().all(|&cand| {
                projection(&dex, &bel[0], &battle_a, &obs[0], cand)
                    == projection(&dex, &bel[1], &battle_b, &obs[1], cand)
            });
            if !agree {
                break;
            }

            // ---- the gate: identical decisions from identical public views
            let cs_a = battle_a.legal_choices(&dex, 0);
            let cs_b = battle_b.legal_choices(&dex, 0);
            assert_eq!(cs_a, cs_b, "own legal choices diverged inside the agreement region");
            let mut picks_a = [None, None];
            let mut picks_b = [None, None];
            if !cs_a.is_empty() {
                let pa = agent_a.choose(&battle_a, &dex, 0, &cs_a);
                let pb = agent_b.choose(&battle_b, &dex, 0, &cs_b);
                assert_eq!(
                    pa, pb,
                    "GATE B LEAK: blind decision diverged between hidden truths \
                     (seed {battle_seed}, turn {}, checkpoint {points})",
                    battle_a.turn
                );
                total_checkpoints += 1;
                if bel[0].candidate_count() >= 2
                    && !matches!(cs_a[0], SearchChoice::Team(_))
                {
                    ambiguous_battle_checkpoints += 1;
                }
                picks_a[0] = Some(pa);
                picks_b[0] = Some(pb);
            }

            // ---- scripted opponent: first choice legal under BOTH truths
            let os_a = battle_a.legal_choices(&dex, 1);
            let os_b = battle_b.legal_choices(&dex, 1);
            assert_eq!(
                os_a.is_empty(),
                os_b.is_empty(),
                "request kinds diverged inside the agreement region"
            );
            if !os_a.is_empty() {
                let Some(c) = os_a.iter().copied().find(|c| os_b.contains(c)) else {
                    break; // no shared legal action left: region ends here
                };
                picks_a[1] = Some(c);
                picks_b[1] = Some(c);
            }
            battle_a.apply_choices(&dex, picks_a).unwrap();
            battle_b.apply_choices(&dex, picks_b).unwrap();
            points += 1;
        }
    }

    eprintln!(
        "gate B: {total_checkpoints} identical-decision checkpoints \
         ({ambiguous_battle_checkpoints} in-battle with live ambiguity)"
    );
    assert!(
        total_checkpoints >= 3,
        "only {total_checkpoints} identical-decision checkpoints proven (need preview + \
         in-battle coverage)"
    );
    assert!(
        ambiguous_battle_checkpoints >= 1,
        "no in-battle checkpoint had both collision candidates alive — the ambiguity was \
         never exercised"
    );
}

// -------------------------------------------------------- lifecycle smoke

/// One full pool game (blind P1 vs a scripted greedy P2, log ON): the
/// belief never goes fallback and always keeps the true team; then the SAME
/// agent instance starts a new game against a different team — the preview
/// reset path must rebuild the observer/belief (arena constructs agents
/// fresh per game; this is the safety net for other harnesses).
#[test]
fn blind_lifecycle_and_reset() {
    let dex = load_dex();
    let meta = pool();
    let cfg = RmConfig { iterations: 32, ..Default::default() };
    let mut agent = BlindAgent::new(cfg, meta.clone(), None, 77);

    for (game, truth_idx) in [(0usize, 1usize), (1, 5)] {
        let mut battle = Battle::from_fixture(
            &dex,
            "2,4,6,8",
            &meta.teams[0].sets,
            &meta.teams[truth_idx].sets,
        )
        .unwrap();
        let mut opp = nc2000_bot::MaxDamageAgent::new();
        let mut decisions = 0usize;
        while battle.outcome().is_none() && battle.turn <= 100 {
            let mut picks = [None, None];
            let cs = battle.legal_choices(&dex, 0);
            if !cs.is_empty() {
                picks[0] = Some(agent.choose(&battle, &dex, 0, &cs));
                decisions += 1;
                let bel = agent.belief().expect("belief exists after a decision");
                assert!(!bel.is_fallback(), "pool opponent must never go fallback");
                assert!(
                    bel.alive().contains(&truth_idx),
                    "game {game}: true team {truth_idx} filtered (alive {:?})",
                    bel.alive()
                );
            }
            let os = battle.legal_choices(&dex, 1);
            if !os.is_empty() {
                picks[1] = Some(opp.choose(&battle, &dex, 1, &os));
            }
            battle.apply_choices(&dex, picks).unwrap();
        }
        assert!(decisions > 5, "game {game}: too few decisions ({decisions})");
    }
}
