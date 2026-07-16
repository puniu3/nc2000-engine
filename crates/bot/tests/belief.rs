//! M10a belief machinery tests:
//! (a) ground truth — over real pool-vs-pool games the true opponent team
//!     stays in the belief's candidate set at every decision point and the
//!     candidate count is monotonically non-increasing;
//! (b) determinized battles are legal and playable to completion from any
//!     mid-game point, and never disturb the observer's own legal choices;
//! (c) leak check — with ≥ 2 candidates alive (the pool's known
//!     species+level collision), determinized samples differ in hidden
//!     fields while agreeing on all public ones;
//! (d) the fallback path (non-pool opponent) never panics.

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::{load_meta_pool, MetaPool, RolloutAgent};
use nc2000_bot::{play_game, Agent, Belief, Observer, RandomAgent, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn pool() -> MetaPool {
    load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"))
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

/// Drive one tracked game (protocol log ON — the observer's log channel is
/// exercised), calling `at_point` after observe+sync at every decision
/// point. Sides play `RolloutAgent`.
#[allow(clippy::too_many_arguments)]
fn drive(
    dex: &Dex,
    meta: &MetaPool,
    p1: &[PokemonSet],
    p2: &[PokemonSet],
    seed: &str,
    agent_seed: u64,
    max_turns: u16,
    mut at_point: impl FnMut(&Battle, [&Observer; 2], [&Belief; 2], usize),
) {
    let mut battle = Battle::from_fixture(dex, seed, p1, p2).unwrap();
    let mut a0 = RolloutAgent::new(0.35, agent_seed);
    let mut a1 = RolloutAgent::new(0.35, agent_seed ^ 0x9E37_79B9);
    let mut obs = [Observer::new(&battle, 0), Observer::new(&battle, 1)];
    let mut bel = [
        Belief::new(dex, meta, &obs[0]),
        Belief::new(dex, meta, &obs[1]),
    ];
    let mut points = 0usize;
    while battle.outcome().is_none() && battle.turn <= max_turns {
        for s in 0..2 {
            obs[s].observe(&battle, dex);
            bel[s].sync(dex, &obs[s]);
        }
        at_point(&battle, [&obs[0], &obs[1]], [&bel[0], &bel[1]], points);
        points += 1;
        let mut picks = [None, None];
        for s in 0..2 {
            let cs = battle.legal_choices(dex, s);
            if !cs.is_empty() {
                let agent: &mut dyn Agent = if s == 0 { &mut a0 } else { &mut a1 };
                picks[s] = Some(agent.choose(&battle, dex, s, &cs));
            }
        }
        battle.apply_choices(dex, picks).unwrap();
    }
}

/// Play a determinized battle to completion with random agents.
fn playable(dex: &Dex, mut det: Battle) {
    let mut r0 = RandomAgent::new(11);
    let mut r1 = RandomAgent::new(13);
    play_game(dex, &mut det, &mut [&mut r0, &mut r1], 400)
        .expect("determinized battle must play to completion");
}

// ------------------------------------------------- (a) + (b): ground truth

#[test]
fn ground_truth_stays_consistent_and_determinizations_play_out() {
    let dex = load_dex();
    let meta = pool();
    let (ci, cj) = collision_pair(&meta);
    // 26 = the team with an item-less mon (exercises the preview item flag)
    let pairs = [(0usize, 1usize), (ci, cj), (5, 12), (0, 26), (3, 17), (9, 20)];
    for (k, &(ti, tj)) in pairs.iter().enumerate() {
        let mut det_rng = SplitMix64::new(0xD57 + k as u64);
        let mut prev = [usize::MAX, usize::MAX];
        drive(
            &dex,
            &meta,
            &meta.teams[ti].sets,
            &meta.teams[tj].sets,
            "11,22,33,44",
            777 + k as u64,
            150,
            |battle, obs, bel, points| {
                // the true team is never filtered out
                assert!(
                    bel[0].alive().contains(&tj),
                    "pair {ti} vs {tj}: true opp team {tj} filtered at point {points} \
                     (alive {:?})",
                    bel[0].alive()
                );
                assert!(
                    bel[1].alive().contains(&ti),
                    "pair {ti} vs {tj}: true opp team {ti} filtered at point {points} \
                     (alive {:?})",
                    bel[1].alive()
                );
                // candidate count is monotonically non-increasing
                for s in 0..2 {
                    let n = bel[s].candidate_count();
                    assert!(n <= prev[s], "candidate count grew at point {points}");
                    prev[s] = n;
                }
                // the collision keeps both candidates at preview
                if points == 0 && (ti, tj) == (ci, cj) {
                    assert!(bel[0].alive().contains(&ci) && bel[0].alive().contains(&cj));
                    assert_eq!(bel[0].candidate_count(), 2, "alive {:?}", bel[0].alive());
                }
                // (b) determinized battles play out from any point, and the
                // observer's own legal choices are untouched
                if points % 8 == 0 {
                    for s in 0..2 {
                        let det = bel[s].determinize(&dex, battle, obs[s], &mut det_rng);
                        let mut t = battle.clone();
                        let mut d = det.clone();
                        assert_eq!(
                            t.legal_choices(&dex, s),
                            d.legal_choices(&dex, s),
                            "determinization changed side {s}'s own legal choices"
                        );
                        playable(&dex, det);
                    }
                }
            },
        );
    }
}

// ------------------------------------------------------- (c): leak check

#[test]
fn determinized_samples_differ_hidden_agree_public() {
    let dex = load_dex();
    let meta = pool();
    let (ci, cj) = collision_pair(&meta);
    let mut checks = 0usize;
    let mut pp_checks = 0usize;
    drive(
        &dex,
        &meta,
        &meta.teams[0].sets,
        &meta.teams[ci].sets,
        "5,6,7,8",
        4242,
        150,
        |battle, obs, bel, _points| {
            // the true team (ci) is always consistent — determinize with it
            // and check the PP-preservation contract at every point
            let mut r1 = SplitMix64::new(99);
            let d1 = bel[0].determinize_with(&dex, battle, obs[0], Some(ci), &mut r1);
            let truth = &battle.sides[1].roster;
            for (slot, mo) in obs[0].mons().iter().enumerate() {
                for s in d1.sides[1].roster[slot].base_move_slots.iter() {
                    if mo.revealed_moves.contains(&s.id) {
                        let tslot = truth[slot]
                            .base_move_slots
                            .iter()
                            .find(|t| t.id == s.id)
                            .expect("revealed move exists in the true set");
                        assert_eq!(s.pp, tslot.pp, "revealed move lost its PP usage");
                        pp_checks += 1;
                    } else {
                        assert_eq!(s.pp, s.maxpp, "unrevealed move not at full PP");
                    }
                }
            }

            // leak check proper: only while both collision candidates live
            if !(bel[0].alive().contains(&ci) && bel[0].alive().contains(&cj)) {
                return;
            }
            // identical rng streams: identical reseed + pick-identity draws,
            // so roster slots line up 1:1 and only the candidate differs
            let mut r1 = SplitMix64::new(99);
            let mut r2 = SplitMix64::new(99);
            let d1 = bel[0].determinize_with(&dex, battle, obs[0], Some(ci), &mut r1);
            let d2 = bel[0].determinize_with(&dex, battle, obs[0], Some(cj), &mut r2);

            // my own side is untouched, bit for bit
            assert_eq!(
                format!("{:?}", d1.sides[0]),
                format!("{:?}", battle.sides[0]),
                "determinization touched the observer's own side"
            );
            assert_eq!(format!("{:?}", d1.sides[0]), format!("{:?}", d2.sides[0]));

            // public agreement on the opponent side
            for (m1, m2) in d1.sides[1].roster.iter().zip(d2.sides[1].roster.iter()) {
                assert_eq!(m1.species, m2.species);
                assert_eq!(m1.level, m2.level);
                assert_eq!(m1.status, m2.status);
                assert_eq!(m1.boosts, m2.boosts);
                if m1.previously_switched_in > 0 || m1.is_active {
                    // exact HP is public once seen
                    assert_eq!(m1.hp, m2.hp, "appeared mon HP diverged");
                }
            }
            assert_eq!(d1.sides[1].party, d2.sides[1].party);

            // hidden fields actually differ between the two candidates
            let differs = d1.sides[1].roster.iter().zip(d2.sides[1].roster.iter()).any(
                |(m1, m2)| {
                    let mv = |p: &nc2000_engine::state::Pokemon| {
                        let mut v: Vec<u16> =
                            p.base_move_slots.iter().map(|s| s.id.0).collect();
                        v.sort_unstable();
                        v
                    };
                    mv(m1) != mv(m2) || m1.item != m2.item || m1.set_ivs != m2.set_ivs
                },
            );
            assert!(differs, "collision candidates produced identical hidden state");
            checks += 1;
        },
    );
    assert!(checks > 0, "collision matchup never had both candidates alive");
    assert!(pp_checks > 0, "no revealed move was ever PP-checked");
}

// ---------------------------------------------------------- (d): fallback

#[test]
fn fallback_never_panics() {
    let dex = load_dex();
    let meta = pool();

    // d1: level changed → preview-inconsistent from the start
    let mut custom = meta.teams[0].sets.clone();
    custom[3].level -= 1;
    let mut det_rng = SplitMix64::new(31337);
    let mut saw_fallback = false;
    drive(&dex, &meta, &meta.teams[1].sets, &custom, "3,1,4,1", 2024, 120, |battle, obs, bel, points| {
        assert!(bel[0].is_fallback(), "level-modified team must not match the pool");
        saw_fallback = true;
        if points % 6 == 0 {
            let det = bel[0].determinize(&dex, battle, obs[0], &mut det_rng);
            playable(&dex, det);
        }
    });
    assert!(saw_fallback);

    // d2: same species+levels but a foreign move — consistent at preview,
    // degrades to fallback once the move is revealed
    let mut custom = meta.teams[0].sets.clone();
    let snorlax = custom
        .iter_mut()
        .find(|s| s.species == "Snorlax")
        .expect("team 0 runs Snorlax");
    snorlax.moves[0] = "Hyper Beam".to_string(); // max-damage bait: gets used
    let mut det_rng = SplitMix64::new(97);
    let mut went_fallback = false;
    // Snorlax must be picked and must move — retry seeds until the reveal
    for attempt in 0..8u64 {
        drive(
            &dex,
            &meta,
            &meta.teams[1].sets,
            &custom,
            "8,8,8,8",
            555 + attempt,
            150,
            |battle, obs, bel, points| {
                // team 0 is the only preview-consistent candidate (until the reveal)
                assert!(
                    bel[0].is_fallback() || bel[0].alive() == [0],
                    "unexpected candidates {:?}",
                    bel[0].alive()
                );
                went_fallback |= bel[0].is_fallback();
                if points % 6 == 0 {
                    let det = bel[0].determinize(&dex, battle, obs[0], &mut det_rng);
                    playable(&dex, det);
                }
            },
        );
        if went_fallback {
            break;
        }
    }
    assert!(went_fallback, "the foreign move never got revealed — weak test drive");
}
// --------------------------------------------- observation channel coverage

/// Regression net for the observer itself: across a handful of pool games
/// both channels must actually fire — move reveals and item consumption
/// (state-diff channel), and at least one *held*-item reveal, which only
/// the log channel can produce (Leftovers heal / Focus Band attribution).
/// A silent log-format drift would otherwise just weaken the belief without
/// failing any consistency test.
#[test]
fn observation_channels_fire() {
    let dex = load_dex();
    let meta = pool();
    let (ci, _cj) = collision_pair(&meta);
    let mut held_reveals = 0usize; // log channel only
    let mut consumed = 0usize;
    let mut move_reveals = 0usize;
    for (g, (ti, tj)) in
        [(0usize, 1usize), (0, ci), (5, 12), (3, 17), (9, 20), (0, 26)].iter().enumerate()
    {
        let mut last = (0usize, 0usize, 0usize);
        drive(
            &dex,
            &meta,
            &meta.teams[*ti].sets,
            &meta.teams[*tj].sets,
            "5,6,7,8",
            4242 + g as u64,
            200,
            |_battle, obs, _bel, _points| {
                let mut rev = 0;
                let mut held = 0;
                let mut gone = 0;
                for m in obs[0].mons().iter().chain(obs[1].mons().iter()) {
                    rev += m.revealed_moves.len();
                    match m.item.current {
                        Some(Some(_)) => held += 1,
                        Some(None) => gone += 1,
                        None => {}
                    }
                }
                last = (rev, held, gone);
            },
        );
        move_reveals += last.0;
        held_reveals += last.1;
        consumed += last.2;
    }
    assert!(move_reveals > 15, "only {move_reveals} move reveals across 6 games");
    assert!(held_reveals > 0, "log channel (Leftovers/Focus Band) never fired");
    assert!(consumed > 0, "no item consumption observed");
}
