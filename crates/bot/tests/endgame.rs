use conformance::load_dex;
use nc2000_bot::endgame::{EndgameConfig, EndgameSearch, EndgameSession};
use nc2000_bot::preview::{MetaPool, MetaTeam};
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig, SkuctSearch, SplitMix64};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn mon(moves: &[&str]) -> PokemonSet {
    PokemonSet {
        name: "Pikachu".into(),
        species: "Pikachu".into(),
        item: String::new(),
        ability: String::new(),
        moves: moves.iter().map(|m| (*m).into()).collect(),
        level: 50,
        evs: None,
        ivs: None,
        happiness: None,
        gender: None,
    }
}

fn position(dex: &Dex, own: &[&str], foe: &[&str], counts: [usize; 2]) -> Battle {
    let mut root = Battle::from_fixture(
        dex,
        "1,2,3,4",
        &vec![mon(own); counts[0]],
        &vec![mon(foe); counts[1]],
    )
    .unwrap();
    let preview = [0, 1].map(|side| root.legal_choices(dex, side).first().copied());
    root.apply_choices(dex, preview).unwrap();
    root.turn = 1000;
    for side in 0..2 {
        let id = root.active_id(side).unwrap();
        root.poke_mut(id).hp = 1;
    }
    root
}

fn session() -> EndgameSession {
    EndgameSession::new(EndgameConfig::default(), 0.0, 10_000.0).unwrap()
}

#[test]
fn native_root_corrects_wrong_control_and_preserves_right_control() {
    let dex = load_dex();
    let attack = SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
    let mut changed = 0;
    for moves in [["Seismic Toss", "Splash"], ["Splash", "Seismic Toss"]] {
        let root = position(&dex, &moves, &["Splash"], [1, 1]);
        let mut search = EndgameSearch::new(&root, &dex, 0, RmConfig::default(), 7, session());
        let baseline = search.best().unwrap();
        for _ in 0..1000 {
            search.step(&dex, 1, &mut || 0.0);
            if search.ready(0.0) {
                break;
            }
        }
        assert!(search.policy().unwrap().max_loss <= 0.02);
        assert!(search.best().is_none());
        assert_eq!(search.commit(), Some(attack));
        let stats = search.session.stats.clone();
        assert_eq!(stats.root_overrides, 1);
        assert_eq!(stats.root_changes, usize::from(baseline != attack));
        changed += stats.root_changes;
        for _ in 0..20 {
            assert_eq!(search.best(), Some(attack));
            assert_eq!(search.commit(), Some(attack));
            search.step(&dex, 5, &mut || 0.0);
            assert_eq!(search.session.stats, stats);
        }
    }
    assert_eq!(changed, 1);
}

#[test]
fn protocol_endgame_decisions_ignore_hidden_truth_including_durations() {
    use nc2000_bot::player::PlayerChannel;
    use nc2000_bot::ProtocolAgent;
    use nc2000_engine::state::{Status, DK};
    let dex = load_dex();
    let own = vec![mon(&["Splash", "Seismic Toss"])];
    let foe = vec![mon(&["Splash"])];
    let pool = MetaPool {
        teams: vec![MetaTeam {
            id: "foe".into(),
            sets: foe,
            weight: None,
        }],
    };
    let mut replacements = 0;
    for asleep in [false, true] {
        let mut a = position(&dex, &["Splash", "Seismic Toss"], &["Splash"], [1, 1]);
        a.turn = 999;
        a.log.push("|turn|999".into());
        if asleep {
            let active = a.active_id(1).unwrap();
            a.poke_mut(active).status = Status::Slp;
            a.poke_mut(active).status_state.set_int(DK::Time, 2);
            a.log.push("|-status|p2a: Pikachu|slp".into());
        }
        let mut b = a.clone();
        for p in &mut b.sides[1].roster {
            p.happiness = 0;
            p.set_ivs = [0; 6];
            p.stored_stats = [400; 5];
            p.item = dex.items.id("quickclaw");
            p.status_state.set_int(DK::Time, 7);
            for m in p.move_slots.iter_mut() {
                m.id = dex.moves.id("explosion").unwrap();
            }
            for (_, state) in p.volatiles.iter_mut() {
                state.duration = Some(7);
            }
        }
        b.quick_claw_roll = !a.quick_claw_roll;
        b.reseed(1234567);
        let frames = [a, b].map(|mut truth| PlayerChannel::new(0).frame(&mut truth, &dex).unwrap());
        assert_eq!(frames[0], frames[1]);
        let results = frames.map(|frame| {
            let mut agent = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 73);
            agent.set_own_team(own.clone());
            for line in frame.lines {
                agent.push_line(&dex, &line);
            }
            assert!(agent.on_request(&dex, &frame.request.to_string()).unwrap());
            let mut session = session();
            agent
                .step_with_endgame(&dex, 96, &mut session, &mut || 0.0)
                .unwrap();
            let best = agent.best(&dex);
            let search = agent.search().unwrap();
            (
                best,
                search.visits().to_vec(),
                search.means(),
                session.stats,
            )
        });
        assert_eq!(results[0], results[1]);
        assert!(results[0].3.calls > 0);
        replacements += results[0].3.value_replacements;
    }
    assert!(replacements > 0);
}

#[test]
fn native_agent_commits_and_plays_the_certified_action() {
    use nc2000_bot::{endgame::EndgameAgent, Agent};
    let dex = load_dex();
    let mut root = position(&dex, &["Splash", "Seismic Toss"], &["Splash"], [1, 1]);
    let choices = root.legal_choices(&dex, 0);
    let mut agent =
        EndgameAgent::new(EndgameConfig::default(), RmConfig::default(), 1000.0, 47).unwrap();
    let choice = agent.choose(&root, &dex, 0, &choices);
    assert_eq!(
        choice,
        SearchChoice::Move(dex.moves.id("seismictoss").unwrap())
    );
    let stats = agent.stats.clone();
    assert_eq!(stats.root_overrides, 1);
    assert_eq!(agent.choose(&root, &dex, 0, &choices), choice);
    assert_eq!(agent.stats, stats);
    let policy = agent.root_policy(&root, &dex, 0, &choices);
    assert!((policy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let reply = root.legal_choices(&dex, 1).first().copied();
    root.apply_choices(&dex, [Some(choice), reply]).unwrap();
    assert!(root.outcome().is_some());
}

#[test]
fn all_small_endgame_classes_reach_the_shared_solver() {
    let dex = load_dex();
    for counts in [[1, 1], [1, 2], [2, 1], [2, 2]] {
        let root = position(&dex, &["Splash"], &["Splash"], counts);
        let mut session = session();
        assert!(session.eligible(&root));
        let mut result = None;
        for _ in 0..1000 {
            result = session.conditional_value(&dex, &root, &mut || 0.0);
            if result.is_some() {
                break;
            }
        }
        let value = result.expect("near-cap endgame must close");
        assert!(value.bounds.lo <= 0.5 && value.bounds.hi >= 0.5);
        assert!(value.absolute_error <= 0.01);
        let runs = session.stats.runs;
        assert!(session
            .conditional_value(&dex, &root, &mut || 0.0)
            .is_some());
        assert_eq!(session.stats.runs, runs);
        assert!(session.stats.cache_hits > 0);
        assert_eq!(session.stats.value_replacements, 0);
    }
}

#[test]
fn off_and_ineligible_paths_preserve_search_bit_for_bit() {
    let dex = load_dex();
    for (counts, fraction) in [([1, 1], 0.0), ([3, 3], 0.1)] {
        let mut root = position(&dex, &["Seismic Toss", "Splash"], &["Splash"], counts);
        root.turn = 1;
        let cfg = RmConfig {
            horizon: 2,
            ..Default::default()
        };
        let mut baseline = SkuctSearch::new(&root, &dex, cfg.clone(), 17);
        let mut session = EndgameSession::new(
            EndgameConfig {
                solver_fraction: fraction,
                ..Default::default()
            },
            0.0,
            10_000.0,
        )
        .unwrap();
        let mut candidate = SkuctSearch::new(&root, &dex, cfg.clone(), 17);
        baseline.step(&dex, 48);
        candidate.step_with_leaf(&dex, 48, &mut |sim, rng, rollout| {
            session.leaf(&dex, &cfg, 3, sim, rng, rollout, &mut || 0.0)
        });
        for side in 0..2 {
            assert_eq!(baseline.visits(side), candidate.visits(side));
            assert_eq!(baseline.means(side), candidate.means(side));
            assert_eq!(baseline.best(side), candidate.best(side));
        }
        assert_eq!(session.stats.calls, 0);
        assert_eq!(session.stats.value_replacements, 0);
    }
}

#[test]
fn leaf_values_are_replaced_in_native_and_blind_search() {
    let dex = load_dex();
    let mut root = position(&dex, &["Splash"], &["Splash"], [1, 1]);
    root.turn = 999;
    let cfg = RmConfig::default();
    let mut native_session = session();
    let mut native = SkuctSearch::new(&root, &dex, cfg.clone(), 31);
    native.step_with_leaf(&dex, 12, &mut |sim, rng, rollout| {
        native_session.leaf(&dex, &cfg, 1100, sim, rng, rollout, &mut || 0.0)
    });
    assert!(
        native_session.stats.value_replacements > 0,
        "{:?}",
        native_session.stats
    );
    let obs = Observer::new(&root, 0);
    let pool = MetaPool {
        teams: vec![MetaTeam {
            id: "foe".into(),
            sets: vec![mon(&["Splash"])],
            weight: None,
        }],
    };
    let belief = Belief::new(&dex, &pool, &obs);
    let mut blind_session = session();
    let mut blind = BlindSearch::new(&root, &dex, cfg, 0, 31);
    blind.step_with_endgame(&dex, &belief, &obs, 12, &mut blind_session, &mut || 0.0);
    assert!(blind_session.stats.value_replacements > 0);
    assert_eq!(blind_session.stats.root_overrides, 0);
    assert!(blind_session.stats.max_value_error <= 0.01);
}

#[test]
fn exhausted_solver_share_does_not_keep_probing_cached_states() {
    let dex = load_dex();
    let root = position(&dex, &["Splash"], &["Splash"], [1, 1]);
    let mut session = EndgameSession::new(EndgameConfig::default(), 0.0, 10.0).unwrap();
    let mut elapsed = 0.0;
    session.conditional_value(&dex, &root, &mut || {
        elapsed += 1.0;
        elapsed
    });
    assert!(session.stats.solver_ms >= 1.0);
    let exhausted = session.stats.clone();
    for _ in 0..100 {
        assert!(session.conditional_value(&dex, &root, &mut || 2.0).is_none());
    }
    assert_eq!(session.stats, exhausted);
    session.extend_budget(100.0).unwrap();
    assert!(session.conditional_value(&dex, &root, &mut || 2.0).is_some());
    assert!(session.stats.cache_hits > exhausted.cache_hits);
}

#[test]
fn budget_and_memory_exhaustion_leave_legal_fallbacks() {
    let dex = load_dex();
    let mut root = position(&dex, &["Rest", "Tackle"], &["Rest", "Tackle"], [1, 1]);
    root.turn = 10;
    let cfg = EndgameConfig {
        max_nodes: 2,
        slice_runs: 1,
        decision_runs: 2,
        ..Default::default()
    };
    let mut search = EndgameSearch::new(
        &root,
        &dex,
        0,
        RmConfig::default(),
        37,
        EndgameSession::new(cfg, 0.0, 10.0).unwrap(),
    );
    let mut elapsed = 0.0;
    search.step(&dex, 100, &mut || {
        elapsed += 0.1;
        elapsed
    });
    assert!(search.ready(10.0));
    assert!(search.policy().is_none());
    assert!(root
        .legal_choices(&dex, 0)
        .contains(&search.commit().unwrap()));
    assert!(search.session.stats.runs <= 2);
    assert!(search.session.stats.peak_entries <= 2);
    let runs = search.session.stats.runs;
    search.step(&dex, 10, &mut || 100.0);
    assert_eq!(search.session.stats.runs, runs);
}

#[test]
fn probe_does_not_mutate_battle_or_consume_search_randomness() {
    let dex = load_dex();
    let mut root = position(&dex, &["Splash"], &["Splash"], [1, 1]);
    let key = root.state_key128();
    let mut rng = SplitMix64::new(5);
    let mut session = session();
    for _ in 0..1000 {
        if session
            .conditional_value(&dex, &root, &mut || 0.0)
            .is_some()
        {
            break;
        }
    }
    let value = session.leaf(
        &dex,
        &RmConfig::default(),
        1001,
        &mut root,
        &mut rng,
        true,
        &mut || 0.0,
    );
    assert_eq!(value, 0.5);
    assert_eq!(key, root.state_key128());
    assert_eq!(rng.0, 5);
    assert_eq!(session.stats.value_replacements, 1);
}

#[test]
fn portable_solver_cache_requires_matching_dex_and_exact_state() {
    use nc2000_bot::bounds::{BoundConfig, BoundSolver};
    let dex = load_dex();
    let root = position(&dex, &["Splash"], &["Splash"], [1, 1]);
    let cfg = BoundConfig {
        dead_damage_quotient: false,
        ..Default::default()
    };
    let mut solver = BoundSolver::new(&dex, cfg.clone());
    let value = solver.solve(&root, None).bounds;
    assert!(value.width() <= 0.02);
    let mut restored = BoundSolver::from_cache(&dex, cfg.clone(), solver.into_cache());
    assert_eq!(restored.solve(&root, None).runs, 0);
    let mut changed = root.clone();
    changed.turn -= 1;
    assert!(restored.peek(&changed).is_none());
    changed = root.clone();
    let id = changed.active_id(0).unwrap();
    changed.poke_mut(id).hp += 1;
    assert!(restored.peek(&changed).is_none());
    let mut other_dex = load_dex();
    other_dex.replay_fingerprint[0] ^= 1;
    let changed_rules = BoundSolver::from_cache(&other_dex, cfg, restored.into_cache());
    assert!(changed_rules.peek(&root).is_none());
    assert_eq!(changed_rules.node_count(), 0);
}
