use conformance::{
    fixture::{corpus_files, repo_root, Fixture},
    load_dex,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    prng::Prng,
    replay::{Recorder, Replay},
    state::Battle,
};

fn assert_same(a: &Battle, b: &Battle) {
    assert_eq!(a.state_key128(), b.state_key128());
    assert_eq!(a.prng.seed_str(), b.prng.seed_str());
    assert_eq!(a.log, b.log);
}

#[test]
fn frozen_compact_replay_remains_compatible() {
    let mut dex = load_dex();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/replay/transition-compatible.json"
    )).unwrap();
    let code = fixture["code"].as_str().unwrap();
    let replay = Replay::decode(&dex, code).unwrap();
    assert_eq!(replay.encode(&dex).unwrap(), code);
    let battle = replay.at(&dex, replay.rounds.len()).unwrap();
    assert_eq!(battle.state_key128().to_string(), fixture["state_key128"]);
    assert_eq!(battle.prng.seed_str(), fixture["prng"]);
    assert_eq!(battle.turn as u64, fixture["turn"].as_u64().unwrap());
    dex.replay_fingerprint[0] ^= 1;
    assert_eq!(Replay::decode(&dex, code).err().as_deref(), Some("incompatible replay version"));
}

#[test]
fn numbered_moves_record_the_effective_locked_action() {
    let dex = load_dex();
    let fixture =
        Fixture::load(&corpus_files(&repo_root().join("fixtures/corpus-v1/full"))[0]).unwrap();
    let mut mon = fixture.p1team[0].clone();
    mon.moves = vec!["fly".into(), "rest".into()];
    let teams = [vec![mon.clone()], vec![mon]];
    let mut battle = Battle::from_fixture(&dex, &fixture.seed, &teams[0], &teams[1]).unwrap();
    let mut recorder = Recorder::new(&dex, &battle, teams, &fixture.seed);
    for input in ["team 1", "move 1", "move 2"] {
        for side in [0, 1] {
            recorder.choose(&dex, &mut battle, side, input).unwrap();
        }
    }
    let code = recorder.replay.encode(&dex).unwrap();
    let decoded = Replay::decode(&dex, &code).unwrap();
    assert_same(&battle, &decoded.at(&dex, decoded.rounds.len()).unwrap());
    for pick in decoded.rounds[2].iter().flatten() {
        assert_eq!(pick.to_input(&dex), "move fly");
    }
}

#[test]
fn compact_replay_preserves_every_decision_state() {
    let dex = load_dex();
    let files = corpus_files(&repo_root().join("fixtures/corpus-v1/full"));
    for (i, path) in files.iter().enumerate() {
        let fixture = Fixture::load(path).unwrap();
        let mut teams = [fixture.p1team, fixture.p2team];
        if i % 2 == 0 {
            for team in &mut teams {
                for p in team {
                    p.gender = None;
                }
            }
        }
        let seed = Prng::new(173 * i as u64 + 9).seed_str();
        let mut original = Battle::from_fixture(&dex, &seed, &teams[0], &teams[1]).unwrap();
        let mut recorder = Recorder::new(&dex, &original, teams, &seed);
        let mut rng = Prng::new(i as u64 + 72);
        let mut states = vec![(original.state_key128(), original.prng.seed_str())];
        for round in 0..3010 {
            if original.ended {
                break;
            }
            let picks: Vec<_> = (0..2)
                .filter_map(|side| {
                    let legal = original.legal_choices(&dex, side);
                    (!legal.is_empty())
                        .then(|| (side, legal[rng.sample_index(legal.len())].to_input(&dex)))
                })
                .collect();
            for (side, input) in if round % 2 == 0 {
                picks
            } else {
                picks.into_iter().rev().collect()
            } {
                recorder.choose(&dex, &mut original, side, &input).unwrap();
            }
            states.push((original.state_key128(), original.prng.seed_str()));
        }
        assert!(original.ended);
        let code = recorder.replay.encode(&dex).unwrap();
        assert!(code.len() <= 2090, "{} characters", code.len());
        let decoded = Replay::decode(&dex, &code).unwrap();
        assert_eq!(decoded.encode(&dex).unwrap(), code);
        let mut replayed = decoded.initial(&dex).unwrap();
        for (round, state) in states.iter().enumerate() {
            assert_eq!(
                &(replayed.state_key128(), replayed.prng.seed_str()),
                state,
                "{} round {round}",
                path.display()
            );
            if round < decoded.rounds.len() {
                decoded.apply(&dex, &mut replayed, round).unwrap();
            }
        }
        assert_same(&original, &replayed);
    }
}

#[test]
fn replacements_and_baton_pass_fit_the_1000_turn_envelope() {
    let dex = load_dex();
    let mon = |species: &str, name: &str| PokemonSet {
        name: name.into(),
        species: species.into(),
        moves: vec![
            "batonpass".into(),
            "rest".into(),
            "protect".into(),
            "sleeptalk".into(),
        ],
        item: "leftovers".into(),
        level: 50,
        ability: String::new(),
        evs: None,
        ivs: None,
        happiness: None,
        gender: None,
    };
    let team: Vec<_> = [
        "eevee", "vaporeon", "jolteon", "flareon", "espeon", "umbreon",
    ]
    .iter()
    .enumerate()
    .map(|(i, s)| mon(s, &format!("日本語の名前{i}")))
    .collect();
    let seed = "1,2,3,4";
    let mut b = Battle::from_fixture(&dex, seed, &team, &team).unwrap();
    let mut recorder = Recorder::new(&dex, &b, [team.clone(), team], seed);
    while !b.ended {
        let choices: Vec<_> = (0..2)
            .filter_map(|s| {
                let legal = b.legal_choices(&dex, s);
                let pick = legal
                    .iter()
                    .find(|c| c.to_input(&dex) == "move batonpass")
                    .or_else(|| legal.iter().find(|c| matches!(c, SearchChoice::Switch(_))))
                    .or_else(|| legal.first());
                pick.map(|c| (s, c.to_input(&dex)))
            })
            .collect();
        for (s, c) in choices {
            recorder.choose(&dex, &mut b, s, &c).unwrap();
        }
    }
    assert_eq!(b.turn, 1001);
    assert!(recorder.replay.rounds.len() > 1001);
    let code = recorder.replay.encode(&dex).unwrap();
    assert!(code.len() <= 2090);
    let replay = Replay::decode(&dex, &code).unwrap();
    assert_same(&b, &replay.at(&dex, replay.rounds.len()).unwrap());
}

#[test]
fn only_committed_choices_are_recorded_and_corruption_is_rejected() {
    let dex = load_dex();
    let fixture =
        Fixture::load(&corpus_files(&repo_root().join("fixtures/corpus-v1/full"))[0]).unwrap();
    let teams = [fixture.p1team, fixture.p2team];
    let mut b = Battle::from_fixture(&dex, &fixture.seed, &teams[0], &teams[1]).unwrap();
    let initial = b.clone();
    let mut recorder = Recorder::new(&dex, &b, teams, &fixture.seed);
    for s in [0, 0, 0] {
        let choice = b.legal_choices(&dex, s)[0].to_input(&dex);
        recorder.choose(&dex, &mut b, s, &choice).unwrap();
    }
    assert!(recorder.replay.rounds.is_empty());
    let code = recorder.replay.encode(&dex).unwrap();
    assert_same(
        &initial,
        &Replay::decode(&dex, &code).unwrap().initial(&dex).unwrap(),
    );
    for i in 4..code.len() {
        let mut changed = code.clone().into_bytes();
        changed[i] = if changed[i] == b'A' { b'B' } else { b'A' };
        if let Ok(decoded) = Replay::decode(&dex, std::str::from_utf8(&changed).unwrap()) {
            assert_same(&initial, &decoded.initial(&dex).unwrap());
        }
    }
    let json = std::fs::read_to_string(repo_root().join("data/gen2stadium2.json")).unwrap();
    let other = nc2000_engine::dex::Dex::from_json(&(json + "\n")).unwrap();
    assert!(Replay::decode(&other, &code)
        .err()
        .unwrap()
        .contains("incompatible"));
}
