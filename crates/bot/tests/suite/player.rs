use conformance::fixture::{corpus_files, repo_root, Fixture};
use nc2000_bot::player::{action_input, PlayerChannel};
use nc2000_bot::Request;
use nc2000_engine::state::Battle;

#[test]
fn player_requests_cover_the_golden_choice_sequences() {
    let dex = super::common::dex();
    let mut decisions = 0;
    let mut replacements = 0;
    for pool in ["full", "puredata"] {
        for path in corpus_files(&repo_root().join("fixtures/corpus-v1").join(pool)) {
            let fixture = Fixture::load(&path).unwrap();
            let mut b = Battle::from_fixture(&dex, &fixture.seed, &fixture.p1team, &fixture.p2team)
                .unwrap();
            let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
            for choice in &fixture.choices {
                let side = if choice.side == "p1" { 0 } else { 1 };
                let frame = channels[side].frame(&mut b, &dex).unwrap();
                let req = Request::parse(&dex, &frame.request.to_string()).unwrap();
                if !req.team_preview {
                    let mut expected = frame.legal_actions.clone();
                    let mut actual = req.legal_inputs();
                    expected.sort();
                    actual.sort();
                    actual.dedup();
                    assert_eq!(
                        actual,
                        expected,
                        "{} turn {} side {side}",
                        path.display(),
                        b.turn
                    );
                }
                replacements += usize::from(req.force_switch);
                decisions += 1;
                b.choose(&dex, side, &choice.choice).unwrap();
            }
        }
    }
    assert!(decisions > 2000, "decisions={decisions}");
    assert!(replacements > 50, "replacements={replacements}");
}

#[test]
fn hidden_opponent_fields_do_not_change_the_player_frame() {
    let dex = super::common::dex();
    let path = corpus_files(&repo_root().join("fixtures/corpus-v1/full"))[0].clone();
    let fixture = Fixture::load(&path).unwrap();
    let mut a =
        Battle::from_fixture(&dex, &fixture.seed, &fixture.p1team, &fixture.p2team).unwrap();
    let mut b = a.clone();
    b.reseed(81828384);
    for p in &mut b.sides[1].roster {
        p.happiness = 0;
        p.set_ivs = [0; 6];
        p.item = dex.items.id("quickclaw");
        p.stored_stats = [400; 5];
        for m in p.move_slots.iter_mut() {
            m.id = dex.moves.id("splash").unwrap();
        }
    }
    let fa = PlayerChannel::new(0).frame(&mut a, &dex).unwrap();
    let fb = PlayerChannel::new(0).frame(&mut b, &dex).unwrap();
    assert_eq!(fa, fb);
    assert!(!fa.lines.iter().any(|l| l.starts_with("|split|")));
}

#[test]
fn split_log_keeps_only_the_players_own_exact_health() {
    let dex = super::common::dex();
    let fixture =
        Fixture::load(&corpus_files(&repo_root().join("fixtures/corpus-v1/full"))[0]).unwrap();
    let mut b =
        Battle::from_fixture(&dex, &fixture.seed, &fixture.p1team, &fixture.p2team).unwrap();
    b.log = vec![
        "|split|p1".into(),
        "own-exact".into(),
        "own-percent".into(),
        "|split|p2".into(),
        "foe-exact".into(),
        "foe-percent".into(),
    ];
    let mut channel = PlayerChannel::new(0);
    assert_eq!(
        channel.frame(&mut b, &dex).unwrap().lines,
        ["own-exact", "foe-percent"]
    );
    assert!(channel.frame(&mut b, &dex).unwrap().lines.is_empty());
}
