use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    fork::{play_out, Arm, ForkSpec, Info, ProtocolSeat, Seat, SCHEMA},
    position::PositionSpec,
    preview::load_meta_pool,
    smmcts::{RmAgent, RmConfig, SelRule},
};
use nc2000_engine::battle::{Outcome, PokemonSet};

fn fork_4296(info: Info) -> ForkSpec {
    let dir = repo_root().join("data/report-4296");
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap();
    let position = PositionSpec::parse(&read("turn-11.json")).unwrap();
    let opponent_team: Vec<PokemonSet> = serde_json::from_str(&read("opponent-team.json")).unwrap();
    ForkSpec {
        exact_replay: None,
        schema: SCHEMA.into(),
        label: "4296 T11".into(),
        info,
        position,
        opponent_team,
        opponent_picks: Vec::new(),
        opponent_position: None,
        arms: ["move earthquake", "switch 2", "move hiddenpower"]
            .iter()
            .map(|&input| Arm { input: input.into(), label: String::new() })
            .collect(),
    }
}

fn cfg(iterations: u32) -> RmConfig {
    RmConfig { iterations, rule: SelRule::Ucb, ..RmConfig::default() }
}

#[test]
fn replay_forks_preserve_true_state_and_rebuild_both_information_sets() {
    use nc2000_engine::{replay::{Recorder, Replay}, state::Battle};
    use nc2000_bot::preview::MetaPool;
    let dex = load_dex();
    let mon = |species: &str| PokemonSet { name: species.into(), species: species.into(),
        moves: ["batonpass", "rest", "sleeptalk", "tackle"].map(String::from).to_vec(),
        item: String::new(), level: 50,
        evs: Some(["hp", "atk", "def", "spa", "spd", "spe"].map(|k| (k.into(), 255)).into()),
        ivs: None, ability: String::new(), happiness: None, gender: Some("M".into()) };
    let team: Vec<_> = ["eevee", "vaporeon", "jolteon", "flareon", "espeon", "umbreon"].map(mon).to_vec();
    let seed = "19,20,21,22";
    let mut battle = Battle::from_fixture(&dex, seed, &team, &team).unwrap();
    let mut recorder = Recorder::new(&dex, &battle, [team.clone(), team], seed);
    for _ in 0..8 {
        let choices: Vec<_> = (0..2).filter_map(|side| battle.legal_choices(&dex,side).first().copied().map(|c| (side,c.to_input(&dex)))).collect();
        for (s,c) in choices { recorder.choose(&dex,&mut battle,s,&c).unwrap(); }
    }
    for open in [true,false] {
        recorder.replay.open = open;
        let code = recorder.replay.encode(&dex).unwrap();
        let replay = Replay::decode(&dex,&code).unwrap();
        let pool = MetaPool { teams: Vec::new() };
        for round in 1..replay.rounds.len() {
            for side in 0..2 {
                let Some(played) = replay.rounds[round][side] else { continue };
                let original = replay.at(&dex,round).unwrap();
                let fork = ForkSpec::from_replay_side(&dex,&code,round,side,&played.to_input(&dex),&pool).unwrap();
                assert_eq!(fork.bot_side(), side);
                for future_seed in [7,99] {
                    let restored = fork.battle(&dex,future_seed).unwrap();
                    assert_eq!(original.state_key128(),restored.state_key128(),"round {round} side {side}");
                    assert!(restored.log.is_empty());
                    assert_eq!(restored.prng.seed_str(),nc2000_engine::prng::Prng::new(future_seed).seed_str());
                }
                fork.bot_agent(&dex,pool.clone(),cfg(1),0).unwrap();
                fork.opponent_agent(&dex,pool.clone(),cfg(1),0).unwrap();
                fork.check(&dex).unwrap();
            }
        }
    }
}

#[test]
fn fork_battle_is_seed_deterministic_and_resolves_arms() {
    let dex = load_dex();
    let fork = fork_4296(Info::Blind);
    let arms = fork.check(&dex).unwrap();
    assert_eq!(arms.len(), 3);
    let a = fork.battle(&dex, 7).unwrap();
    let b = fork.battle(&dex, 7).unwrap();
    assert_eq!(a.state_key128(), b.state_key128());
    assert!(a.log.is_empty() && a.log_enabled);
    assert_eq!(a.needs_choice(), [true, true]);
    let roundtrip = ForkSpec::parse(&serde_json::to_string(&fork).unwrap()).unwrap();
    assert_eq!(roundtrip.battle(&dex, 7).unwrap().state_key128(), a.state_key128());
}

#[test]
fn fork_rejects_what_it_cannot_reconstruct() {
    let dex = load_dex();
    let mut fork = fork_4296(Info::Blind);
    fork.arms.push(Arm { input: "move thunderbolt".into(), label: String::new() });
    assert!(fork.check(&dex).unwrap_err().contains("not legal"));

    let mut fork = fork_4296(Info::Blind);
    fork.opponent_picks = vec!["starmie".into(), "shuckle".into(), "zapdos".into()];
    assert!(fork.check(&dex).unwrap_err().contains("blissey"));

    let mut fork = fork_4296(Info::Blind);
    fork.opponent_team.swap(0, 1);
    assert!(fork.check(&dex).is_err());

    let mut fork = fork_4296(Info::Blind);
    fork.schema = "nc2000-fork-v0".into();
    assert!(ForkSpec::parse(&serde_json::to_string(&fork).unwrap()).is_err());
}

#[test]
fn stated_opponent_picks_replace_the_unseen_ones() {
    let dex = load_dex();
    let mut fork = fork_4296(Info::Blind);
    let opp = fork.opponent_side();
    let blissey = fork.position.sides[opp].mons.iter().position(|m| m.species == "blissey").unwrap();
    fork.position.sides[opp].mons[blissey].appeared = false;
    fork.position.sides[opp].mons[blissey].appear_count = 0;
    assert!(fork.check(&dex).unwrap_err().contains("opponent_picks"));
    fork.opponent_picks = vec!["snorlax".into(), "shuckle".into(), "starmie".into()];
    let b = fork.battle(&dex, 3).unwrap();
    let party: Vec<&str> = b.sides[opp]
        .party
        .iter()
        .map(|&s| dex.species.key(b.sides[opp].roster[s as usize].species))
        .collect();
    assert_eq!(party[0], "shuckle");
    assert!(party.contains(&"snorlax") && party.contains(&"starmie") && !party.contains(&"blissey"));
}

#[test]
fn protocol_seat_plays_every_arm_to_the_end() {
    let dex = load_dex();
    let pool = load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"));
    for info in [Info::Blind, Info::Open] {
        let fork = fork_4296(info);
        let bot = fork.bot_side();
        for seed in 0..3u64 {
            let mut outcomes = Vec::new();
            for (i, _) in fork.arms.iter().enumerate() {
                let mut battle = fork.battle(&dex, seed).unwrap();
                let arm = fork.arm_choices(&dex, &mut battle).unwrap()[i];
                let agent = fork.bot_agent(&dex, pool.clone(), cfg(60), seed).unwrap();
                let mut seats: [Seat; 2] = std::array::from_fn(|side| {
                    if side == bot {
                        Seat::Engine(Box::new(RmAgent::new(cfg(1), 0)))
                    } else {
                        Seat::Engine(Box::new(RmAgent::new(cfg(60), seed)))
                    }
                });
                seats[bot] = Seat::Protocol(ProtocolSeat::new(agent, 60));
                let result = play_out(&mut battle, &dex, bot, arm, &mut seats, 3000).unwrap();
                let Seat::Protocol(seat) = &seats[bot] else { unreachable!() };
                assert_eq!(seat.agent.legality_drift, 0, "{info:?} seed {seed} arm {i}");
                assert_eq!(seat.agent.projections, 0, "{info:?} seed {seed} arm {i}");
                outcomes.push(result.outcome.expect("capped"));
            }
            assert!(outcomes.iter().all(|o| matches!(o, Outcome::P1Win | Outcome::P2Win | Outcome::Tie)));
        }
    }
}
