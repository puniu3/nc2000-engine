//! Product boundary regressions use the independent player frame as oracle.
use conformance::fixture::repo_root;
use nc2000_bot::{
    import::announce_hp,
    player::PlayerChannel,
    player_state::LocalPlayer,
    preview::{load_meta_pool, MetaPool, MetaTeam},
    smmcts::SelRule,
};
use nc2000_bot::{
    play_game, Agent, Belief, BlindAgent, BlindSearch, OpenAgent, ProtocolAgent, RmConfig,
    SplitMix64,
};
use nc2000_engine::{
    battle::PokemonSet,
    state::{Battle, Status, DK},
};
use serde_json::json;
use std::sync::Arc;

fn config() -> RmConfig {
    RmConfig {
        iterations: std::env::var("NC2000_BOUNDARY_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512),
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    }
}

#[test]
fn product_search_ignores_real_hp_counters_moves_and_battle_rng_on_both_sides() {
    let dex = super::common::dex();
    let pool = load_meta_pool(&repo_root().join("data/belief-pool-v3/belief-pool.json"));
    let catalog = load_meta_pool(&repo_root().join("data/team-pool-v2/team-pool.json"));
    for side in 0..2 {
        let own = &catalog.teams[0].sets;
        let foe = &pool.teams[0].sets;
        let teams = if side == 0 { [own, foe] } else { [foe, own] };
        let base = Battle::from_fixture(&dex, "1,2,3,4", teams[0], teams[1]).unwrap();
        let mut agents =
            [0, 1].map(|_| BlindAgent::new(config(), Arc::new(pool.clone()), None, 97));
        let mut roots = [base.clone(), base];
        for (i, b) in roots.iter_mut().enumerate() {
            let cs = b.legal_choices(&dex, side);
            agents[i].choose(b, &dex, side, &cs); // Capture only the submitted own set at preview.
            for player in 0..2 {
                b.choose(
                    &dex,
                    player,
                    if player != side && i == 1 {
                        "team 1,4,5"
                    } else {
                        "team 1,2,3"
                    },
                )
                .unwrap();
            }
            let mine = b.active_id(side).unwrap();
            b.restore_status(&dex, mine, Status::Slp, b.active_id(1 - side));
            b.poke_mut(mine)
                .status_state
                .set_int(DK::Time, if i == 0 { 1 } else { 3 });
            let subject = format!("p{}a: {}", side + 1, b.poke(mine).name.as_str());
            b.add(&["-status", &subject, "slp"]);
            let target = b.active_id(1 - side).unwrap();
            assert_eq!(b.poke(target).maxhp, 166);
            b.poke_mut(target).hp = 164 + i as i32;
            // Unused move identity, hidden current speed and upcoming chance
            // are all absent from the player frame.
            if i == 1 {
                let splash = dex.moves.id("splash").unwrap();
                b.poke_mut(target).base_move_slots[0].id = splash;
                b.poke_mut(target).move_slots[0].id = splash;
                b.poke_mut(target).stored_stats[4] = 1;
                b.poke_mut(target).item = dex.items.id("quickclaw");
            }
            let subject = format!("p{}a: {}", 2 - side, b.poke(target).name.as_str());
            let (secret, shared) = b.get_health(target);
            b.add_split(
                if side == 0 { "p2" } else { "p1" },
                &["-damage", &subject, &secret],
                &["-damage", &subject, &shared],
            );
            b.quick_claw_roll = i == 1;
            b.reseed(100 + i as u64);
            // The other player's unannounced pending input is not knowledge.
            let pending = b.legal_choices(&dex, 1 - side);
            b.choose(&dex, 1 - side, &pending[i % pending.len()].to_input(&dex))
                .unwrap();
        }
        let frames = roots.each_ref().map(|b| {
            PlayerChannel::new(side)
                .frame(&mut b.clone(), &dex)
                .unwrap()
        });
        assert_eq!(frames[0], frames[1]);
        let mut models = Vec::new();
        let mut values = Vec::new();
        let mut picks = Vec::new();
        for (i, b) in roots.iter_mut().enumerate() {
            let (local, obs) = LocalPlayer::new(b, &dex, side).unwrap();
            let mut belief = Belief::new(&dex, &pool, &obs);
            belief.sync(&dex, &obs);
            let position = local
                .position(&dex, &mut belief, &obs, &mut SplitMix64::new(17))
                .unwrap();
            models.push(format!("{:?}", position.model()));
            let mut search = BlindSearch::from_observation(&position, &dex, config(), side, 97);
            search.step(&dex, &belief, &obs, config().iterations);
            values.push((
                search.visits().to_vec(),
                search.means(),
                search.dominated().to_vec(),
                search.best(),
            ));
            let cs = b.legal_choices(&dex, side);
            picks.push(agents[i].choose(b, &dex, side, &cs));
        }
        assert_eq!(models[0], models[1]);
        assert_eq!(values[0], values[1]);
        assert_eq!(picks[0], picks[1]);
    }
}

fn mon(species: &str, item: &str, speed: u16, hp: u16) -> PokemonSet {
    serde_json::from_value(
        json!({"name":species,"species":species,"level":50,"gender":"M","item":item,
        "moves":if species=="Starmie" {vec!["Recover","Surf"]} else {vec!["Body Slam"]},
        "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
        "evs":{"hp":hp,"atk":255,"def":255,"spa":255,"spd":255,"spe":speed}}),
    )
    .unwrap()
}

#[test]
fn protocol_never_excludes_recover_on_one_guessed_item_or_speed() {
    let dex = super::common::dex();
    for (own, candidates) in [
        (
            mon("Starmie", "", 255, 255),
            [
                mon("Snorlax", "Leftovers", 255, 255),
                mon("Snorlax", "Quick Claw", 255, 255),
            ],
        ),
        (
            mon("Starmie", "", 128, 255),
            [
                mon("Starmie", "Leftovers", 0, 255),
                mon("Starmie", "Leftovers", 255, 255),
            ],
        ),
    ] {
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &[own.clone()], &[candidates[0].clone()])
            .unwrap();
        b.choose(&dex, 0, "team 1").unwrap();
        b.choose(&dex, 1, "team 1").unwrap();
        let f = PlayerChannel::new(0).frame(&mut b, &dex).unwrap();
        for reverse in [false, true] {
            let mut pool = MetaPool {
                teams: candidates
                    .iter()
                    .enumerate()
                    .map(|(i, s)| MetaTeam {
                        id: i.to_string(),
                        sets: vec![s.clone()],
                        weight: Some(1.),
                    })
                    .collect(),
            };
            if reverse {
                pool.teams.reverse();
            }
            let mut agent = ProtocolAgent::new(&dex, 0, pool, config(), 97);
            agent.set_own_team(vec![own.clone()]);
            for l in &f.lines {
                agent.push_line(&dex, l);
            }
            agent.on_request(&dex, &f.request.to_string()).unwrap();
            assert_eq!(agent.belief().unwrap().candidate_count(), 2);
            assert_eq!(agent.search().unwrap().dominated(), [false, false]);
        }
    }
}

#[test]
fn every_protocol_candidate_reannounces_the_original_hp_bucket() {
    let dex = super::common::dex();
    let own = mon("Starmie", "", 255, 255);
    let candidates = [
        mon("Snorlax", "Leftovers", 255, 255),
        mon("Snorlax", "Leftovers", 255, 0),
    ];
    let pool = MetaPool {
        teams: candidates
            .iter()
            .enumerate()
            .map(|(i, s)| MetaTeam {
                id: i.to_string(),
                sets: vec![s.clone()],
                weight: Some(1.),
            })
            .collect(),
    };
    for den in [48, 100] {
        for cur in [0, 1, den / 2, den - 1, den] {
            let mut b =
                Battle::from_fixture(&dex, "1,2,3,4", &[own.clone()], &[candidates[0].clone()])
                    .unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            b.add(&["-damage", "p2a: Snorlax", &format!("{cur}/{den}")]);
            let f = PlayerChannel::new(0).frame(&mut b, &dex).unwrap();
            let mut a = ProtocolAgent::new(&dex, 0, pool.clone(), config(), 97);
            a.set_own_team(vec![own.clone()]);
            for l in &f.lines {
                a.push_line(&dex, l);
            }
            a.on_request(&dex, &f.request.to_string()).unwrap();
            let bel = a.belief().unwrap();
            for &pick in bel.alive() {
                let model = bel.determinize_with(
                    &dex,
                    a.battle().unwrap(),
                    a.observer().unwrap(),
                    Some(pick),
                    &mut SplitMix64::new(7),
                );
                let p = model.poke(model.active_id(1).unwrap());
                assert_eq!(
                    announce_hp(p.hp, p.maxhp, den),
                    cur,
                    "{cur}/{den} pick {pick}"
                );
            }
        }
    }
}

#[test]
fn protocol_preserves_public_relaxed_caps_and_unequal_roster_sizes() {
    let dex = super::common::dex();
    let set = |species: &str, level: u8, mov: &str| -> PokemonSet {
        serde_json::from_value(json!({"name":species,"species":species,"level":level,
            "item":"","ability":"","moves":[mov]}))
        .unwrap()
    };
    let teams = [
        vec![set("Mewtwo", 100, "Psychic"); 3],
        vec![set("Magikarp", 1, "Splash")],
    ];
    let mut real = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
    real.preview_level_caps[0] = None;
    let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
    let mut agents = [0, 1].map(|side| {
        let mut a = ProtocolAgent::new(&dex, side, MetaPool { teams: vec![] }, config(), 97);
        a.set_own_team(teams[side].clone());
        a
    });
    for _ in 0..4 {
        if real.outcome().is_some() {
            break;
        }
        let mut choices = [None, None];
        for side in 0..2 {
            let legal = real.legal_choices(&dex, side);
            if legal.is_empty() {
                continue;
            }
            let frame = channels[side].frame(&mut real, &dex).unwrap();
            for line in frame.lines {
                agents[side].push_line(&dex, &line);
            }
            agents[side]
                .on_request(&dex, &frame.request.to_string())
                .unwrap();
            let model = agents[side].battle().unwrap();
            assert_eq!(model.preview_level_caps, real.preview_level_caps);
            assert_eq!(model.sides[1 - side].roster.len(), teams[1 - side].len());
            let party = &model.sides[side].party;
            assert_eq!(
                party
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                party.len()
            );
            agents[side].step(&dex, 8).unwrap();
            choices[side] = Some(
                nc2000_bot::fork::find_choice(&dex, &legal, &agents[side].best(&dex).unwrap())
                    .unwrap(),
            );
        }
        real.apply_choices(&dex, choices).unwrap();
    }
    assert_eq!(real.outcome(), Some(nc2000_engine::battle::Outcome::P1Win));
}

#[test]
fn selecting_self_ko_while_unable_to_move_is_not_a_certain_loss() {
    let dex = super::common::dex();
    let mut own = mon("Golem", "", 255, 255);
    own.moves = vec!["Explosion".into(), "Splash".into()];
    let mut foe = mon("Snorlax", "", 255, 255);
    foe.moves = vec!["Splash".into()];
    let mut real = Battle::from_fixture(&dex, "1,2,3,4", &[own], &[foe]).unwrap();
    real.choose(&dex, 0, "team 1").unwrap();
    real.choose(&dex, 1, "team 1").unwrap();
    let me = real.active_id(0).unwrap();
    let enemy = real.active_id(1).unwrap();
    real.restore_status(&dex, me, Status::Slp, Some(enemy));
    real.poke_mut(me).status_state.set_int(DK::Time, 3);
    real.restore_status(&dex, enemy, Status::Psn, Some(me));
    real.poke_mut(enemy).hp = 1;
    real.add(&["-status", "p1a: Golem", "slp"]);
    real.add(&["-status", "p2a: Snorlax", "psn"]);
    let (secret, shared) = real.get_health(enemy);
    real.add_split(
        "p2",
        &["-damage", "p2a: Snorlax", &secret],
        &["-damage", "p2a: Snorlax", &shared],
    );
    let (local, obs) = LocalPlayer::new(&real, &dex, 0).unwrap();
    let mut belief = Belief::new(&dex, &MetaPool { teams: vec![] }, &obs);
    let position = local
        .position(&dex, &mut belief, &obs, &mut SplitMix64::new(17))
        .unwrap();
    let search = BlindSearch::from_observation(&position, &dex, config(), 0, 97);
    // Independent mechanics oracle: the selected move is never executed;
    // the opponent faints to poison and the user remains alive.
    real.choose(&dex, 0, "move explosion").unwrap();
    real.choose(&dex, 1, "move splash").unwrap();
    assert_eq!(real.outcome(), Some(nc2000_engine::battle::Outcome::P1Win));
    assert!(search.dominated().iter().all(|excluded| !excluded));
}

#[test]
fn open_agent_matches_singleton_pool_blind_agent_bit_for_bit() {
    let dex = super::common::dex();
    let catalog = load_meta_pool(&repo_root().join("data/team-pool-v2/team-pool.json"));
    let p1_team = &catalog.teams[0];
    let p2_team = &catalog.teams[1];
    let singleton_pool = Arc::new(MetaPool {
        teams: vec![p2_team.clone()],
    });
    let mut cfg = config();
    cfg.iterations = 64;

    for adaptive_probe in [None, Some(16)] {
        let mut blind_b =
            Battle::from_fixture(&dex, "1,2,3,4", &p1_team.sets, &p2_team.sets).unwrap();
        let mut open_b = blind_b.clone();

        let mut blind = BlindAgent::new(cfg.clone(), singleton_pool.clone(), None, 4242);
        let mut open = OpenAgent::new(cfg.clone(), None, 4242);
        if let Some(probes) = adaptive_probe {
            blind = blind.with_adaptive_mixture(probes);
            open = open.with_adaptive_mixture(probes);
        }

        for turn in 0..4 {
            let cs_b = blind_b.legal_choices(&dex, 0);
            let cs_o = open_b.legal_choices(&dex, 0);
            assert_eq!(cs_b, cs_o, "turn {turn} choices diverged");
            if cs_b.is_empty() || blind_b.outcome().is_some() {
                break;
            }
            let pick_b = blind.choose(&blind_b, &dex, 0, &cs_b);
            let pick_o = open.choose(&open_b, &dex, 0, &cs_o);
            assert_eq!(pick_b, pick_o, "choice diverged on turn {turn}");
            assert_eq!(
                blind.search_rng_state(),
                open.search_rng_state(),
                "search RNG state diverged on turn {turn}"
            );
            assert_eq!(blind.root_search_stats(), open.root_search_stats());

            let opp_cs = blind_b.legal_choices(&dex, 1);
            let opp_choice = opp_cs.first().copied();
            blind_b.apply_choices(&dex, [Some(pick_b), opp_choice]).unwrap();
            open_b.apply_choices(&dex, [Some(pick_o), opp_choice]).unwrap();
        }
    }
}

#[test]
fn open_agent_has_zero_hidden_runtime_leaks() {
    let dex = super::common::dex();
    let catalog = load_meta_pool(&repo_root().join("data/team-pool-v2/team-pool.json"));
    let mut cfg = config();
    cfg.iterations = 64;

    // 1. Fail-closed when battle.log_enabled is false.
    let mut logless =
        Battle::from_fixture(&dex, "1,2,3,4", &catalog.teams[0].sets, &catalog.teams[1].sets)
            .unwrap();
    logless.set_log_enabled(false);
    let logless_cs = logless.legal_choices(&dex, 0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut agent = OpenAgent::new(cfg.clone(), None, 97);
        let _ = agent.choose(&logless, &dex, 0, &logless_cs);
    }));
    assert!(result.is_err(), "OpenAgent must fail closed when log is disabled");

    // 2. Identical choices & RNG transitions under hidden runtime mutations.
    for side in 0..2 {
        let own = &catalog.teams[0].sets;
        let foe = &catalog.teams[1].sets;
        let teams = if side == 0 { [own, foe] } else { [foe, own] };
        let base = Battle::from_fixture(&dex, "1,2,3,4", teams[0], teams[1]).unwrap();
        let mut agents = [0, 1].map(|_| OpenAgent::new(cfg.clone(), None, 97));
        let mut roots = [base.clone(), base];
        for (i, b) in roots.iter_mut().enumerate() {
            let cs = b.legal_choices(&dex, side);
            agents[i].choose(b, &dex, side, &cs);
            for player in 0..2 {
                b.choose(
                    &dex,
                    player,
                    if player != side && i == 1 {
                        "team 1,4,5"
                    } else {
                        "team 1,2,3"
                    },
                )
                .unwrap();
            }
            let mine = b.active_id(side).unwrap();
            b.restore_status(&dex, mine, Status::Slp, b.active_id(1 - side));
            b.poke_mut(mine)
                .status_state
                .set_int(DK::Time, if i == 0 { 1 } else { 3 });
            let subject = format!("p{}a: {}", side + 1, b.poke(mine).name.as_str());
            b.add(&["-status", &subject, "slp"]);

            let target = b.active_id(1 - side).unwrap();
            b.poke_mut(target).hp = b.poke(target).maxhp - 2 + i as i32;
            b.poke_mut(target).stored_stats[4] = if i == 1 { 1 } else { b.poke(target).stored_stats[4] };
            b.poke_mut(target).move_slots[0].pp = if i == 1 { 0 } else { b.poke(target).move_slots[0].pp };
            let t_sub = format!("p{}a: {}", 2 - side, b.poke(target).name.as_str());
            let (secret, shared) = b.get_health(target);
            b.add_split(
                if side == 0 { "p2" } else { "p1" },
                &["-damage", &t_sub, &secret],
                &["-damage", &t_sub, &shared],
            );
            b.quick_claw_roll = i == 1;
            b.reseed(100 + i as u64);
            let pending = b.legal_choices(&dex, 1 - side);
            b.choose(&dex, 1 - side, &pending[i % pending.len()].to_input(&dex))
                .unwrap();
        }
        let cs0 = roots[0].legal_choices(&dex, side);
        let cs1 = roots[1].legal_choices(&dex, side);
        let p0 = agents[0].choose(&roots[0], &dex, side, &cs0);
        let p1 = agents[1].choose(&roots[1], &dex, side, &cs1);
        assert_eq!(p0, p1);
        assert_eq!(agents[0].search_rng_state(), agents[1].search_rng_state());
    }
}

#[test]
fn open_agent_forced_preview_and_berserk_gene_rollback() {
    let dex = super::common::dex();
    let mut cfg = config();
    cfg.iterations = 32;

    let make_subteam = |lead_item: &str| -> Vec<PokemonSet> {
        vec![
            mon("Snorlax", lead_item, 255, 255),
            mon("Starmie", "Leftovers", 255, 255),
            mon("Zapdos", "Leftovers", 255, 255),
        ]
    };

    let t1 = make_subteam("Berserk Gene");
    let t2 = make_subteam("Berserk Gene");
    let mut b1 = Battle::from_fixture(&dex, "1,2,3,4", &t1, &t2).unwrap();

    // Path A: forced preview via init_at_preview on a 3-mon subteam before Berserk Gene activates.
    let mut oa_init = OpenAgent::new(cfg.clone(), None, 77);
    let mut ob_init = OpenAgent::new(cfg.clone(), None, 88);
    oa_init.init_at_preview(&b1, 0);
    ob_init.init_at_preview(&b1, 1);

    b1.choose(&dex, 0, "team 1,2,3").unwrap();
    b1.choose(&dex, 1, "team 1,2,3").unwrap();
    // Both Berserk Genes were consumed upon Turn-1 switch-in!
    assert_eq!(b1.turn, 1);
    assert!(b1.sides[0].roster[0].item.is_none());
    assert!(b1.sides[1].roster[0].item.is_none());

    // Path B: lazy Turn-1 entry without init_at_preview recovers Turn-0 items from log.
    let mut oa_lazy = OpenAgent::new(cfg.clone(), None, 77);
    let cs = b1.legal_choices(&dex, 0);
    let pick_init = oa_init.choose(&b1, &dex, 0, &cs);
    let pick_lazy = oa_lazy.choose(&b1, &dex, 0, &cs);
    assert_eq!(pick_init, pick_lazy);
    assert_eq!(oa_init.search_rng_state(), oa_lazy.search_rng_state());

    for oa in [&oa_init, &oa_lazy] {
        let pinned = oa.belief().unwrap().pinned_sets().unwrap();
        assert_eq!(pinned.len(), 3);
        assert_eq!(pinned[0].item, "berserkgene");
    }
    let (_, turns) = oa_init.move_telemetry();
    assert_eq!(turns[0], 1, "move telemetry must record Turn-1 active slot");

    // Path C: cross-battle Turn-1 reuse must be rejected by LocalPlayer::is_compatible
    // and cleanly rebuilt even when no init_at_preview is called between battles.
    let res = play_game(&dex, &mut b1, &mut [&mut oa_init, &mut ob_init], 2);
    let _ = res;

    let mut b2 = Battle::from_fixture(
        &dex,
        "9,10,11,12",
        &make_subteam("Leftovers"),
        &[mon("Golem", "", 255, 255), mon("Starmie", "Leftovers", 255, 255), mon("Zapdos", "Leftovers", 255, 255)],
    )
    .unwrap();
    b2.choose(&dex, 0, "team 1,2,3").unwrap();
    b2.choose(&dex, 1, "team 1,2,3").unwrap();
    let cs2 = b2.legal_choices(&dex, 0);
    let mut fresh = OpenAgent::new(cfg, None, oa_lazy.search_rng_state());
    let picked_reused = oa_lazy.choose(&b2, &dex, 0, &cs2);
    let picked_fresh = fresh.choose(&b2, &dex, 0, &cs2);
    assert!(cs2.contains(&picked_reused));
    assert_eq!(picked_reused, picked_fresh);
    assert_eq!(
        oa_lazy.belief().unwrap().pinned_sets().unwrap()[0].species,
        "golem"
    );
}
