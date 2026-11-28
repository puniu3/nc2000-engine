use std::collections::BTreeSet;

use conformance::load_dex;
use nc2000_bot::belief::supplemental_prior_count;
use nc2000_bot::blind::resample_hidden_timers;
use nc2000_bot::duel::DuelSpec;
use nc2000_bot::fork::ForkSpec;
use nc2000_bot::import::{announce_hp_mode, impute_hp_mode, ProtocolAgent, ProtocolTracker};
use nc2000_bot::player_state::{HiddenTimerKind, HiddenTimerSpec};
use nc2000_bot::position::{max_pp_for_ruleset, PositionSpec};
use nc2000_bot::preview::{BakeCfg, MetaPool, MetaTeam};
use nc2000_bot::runner::{effective_turn_cap, judge_castle_turn_limit};
use nc2000_bot::teamgen::TeamGen;
use nc2000_bot::{Belief, Observer, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::replay::Recorder;
use nc2000_engine::rules::Ruleset;
use nc2000_engine::state::{ActionKind, Battle, Scalar, Status, DK};

fn make_set(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": species,
        "species": species,
        "level": level,
        "item": item,
        "gender": "M",
        "moves": moves,
        "ivs": { "hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30 },
        "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
    }))
    .unwrap()
}

fn make_set_evs(species: &str, level: u8, item: &str, moves: &[&str], hp_ev: u16, hp_iv: u8) -> PokemonSet {
    let hp_dv = (hp_iv >> 1) & 15;
    let atk_iv = if (hp_dv & 8) != 0 { 30 } else { 28 };
    let def_iv = if (hp_dv & 4) != 0 { 30 } else { 28 };
    let spe_iv = if (hp_dv & 2) != 0 { 30 } else { 28 };
    let spc_iv = if (hp_dv & 1) != 0 { 30 } else { 28 };
    serde_json::from_value(serde_json::json!({
        "name": species,
        "species": species,
        "level": level,
        "item": item,
        "gender": "M",
        "moves": moves,
        "ivs": { "hp": hp_iv, "atk": atk_iv, "def": def_iv, "spa": spc_iv, "spd": spc_iv, "spe": spe_iv },
        "evs": { "hp": hp_ev, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
    }))
    .unwrap()
}

fn sample_six(dex: &Dex) -> Vec<PokemonSet> {
    let _ = dex;
    vec![
        make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
        make_set("Zapdos", 50, "Mint Berry", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        make_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
        make_set("Gengar", 50, "Gold Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
        make_set("Raikou", 50, "Magnet", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
        make_set("Exeggutor", 50, "Pink Bow", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
    ]
}

#[test]
fn observer_castle_iceberry_dual_cure_and_bitterberry_inert() {
    let dex = load_dex();
    let team = vec![make_set("Snorlax", 50, "Leftovers", &["Body Slam", "Rest", "Curse", "Earthquake"])];

    // PS mode: confusion excludes bitterberry, NOT iceberry
    let b_ps = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &team, &team, Ruleset::PsStadium2).unwrap();
    let mut obs_ps = Observer::new(&b_ps, 0);
    assert_eq!(obs_ps.ruleset(), Ruleset::PsStadium2);
    for line in [
        "|switch|p1a: Snorlax|Snorlax, L50, M|100/100",
        "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
        "|-start|p2a: Snorlax|confusion",
        "|turn|2",
    ] {
        obs_ps.ingest_line(line, &dex);
    }
    assert!(Observer::is_item_excluded_for_slot(&dex, obs_ps.mons(), 0, dex.items.id("bitterberry")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_ps.mons(), 0, dex.items.id("iceberry")));

    // Castle mode: confusion excludes iceberry (ID 81) and NEVER excludes bitterberry (ID 84)
    let b_castle = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &team,
        &team,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let mut obs_castle = Observer::new(&b_castle, 0);
    assert!(obs_castle.ruleset().is_castle());
    for line in [
        "|switch|p1a: Snorlax|Snorlax, L50, M|266/266",
        "|switch|p2a: Snorlax|Snorlax, L50, M|266/266",
        "|-start|p2a: Snorlax|confusion",
        "|turn|2",
    ] {
        obs_castle.ingest_line(line, &dex);
    }
    assert!(Observer::is_item_excluded_for_slot(&dex, obs_castle.mons(), 0, dex.items.id("iceberry")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_castle.mons(), 0, dex.items.id("bitterberry")));

    // Castle mode: Burn ALSO excludes iceberry (ID 81) and NEVER excludes bitterberry
    let mut obs_brn = Observer::new(&b_castle, 0);
    for line in [
        "|switch|p1a: Snorlax|Snorlax, L50, M|266/266",
        "|switch|p2a: Snorlax|Snorlax, L50, M|266/266",
        "|-status|p2a: Snorlax|brn",
        "|turn|2",
    ] {
        obs_brn.ingest_line(line, &dex);
    }
    assert!(Observer::is_item_excluded_for_slot(&dex, obs_brn.mons(), 0, dex.items.id("iceberry")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_brn.mons(), 0, dex.items.id("bitterberry")));

    // Castle mode: curing confusion via "Bitter Berry" display disguise maps to iceberry!
    let mut obs_cure = Observer::new(&b_castle, 0);
    for line in [
        "|switch|p1a: Snorlax|Snorlax, L50, M|266/266",
        "|switch|p2a: Snorlax|Snorlax, L50, M|266/266",
        "|-start|p2a: Snorlax|confusion",
        "|-enditem|p2a: Snorlax|Bitter Berry",
        "|-end|p2a: Snorlax|confusion",
    ] {
        obs_cure.ingest_line(line, &dex);
    }
    assert_eq!(obs_cure.mons()[0].item.original, Some(dex.items.id("iceberry")));
    assert_eq!(obs_cure.mons()[0].item.current, Some(None));
}

#[test]
fn observer_castle_observed_maxhp_and_belief_exact_hp_filtering() {
    let dex = load_dex();
    let lax_full = make_set_evs("Snorlax", 50, "Leftovers", &["Body Slam", "Rest", "Curse", "Earthquake"], 255, 30);
    // Castle-legal low-HP Snorlax (0 EVs across all stats, even DVs -> derived HP DV = 0).
    let lax_low: PokemonSet = serde_json::from_value(serde_json::json!({
        "name": "Snorlax",
        "species": "Snorlax",
        "level": 50,
        "item": "Leftovers",
        "gender": "M",
        "moves": ["Body Slam", "Rest", "Curse", "Earthquake"],
        "ivs": { "hp": 0, "atk": 28, "def": 28, "spa": 28, "spd": 28, "spe": 28 },
        "evs": { "hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0 }
    }))
    .unwrap();

    let b_full = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        std::slice::from_ref(&lax_full),
        std::slice::from_ref(&lax_full),
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let maxhp_full = b_full.sides[1].roster[0].base_maxhp;
    let b_low = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        std::slice::from_ref(&lax_low),
        std::slice::from_ref(&lax_low),
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let maxhp_low = b_low.sides[1].roster[0].base_maxhp;
    assert_ne!(maxhp_full, maxhp_low);

    let pool = MetaPool {
        teams: vec![
            MetaTeam { id: "full".into(), sets: vec![lax_full.clone()], weight: None },
            MetaTeam { id: "low".into(), sets: vec![lax_low.clone()], weight: None },
        ],
    };

    let mut battle = b_full.clone();
    battle.sides[1].roster[0].is_active = true;
    battle.sides[1].roster[0].previously_switched_in = 1;
    // Pick an HP value that PS 100-step quantization would perturb:
    // For maxhp = 266, cur = 134 -> announce_hp(134, 266, 100) = 51 -> impute_hp(51, 100, 266) = 135 != 134.
    let target_hp = (1..maxhp_full)
        .find(|&h| {
            let pct = nc2000_bot::import::announce_hp(h, maxhp_full, 100);
            nc2000_bot::import::impute_hp(pct, 100, maxhp_full) != h
        })
        .expect("must have at least one quantized HP mismatch");
    battle.sides[1].roster[0].hp = target_hp;

    let mut obs = Observer::new(&battle, 0);
    assert_eq!(obs.mons()[0].observed_maxhp, None);
    let mut belief = Belief::new(&dex, &pool, &obs);
    // Before appearing, both candidates match preview (species, level, gender, item flag).
    assert_eq!(belief.alive(), &[0, 1]);

    // After observing in Castle mode, exact maxhp is revealed -> low-HP candidate rejected!
    obs.observe(&battle, &dex);
    assert_eq!(obs.mons()[0].observed_maxhp, Some(maxhp_full));
    belief.sync(&dex, &obs);
    assert_eq!(belief.alive(), &[0]);

    // Castle determinization preserves exact HP without 100-step quantization!
    let mut rng = SplitMix64::new(42);
    let det_castle = belief.determinize(&dex, &battle, &obs, &mut rng);
    assert_eq!(det_castle.sides[1].roster[0].hp, target_hp);

    // PS determinization still quantizes through 100-step percentages identically.
    let mut battle_ps = battle.clone();
    battle_ps.ruleset = Ruleset::PsStadium2;
    let mut obs_ps = Observer::new(&battle_ps, 0);
    obs_ps.observe(&battle_ps, &dex);
    assert_eq!(obs_ps.mons()[0].observed_maxhp, None);
    let mut belief_ps = Belief::new(&dex, &pool, &obs_ps);
    belief_ps.sync(&dex, &obs_ps);
    assert_eq!(belief_ps.alive(), &[0, 1]);
    let det_ps = belief_ps.determinize_with(&dex, &battle_ps, &obs_ps, Some(0), &mut rng);
    assert_ne!(det_ps.sides[1].roster[0].hp, target_hp);
}

#[test]
fn observer_castle_mystery_berry_pp_over_maxpp_and_turn_cut_upkeep() {
    let dex = load_dex();
    let team = vec![make_set("Smeargle", 50, "Mystery Berry", &["Sketch", "Spore", "Baton Pass", "Recover"])];
    let mut battle = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &team,
        &team,
        Ruleset::castle_tournament(),
    )
    .unwrap();

    // Simulate Castle Mystery Berry restoring Sketch PP from 0 to 5 (> maxpp 1) without `used` flag
    battle.sides[1].roster[0].base_move_slots[0].maxpp = 1;
    battle.sides[1].roster[0].base_move_slots[0].pp = 5;
    battle.sides[1].roster[0].base_move_slots[0].used = false;
    battle.sides[1].roster[0].is_active = true;

    let mut obs = Observer::new(&battle, 0);
    obs.observe(&battle, &dex);
    let sketch_id = dex.moves.id("sketch").unwrap();
    assert!(
        obs.mons()[0].revealed_moves.contains(&sketch_id),
        "Castle observer must detect revealed move when Mystery Berry boosts pp > maxpp"
    );

    // Turn-Cut safety: mid-turn faint + upkeep must NOT exclude Leftovers or Gold Berry on active opp
    let mut obs_tc = Observer::new(&battle, 0);
    for line in [
        "|switch|p1a: Smeargle|Smeargle, L50, M|161/161",
        "|switch|p2a: Smeargle|Smeargle, L50, M|161/161",
        "|turn|1",
        "|-damage|p2a: Smeargle|50/161",
        "|faint|p1a: Smeargle",
        "|upkeep",
    ] {
        obs_tc.ingest_line(line, &dex);
    }
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_tc.mons(), 0, dex.items.id("leftovers")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_tc.mons(), 0, dex.items.id("goldberry")));

    // Normal Castle upkeep at exact floor(171/2) = 85/171: Leftovers excluded, Gold Berry NOT excluded
    let mut obs_b = Observer::new(&battle, 0);
    for line in [
        "|switch|p1a: Smeargle|Smeargle, L50, M|171/171",
        "|switch|p2a: Smeargle|Smeargle, L50, M|171/171",
        "|turn|1",
        "|-damage|p2a: Smeargle|85/171",
        "|upkeep",
    ] {
        obs_b.ingest_line(line, &dex);
    }
    assert!(Observer::is_item_excluded_for_slot(&dex, obs_b.mons(), 0, dex.items.id("leftovers")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_b.mons(), 0, dex.items.id("goldberry")));

    // Strict below half (84/171 < 85): Gold Berry IS excluded, Focus Band & King's Rock stay unexcluded
    obs_b.ingest_line("|turn|2", &dex);
    obs_b.ingest_line("|-damage|p2a: Smeargle|84/171", &dex);
    obs_b.ingest_line("|upkeep", &dex);
    assert!(Observer::is_item_excluded_for_slot(&dex, obs_b.mons(), 0, dex.items.id("goldberry")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_b.mons(), 0, dex.items.id("focusband")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_b.mons(), 0, dex.items.id("kingsrock")));
}

#[test]
fn blind_search_castle_hidden_timer_ranges() {
    let dex = load_dex();
    let team = vec![make_set("Snorlax", 50, "Leftovers", &["Body Slam", "Rest", "Curse", "Earthquake"])];

    let sample_ranges = |ruleset: Ruleset| {
        let mut base = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &team, &team, ruleset).unwrap();
        let pid = nc2000_engine::state::PokeId { side: 1, slot: 0 };
        base.poke_mut(pid).status = Status::Slp;
        base.poke_mut(pid).last_move = dex.moves.id("bodyslam");
        base.add_volatile(&dex, pid, "confusion", None, nc2000_engine::battle::EffectHandle::None);
        base.add_volatile(&dex, pid, "disable", None, nc2000_engine::battle::EffectHandle::None);
        base.add_volatile(&dex, pid, "bide", None, nc2000_engine::battle::EffectHandle::None);

        let specs = [
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Sleep { slept: 0 } },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Confusion { ticks: 0 } },
            HiddenTimerSpec {
                side: 1,
                slot: 0,
                kind: HiddenTimerKind::Disable { upkeeps: 0, acted_on_start: false },
            },
            HiddenTimerSpec { side: 1, slot: 0, kind: HiddenTimerKind::Bide { upkeeps: 0 } },
        ];
        let mut slp = BTreeSet::new();
        let mut conf = BTreeSet::new();
        let mut dis = BTreeSet::new();
        let mut bide = BTreeSet::new();
        let mut rng = SplitMix64::new(12345);
        for _ in 0..400 {
            let mut sim = base.clone();
            resample_hidden_timers(&dex, &mut sim, &specs, &mut rng);
            let p = sim.poke(pid);
            slp.insert(p.status_state.get_int(DK::Time));
            conf.insert(p.volatile(dex.conds_id("confusion").unwrap()).unwrap().get_int(DK::Time));
            dis.insert(p.volatile(dex.conds_id("disable").unwrap()).unwrap().duration.unwrap());
            bide.insert(p.volatile(dex.conds_id("bide").unwrap()).unwrap().duration.unwrap());
        }
        (slp, conf, dis, bide)
    };

    let (ps_slp, ps_conf, ps_dis, ps_bide) = sample_ranges(Ruleset::PsStadium2);
    assert_eq!(ps_slp, BTreeSet::from([2, 3, 4]));
    assert_eq!(ps_conf, BTreeSet::from([2, 3, 4, 5]));
    assert_eq!(ps_dis, BTreeSet::from([2, 3, 4, 5]));
    assert_eq!(ps_bide, BTreeSet::from([3, 4]));

    let (c_slp, c_conf, c_dis, c_bide) = sample_ranges(Ruleset::castle_tournament());
    assert_eq!(c_slp, BTreeSet::from([2, 3, 4, 5]));
    assert_eq!(c_conf, BTreeSet::from([2, 3, 4, 5]));
    assert_eq!(c_dis, BTreeSet::from([1, 2, 3, 4, 5, 6, 7]));
    assert_eq!(c_bide, BTreeSet::from([2, 3]));
}

#[test]
fn import_castle_den_100_exact_hp_and_ruleset_preservation() {
    let dex = load_dex();

    // Pure HP helper checks: Castle exact mode does NOT misread den==100 as percentage mode
    assert_eq!(impute_hp_mode(37, 100, 100, true), 37);
    assert_eq!(impute_hp_mode(49, 100, 200, true), 49);
    assert_eq!(impute_hp_mode(49, 100, 200, false), 98);
    assert_eq!(announce_hp_mode(37, 100, 100, true), 37);
    assert_eq!(announce_hp_mode(99, 266, 266, true), 99);

    // End-to-end ProtocolAgent with Castle ruleset and a real 100-HP opponent (Lv50 Diglett-like exact den==100)
    let own = sample_six(&dex);
    let mut foe = sample_six(&dex);
    // Diglett Lv50 with HP DV=10 (iv=20), HP EV=255 -> baseHP(10) + hpDV(10) + floor(63/2)(31) + 60 = 111;
    // With HP DV=0 (iv=0), HP EV=255 (effort 63 -> +31): 10 + 0 + 31 + 60 = 101; with HP EV=240 (effort 60 -> +30): 100!
    foe[0] = make_set_evs("Diglett", 50, "Focus Band", &["Earthquake", "Rock Slide", "Slash", "Substitute"], 240, 0);
    let check_b = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &own, &foe, Ruleset::castle_tournament()).unwrap();
    assert_eq!(check_b.sides[1].roster[0].base_maxhp, 100);

    let pool = MetaPool { teams: vec![MetaTeam { id: "foe".into(), sets: foe.clone(), weight: None }] };
    let mut agent = ProtocolAgent::new_with_ruleset(
        &dex,
        0,
        pool,
        RmConfig::default(),
        7,
        Ruleset::castle_tournament(),
    );
    agent.set_own_team(own.clone());
    assert_eq!(agent.ruleset(), Ruleset::castle_tournament());

    let mut lines = Vec::new();
    for (p, team) in [("p1", &own), ("p2", &foe)] {
        for s in team {
            lines.push(format!("|poke|{p}|{}, L{}, M|item", s.species, s.level));
        }
    }
    lines.extend([
        "|teampreview|3".to_string(),
        "|start".to_string(),
        "|switch|p1a: Snorlax|Snorlax, L55, M|292/292".to_string(),
        "|switch|p2a: Diglett|Diglett, L50, M|100/100".to_string(),
        "|turn|1".to_string(),
        "|-damage|p2a: Diglett|37/100".to_string(),
    ]);
    for l in &lines {
        agent.push_line(&dex, l);
    }

    let req_json = serde_json::json!({
        "active": [{
            "moves": [
                { "move": "Body Slam", "id": "bodyslam", "pp": 24, "maxpp": 24, "disabled": false },
                { "move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "disabled": false },
                { "move": "Curse", "id": "curse", "pp": 16, "maxpp": 16, "disabled": false },
                { "move": "Rest", "id": "rest", "pp": 16, "maxpp": 16, "disabled": false }
            ]
        }],
        "side": {
            "name": "P1",
            "id": "p1",
            "pokemon": [
                { "ident": "p1: Snorlax", "details": "Snorlax, L55, M", "condition": "292/292", "active": true, "item": "leftovers" },
                { "ident": "p1: Zapdos", "details": "Zapdos, L50, M", "condition": "197/197", "active": false, "item": "mintberry" },
                { "ident": "p1: Starmie", "details": "Starmie, L50, M", "condition": "167/167", "active": false, "item": "miracleberry" }
            ]
        }
    })
    .to_string();
    assert!(agent.on_request(&dex, &req_json).unwrap());

    let synth = agent.battle().expect("synthesized battle");
    assert_eq!(synth.ruleset, Ruleset::castle_tournament());
    assert_eq!(synth.sides[1].roster[0].base_maxhp, 100);
    assert_eq!(synth.sides[1].roster[0].hp, 37);
    assert_eq!(agent.observer().unwrap().mons()[0].observed_maxhp, Some(100));

    // Castle switch-in faint Turn-Cut (Spikes/switch-in poison KO before moves) omits Residual from queue
    let mut tracker = ProtocolTracker::new_with_ruleset(0, Ruleset::castle_tournament());
    for l in [
        "|poke|p1|Snorlax, L55, M|item",
        "|poke|p1|Zapdos, L50, M|item",
        "|poke|p1|Starmie, L50, M|item",
        "|poke|p2|Snorlax, L55, M|item",
        "|poke|p2|Zapdos, L50, M|item",
        "|poke|p2|Starmie, L50, M|item",
        "|start",
        "|switch|p1a: Snorlax|Snorlax, L55, M|292/292",
        "|switch|p2a: Snorlax|Snorlax, L55, M|292/292",
        "|turn|1",
        "|switch|p1a: Zapdos|Zapdos, L50, M|10/197",
        "|-damage|p1a: Zapdos|0 fnt|[from] Spikes",
        "|faint|p1a: Zapdos",
    ] {
        tracker.push_line(&dex, l);
    }
    let fswitch_req = nc2000_bot::import::Request::parse(
        &dex,
        &serde_json::json!({
            "forceSwitch": [true],
            "side": {
                "name": "P1",
                "id": "p1",
                "pokemon": [
                    { "ident": "p1: Zapdos", "details": "Zapdos, L50, M", "condition": "0 fnt", "active": true, "item": "mintberry" },
                    { "ident": "p1: Snorlax", "details": "Snorlax, L55, M", "condition": "292/292", "active": false, "item": "leftovers" },
                    { "ident": "p1: Starmie", "details": "Starmie, L50, M", "condition": "167/167", "active": false, "item": "miracleberry" }
                ]
            }
        })
        .to_string(),
    )
    .unwrap();
    let three_own = own[..3].to_vec();
    let base_ref = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &three_own,
        &three_own,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let obs = Observer::new(&base_ref, 0);
    let mut rng = SplitMix64::new(9);
    let tc_b = tracker
        .synthesize(&dex, &three_own, &base_ref.sides[1].roster, &obs, &fswitch_req, &mut rng)
        .unwrap();
    assert!(
        tc_b.queue.iter().any(|a| matches!(a.choice, ActionKind::Residual)),
        "Castle switch-in Turn-Cut faint synthesis must preserve ActionKind::Residual to match Battle::faint_messages (turn.rs:344-354)"
    );
    assert!(
        !tc_b.queue.iter().any(|a| matches!(a.choice, ActionKind::Move { .. })),
        "Castle switch-in Turn-Cut faint synthesis must not queue moves"
    );
}

#[test]
fn teamgen_castle_item_swaps_and_dummy_items() {
    let dex = load_dex();
    let ls_text = include_str!("../../../data/learnsets-gen2.json");
    let pool_text = include_str!("../../../data/meta-pool-v0/meta-pool.json");

    let gen_ps = TeamGen::new(&dex, ls_text, pool_text).unwrap();
    let gen_castle = TeamGen::new(&dex, ls_text, pool_text)
        .unwrap()
        .with_ruleset(Ruleset::castle_tournament());
    assert_eq!(gen_ps.ruleset(), Ruleset::PsStadium2);
    assert_eq!(gen_castle.ruleset(), Ruleset::castle_tournament());

    // Team with Dragonite (Outrage -> Dragon damaging move) and a teammate holding Miracle Berry
    let raw = vec![
        serde_json::json!({
            "species": "Dragonite",
            "level": 55,
            "item": "",
            "moves": ["Outrage", "Thunderbolt", "Fire Blast", "Wing Attack"]
        }),
        serde_json::json!({
            "species": "Snorlax",
            "level": 50,
            "item": "Miracle Berry",
            "moves": ["Body Slam", "Earthquake", "Curse", "Rest"]
        }),
    ];

    let ps_items: BTreeSet<String> = gen_ps
        .non_dominated_items(&dex, &raw, 0, true)
        .into_iter()
        .map(|s| toid(&s))
        .collect();
    assert!(ps_items.contains("dragonfang"));
    assert!(!ps_items.contains("dragonscale"));
    assert!(ps_items.contains("bitterberry"));
    assert!(!ps_items.contains("iceberry"));
    assert!(!ps_items.contains("mail"));
    assert!(!ps_items.contains("fastball"));
    assert!(!ps_items.contains("firestone"));

    let castle_items: BTreeSet<String> = gen_castle
        .non_dominated_items(&dex, &raw, 0, true)
        .into_iter()
        .map(|s| toid(&s))
        .collect();
    assert!(castle_items.contains("dragonscale"), "Castle must offer Dragon Scale for Dragon moves");
    assert!(!castle_items.contains("dragonfang"), "Castle must NOT offer inert Dragon Fang");
    assert!(castle_items.contains("iceberry"), "Castle must offer Ice Berry (ID 81) when Miracle Berry is held");
    assert!(!castle_items.contains("bitterberry"), "Castle must NOT offer inert Bitter Berry (ID 84)");
    assert!(castle_items.contains("mail"), "Castle must permit Mail as dummy Thief-bait item");
    assert!(castle_items.contains("fastball"), "Castle must permit Apricorn balls as dummy items");
    assert!(castle_items.contains("firestone"), "Castle must permit stones as dummy items");

    // Non-Dragon move holder must NOT receive Dragon Scale in Castle
    let castle_lax_items: BTreeSet<String> = gen_castle
        .non_dominated_items(&dex, &raw, 1, true)
        .into_iter()
        .map(|s| toid(&s))
        .collect();
    assert!(!castle_lax_items.contains("dragonscale"));
}

#[test]
fn runner_duel_preview_castle_turn_limit_and_probe_12a_judge() {
    let dex = load_dex();

    assert_eq!(effective_turn_cap(Ruleset::PsStadium2, 500), 500);
    assert_eq!(effective_turn_cap(Ruleset::PsStadium2, 1500), 1000);
    assert_eq!(effective_turn_cap(Ruleset::castle_tournament(), 500), 80);
    assert_eq!(effective_turn_cap(Ruleset::castle_free(), 500), 500);

    let spec = DuelSpec::for_ruleset(10, 1, Ruleset::castle_tournament());
    assert_eq!(spec.max_turns, 80);

    let bake = BakeCfg {
        screen_games: 1,
        refine_games: 1,
        support: 1,
        skuct_iters: 1,
        advisor_iters: 1,
        advisor_runs: 1,
        eps: 0.1,
        max_turns: 500,
        seed: 1,
    };
    assert_eq!(bake.effective_max_turns(Ruleset::castle_tournament()), 80);
    assert_eq!(bake.effective_max_turns(Ruleset::PsStadium2), 500);

    // Probe 12A1/12A2/12A3 exact judge verification
    let p1_team = vec![
        make_set("Blissey", 50, "Leftovers", &["Soft-Boiled"]),
        make_set("Alakazam", 50, "Miracle Berry", &["Psychic"]),
    ];
    let p2_team = vec![
        make_set("Snorlax", 50, "Leftovers", &["Body Slam"]),
        make_set("Snorlax", 50, "Mint Berry", &["Body Slam"]),
    ];
    let mut b = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &p1_team,
        &p2_team,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    // 12A1: P1 (91/361 + 126/126 = 1.252) vs P2 (133/266 + 133/266 = 1.000) -> P1Win
    b.sides[0].roster[0].maxhp = 361;
    b.sides[0].roster[0].hp = 91;
    b.sides[0].roster[1].maxhp = 126;
    b.sides[0].roster[1].hp = 126;
    b.sides[1].roster[0].maxhp = 266;
    b.sides[1].roster[0].hp = 133;
    b.sides[1].roster[1].maxhp = 266;
    b.sides[1].roster[1].hp = 133;
    assert_eq!(judge_castle_turn_limit(&b), Outcome::P1Win);

    // 12A2: seat swapped -> P2Win (no seat bias)
    let mut b_swap = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &p2_team,
        &p1_team,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_swap.sides[0].roster[0].maxhp = 266;
    b_swap.sides[0].roster[0].hp = 133;
    b_swap.sides[0].roster[1].maxhp = 266;
    b_swap.sides[0].roster[1].hp = 133;
    b_swap.sides[1].roster[0].maxhp = 361;
    b_swap.sides[1].roster[0].hp = 91;
    b_swap.sides[1].roster[1].maxhp = 126;
    b_swap.sides[1].roster[1].hp = 126;
    assert_eq!(judge_castle_turn_limit(&b_swap), Outcome::P2Win);

    // 12A3: 2 alive (0.505 total) beats 1 alive (1.000 total)
    b.sides[0].roster[0].hp = 1;
    b.sides[0].roster[1].hp = 63; // 1/361 + 63/126 = 0.5028
    b.sides[1].roster[0].hp = 266;
    b.sides[1].roster[1].hp = 0;
    b.sides[1].roster[1].fainted = true;
    assert_eq!(judge_castle_turn_limit(&b), Outcome::P1Win);
}

#[test]
fn import_castle_stall_counter_and_mimic_pp_and_base40_maxpp() {
    let dex = load_dex();
    let own = vec![
        make_set("Clefable", 55, "Leftovers", &["Protect", "Mimic", "Splash", "Rest"]),
        make_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
        make_set("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
    ];
    let foe = vec![
        make_set("Smeargle", 50, "Gold Berry", &["Splash", "Sketch", "Body Slam", "Rest"]),
        make_set("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
        make_set("Zapdos", 50, "Magnet", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
    ];

    let req = nc2000_bot::import::Request::parse(
        &dex,
        &serde_json::json!({
            "active": [{
                "moves": [
                    { "move": "Protect", "id": "protect", "pp": 13, "maxpp": 16, "disabled": false },
                    { "move": "Splash", "id": "splash", "pp": 8, "maxpp": 64, "disabled": false },
                    { "move": "Splash", "id": "splash", "pp": 64, "maxpp": 64, "disabled": false },
                    { "move": "Rest", "id": "rest", "pp": 16, "maxpp": 16, "disabled": false }
                ]
            }],
            "side": {
                "name": "P1",
                "id": "p1",
                "pokemon": [
                    { "ident": "p1: Clefable", "details": "Clefable, L55, M", "condition": "222/222", "active": true, "item": "leftovers" },
                    { "ident": "p1: Snorlax", "details": "Snorlax, L50, M", "condition": "266/266", "active": false, "item": "mintberry" },
                    { "ident": "p1: Zapdos", "details": "Zapdos, L50, M", "condition": "197/197", "active": false, "item": "miracleberry" }
                ]
            }
        })
        .to_string(),
    )
    .unwrap();

    // Feed 3 consecutive Protects on both sides + Mimic copying Splash on p2a (foe)
    // and verify Castle stall counter == Scalar::Int(3) vs PS float 0.25,
    // plus Castle Mimic PP == 8 (5 + 3 PP Ups) and base-40 maxpp == 64 (vs PS 5 and 61).
    let run_synth = |ruleset: Ruleset, mimic_move: &str| {
        let mut tracker = ProtocolTracker::new_with_ruleset(0, ruleset);
        for l in [
            "|poke|p1|Clefable, L55, M|item",
            "|poke|p1|Snorlax, L50, M|item",
            "|poke|p1|Zapdos, L50, M|item",
            "|poke|p2|Smeargle, L50, M|item",
            "|poke|p2|Snorlax, L50, M|item",
            "|poke|p2|Zapdos, L50, M|item",
            "|start",
            "|switch|p1a: Clefable|Clefable, L55, M|222/222",
            "|switch|p2a: Smeargle|Smeargle, L50, M|161/161",
            "|turn|1",
            "|move|p1a: Clefable|Protect|p1a: Clefable",
            "|-singleturn|p1a: Clefable|Protect",
            "|move|p2a: Smeargle|Mimic|p1a: Clefable",
            &format!("|-activate|p2a: Smeargle|move: Mimic|{mimic_move}"),
            "|turn|2",
            "|move|p1a: Clefable|Protect|p1a: Clefable",
            "|-singleturn|p1a: Clefable|Protect",
            "|turn|3",
            "|move|p1a: Clefable|Protect|p1a: Clefable",
            "|-singleturn|p1a: Clefable|Protect",
            "|turn|4",
        ] {
            tracker.push_line(&dex, l);
        }
        let foe_mimic = vec![
            make_set("Smeargle", 50, "Gold Berry", &["Mimic", "Sketch", "Body Slam", "Rest"]),
            foe[1].clone(),
            foe[2].clone(),
        ];
        let base_ref =
            Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &own, &foe_mimic, ruleset).unwrap();
        let obs = Observer::new(&base_ref, 0);
        let mut rng = SplitMix64::new(5);
        tracker
            .synthesize(&dex, &own, &base_ref.sides[1].roster, &obs, &req, &mut rng)
            .unwrap()
    };

    let b_castle = run_synth(Ruleset::castle_tournament(), "Splash");
    let stall_id = dex.conds_id("stall").unwrap();
    let c_stall = b_castle.sides[0].roster[0].volatile(stall_id).expect("Castle stall volatile");
    assert_eq!(
        c_stall.get(DK::Counter),
        Some(Scalar::Int(3)),
        "Castle stall counter must be integer streak 1..=8"
    );
    let c_mimic_slot = &b_castle.sides[1].roster[0].move_slots[0];
    assert_eq!(c_mimic_slot.id, dex.moves.id("splash").unwrap());
    assert_eq!(c_mimic_slot.maxpp, 64, "Castle base-40 move maxpp must be 64");
    assert_eq!(c_mimic_slot.pp, 8, "Castle Mimic PP must be 5 + 3 PP Ups = 8");

    // Sketch has no_pp_boosts: Castle Mimic PP must clamp to maxpp = 1 (not 8)
    let b_castle_sketch = run_synth(Ruleset::castle_tournament(), "Sketch");
    let c_sketch_slot = &b_castle_sketch.sides[1].roster[0].move_slots[0];
    assert_eq!(c_sketch_slot.id, dex.moves.id("sketch").unwrap());
    assert_eq!(c_sketch_slot.maxpp, 1);
    assert_eq!(c_sketch_slot.pp, 1);

    // PS mode preserves halving float counter (127.0 -> 63.5 -> 31.75) and Mimic PP = 5 / maxpp = 61
    let b_ps = run_synth(Ruleset::PsStadium2, "Splash");
    let ps_stall = b_ps.sides[0].roster[0].volatile(stall_id).expect("PS stall volatile");
    assert_eq!(ps_stall.get(DK::Counter), Some(Scalar::Float(31.75)));
    let ps_mimic_slot = &b_ps.sides[1].roster[0].move_slots[0];
    assert_eq!(ps_mimic_slot.maxpp, 61);
    assert_eq!(ps_mimic_slot.pp, 5);
}

#[test]
fn belief_castle_legality_filtering_and_warning_tolerance() {
    let dex = load_dex();
    let pool: MetaPool =
        serde_json::from_str(include_str!("../../../data/belief-pool-v3/belief-pool.json"))
            .unwrap();
    assert_eq!(pool.teams.len(), 527);

    let team = sample_six(&dex);
    let b_ps = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &team, &team, Ruleset::PsStadium2)
        .unwrap();
    let obs_ps = Observer::new(&b_ps, 0);
    let belief_ps = Belief::new(&dex, &pool, &obs_ps);
    assert_eq!(
        belief_ps.pool_candidate_count(),
        527,
        "PS belief pool must stay 100% unchanged (527 teams)"
    );

    let b_castle = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &team,
        &team,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let obs_castle = Observer::new(&b_castle, 0);
    let belief_castle = Belief::new(&dex, &pool, &obs_castle);
    assert_eq!(
        belief_castle.pool_candidate_count(),
        527 - 19,
        "Castle belief pool must filter exact 19/527 illegal teams (508 kept)"
    );

    // Embedded supplemental.json: PS keeps all 43, Castle rejects exact 17/43 (26 kept)
    assert_eq!(supplemental_prior_count(&dex, Ruleset::PsStadium2), 43);
    assert_eq!(supplemental_prior_count(&dex, Ruleset::castle_tournament()), 43 - 17);

    // Warning-only Castle team (inert Bitter Berry + Dragon Fang, no errors) must NOT be rejected
    // by Belief::new or Belief::pinned_checked!
    let mk_g = |species: &str, level: u8, item: &str, gender: &str, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "name": species,
            "species": species,
            "level": level,
            "item": item,
            "gender": gender,
            "moves": moves,
            "ivs": { "hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30 },
            "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
        }))
        .unwrap()
    };
    let warn_team = vec![
        mk_g("Snorlax", 50, "Bitter Berry", "M", &["Body Slam", "Earthquake", "Curse", "Rest"]),
        mk_g("Dragonite", 55, "Dragon Fang", "M", &["Dragon Breath", "Thunderbolt", "Ice Beam", "Rest"]),
        mk_g("Zapdos", 50, "Mint Berry", "N", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        mk_g("Starmie", 50, "Miracle Berry", "N", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
        mk_g("Raikou", 50, "Magnet", "N", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
        mk_g("Exeggutor", 50, "Leftovers", "M", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
    ];
    let b_warn = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &team,
        &warn_team,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let obs_warn = Observer::new(&b_warn, 0);
    let warn_pool = MetaPool {
        teams: vec![MetaTeam { id: "warn-only".into(), sets: warn_team.clone(), weight: None }],
    };
    let belief_warn = Belief::new(&dex, &warn_pool, &obs_warn);
    assert_eq!(
        belief_warn.pool_candidate_count(),
        1,
        "Castle belief pool must NOT reject warning-only teams"
    );
    assert!(
        Belief::pinned_checked(&dex, "warn-only", &warn_team, &obs_warn).is_ok(),
        "Belief::pinned_checked must accept warning-only Castle teams"
    );

    // NYPC Snorlax Lovely Kiss MUST be rejected by Belief::pinned_checked on Castle
    let mut nypc_team = sample_six(&dex);
    nypc_team[0] =
        make_set("Snorlax", 55, "Leftovers", &["Lovely Kiss", "Body Slam", "Earthquake", "Rest"]);
    let b_nypc = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &team,
        &nypc_team,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let obs_nypc = Observer::new(&b_nypc, 0);
    assert!(
        Belief::pinned_checked(&dex, "nypc", &nypc_team, &obs_nypc).is_err(),
        "Belief::pinned_checked must reject NYPC Snorlax Lovely Kiss on Castle"
    );
}

#[test]
fn teamgen_castle_canonize_learnsets_and_jp_events() {
    let dex = load_dex();
    let ls_text = include_str!("../../../data/learnsets-gen2.json");
    let pool_text = include_str!("../../../data/meta-pool-v0/meta-pool.json");
    let gen_castle = TeamGen::new(&dex, ls_text, pool_text)
        .unwrap()
        .with_ruleset(Ruleset::castle_tournament());
    let gen_ps = TeamGen::new(&dex, ls_text, pool_text).unwrap();

    // Castle non_dominated_moves excludes NYPC Snorlax Lovely Kiss and allows JP-event Donphan Encore
    let castle_lax_moves: BTreeSet<String> = gen_castle
        .non_dominated_moves(&dex, "snorlax", 55, &[])
        .into_iter()
        .map(|m| toid(&m))
        .collect();
    assert!(!castle_lax_moves.contains("lovelykiss"), "Castle teamgen must exclude NYPC Lovely Kiss");

    let castle_donphan_moves: BTreeSet<String> = gen_castle
        .non_dominated_moves(&dex, "donphan", 50, &[])
        .into_iter()
        .map(|m| toid(&m))
        .collect();
    assert!(
        castle_donphan_moves.contains("encore"),
        "Castle teamgen must explore Castle-legal JP-event Donphan Encore"
    );
    let ps_donphan_moves: BTreeSet<String> = gen_ps
        .non_dominated_moves(&dex, "donphan", 50, &[])
        .into_iter()
        .map(|m| toid(&m))
        .collect();
    assert!(!ps_donphan_moves.contains("encore"), "PS teamgen must NOT emit Donphan Encore");

    // TeamGen::canonize accepts warning-only Castle team (inert Bitter Berry) + JP-event Donphan Encore
    let raw_castle_legal = vec![
        serde_json::json!({"species": "Donphan", "level": 50, "item": "Bitter Berry", "moves": ["Encore", "Earthquake", "Rest", "Sleep Talk"]}),
        serde_json::json!({"species": "Snorlax", "level": 55, "item": "Leftovers", "moves": ["Body Slam", "Earthquake", "Curse", "Rest"]}),
        serde_json::json!({"species": "Zapdos", "level": 50, "item": "Mint Berry", "moves": ["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]}),
        serde_json::json!({"species": "Starmie", "level": 50, "item": "Miracle Berry", "moves": ["Surf", "Psychic", "Recover", "Rapid Spin"]}),
        serde_json::json!({"species": "Raikou", "level": 50, "item": "Magnet", "moves": ["Thunderbolt", "Roar", "Rest", "Sleep Talk"]}),
        serde_json::json!({"species": "Exeggutor", "level": 50, "item": "Gold Berry", "moves": ["Psychic", "Giga Drain", "Stun Spore", "Explosion"]}),
    ];
    assert!(
        gen_castle.canonize(&dex, &raw_castle_legal).is_some(),
        "Castle TeamGen::canonize must accept warning-only + JP-event teams"
    );
    assert!(
        gen_ps.canonize(&dex, &raw_castle_legal).is_none(),
        "PS TeamGen::canonize must reject JP-event Donphan Encore"
    );
}

#[test]
fn position_spec_and_replay_origin_preserve_castle_ruleset() {
    let dex = load_dex();

    // Base-40 PP move (Splash): 64 on Castle vs 61 on PS
    let splash_id = dex.moves.id("splash").unwrap();
    assert_eq!(max_pp_for_ruleset(&dex, splash_id, Ruleset::castle_tournament()), 64);
    assert_eq!(max_pp_for_ruleset(&dex, splash_id, Ruleset::PsStadium2), 61);

    // PositionSpec with ruleset: Some("castle-tournament") and Lv50 100-HP Diglett opponent
    let own = sample_six(&dex);
    // Diglett Lv50 with even DVs (derived HP DV = 0) and effort 60 (EV 240) across all stats -> exact maxhp = 100!
    let diglett: PokemonSet = serde_json::from_value(serde_json::json!({
        "name": "Diglett",
        "species": "Diglett",
        "level": 50,
        "item": "Focus Band",
        "gender": "M",
        "moves": ["Earthquake", "Rock Slide", "Slash", "Substitute"],
        "ivs": { "hp": 0, "atk": 28, "def": 28, "spa": 28, "spd": 28, "spe": 28 },
        "evs": { "hp": 240, "atk": 240, "def": 240, "spa": 240, "spd": 240, "spe": 240 }
    }))
    .unwrap();
    let mut foe = sample_six(&dex);
    foe[0] = diglett;

    let spec_json = serde_json::json!({
        "schema": "nc2000-position-v1",
        "ruleset": "castle-tournament",
        "side": 0,
        "turn": 3,
        "own_sets": own,
        "sides": [
            {
                "active": 0,
                "party": [0, 1, 2],
                "mons": [
                    { "species": "Snorlax", "level": 55, "gender": "M", "name": "Snorlax", "item_flag": true, "appeared": true, "appear_count": 1, "active": true, "hp_num": 292, "hp_den": 292, "hp_exact": 292 },
                    { "species": "Zapdos", "level": 50, "gender": "", "name": "Zapdos", "item_flag": true, "appeared": false, "hp_num": 197, "hp_den": 197, "hp_exact": 197 },
                    { "species": "Starmie", "level": 50, "gender": "", "name": "Starmie", "item_flag": true, "appeared": false, "hp_num": 167, "hp_den": 167, "hp_exact": 167 },
                    { "species": "Gengar", "level": 50, "gender": "M", "item_flag": true },
                    { "species": "Raikou", "level": 50, "gender": "", "item_flag": true },
                    { "species": "Exeggutor", "level": 50, "gender": "M", "item_flag": true }
                ]
            },
            {
                "active": 0,
                "party": [0],
                "mons": [
                    { "species": "Diglett", "level": 50, "gender": "M", "name": "Diglett", "item_flag": true, "appeared": true, "appear_count": 1, "active": true, "hp_num": 37, "hp_den": 100 },
                    { "species": "Zapdos", "level": 50, "gender": "", "item_flag": true, "appeared": false, "hp_num": 100, "hp_den": 100 },
                    { "species": "Starmie", "level": 50, "gender": "", "item_flag": true, "appeared": false, "hp_num": 100, "hp_den": 100 },
                    { "species": "Gengar", "level": 50, "gender": "M", "item_flag": true, "appeared": false, "hp_num": 100, "hp_den": 100 },
                    { "species": "Raikou", "level": 50, "gender": "", "item_flag": true, "appeared": false, "hp_num": 100, "hp_den": 100 },
                    { "species": "Exeggutor", "level": 50, "gender": "M", "item_flag": true, "appeared": false, "hp_num": 100, "hp_den": 100 }
                ]
            }
        ]
    })
    .to_string();
    let spec = PositionSpec::parse(&spec_json).unwrap();
    assert_eq!(spec.resolved_ruleset().unwrap(), Ruleset::castle_tournament());

    let obs = Observer::from_position(&dex, &spec).unwrap();
    assert_eq!(obs.ruleset(), Ruleset::castle_tournament());
    // Appeared Diglett at hp_den==100 gets exact observed_maxhp=Some(100); unappeared bench stays None!
    assert_eq!(obs.mons()[0].observed_maxhp, Some(100));
    assert_eq!(obs.mons()[1].observed_maxhp, None);

    let pool = MetaPool { teams: vec![MetaTeam { id: "foe".into(), sets: foe.clone(), weight: None }] };
    let mut agent = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 7);
    agent.set_position(&dex, &spec).unwrap();
    assert_eq!(agent.ruleset(), Ruleset::castle_tournament());
    assert_eq!(agent.battle().unwrap().ruleset, Ruleset::castle_tournament());
    assert_eq!(agent.battle().unwrap().sides[1].roster[0].hp, 37);
    let roundtrip = agent.to_position_spec(&dex).unwrap();
    assert_eq!(roundtrip.ruleset.as_deref(), Some("castle-tournament"));

    // ReplayOrigin::agent & ForkSpec on a Castle NC2- replay must preserve Castle ruleset!
    let mut b_castle = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &own,
        &foe,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let mut rec = Recorder::new(&dex, &b_castle, [own.clone(), foe.clone()], "1,2,3,4");
    rec.replay.open = false;
    rec.replay.bot_side = 0;
    let mut alt_input = String::new();
    let mut fork_round = 0usize;
    for r in 0..4 {
        if b_castle.outcome().is_some() {
            break;
        }
        let needs = b_castle.needs_choice();
        let mut step_picks = Vec::new();
        for side in 0..2 {
            if needs[side] {
                let legal = b_castle.legal_choices(&dex, side);
                if side == 0 && b_castle.turn > 0 && alt_input.is_empty() {
                    fork_round = r;
                    alt_input = legal.last().unwrap().to_input(&dex);
                }
                step_picks.push((side, legal[0].to_input(&dex)));
            }
        }
        for (side, c) in &step_picks {
            rec.choose(&dex, &mut b_castle, *side, c).unwrap();
        }
    }
    let code = rec.replay.encode(&dex).unwrap();
    let fork = ForkSpec::from_replay(&dex, &code, fork_round, &alt_input, &pool).unwrap();
    assert_eq!(fork.position.resolved_ruleset().unwrap(), Ruleset::castle_tournament());
    let fork_agent = fork.bot_agent(&dex, pool, RmConfig::default(), 11).unwrap();
    assert_eq!(fork_agent.ruleset(), Ruleset::castle_tournament());
    assert_eq!(fork_agent.battle().unwrap().ruleset, Ruleset::castle_tournament());
    let fork_battle = fork.battle(&dex, 99).unwrap();
    assert_eq!(fork_battle.ruleset, Ruleset::castle_tournament());
}

#[test]
fn belief_determinize_respects_observed_maxhp_and_hp_evidence() {
    let dex = load_dex();
    let own = vec![make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"])];
    // Zapdos Lv55 with HP Ice (215 HP) vs HP Fire (201 HP)
    let zap_ice = serde_json::from_value::<PokemonSet>(serde_json::json!({
        "name": "Zapdos", "species": "Zapdos", "level": 55, "item": "Leftovers", "gender": "N",
        "moves": ["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"],
        "ivs": { "hp": 30, "atk": 30, "def": 26, "spa": 30, "spd": 30, "spe": 30 },
        "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
    }))
    .unwrap();
    let zap_fire = serde_json::from_value::<PokemonSet>(serde_json::json!({
        "name": "Zapdos", "species": "Zapdos", "level": 55, "item": "Leftovers", "gender": "N",
        "moves": ["Thunderbolt", "Hidden Power Fire", "Rest", "Sleep Talk"],
        "ivs": { "hp": 6, "atk": 28, "def": 24, "spa": 30, "spd": 30, "spe": 30 },
        "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
    }))
    .unwrap();
    let zap_ice_205 = serde_json::from_value::<PokemonSet>(serde_json::json!({
        "name": "Zapdos", "species": "Zapdos", "level": 55, "item": "Leftovers", "gender": "N",
        "moves": ["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"],
        "ivs": { "hp": 30, "atk": 30, "def": 26, "spa": 30, "spd": 30, "spe": 30 },
        "evs": { "hp": 180, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
    }))
    .unwrap();

    // True opponent is Zapdos Lv55 at 205 HP with HP Ice (lower HP EV -> §1 Astra C3!).
    // Pool only contains HP Fire Zapdos (201 HP), forcing fallback + hp_evidence::project.
    let mut true_b = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &own,
        std::slice::from_ref(&zap_ice_205),
        Ruleset::castle_tournament(),
    )
    .unwrap();
    assert_eq!(true_b.sides[1].roster[0].base_maxhp, 205);
    true_b.sides[1].roster[0].is_active = true;
    true_b.sides[1].roster[0].previously_switched_in = 1;
    true_b.sides[1].roster[0].base_move_slots[1].used = true;

    let pool = MetaPool {
        teams: vec![
            MetaTeam { id: "fire".into(), sets: vec![zap_fire], weight: None },
            MetaTeam { id: "ice215".into(), sets: vec![zap_ice], weight: None },
        ],
    };
    let mut obs = Observer::new(&true_b, 0);
    obs.observe(&true_b, &dex);
    assert_eq!(obs.mons()[0].observed_maxhp, Some(205));

    let mut belief = Belief::new(&dex, &pool, &obs);
    belief.sync(&dex, &obs);
    let mut rng = SplitMix64::new(99);
    let det = belief.determinize(&dex, &true_b, &obs, &mut rng);
    assert_eq!(
        det.sides[1].roster[0].base_maxhp, 205,
        "Castle fallback determinization must honor observed_maxhp == 205"
    );
    let hp_type_name = dex.type_name(det.sides[1].roster[0].base_hp_type);
    assert_ne!(
        hp_type_name, "Fire",
        "HP Fire (max 201 HP) must never be determinized at observed_maxhp == 205"
    );
}

#[test]
fn castle_player_channel_exact_hp_with_protocol_open_and_blind_agents() {
    use nc2000_bot::blind::{BlindAgent, OpenAgent};
    use nc2000_bot::player::PlayerChannel;
    use nc2000_bot::Agent;
    use nc2000_engine::battle::SearchChoice;

    let dex = load_dex();
    let own = sample_six(&dex);
    let foe = sample_six(&dex);
    let mut b = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &own,
        &foe,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let mut ch = PlayerChannel::new(0);
    let pool = MetaPool {
        teams: vec![MetaTeam {
            id: "foe".into(),
            sets: foe.clone(),
            weight: None,
        }],
    };
    let mut proto = ProtocolAgent::new_with_ruleset(
        &dex,
        0,
        pool.clone(),
        RmConfig::default(),
        13,
        Ruleset::castle_tournament(),
    );
    proto.set_own_team(own.clone());

    // TeamPreview -> pick 3
    let f0 = ch.frame(&mut b, &dex).unwrap();
    for l in &f0.lines {
        assert!(!l.contains("HP Percentage Mod"), "Castle log must omit HP Percentage Mod: {l}");
        proto.push_line(&dex, l);
    }
    proto.on_request(&dex, &f0.request.to_string()).unwrap();

    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();

    // Turn 1: PlayerChannel must deliver exact opponent maxhp (292/292, NEVER 100/100!)
    let f1 = ch.frame(&mut b, &dex).unwrap();
    assert!(
        f1.lines.iter().any(|l| l.contains("|switch|p2a:") && l.contains("|292/292")),
        "Castle PlayerChannel must emit exact opponent HP 292/292: {:?}",
        f1.lines
    );
    assert!(
        !f1.lines.iter().any(|l| l.contains("|100/100")),
        "Castle PlayerChannel must NOT emit /100 percentages: {:?}",
        f1.lines
    );
    for l in &f1.lines {
        proto.push_line(&dex, l);
    }
    proto.on_request(&dex, &f1.request.to_string()).unwrap();
    let synth = proto.battle().expect("Castle ProtocolAgent synthesized battle");
    assert_eq!(synth.sides[1].roster[0].base_maxhp, 292);
    assert_eq!(synth.sides[1].roster[0].hp, 292);

    // Step 1 turn so both sides take damage and verify exact damaged HP via PlayerChannel + OpenAgent + BlindAgent
    b.choose(&dex, 0, "move 1").unwrap();
    b.choose(&dex, 1, "move 1").unwrap();
    let f2 = ch.frame(&mut b, &dex).unwrap();
    for l in &f2.lines {
        assert!(!l.ends_with("/100"), "Castle damage line must use exact maxhp, not /100: {l}");
        proto.push_line(&dex, l);
    }
    proto.on_request(&dex, &f2.request.to_string()).unwrap();
    let synth2 = proto.battle().expect("Castle ProtocolAgent turn 2 battle");
    assert_eq!(synth2.sides[1].roster[0].hp, b.sides[1].roster[0].hp);

    // OpenAgent & BlindAgent must choose legally without panicking on observed_maxhp
    use std::sync::Arc;
    let mut cfg = RmConfig::default();
    cfg.iterations = 16;
    let mut open_agent = OpenAgent::new(cfg.clone(), None, 42);
    let mut blind_agent = BlindAgent::new(cfg, Arc::new(pool), None, 42);
    let legal = b.legal_choices(&dex, 0);
    let oc: SearchChoice = open_agent.choose(&b, &dex, 0, &legal);
    let bc: SearchChoice = blind_agent.choose(&b, &dex, 0, &legal);
    assert!(legal.contains(&oc), "OpenAgent returned illegal choice {oc:?}");
    assert!(legal.contains(&bc), "BlindAgent returned illegal choice {bc:?}");
}
