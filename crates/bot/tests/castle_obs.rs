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

    // Castle mode: confusion does NOT exclude iceberry on |turn| without |upkeep|
    // (Castle status berries do not trigger on Update), and DOES exclude iceberry
    // on non-faint |upkeep| while NEVER excluding inert bitterberry (ID 84).
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
    assert!(
        !Observer::is_item_excluded_for_slot(&dex, obs_castle.mons(), 0, dex.items.id("iceberry")),
        "Castle must NOT exclude iceberry on Update/|turn| before EOT upkeep"
    );
    obs_castle.ingest_line("|upkeep", &dex);
    assert!(Observer::is_item_excluded_for_slot(&dex, obs_castle.mons(), 0, dex.items.id("iceberry")));
    assert!(!Observer::is_item_excluded_for_slot(&dex, obs_castle.mons(), 0, dex.items.id("bitterberry")));

    // Castle mode: Burn ALSO excludes iceberry (ID 81) on EOT upkeep and NEVER excludes bitterberry
    let mut obs_brn = Observer::new(&b_castle, 0);
    for line in [
        "|switch|p1a: Snorlax|Snorlax, L50, M|266/266",
        "|switch|p2a: Snorlax|Snorlax, L50, M|266/266",
        "|-status|p2a: Snorlax|brn",
        "|upkeep",
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

#[test]
fn castle_observer_self_confusion_ko_preserves_iceberry_and_upkeep_excludes_all_six() {
    let dex = load_dex();
    let own = vec![
        make_set("Lapras", 50, "Leftovers", &["Ice Beam", "Surf", "Rest", "Sleep Talk"]),
        make_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
    ];
    let foe = vec![
        make_set("Dragonite", 55, "Ice Berry", &["Outrage", "Thunderbolt", "Fire Blast", "Rest"]),
        make_set("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
    ];
    let b = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &own,
        &foe,
        Ruleset::castle_free(),
    )
    .unwrap();
    let lap_hp = b.sides[0].roster[0].base_maxhp;
    let dni_hp = b.sides[1].roster[0].base_maxhp;
    let lax_hp = b.sides[1].roster[1].base_maxhp;
    let mut obs = Observer::new(&b, 0);
    // Dragonite self-confuses after Outrage, then Lapras KOs it with Ice Beam before EOT:
    // Castle never gave Dragonite a living-foe or EOT opportunity to eat Ice Berry.
    for line in [
        format!("|switch|p1a: Lapras|Lapras, L50, M|{lap_hp}/{lap_hp}"),
        format!("|switch|p2a: Dragonite|Dragonite, L55, M|{dni_hp}/{dni_hp}"),
        "|turn|1".to_string(),
        "|move|p2a: Dragonite|Outrage|p1a: Lapras".to_string(),
        format!("|-damage|p1a: Lapras|{}/{lap_hp}", lap_hp - 60),
        "|-start|p2a: Dragonite|confusion|[fatigue]".to_string(),
        "|move|p1a: Lapras|Ice Beam|p2a: Dragonite".to_string(),
        "|-damage|p2a: Dragonite|0 fnt".to_string(),
        "|faint|p2a: Dragonite".to_string(),
        "|upkeep".to_string(),
        format!("|switch|p2a: Snorlax|Snorlax, L50, M|{lax_hp}/{lax_hp}"),
        "|turn|2".to_string(),
    ] {
        obs.ingest_line(&line, &dex);
    }
    assert!(
        !Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("iceberry")),
        "Castle must NOT falsely exclude Ice Berry after Outrage self-confusion + KO"
    );
    let pool = MetaPool {
        teams: vec![MetaTeam { id: "true-dnite".into(), sets: foe.clone(), weight: None }],
    };
    let mut belief = Belief::new(&dex, &pool, &obs);
    belief.sync(&dex, &obs);
    assert_eq!(belief.alive(), &[0], "True Ice Berry Dragonite candidate must survive");

    // Non-faint Castle upkeep excludes PRZCure/PSNCure/Ice/Burnt/Mint/Miracle cleanly
    // while keeping unrelated status berries and inert Bitter Berry eligible.
    let all_status_berries = [
        "przcureberry",
        "psncureberry",
        "iceberry",
        "burntberry",
        "mintberry",
        "bitterberry",
    ];
    for (status_line, expected_excluded) in [
        ("|-status|p2a: Dragonite|par", "przcureberry"),
        ("|-status|p2a: Dragonite|psn", "psncureberry"),
        ("|-status|p2a: Dragonite|tox", "psncureberry"),
        ("|-status|p2a: Dragonite|brn", "iceberry"),
        ("|-status|p2a: Dragonite|frz", "burntberry"),
        ("|-status|p2a: Dragonite|slp", "mintberry"),
    ] {
        let mut o = Observer::new(&b, 0);
        for line in [
            format!("|switch|p1a: Lapras|Lapras, L50, M|{lap_hp}/{lap_hp}"),
            format!("|switch|p2a: Dragonite|Dragonite, L55, M|{dni_hp}/{dni_hp}"),
            "|turn|1".to_string(),
            status_line.to_string(),
            "|upkeep".to_string(),
        ] {
            o.ingest_line(&line, &dex);
        }
        assert!(
            Observer::is_item_excluded_for_slot(&dex, o.mons(), 0, dex.items.id(expected_excluded)),
            "Castle non-faint upkeep must exclude {expected_excluded} for {status_line}"
        );
        assert!(
            Observer::is_item_excluded_for_slot(&dex, o.mons(), 0, dex.items.id("miracleberry")),
            "Castle non-faint upkeep must exclude miracleberry for {status_line}"
        );
        for other in all_status_berries {
            if other != expected_excluded {
                assert!(
                    !Observer::is_item_excluded_for_slot(&dex, o.mons(), 0, dex.items.id(other)),
                    "Castle upkeep for {status_line} must NOT exclude unrelated berry {other}"
                );
            }
        }
    }
}

#[test]
fn castle_observer_and_importer_sketch_preserves_true_team_and_replaces_slot() {
    let dex = load_dex();
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
    let own = vec![
        mk_g("Jolteon", 50, "Leftovers", "M", &["Agility", "Thunderbolt", "Baton Pass", "Substitute"]),
        mk_g("Snorlax", 55, "Mint Berry", "M", &["Body Slam", "Earthquake", "Curse", "Rest"]),
        mk_g("Zapdos", 50, "Miracle Berry", "N", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        mk_g("Starmie", 50, "Gold Berry", "N", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
        mk_g("Raikou", 50, "Magnet", "N", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
        mk_g("Exeggutor", 50, "Pink Bow", "M", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
    ];
    let foe = vec![
        mk_g("Smeargle", 50, "Gold Berry", "M", &["Sketch", "Spore", "Baton Pass", "Recover"]),
        mk_g("Snorlax", 55, "Leftovers", "M", &["Body Slam", "Earthquake", "Curse", "Rest"]),
        mk_g("Zapdos", 50, "Magnet", "N", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        mk_g("Starmie", 50, "Miracle Berry", "N", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
        mk_g("Raikou", 50, "Mint Berry", "N", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
        mk_g("Exeggutor", 50, "Pink Bow", "M", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
    ];

    // 1. Live engine battle + Observer::observe() + pinned Belief
    let mut b = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &own,
        &foe,
        Ruleset::castle_free(),
    )
    .unwrap();
    let jolt_hp = b.sides[0].roster[0].base_maxhp;
    let lax_own_hp = b.sides[0].roster[1].base_maxhp;
    let zap_own_hp = b.sides[0].roster[2].base_maxhp;
    let smear_hp = b.sides[1].roster[0].base_maxhp;
    let lax_foe_hp = b.sides[1].roster[1].base_maxhp;
    let mut obs = Observer::new(&b, 0);
    let mut belief = Belief::pinned_checked(&dex, "true-smeargle", &foe, &obs).unwrap();
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    obs.observe(&b, &dex);
    belief.sync_checked(&dex, &obs).unwrap();

    // Turn 1: Jolteon uses Agility (faster), Smeargle uses Sketch -> copies Agility
    b.choose(&dex, 0, "move 1").unwrap();
    b.choose(&dex, 1, "move 1").unwrap();
    obs.observe(&b, &dex);
    belief.sync_checked(&dex, &obs).unwrap();

    // Turn 2: Smeargle uses its Sketched Agility (slot 1 is now Agility!)
    b.choose(&dex, 0, "move 1").unwrap();
    b.choose(&dex, 1, "move 1").unwrap();
    obs.observe(&b, &dex);
    belief
        .sync_checked(&dex, &obs)
        .expect("Using a Castle Sketched move must NOT eliminate the true submitted team");
    let sketch_id = dex.moves.id("sketch").unwrap();
    let agility_id = dex.moves.id("agility").unwrap();
    assert!(obs.mons()[0].revealed_moves.contains(&sketch_id));
    assert!(!obs.mons()[0].revealed_moves.contains(&agility_id));

    // 2. ProtocolAgent + impossible duplicate-move candidate rejection + determinization + PositionSpec round-trip
    let mut impossible_foe = foe.clone();
    impossible_foe[0] =
        mk_g("Smeargle", 50, "Gold Berry", "M", &["Sketch", "Agility", "Baton Pass", "Recover"]);
    let pool = MetaPool {
        teams: vec![
            MetaTeam { id: "true-smeargle".into(), sets: foe.clone(), weight: None },
            MetaTeam { id: "impossible-dup-agility".into(), sets: impossible_foe, weight: None },
        ],
    };
    let mut agent = ProtocolAgent::new_with_ruleset(
        &dex,
        0,
        pool.clone(),
        RmConfig::default(),
        17,
        Ruleset::castle_free(),
    );
    agent.set_own_team(own.clone());
    let mut proto_lines: Vec<String> = Vec::new();
    for (p, team) in [("p1", &own), ("p2", &foe)] {
        for s in team {
            let g = s.gender.as_deref().unwrap_or("");
            let gpart = if g == "M" || g == "F" { format!(", {g}") } else { String::new() };
            proto_lines.push(format!("|poke|{p}|{}, L{}{gpart}|item", s.species, s.level));
        }
    }
    proto_lines.extend([
        "|start".to_string(),
        format!("|switch|p1a: Jolteon|Jolteon, L50, M|{jolt_hp}/{jolt_hp}"),
        format!("|switch|p2a: Smeargle|Smeargle, L50, M|{smear_hp}/{smear_hp}"),
        "|turn|1".to_string(),
        "|move|p1a: Jolteon|Agility|p1a: Jolteon".to_string(),
        "|move|p2a: Smeargle|Sketch|p1a: Jolteon".to_string(),
        "|-activate|p2a: Smeargle|move: Sketch|Agility".to_string(),
        "|upkeep".to_string(),
        "|turn|2".to_string(),
        format!("|switch|p2a: Snorlax|Snorlax, L55, M|{lax_foe_hp}/{lax_foe_hp}"),
        "|move|p1a: Jolteon|Agility|p1a: Jolteon".to_string(),
        "|upkeep".to_string(),
        "|turn|3".to_string(),
        format!("|switch|p2a: Smeargle|Smeargle, L50, M|{smear_hp}/{smear_hp}"),
        "|move|p1a: Jolteon|Agility|p1a: Jolteon".to_string(),
        "|upkeep".to_string(),
        "|turn|4".to_string(),
        "|move|p1a: Jolteon|Agility|p1a: Jolteon".to_string(),
        "|move|p2a: Smeargle|Agility|p2a: Smeargle".to_string(),
        "|upkeep".to_string(),
        "|turn|5".to_string(),
    ]);
    for l in &proto_lines {
        agent.push_line(&dex, l);
    }
    let req_json = serde_json::json!({
        "active": [{
            "moves": [
                { "move": "Agility", "id": "agility", "pp": 44, "maxpp": 48, "disabled": false },
                { "move": "Thunderbolt", "id": "thunderbolt", "pp": 24, "maxpp": 24, "disabled": false },
                { "move": "Baton Pass", "id": "batonpass", "pp": 64, "maxpp": 64, "disabled": false },
                { "move": "Substitute", "id": "substitute", "pp": 16, "maxpp": 16, "disabled": false }
            ]
        }],
        "side": {
            "name": "P1",
            "id": "p1",
            "pokemon": [
                { "ident": "p1: Jolteon", "details": "Jolteon, L50, M", "condition": format!("{jolt_hp}/{jolt_hp}"), "active": true, "item": "leftovers" },
                { "ident": "p1: Snorlax", "details": "Snorlax, L55, M", "condition": format!("{lax_own_hp}/{lax_own_hp}"), "active": false, "item": "mintberry" },
                { "ident": "p1: Zapdos", "details": "Zapdos, L50", "condition": format!("{zap_own_hp}/{zap_own_hp}"), "active": false, "item": "miracleberry" }
            ]
        }
    })
    .to_string();
    assert!(agent.on_request(&dex, &req_json).unwrap());
    assert_eq!(
        agent.belief().unwrap().alive(),
        &[0],
        "Candidate submitting both Sketch and copied Agility must be rejected"
    );
    let synth = agent.battle().unwrap();
    let sm_slot0 = &synth.sides[1].roster[0].move_slots[0];
    assert_eq!(sm_slot0.id, agility_id, "Synthesized Castle Smeargle must hold Sketched Agility");
    assert_eq!(sm_slot0.maxpp, 48, "Agility maxpp on Castle = 30 * 8 / 5 = 48");
    assert_eq!(sm_slot0.pp, 47, "One post-Sketch use must deduct 1 PP from Agility");

    // Determinization MUST preserve Sketched Agility (47/48) instead of restoring spent Sketch!
    let mut det_rng = SplitMix64::new(33);
    let det = agent
        .belief()
        .unwrap()
        .determinize(&dex, synth, agent.observer().unwrap(), &mut det_rng);
    let det_sm_moves: Vec<_> = det.sides[1].roster[0].move_slots.iter().map(|m| (m.id, m.pp, m.maxpp)).collect();
    assert!(
        det_sm_moves.iter().any(|&(id, pp, maxpp)| id == agility_id && pp == 47 && maxpp == 48),
        "Determinized Castle Smeargle must retain Sketched Agility at 47/48 PP: {det_sm_moves:?}"
    );
    assert!(
        !det_sm_moves.iter().any(|&(id, _, _)| id == sketch_id),
        "Determinized Castle Smeargle must NOT restore spent Sketch: {det_sm_moves:?}"
    );
    // Search root stepping must consider Agility and never spent Sketch for opponent Smeargle
    agent.step(&dex, 128);
    let replies: Vec<String> = agent
        .search()
        .unwrap()
        .root_matrix()
        .into_iter()
        .map(|(_, c, _, _)| c.to_input(&dex))
        .collect();
    assert!(
        replies.iter().any(|r| r == "move agility"),
        "Castle search must offer Sketched Agility in opponent replies: {replies:?}"
    );
    assert!(
        !replies.iter().any(|r| r == "move sketch"),
        "Castle search must NOT offer spent Sketch in opponent replies: {replies:?}"
    );

    // PositionSpec round-trip: reload and feed ANOTHER Agility use -> still not recorded as submitted move!
    let spec = agent.to_position_spec(&dex).unwrap();
    assert_eq!(spec.sides[1].mons[0].sketch_overlay.as_deref(), Some("agility"));
    let mut obs_rt = Observer::from_position(&dex, &spec).unwrap();
    obs_rt.ingest_line("|move|p2a: Smeargle|Agility|p2a: Smeargle", &dex);
    assert!(
        !obs_rt.mons()[0].revealed_moves.contains(&agility_id),
        "Restored Observer from PositionSpec must continue ignoring Sketched Agility uses"
    );
    // PS PositionSpec must fail-closed on sketch_overlay
    let mut ps_spec = spec.clone();
    ps_spec.ruleset = None;
    assert!(ps_spec.check().is_err(), "PS PositionSpec must reject sketch_overlay");

    // 3. Own-side Smeargle PositionSpec reload with `pp` omitted must offer Agility 48/48, NOT Struggle!
    let mut own_smear_team = foe.clone();
    own_smear_team[0] = mk_g("Smeargle", 50, "Gold Berry", "M", &["Sketch"]);
    let mut own_agent = ProtocolAgent::new_with_ruleset(
        &dex,
        1,
        MetaPool { teams: vec![MetaTeam { id: "own".into(), sets: own.clone(), weight: None }] },
        RmConfig::default(),
        19,
        Ruleset::castle_free(),
    );
    own_agent.set_own_team(own_smear_team.clone());
    for l in [
        "|poke|p1|Jolteon, L50, M|item",
        "|poke|p1|Snorlax, L55, M|item",
        "|poke|p1|Zapdos, L50|item",
        "|poke|p1|Starmie, L50|item",
        "|poke|p1|Raikou, L50|item",
        "|poke|p1|Exeggutor, L50, M|item",
        "|poke|p2|Smeargle, L50, M|item",
        "|poke|p2|Snorlax, L55, M|item",
        "|poke|p2|Zapdos, L50|item",
        "|poke|p2|Starmie, L50|item",
        "|poke|p2|Raikou, L50|item",
        "|poke|p2|Exeggutor, L50, M|item",
        "|start",
        &format!("|switch|p1a: Jolteon|Jolteon, L50, M|{jolt_hp}/{jolt_hp}"),
        &format!("|switch|p2a: Smeargle|Smeargle, L50, M|{smear_hp}/{smear_hp}"),
        "|turn|1",
        "|move|p1a: Jolteon|Agility|p1a: Jolteon",
        "|move|p2a: Smeargle|Sketch|p1a: Jolteon",
        "|-activate|p2a: Smeargle|move: Sketch|Agility",
        "|upkeep",
        "|turn|2",
    ] {
        own_agent.push_line(&dex, l);
    }
    let own_req = serde_json::json!({
        "active": [{
            "moves": [
                { "move": "Agility", "id": "agility", "pp": 48, "maxpp": 48, "disabled": false }
            ]
        }],
        "side": {
            "name": "P2",
            "id": "p2",
            "pokemon": [
                { "ident": "p2: Smeargle", "details": "Smeargle, L50, M", "condition": format!("{smear_hp}/{smear_hp}"), "active": true, "item": "goldberry" },
                { "ident": "p2: Snorlax", "details": "Snorlax, L55, M", "condition": format!("{lax_foe_hp}/{lax_foe_hp}"), "active": false, "item": "leftovers" },
                { "ident": "p2: Zapdos", "details": "Zapdos, L50", "condition": format!("{zap_own_hp}/{zap_own_hp}"), "active": false, "item": "magnet" }
            ]
        }
    })
    .to_string();
    assert!(own_agent.on_request(&dex, &own_req).unwrap());
    let mut own_spec = own_agent.to_position_spec(&dex).unwrap();
    own_spec.sides[1].mons[0].pp.clear(); // Omit explicit pp rows!
    let mut reloaded = ProtocolAgent::new_with_ruleset(
        &dex,
        1,
        MetaPool { teams: vec![MetaTeam { id: "own".into(), sets: own, weight: None }] },
        RmConfig::default(),
        23,
        Ruleset::castle_free(),
    );
    reloaded.set_position(&dex, &own_spec).unwrap();
    let re_battle = reloaded.battle().unwrap();
    assert_eq!(re_battle.sides[1].roster[0].move_slots[0].id, agility_id);
    assert_eq!(re_battle.sides[1].roster[0].move_slots[0].pp, 48);
    let own_actions: Vec<String> = reloaded
        .search()
        .unwrap()
        .actions()
        .iter()
        .map(|c| c.to_input(&dex))
        .collect();
    assert!(
        own_actions.iter().any(|a| a == "move agility"),
        "Reloaded own-side Smeargle with omitted pp must offer Agility, not Struggle: {own_actions:?}"
    );
    assert!(!own_actions.iter().any(|a| a == "move struggle"));
}

#[test]
fn belief_fallback_set_filters_impossible_hp_carrier_by_observed_maxhp() {
    let dex = load_dex();
    let own = vec![make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"])];
    // Discriminating off-pool carriers with distinct accompanying moves:
    // - High-weight HP Fire Zapdos (201 HP max) carries Drill Peck
    // - Lower-weight HP Ice Zapdos (215 HP max) carries Whirlwind
    let zap_fire_dp = serde_json::from_value::<PokemonSet>(serde_json::json!({
        "name": "Zapdos", "species": "Zapdos", "level": 55, "item": "Leftovers", "gender": "N",
        "moves": ["Thunderbolt", "Hidden Power Fire", "Drill Peck", "Rest"],
        "ivs": { "hp": 6, "atk": 28, "def": 24, "spa": 30, "spd": 30, "spe": 30 },
        "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
    }))
    .unwrap();
    let zap_ice_ww = serde_json::from_value::<PokemonSet>(serde_json::json!({
        "name": "Zapdos", "species": "Zapdos", "level": 55, "item": "Leftovers", "gender": "N",
        "moves": ["Thunderbolt", "Hidden Power Ice", "Whirlwind", "Rest"],
        "ivs": { "hp": 30, "atk": 30, "def": 26, "spa": 30, "spd": 30, "spe": 30 },
        "evs": { "hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255 }
    }))
    .unwrap();
    // Opponent is off-pool (uses Snorlax Lv50 companion not matching the pool's companion).
    let foe_true = vec![
        zap_ice_ww.clone(),
        make_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
    ];
    let mut b = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &own,
        &foe_true,
        Ruleset::castle_free(),
    )
    .unwrap();
    b.sides[1].roster[0].is_active = true;
    b.sides[1].roster[0].previously_switched_in = 1;
    b.sides[1].roster[0].base_move_slots[0].used = true; // Thunderbolt revealed

    let dummy = make_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]);
    let pool = MetaPool {
        teams: vec![
            MetaTeam { id: "fire-heavy".into(), sets: vec![zap_fire_dp, dummy.clone()], weight: Some(100.0) },
            MetaTeam { id: "ice-light".into(), sets: vec![zap_ice_ww, dummy], weight: Some(1.0) },
        ],
    };
    let mut obs = Observer::new(&b, 0);
    obs.observe(&b, &dex);
    assert_eq!(obs.mons()[0].observed_maxhp, Some(215));

    let mut belief = Belief::new(&dex, &pool, &obs);
    belief.sync(&dex, &obs);
    assert!(belief.is_fallback());
    let mut rng = SplitMix64::new(77);
    let det = belief.determinize(&dex, &b, &obs, &mut rng);
    let ww_id = dex.moves.id("whirlwind").unwrap();
    let dp_id = dex.moves.id("drillpeck").unwrap();
    let det_moves: Vec<_> = det.sides[1].roster[0].base_move_slots.iter().map(|m| m.id).collect();
    assert!(
        det_moves.contains(&ww_id) && !det_moves.contains(&dp_id),
        "At observed_maxhp == 215, impossible HP Fire carrier (Drill Peck) must not win fallback loadout: {det_moves:?}"
    );

    // Novel move combination (Thunderbolt + Detect + Rest revealed): impossible HP Fire
    // carrier must NOT push Drill Peck into Castle co-occurrence scoring (across 32 draws).
    let mut foe_novel = foe_true.clone();
    foe_novel[0].moves = vec![
        "Thunderbolt".into(),
        "Detect".into(),
        "Rest".into(),
        "Whirlwind".into(),
    ];
    let mut b_novel = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &own,
        &foe_novel,
        Ruleset::castle_free(),
    )
    .unwrap();
    b_novel.sides[1].roster[0].is_active = true;
    b_novel.sides[1].roster[0].previously_switched_in = 1;
    b_novel.sides[1].roster[0].base_move_slots[0].used = true; // Thunderbolt
    b_novel.sides[1].roster[0].base_move_slots[1].used = true; // Detect
    b_novel.sides[1].roster[0].base_move_slots[2].used = true; // Rest
    let mut obs_novel = Observer::new(&b_novel, 0);
    obs_novel.observe(&b_novel, &dex);
    let mut belief_novel = Belief::new(&dex, &pool, &obs_novel);
    belief_novel.sync(&dex, &obs_novel);
    let mut novel_rng = SplitMix64::new(79);
    let mut saw_ww = false;
    for _ in 0..32 {
        let det_novel = belief_novel.determinize(&dex, &b_novel, &obs_novel, &mut novel_rng);
        let novel_moves: Vec<_> =
            det_novel.sides[1].roster[0].base_move_slots.iter().map(|m| m.id).collect();
        assert!(
            !novel_moves.contains(&dp_id),
            "Castle novel move combination must exclude impossible HP Fire carrier (Drill Peck): {novel_moves:?}"
        );
        if novel_moves.contains(&ww_id) {
            saw_ww = true;
        }
    }
    assert!(saw_ww, "Castle novel move combination should sample Whirlwind from legal HP Ice carrier");
}

#[test]
fn castle_return_and_frustration_fixed_102_bp_across_compact_and_evaluators() {
    use nc2000_bot::eval::expected_hit_fraction;
    use nc2000_engine::compact::CompactBattle;

    let dex = load_dex();
    let mut ret_h0 = make_set("Snorlax", 50, "Leftovers", &["Return", "Rest"]);
    ret_h0.happiness = Some(0);
    let mut frust_h255 = make_set("Snorlax", 50, "Leftovers", &["Frustration", "Rest"]);
    frust_h255.happiness = Some(255);
    let mut ret_h255 = make_set("Snorlax", 50, "Leftovers", &["Return", "Rest"]);
    ret_h255.happiness = Some(255);

    let b_cs_ret = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        std::slice::from_ref(&ret_h0),
        std::slice::from_ref(&ret_h255),
        Ruleset::castle_free(),
    )
    .unwrap();
    let b_cs_frust = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        std::slice::from_ref(&frust_h255),
        std::slice::from_ref(&ret_h255),
        Ruleset::castle_free(),
    )
    .unwrap();
    let b_ps_ret = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        std::slice::from_ref(&ret_h0),
        std::slice::from_ref(&ret_h255),
        Ruleset::PsStadium2,
    )
    .unwrap();

    let (t_cs_ret, _) = CompactBattle::from_battle(&b_cs_ret, &dex);
    let (t_cs_frust, _) = CompactBattle::from_battle(&b_cs_frust, &dex);
    let (t_ps_ret, _) = CompactBattle::from_battle(&b_ps_ret, &dex);
    assert_eq!(t_cs_ret.move_specs[0][0].base_power, 102, "Castle Return must be 102 BP even at happiness=0");
    assert_eq!(t_cs_frust.move_specs[0][0].base_power, 102, "Castle Frustration must be 102 BP even at happiness=255");
    assert_eq!(t_ps_ret.move_specs[0][0].base_power, 0, "PS Return at happiness=0 stays 0 BP");

    let p0 = nc2000_engine::state::PokeId { side: 0, slot: 0 };
    let p1 = nc2000_engine::state::PokeId { side: 1, slot: 0 };
    let ret_id = dex.moves.id("return").unwrap();
    let frust_id = dex.moves.id("frustration").unwrap();
    let e_cs_ret = expected_hit_fraction(&b_cs_ret, &dex, p0, p1, ret_id);
    let e_cs_frust = expected_hit_fraction(&b_cs_frust, &dex, p0, p1, frust_id);
    let e_ps_ret = expected_hit_fraction(&b_ps_ret, &dex, p0, p1, ret_id);
    assert!(e_cs_ret > 0.25, "Castle eval Return (happiness=0) must compute 102 BP damage, got {e_cs_ret}");
    assert!((e_cs_ret - e_cs_frust).abs() < 1e-9, "Castle Return and Frustration must match identically");
    assert_eq!(e_ps_ret, 0.0, "PS eval Return (happiness=0) must remain 0.0");

    // RosterCompactCache cross-ruleset reuse must honor guard_is_castle
    use nc2000_engine::compact::RosterCompactCache;
    let ps_cache = RosterCompactCache::from_battle(&b_ps_ret, &dex);
    let (t_cs_with_ps_cache, _) =
        CompactBattle::from_battle_with_roster_cache(&b_cs_ret, &dex, Some(&ps_cache));
    assert_eq!(
        t_cs_with_ps_cache.move_specs[0][0].base_power, 102,
        "Castle CompactBattle must not reuse PS 0-BP RosterCompactCache entry"
    );
    let cs_cache = RosterCompactCache::from_battle(&b_cs_ret, &dex);
    let (t_ps_with_cs_cache, _) =
        CompactBattle::from_battle_with_roster_cache(&b_ps_ret, &dex, Some(&cs_cache));
    assert_eq!(
        t_ps_with_cs_cache.move_specs[0][0].base_power, 0,
        "PS CompactBattle must not reuse Castle 102-BP RosterCompactCache entry"
    );
}

#[test]
fn castle_native_blind_and_open_agents_ignore_ps_baked_tables() {
    use nc2000_bot::blind::{BlindAgent, OpenAgent};
    use nc2000_bot::preview::{MatrixEst, PairSolution, PairTable, TableSet, SPACE_VERSION};
    use nc2000_bot::Agent;
    use nc2000_engine::battle::SearchChoice;
    use std::sync::Arc;

    let dex = load_dex();
    let own = sample_six(&dex);
    let foe = sample_six(&dex);
    let pool = Arc::new(MetaPool {
        teams: vec![MetaTeam { id: "t0".into(), sets: own.clone(), weight: None }],
    });
    let mut table_set = TableSet::from_pool(&dex, &pool);
    let actions = table_set.actions().to_vec();
    let sentinel = [6u8, 4, 5];
    let sent_idx = actions.iter().position(|&a| a == sentinel).unwrap();
    let mut p = vec![0.0f64; actions.len()];
    p[sent_idx] = 1.0;
    table_set
        .add_pair(PairTable {
            team_a: "t0".into(),
            team_b: "t0".into(),
            actions,
            space_version: SPACE_VERSION,
            screen: MatrixEst::new(1, 1),
            support: [vec![sent_idx], vec![sent_idx]],
            refine: MatrixEst::new(1, 1),
            sol: PairSolution {
                p_a: p.clone(),
                p_b: p,
                argmax_a: sent_idx,
                argmax_b: sent_idx,
                value: 0.5,
                guarantee_mixed_a: 0.5,
                guarantee_argmax_a: 0.5,
                guarantee_mixed_b: 0.5,
                guarantee_argmax_b: 0.5,
            },
            cfg: BakeCfg {
                screen_games: 1,
                refine_games: 1,
                support: 1,
                skuct_iters: 1,
                advisor_iters: 1,
                advisor_runs: 1,
                eps: 0.1,
                max_turns: 500,
                seed: 1,
            },
            secs: 0.0,
        })
        .unwrap();
    let tables = Arc::new(table_set);

    // PS mode uses the baked table and returns the sentinel policy [6, 5, 4] with prob 1.0
    let mut b_ps = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &own, &foe, Ruleset::PsStadium2)
        .unwrap();
    let legal_ps = b_ps.legal_choices(&dex, 0);
    let mut cfg = RmConfig::default();
    cfg.iterations = 16;
    let mut open_ps = OpenAgent::new(cfg.clone(), Some(tables.clone()), 1);
    let pol_ps = open_ps.root_policy(&b_ps, &dex, 0, &legal_ps);
    let sent_pos = legal_ps.iter().position(|c| *c == SearchChoice::Team(sentinel)).unwrap();
    assert!((pol_ps[sent_pos] - 1.0).abs() < 1e-9, "PS OpenAgent must still serve baked table");

    // Castle mode MUST ignore PS baked tables in both choose() and root_policy()
    let mut b_cs = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &own, &foe, Ruleset::castle_free())
        .unwrap();
    let legal_cs = b_cs.legal_choices(&dex, 0);
    let mut open_cs = OpenAgent::new(cfg.clone(), Some(tables.clone()), 1);
    let mut open_cs_notbl = OpenAgent::new(cfg.clone(), None, 1);
    assert_eq!(
        open_cs.choose(&b_cs, &dex, 0, &legal_cs),
        open_cs_notbl.choose(&b_cs, &dex, 0, &legal_cs),
        "Castle OpenAgent must ignore supplied PS baked tables"
    );
    let mut blind_cs = BlindAgent::new(cfg.clone(), pool.clone(), Some(tables), 1);
    let mut blind_cs_notbl = BlindAgent::new(cfg, pool, None, 1);
    assert_eq!(
        blind_cs.root_policy(&b_cs, &dex, 0, &legal_cs),
        blind_cs_notbl.root_policy(&b_cs, &dex, 0, &legal_cs),
        "Castle BlindAgent::root_policy must ignore supplied PS baked tables"
    );
}

#[test]
fn castle_legality_cache_separates_team_size_and_same_id_mutations() {
    let dex = load_dex();
    let full = sample_six(&dex);
    let sub2 = full[..2].to_vec();
    let mut illegal_same_id = full.clone();
    illegal_same_id[0] =
        make_set("Snorlax", 55, "Leftovers", &["Lovely Kiss", "Body Slam", "Earthquake", "Rest"]);

    // 1. Sub-2-mon team is allowed against a 2-mon synthetic obs, but rejected against a 6-mon obs!
    let b2 = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &sub2, &sub2, Ruleset::castle_free())
        .unwrap();
    let b6 = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &full, &full, Ruleset::castle_free())
        .unwrap();
    let obs2 = Observer::new(&b2, 0);
    let obs6 = Observer::new(&b6, 0);
    let sub_pool = MetaPool {
        teams: vec![MetaTeam { id: "same-id".into(), sets: sub2.clone(), weight: None }],
    };
    assert_eq!(Belief::new(&dex, &sub_pool, &obs2).pool_candidate_count(), 1);
    assert_eq!(Belief::new(&dex, &sub_pool, &obs6).pool_candidate_count(), 0);
    // Repeat after cache warm-up to prove cached entry preserves team-size distinction
    assert_eq!(Belief::new(&dex, &sub_pool, &obs2).pool_candidate_count(), 1);
    assert_eq!(Belief::new(&dex, &sub_pool, &obs6).pool_candidate_count(), 0);

    // 2. Two teams sharing the exact same team id ("same-id") with legal vs NYPC-illegal sets
    let legal_pool = MetaPool {
        teams: vec![MetaTeam { id: "same-id".into(), sets: full, weight: None }],
    };
    let illegal_pool = MetaPool {
        teams: vec![MetaTeam { id: "same-id".into(), sets: illegal_same_id, weight: None }],
    };
    assert_eq!(Belief::new(&dex, &legal_pool, &obs6).pool_candidate_count(), 1);
    assert_eq!(Belief::new(&dex, &illegal_pool, &obs6).pool_candidate_count(), 0);
}
