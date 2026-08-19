//! Mechanical premises behind the player's 正着 claims for battles 4069/4070,
//! checked against THIS engine (not against general Pokemon knowledge).
//! Scratch verification harness for tmp/verdicts-4069-4070/.

use nc2000_engine::battle::{EffectHandle, PokemonSet, RV};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, PokeId, Status};

fn dex() -> Dex {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/gen2stadium2.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    Dex::from_json(&json).expect("dex JSON must parse")
}

fn mk(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": species, "species": species, "item": item, "ability": "No Ability",
        "moves": moves, "level": level,
        "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
        "happiness": 255
    }))
    .unwrap()
}

fn start(seed: &str, p1: &[PokemonSet], p2: &[PokemonSet]) -> (Dex, Battle) {
    let d = dex();
    let mut b = Battle::from_fixture(&d, seed, p1, p2).unwrap();
    b.choose(&d, 0, "team 1,2,3").unwrap();
    b.choose(&d, 1, "team 1,2,3").unwrap();
    (d, b)
}

fn act(b: &Battle, side: usize) -> PokeId {
    b.active_id(side).unwrap()
}

fn has_vol(b: &Battle, d: &Dex, id: PokeId, key: &str) -> bool {
    let c = d.conds_id(key).unwrap();
    b.poke(id).has_volatile(c)
}

// --------------------------------------------------------------------- Q1

/// Mean Look sets `trapped`; the victim cannot choose a switch.
#[test]
fn q1_meanlook_sets_trapped_and_blocks_switch() {
    let p1 = vec![
        mk("Umbreon", 50, "Leftovers", &["Mean Look", "Toxic", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse", "Milk Drink"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move meanlook").unwrap();
    b.choose(&d, 1, "move curse").unwrap();

    let victim = act(&b, 1);
    assert!(has_vol(&b, &d, victim, "trapped"), "victim has the trapped volatile");
    assert!(has_vol(&b, &d, act(&b, 0), "trapper"), "trapper has the trapper volatile");
    assert!(b.poke(victim).trapped, "victim.trapped flag set at end of turn");
    let err = b.choose(&d, 1, "switch 2");
    assert!(err.is_err(), "trapped victim must not be allowed to switch: {err:?}");
    assert!(format!("{err:?}").contains("trapped"), "{err:?}");
}

/// Trapper switches out -> the linked volatile is torn down, victim freed.
#[test]
fn q1_trapper_switching_out_frees_the_victim() {
    let p1 = vec![
        mk("Umbreon", 50, "Leftovers", &["Mean Look", "Toxic", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse", "Milk Drink"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move meanlook").unwrap();
    b.choose(&d, 1, "move curse").unwrap();
    let victim = act(&b, 1);
    assert!(b.poke(victim).trapped);

    b.choose(&d, 0, "switch 2").unwrap(); // trapper leaves
    b.choose(&d, 1, "move curse").unwrap();
    assert!(!has_vol(&b, &d, victim, "trapped"), "trapped volatile gone once trapper left");
    assert!(!b.poke(victim).trapped, "victim free after trapper switched out");
    assert!(b.choose(&d, 1, "switch 2").is_ok(), "victim may switch again");
}

/// The trapped victim can still USE Whirlwind, the drag lands on a Ghost
/// trapper (ignoreImmunity), and the drag frees the victim.
#[test]
fn q1_trapped_mon_can_whirlwind_the_ghost_trapper_out_and_is_freed() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Mean Look", "Perish Song", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move meanlook").unwrap();
    b.choose(&d, 1, "move thunderbolt").unwrap();
    let victim = act(&b, 1);
    assert!(b.poke(victim).trapped, "Zapdos trapped");
    let ghost_slot = act(&b, 0).slot;

    let before = b.log.len();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move whirlwind").unwrap(); // legal while trapped
    let tail: Vec<&String> = b.log[before..].iter().collect();
    assert!(
        tail.iter().any(|l| l.starts_with("|drag|")),
        "whirlwind must drag the Ghost trapper out: {tail:?}"
    );
    assert_ne!(act(&b, 0).slot, ghost_slot, "a different p1 mon is now active");
    assert!(!has_vol(&b, &d, victim, "trapped"), "trapped volatile removed by the drag");
    assert!(!b.poke(victim).trapped, "victim free after the trapper was dragged out");
}

/// Trapper faints -> victim freed (faint_messages calls clear_volatile).
#[test]
fn q1_trapper_fainting_frees_the_victim() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Mean Look", "Perish Song", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Alakazam", 50, "Leftovers", &["Psychic", "Thunder Punch"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move meanlook").unwrap();
    b.choose(&d, 1, "move psychic").unwrap();
    let victim = act(&b, 1);
    assert!(b.poke(victim).trapped, "Alakazam trapped");
    let trapper = act(&b, 0);
    // KO the trapper outright
    b.poke_mut(trapper).hp = 1;
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move psychic").unwrap();
    assert!(b.poke(trapper).fainted, "trapper fainted: {:?}", b.log);
    assert!(
        !has_vol(&b, &d, victim, "trapped"),
        "trapped volatile removed the instant the trapper faints"
    );
    // NOTE: `Pokemon::trapped` is a CACHE recomputed only in end_turn
    // (turn.rs:450). While the forced-replacement request is pending it is
    // still stale-true; no Move request (and hence no switch validation) can
    // happen in that window.
    assert!(b.poke(victim).trapped, "the cached flag is still stale here");
    b.choose(&d, 0, "switch 2").unwrap(); // replacement -> end_turn runs
    assert!(!b.poke(victim).trapped, "victim free after trapper fainted");
    assert!(b.choose(&d, 1, "switch 2").is_ok(), "victim may switch again");
}

// --------------------------------------------------------------------- Q2

/// Perish Song counter is a per-Pokemon volatile; switching out drops it and
/// the mon comes back clean, while the mon that keeps the volatile faints.
#[test]
fn q2_perish_counter_is_a_volatile_dropped_on_switch_out() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Mean Look", "Perish Song", "Rest"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
    ];
    let p2 = vec![
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse", "Milk Drink"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move meanlook").unwrap();
    b.choose(&d, 1, "move curse").unwrap();
    let victim = act(&b, 1);
    let singer = act(&b, 0);

    b.choose(&d, 0, "move perishsong").unwrap();
    b.choose(&d, 1, "move curse").unwrap();
    assert!(has_vol(&b, &d, singer, "perishsong"), "singer counts down too");
    assert!(has_vol(&b, &d, victim, "perishsong"));

    // turn 3, 4: nothing
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move curse").unwrap();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move curse").unwrap();
    assert!(b.log.iter().any(|l| l.contains("perish1")), "{:?}", b.log);

    // turn 5: the singer escapes by switching, the trapped victim cannot
    b.choose(&d, 0, "switch 2").unwrap();
    b.choose(&d, 1, "move return").unwrap();
    assert!(!has_vol(&b, &d, singer, "perishsong"), "counter dropped on switch out");
    assert!(b.poke(victim).fainted, "perish0 fainted the volatile holder: {:?}", b.log);
    assert!(!b.poke(singer).fainted, "the mon that switched out survives");
    assert!(b.log.iter().any(|l| l.contains("perish0")));
}

/// A phazed-out Perish Song holder loses the counter and comes back clean.
#[test]
fn q2_phazed_out_mon_returns_without_a_counter() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Perish Song", "Rest", "Night Shade"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
    ];
    let p2 = vec![
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move perishsong").unwrap();
    b.choose(&d, 1, "move thunderbolt").unwrap();
    let singer = act(&b, 0);
    assert!(has_vol(&b, &d, singer, "perishsong"));

    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move whirlwind").unwrap();
    assert!(b.log.iter().any(|l| l.starts_with("|drag|")), "{:?}", b.log);
    assert!(!has_vol(&b, &d, singer, "perishsong"), "drag removed the counter");
    // and the p2 mon that stayed in still has its own counter
    assert!(has_vol(&b, &d, act(&b, 1), "perishsong"));
}

// --------------------------------------------------------------------- Q3

/// Rest = 3 sleep "time" units: two turns fully skipped, acting on the third.
/// On the wake-up turn a FASTER foe's move resolves while the target is still
/// asleep -- the cure line comes after the damage line.
#[test]
fn q3_rest_sleeps_two_turns_and_wakes_inside_its_own_beforemove() {
    // Suicune L51 (base spe 85) outspeeds Umbreon L50 (base spe 65).
    let p1 = vec![
        mk("Umbreon", 50, "Leftovers", &["Rest", "Toxic", "Charm"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Suicune", 51, "Bright Powder", &["Surf", "Ice Beam", "Rest"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Steelix", 50, "Leftovers", &["Earthquake", "Rest"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    // chip Umbreon so Rest is legal
    b.choose(&d, 0, "move charm").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    let sleeper = act(&b, 0);
    assert_eq!(b.poke(sleeper).status, Status::Slp);
    assert_eq!(
        b.poke(sleeper).status_state.get_int(nc2000_engine::state::DK::Time),
        3,
        "Rest sets sleep time 3"
    );

    // turn A: skipped
    let m0 = b.log.len();
    b.choose(&d, 0, "move charm").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    assert!(b.log[m0..].iter().any(|l| l.contains("|cant|") && l.contains("slp")));
    assert_eq!(b.poke(sleeper).status, Status::Slp);

    // turn B: skipped
    let m1 = b.log.len();
    b.choose(&d, 0, "move charm").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    assert!(b.log[m1..].iter().any(|l| l.contains("|cant|") && l.contains("slp")));
    assert_eq!(b.poke(sleeper).status, Status::Slp);

    // turn C: wakes, and the faster foe's Surf lands on a still-asleep target
    let m2 = b.log.len();
    b.choose(&d, 0, "move charm").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    let tail: Vec<String> = b.log[m2..].to_vec();
    let dmg = tail.iter().position(|l| l.starts_with("|-damage|p1a")).expect(&format!("{tail:?}"));
    let cure = tail
        .iter()
        .position(|l| l.starts_with("|-curestatus|p1a") && l.contains("slp"))
        .expect(&format!("{tail:?}"));
    assert!(dmg < cure, "faster foe's move resolves BEFORE the wake-up: {tail:?}");
    assert!(tail[dmg].contains("slp"), "target still shows slp when hit: {tail:?}");
    assert_eq!(b.poke(sleeper).status, Status::None, "awake at end of turn C");
}

// --------------------------------------------------------------------- Q4

/// Sleep Talk used while awake fails outright AND still costs 1 PP.
#[test]
fn q4_awake_sleep_talk_fails_and_still_spends_pp() {
    let p1 = vec![
        mk("Umbreon", 50, "Leftovers", &["Toxic", "Charm", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Suicune", 51, "Bright Powder", &["Surf", "Ice Beam", "Sleep Talk", "Rest"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Steelix", 50, "Leftovers", &["Earthquake", "Rest"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    let user = act(&b, 1);
    let st = d.moves.id("sleeptalk").unwrap();
    let pp_before = b.poke(user).move_slots.iter().find(|s| s.id == st).unwrap().pp;
    let m0 = b.log.len();
    b.choose(&d, 0, "move charm").unwrap();
    b.choose(&d, 1, "move sleeptalk").unwrap();
    let pp_after = b.poke(user).move_slots.iter().find(|s| s.id == st).unwrap().pp;
    assert_eq!(b.poke(user).status, Status::None, "user was awake");
    assert_eq!(pp_after, pp_before - 1, "the failed awake Sleep Talk still spent PP");
    let tail: Vec<String> = b.log[m0..].to_vec();
    assert!(
        tail.iter().any(|l| l.starts_with("|move|p2a") && l.contains("Sleep Talk")),
        "{tail:?}"
    );
    assert!(
        !tail.iter().any(|l| l.contains("[from] Sleep Talk")),
        "no move was called: {tail:?}"
    );
}

// --------------------------------------------------------------------- Q5

/// A target that already has a status cannot be frozen (trySetStatus re-sets
/// the SAME status and bails), so a sleeping foe is freeze-proof.
#[test]
fn q5_sleeping_or_statused_target_cannot_be_frozen() {
    let p1 = vec![
        mk("Umbreon", 50, "Leftovers", &["Rest", "Toxic", "Charm"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Suicune", 51, "Bright Powder", &["Surf", "Ice Beam", "Rest"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Steelix", 50, "Leftovers", &["Earthquake", "Rest"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move charm").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    let target = act(&b, 0);
    let src = act(&b, 1);
    assert_eq!(b.poke(target).status, Status::Slp);
    let rv = b.try_set_status(&d, target, "frz", Some(src), EffectHandle::None);
    assert_eq!(rv, RV::False, "a sleeping target cannot be frozen");
    assert_eq!(b.poke(target).status, Status::Slp);
}

/// Freeze Clause Mod: an ALIVE frozen party member blocks a second freeze;
/// once that member faints and its replacement switches in, its status is
/// wiped (actions.rs switch_in) and the clause no longer blocks.
#[test]
fn q5_freeze_clause_alive_blocks_fainted_does_not() {
    let p1 = vec![
        mk("Snorlax", 55, "Leftovers", &["Body Slam", "Rest"]),
        mk("Umbreon", 50, "Mint Berry", &["Toxic", "Charm", "Rest"]),
        mk("Marowak", 50, "Thick Club", &["Bonemerang", "Swords Dance"]),
    ];
    let p2 = vec![
        mk("Suicune", 51, "Bright Powder", &["Surf", "Ice Beam", "Rest"]),
        mk("Steelix", 50, "Leftovers", &["Earthquake", "Rest"]),
        mk("Alakazam", 50, "Leftovers", &["Psychic", "Recover"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    let frozen = act(&b, 0);
    let src = act(&b, 1);
    assert_eq!(
        b.try_set_status(&d, frozen, "frz", Some(src), EffectHandle::None),
        RV::True
    );
    assert_eq!(b.poke(frozen).status, Status::Frz);

    // control: while it is ALIVE, a second freeze on the same side is blocked
    let other = PokeId { side: 0, slot: b.sides[0].party[1] };
    assert_eq!(
        b.try_set_status(&d, other, "frz", Some(src), EffectHandle::None),
        RV::False,
        "clause blocks while the frozen mon is alive"
    );
    assert!(b.log.iter().any(|l| l.contains("Freeze Clause activated.")));

    // KO the frozen mon; its replacement switches in
    b.poke_mut(frozen).hp = 1;
    b.choose(&d, 0, "move bodyslam").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    assert!(b.poke(frozen).fainted, "{:?}", b.log);
    if b.sides[0].request_state().is_some() {
        b.choose(&d, 0, "switch 2").unwrap();
    }
    assert_eq!(
        b.poke(frozen).status,
        Status::None,
        "a fainted mon's status is wiped when its replacement switches in"
    );
    let newmon = act(&b, 0);
    let src = act(&b, 1);
    assert_eq!(
        b.try_set_status(&d, newmon, "frz", Some(src), EffectHandle::None),
        RV::True,
        "clause no longer blocks once the frozen mon has fainted and been replaced"
    );
}

// --------------------------------------------------------------------- Q6

/// Normal -> Ghost is a hard immunity; Ground -> Ghost is neutral.
#[test]
fn q6_return_is_immune_but_earthquake_is_not() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Mean Look", "Perish Song", "Rest"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
    ];
    let p2 = vec![
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse", "Milk Drink"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Rest"]),
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    let ghost = act(&b, 0);
    let hp0 = b.poke(ghost).hp;
    let m0 = b.log.len();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move return").unwrap();
    assert!(b.log[m0..].iter().any(|l| l.starts_with("|-immune|p1a")), "{:?}", &b.log[m0..]);
    assert_eq!(b.poke(ghost).hp, hp0, "Return does exactly 0 to a Ghost");

    // Body Slam from Snorlax: same immunity
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "switch 2").unwrap();
    let m1 = b.log.len();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move bodyslam").unwrap();
    assert!(b.log[m1..].iter().any(|l| l.starts_with("|-immune|p1a")), "{:?}", &b.log[m1..]);
    assert_eq!(b.poke(ghost).hp, hp0, "Body Slam does exactly 0 to a Ghost");

    // Earthquake: not immune, real damage
    let m2 = b.log.len();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move earthquake").unwrap();
    assert!(
        b.log[m2..].iter().any(|l| l.starts_with("|-damage|p1a")),
        "Earthquake must damage a Ghost: {:?}",
        &b.log[m2..]
    );
    assert!(!b.log[m2..].iter().any(|l| l.starts_with("|-immune|p1a")));
    assert!(b.poke(ghost).hp < hp0);
    assert!(
        !b.log[m2..].iter().any(|l| l.contains("-resisted") || l.contains("-supereffective")),
        "Ground vs Ghost is neutral: {:?}",
        &b.log[m2..]
    );
}

// --------------------------------------------------------------------- Q7

/// Curse from a non-Ghost: +1 Atk, +1 Def, -1 Spe on the user, no HP cost,
/// and one use is enough to lose the speed race against Misdreavus L50.
#[test]
fn q7_non_ghost_curse_boosts_and_costs_speed() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Mean Look", "Perish Song", "Rest"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
    ];
    let p2 = vec![
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse", "Milk Drink"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Rest"]),
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    let user = act(&b, 1);
    let foe = act(&b, 0);
    let hp0 = b.poke(user).hp;
    let spe0 = b.get_stat(&d, user, 4, false, false, false);
    let ghost_spe = b.get_stat(&d, foe, 4, false, false, false);
    let sky_slot = b.sides[0].party[1];
    let skarm = PokeId { side: 0, slot: sky_slot };
    let skarm_spe = b.get_stat(&d, skarm, 4, false, false, false);

    let m0 = b.log.len();
    b.choose(&d, 0, "move rest").unwrap();
    b.choose(&d, 1, "move curse").unwrap();
    assert_eq!(b.poke(user).boosts[0], 1, "atk +1");
    assert_eq!(b.poke(user).boosts[1], 1, "def +1");
    assert_eq!(b.poke(user).boosts[4], -1, "spe -1");
    assert_eq!(b.poke(user).hp, hp0, "no HP cost for a non-Ghost Curse");
    assert!(b.log[m0..].iter().any(|l| l.contains("|-unboost|p2a") && l.contains("spe")));

    let spe1 = b.get_stat(&d, user, 4, false, false, false);
    eprintln!(
        "SPEED miltank {spe0} -> {spe1} (1 curse); misdreavus L50 {ghost_spe}; skarmory L55 {skarm_spe}"
    );
    assert!(spe1 < spe0);
    assert!(spe0 > ghost_spe, "uncursed Miltank outspeeds Misdreavus");
    assert!(spe1 < ghost_spe, "ONE curse already loses the race to Misdreavus");
    assert!(spe0 > skarm_spe, "uncursed Miltank outspeeds Skarmory L55");
    assert!(spe1 < skarm_spe, "ONE curse also loses the race to Skarmory L55");
}

// --------------------------------------------------------------------- Q8

/// Bright Powder subtracts 20 from the 0-255 accuracy value (post-scaling),
/// so a 100%-accuracy move goes from never-miss (255) to 235/256 = 91.8%.
#[test]
fn q8_bright_powder_turns_a_never_miss_move_into_235_of_256() {
    fn miss_rate(item: &str, n: usize) -> (usize, usize) {
        let d = dex();
        let mut misses = 0;
        for s in 0..n {
            let p1 = vec![
                mk("Umbreon", 50, item, &["Charm", "Toxic", "Rest"]),
                mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
                mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
            ];
            let p2 = vec![
                mk("Snorlax", 55, "Mint Berry", &["Body Slam", "Rest"]),
                mk("Steelix", 50, "Leftovers", &["Earthquake", "Rest"]),
                mk("Alakazam", 50, "Leftovers", &["Psychic", "Recover"]),
            ];
            let seed = format!("{},{},{},{}", s % 251, (s * 7) % 251, (s * 13) % 251, (s * 29) % 251);
            let mut b = Battle::from_fixture(&d, &seed, &p1, &p2).unwrap();
            b.choose(&d, 0, "team 1,2,3").unwrap();
            b.choose(&d, 1, "team 1,2,3").unwrap();
            let m0 = b.log.len();
            b.choose(&d, 0, "move charm").unwrap();
            b.choose(&d, 1, "move bodyslam").unwrap();
            if b.log[m0..].iter().any(|l| l.starts_with("|-miss|p2a")) {
                misses += 1;
            }
        }
        (misses, n)
    }
    let n = 2000;
    let (with, _) = miss_rate("Bright Powder", n);
    let (without, _) = miss_rate("Leftovers", n);
    eprintln!("BRIGHTPOWDER miss {with}/{n}; LEFTOVERS miss {without}/{n}");
    assert_eq!(without, 0, "a 100%-accuracy move never misses without Bright Powder");
    let rate = with as f64 / n as f64;
    assert!(
        (rate - 21.0 / 256.0).abs() < 0.02,
        "expected ~8.2% miss with Bright Powder, got {rate}"
    );
}

/// Trapping blocks only VOLUNTARY switches: the trapper can still phaze its
/// own victim out with Whirlwind, and the incoming mon is not trapped.
#[test]
fn q1_trapped_victim_can_still_be_dragged_out() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Mean Look", "Whirlwind", "Rest"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
    ];
    let p2 = vec![
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Thunder Wave"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    b.choose(&d, 0, "move meanlook").unwrap();
    b.choose(&d, 1, "move thunderbolt").unwrap();
    let victim = act(&b, 1);
    assert!(b.poke(victim).trapped, "Zapdos trapped");

    let m0 = b.log.len();
    b.choose(&d, 0, "move whirlwind").unwrap();
    b.choose(&d, 1, "move thunderbolt").unwrap();
    assert!(
        b.log[m0..].iter().any(|l| l.starts_with("|drag|p2a")),
        "a trapped mon is still draggable: {:?}",
        &b.log[m0..]
    );
    assert_ne!(act(&b, 1), victim, "a different p2 mon is active");
    assert!(!has_vol(&b, &d, victim, "trapped"));
    assert!(!b.poke(act(&b, 1)).trapped, "the incoming mon inherits nothing");
    assert!(!has_vol(&b, &d, act(&b, 0), "trapper"), "the trapper link is gone too");
}

/// Whirlwind/Roar fail when a move action is still queued (gen2 onTryHit):
/// two -1-priority phazes in the same turn -> the FASTER one fails.
#[test]
fn q1_faster_whirlwind_fails_when_the_slower_one_is_still_queued() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Whirlwind", "Rest", "Night Shade"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
    ];
    let p2 = vec![
        mk("Zapdos", 50, "Miracle Berry", &["Whirlwind", "Thunderbolt"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Miltank", 55, "Bright Powder", &["Return", "Curse"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    let m0 = b.log.len();
    b.choose(&d, 0, "move whirlwind").unwrap();
    b.choose(&d, 1, "move whirlwind").unwrap();
    let tail: Vec<String> = b.log[m0..].to_vec();
    // (|drag| appears twice: the |split| secret/shared log pair)
    let fast = tail.iter().position(|l| l.starts_with("|move|p2a") && l.contains("Whirlwind"));
    let slow = tail.iter().position(|l| l.starts_with("|move|p1a") && l.contains("Whirlwind"));
    let (fast, slow) = (fast.unwrap(), slow.unwrap());
    assert!(fast < slow, "Zapdos L50 is the faster phazer: {tail:?}");
    assert_eq!(tail[fast + 1], "|-fail|p1a: Misdreavus", "faster phaze fails: {tail:?}");
    assert!(
        tail[slow..].iter().any(|l| l.starts_with("|drag|p2a")),
        "the slower phaze lands: {tail:?}"
    );
    assert!(
        !tail[..slow].iter().any(|l| l.starts_with("|drag|")),
        "no drag before the slower phaze: {tail:?}"
    );
}

/// If both sides sit still, BOTH holders faint at perish0 (each holds its own
/// volatile); the phazed/switched mon re-enters with no counter at all.
#[test]
fn q2_both_holders_faint_and_reentry_is_clean() {
    let p1 = vec![
        mk("Misdreavus", 50, "Miracle Berry", &["Perish Song", "Rest", "Night Shade"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
    ];
    let p2 = vec![
        mk("Miltank", 55, "Bright Powder", &["Curse", "Milk Drink", "Double Team"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Zapdos", 50, "Miracle Berry", &["Thunderbolt", "Whirlwind"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    let singer = act(&b, 0);
    let foe = act(&b, 1);
    b.choose(&d, 0, "move perishsong").unwrap();
    b.choose(&d, 1, "move curse").unwrap();
    for _ in 0..3 {
        if b.ended || b.poke(singer).fainted {
            break;
        }
        b.choose(&d, 0, "move rest").unwrap();
        b.choose(&d, 1, "move curse").unwrap();
    }
    assert!(b.poke(singer).fainted, "singer faints too: {:?}", b.log);
    assert!(b.poke(foe).fainted, "foe faints too: {:?}", b.log);

    // re-entry after a voluntary switch-out is clean
    let p1b = p1.clone();
    let p2b = p2.clone();
    let (d2, mut c) = start("5,6,7,8", &p1b, &p2b);
    let s2 = act(&c, 0);
    c.choose(&d2, 0, "move perishsong").unwrap();
    c.choose(&d2, 1, "move curse").unwrap();
    assert!(has_vol(&c, &d2, s2, "perishsong"));
    c.choose(&d2, 0, "switch 2").unwrap();
    c.choose(&d2, 1, "move curse").unwrap();
    assert!(!has_vol(&c, &d2, s2, "perishsong"));
    c.choose(&d2, 0, "switch 2").unwrap(); // back in
    c.choose(&d2, 1, "move curse").unwrap();
    assert_eq!(act(&c, 0), s2, "the singer is back on the field");
    assert!(!has_vol(&c, &d2, s2, "perishsong"), "returns with no counter");
    assert!(!c.poke(s2).fainted);
}

/// Converse of q3: if the SLEEPER is the faster mon it wakes inside its own
/// BeforeMove and is already awake when the slower foe's move resolves.
#[test]
fn q3_slower_attacker_hits_an_already_awake_sleeper() {
    let p1 = vec![
        mk("Umbreon", 50, "Leftovers", &["Charm", "Toxic", "Rest"]),
        mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
        mk("Skarmory", 55, "Leftovers", &["Drill Peck", "Rest"]),
    ];
    let p2 = vec![
        mk("Suicune", 51, "Bright Powder", &["Surf", "Ice Beam", "Rest"]),
        mk("Snorlax", 50, "Mint Berry", &["Body Slam", "Rest"]),
        mk("Steelix", 50, "Leftovers", &["Earthquake", "Rest"]),
    ];
    let (d, mut b) = start("1,2,3,4", &p1, &p2);
    let sleeper = act(&b, 1); // Suicune, the FASTER side
    b.poke_mut(sleeper).hp = b.poke(sleeper).maxhp / 2;
    b.choose(&d, 0, "move charm").unwrap();
    b.choose(&d, 1, "move rest").unwrap();
    assert_eq!(b.poke(sleeper).status, Status::Slp);
    for _ in 0..2 {
        b.choose(&d, 0, "move charm").unwrap();
        b.choose(&d, 1, "move surf").unwrap();
        assert_eq!(b.poke(sleeper).status, Status::Slp);
    }
    let m = b.log.len();
    b.choose(&d, 0, "move toxic").unwrap();
    b.choose(&d, 1, "move surf").unwrap();
    let tail: Vec<String> = b.log[m..].to_vec();
    let cure = tail
        .iter()
        .position(|l| l.starts_with("|-curestatus|p2a") && l.contains("slp"))
        .expect(&format!("{tail:?}"));
    let foe_move = tail
        .iter()
        .position(|l| l.starts_with("|move|p1a"))
        .expect(&format!("{tail:?}"));
    assert!(cure < foe_move, "the faster sleeper wakes first: {tail:?}");
    assert_ne!(
        b.poke(sleeper).status,
        Status::Slp,
        "already awake when the slower foe moved: {tail:?}"
    );
}

/// Double Team (multiplicative, on the 0-255 scale) and Bright Powder (flat
/// -20 AFTER it) stack: 1 Double Team + Bright Powder = 171/256 to be hit.
#[test]
fn q8_double_team_and_bright_powder_stack() {
    let d = dex();
    let n = 2000usize;
    let mut misses = 0;
    for s in 0..n {
        let p1 = vec![
            mk("Miltank", 55, "Bright Powder", &["Double Team", "Return", "Milk Drink"]),
            mk("Blissey", 50, "Leftovers", &["Soft-Boiled", "Toxic"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Rest"]),
        ];
        let p2 = vec![
            mk("Snorlax", 55, "Mint Berry", &["Body Slam", "Rest"]),
            mk("Steelix", 50, "Leftovers", &["Earthquake", "Rest"]),
            mk("Alakazam", 50, "Leftovers", &["Psychic", "Recover"]),
        ];
        let seed = format!("{},{},{},{}", s % 251, (s * 7) % 251, (s * 13) % 251, (s * 29) % 251);
        let mut b = Battle::from_fixture(&d, &seed, &p1, &p2).unwrap();
        b.choose(&d, 0, "team 1,2,3").unwrap();
        b.choose(&d, 1, "team 1,2,3").unwrap();
        b.choose(&d, 0, "move doubleteam").unwrap();
        b.choose(&d, 1, "move rest").unwrap();
        let m0 = b.log.len();
        b.choose(&d, 0, "move milkdrink").unwrap();
        b.choose(&d, 1, "move bodyslam").unwrap();
        if b.log[m0..].iter().any(|l| l.starts_with("|-miss|p2a")) {
            misses += 1;
        }
    }
    let rate = misses as f64 / n as f64;
    eprintln!("DT1+BRIGHTPOWDER miss {misses}/{n} = {rate:.4} (expected {:.4})", 85.0 / 256.0);
    assert!((rate - 85.0 / 256.0).abs() < 0.03, "got {rate}");
}
