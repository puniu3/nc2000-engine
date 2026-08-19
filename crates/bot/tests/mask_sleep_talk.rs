//! What the awake-Sleep-Talk mask rule (`noop_reason`, smmcts.rs) must and
//! must not do, and what its census gate cannot see.
//!
//! Written as a refutation attempt on commit f1e3d5c and kept because it
//! caught a real one: the rule was first written with `verdict!`, whose early
//! return shadowed every downstream rule for sleeptalk/snore, so an ASLEEP
//! user aiming Snore at a stranded Ghost stopped being refused by the
//! type-immunity arm. `no_shadowing_of_downstream_rules` is the guard against
//! that shape coming back — any future rule placed above the foe-reading arms
//! has the same hazard.
//!
//! The last two tests are not pass/fail claims about the mask. They pin WHY
//! `noop_census` cannot score this particular rule: a working Sleep Talk logs
//! the called move on its own `|move|` line, which is the token the census
//! breaks its window on, and the failure markers it does see are
//! subject-blind. "ENGINE DISAGREED 0" over this rule's firings is therefore
//! silence, not evidence.
//!
//! Both of those holes are now REPAIRED in `noop_census::read_outcome`: the
//! window follows a `|move|` line tagged `[from] <the masked move>` instead
//! of breaking on it, and `cant`/`-miss` only count when they name the actor.
//! Measured over the 570-battle corpus, the repair moved 245 rows out of
//! "engine confirmed", every one of them credited from a foe `|cant|` line,
//! and it takes 516 of the 863 real Sleep-Talk windows in that corpus from
//! "no evidence" to a scoreable DISAGREED. The inline replicas below are
//! deliberately left as the OLD reader — they are the record of what the
//! census used to see, and `read_outcome_legacy` is the same code kept in
//! the harness so the two can be scored side by side on every run.

use nc2000_bot::smmcts::{dominated_actions, dominated_actions_with, MaskRules};
use nc2000_engine::state::{Battle, DK, PokeId, Status};
use nc2000_engine::battle::{EffectHandle, PokemonSet};

fn team_snore_vs_ghost() -> (Vec<PokemonSet>, Vec<PokemonSet>) {
    // side 0: Suicune (base Spe 85) with Snore + Sleep Talk + Surf.
    let mine: Vec<PokemonSet> = serde_json::from_str(
        r#"[
        {"name":"Suicune","species":"Suicune","item":"","ability":"No Ability",
         "moves":["Surf","Sleep Talk","Snore","Rest"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"N","level":50},
        {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
         "moves":["Splash","Sleep Powder","Body Slam","Rest"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
         "moves":["Splash","Sleep Powder","Psychic","Rest"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
    ]"#,
    )
    .unwrap();
    // side 1: Misdreavus (Ghost, base Spe 85 -> slower than Suicune? both 85).
    // Use Gengar? base Spe 110 (faster). Use Haunter? 95. Need a GHOST that is
    // SLOWER than Suicune (85): Misdreavus is 85 (tie -> faster_than_foe false).
    // Snorlax-speed ghost: none. Use Misdreavus but drop its speed with a boost.
    let theirs: Vec<PokemonSet> = serde_json::from_str(
        r#"[
        {"name":"Misdreavus","species":"Misdreavus","item":"","ability":"No Ability",
         "moves":["Splash","Rest","Curse","Perish Song"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"F","level":50},
        {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
         "moves":["Splash","Body Slam","Rest","Sleep Powder"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
        {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
         "moves":["Splash","Psychic","Rest","Sleep Powder"],
         "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
    ]"#,
    )
    .unwrap();
    (mine, theirs)
}

fn setup() -> (nc2000_engine::dex::Dex, Battle) {
    let dex = conformance::load_dex();
    let (a, b) = team_snore_vs_ghost();
    let mut bt = Battle::from_fixture(&dex, "7,8,9,10", &a, &b).unwrap();
    bt.set_log_enabled(false);
    bt.choose(&dex, 0, "team 1, 2, 3").unwrap();
    bt.choose(&dex, 1, "team 1, 2, 3").unwrap();
    (dex, bt)
}

fn strand(b: &mut Battle, side: usize) {
    let party = b.sides[side].party.clone();
    for &slot in party.iter().skip(1) {
        let id = PokeId { side: side as u8, slot };
        b.poke_mut(id).hp = 0;
        b.poke_mut(id).fainted = true;
    }
    b.sides[side].pokemon_left = 1;
}

fn reasons(v: Vec<(nc2000_engine::battle::SearchChoice, &'static str)>, dex: &nc2000_engine::dex::Dex, key: &str) -> Vec<&'static str> {
    let id = dex.moves.id(key).unwrap();
    v.into_iter()
        .filter(|(c, _)| matches!(c, nc2000_engine::battle::SearchChoice::Move(m) if *m == id))
        .map(|(_, w)| w)
        .collect()
}

/// The rule must not cost coverage. Snore is Normal and physical, so an
/// ASLEEP user aiming it at a stranded Ghost has to keep reaching the
/// type-immunity arm below — as must an AWAKE but SLOWER one, whose own
/// status the rule cannot prove anything about.
#[test]
fn no_shadowing_of_downstream_rules() {
    let (dex, mut b) = setup();
    strand(&mut b, 1);
    let me = b.active_id(0).unwrap();
    let foe = b.active_id(1).unwrap();
    eprintln!(
        "speeds: me {} foe {}",
        b.get_pokemon_action_speed(&dex, me),
        b.get_pokemon_action_speed(&dex, foe)
    );
    eprintln!("foe types ghost? {}", b.poke(foe).types.iter().any(|t| t == dex.known_types.ghost));

    // (a) user ASLEEP: the new rule's verdict! returns None and stops the scan.
    let mut asleep = b.clone();
    asleep.set_status(&dex, me, "slp", Some(me), EffectHandle::None, true);
    asleep.poke_mut(me).status_state.set_int(DK::Time, 3);
    let shipped = reasons(dominated_actions(&asleep, &dex, 0), &dex, "snore");
    let legacy = reasons(
        dominated_actions_with(&asleep, &dex, 0, MaskRules { sleep_talk_awake: false }),
        &dex,
        "snore",
    );
    eprintln!("ASLEEP  snore: shipped={shipped:?}  legacy={legacy:?}");

    // (b) user AWAKE but SLOWER (speed tie counts as slower): same shadowing.
    let mut slow = b.clone();
    slow.poke_mut(foe).boosts[4] = 6;
    let shipped_s = reasons(dominated_actions(&slow, &dex, 0), &dex, "snore");
    let legacy_s = reasons(
        dominated_actions_with(&slow, &dex, 0, MaskRules { sleep_talk_awake: false }),
        &dex,
        "snore",
    );
    eprintln!("AWAKE-SLOWER snore: shipped={shipped_s:?}  legacy={legacy_s:?}");

    assert!(!legacy.is_empty(), "legacy mask must refuse Snore into a stranded Ghost");
    assert_eq!(shipped, legacy, "the awake-Sleep-Talk rule shadowed the type-immunity rule");
    assert_eq!(shipped_s, legacy_s, "same, for an awake but slower user");
}

/// A WORKING Sleep Talk logs the called move on its own `|move|`
/// line, which is exactly the token `noop_census::read_outcome` breaks its
/// window on. So a false positive of this rule is invisible to the census.
#[test]
fn census_window_cannot_see_a_working_sleep_talk() {
    let (dex, b) = setup();
    let me = b.active_id(0).unwrap();
    let mut asleep = b.clone();
    asleep.set_log_enabled(true);
    asleep.set_status(&dex, me, "slp", Some(me), EffectHandle::None, true);
    asleep.poke_mut(me).status_state.set_int(DK::Time, 3);
    asleep.log.clear();
    asleep.choose(&dex, 0, "move sleeptalk").unwrap();
    asleep.choose(&dex, 1, "move splash").unwrap();
    eprintln!("--- WORKING SLEEP TALK LOG ---");
    for l in &asleep.log {
        eprintln!("{l}");
    }
    // replicate read_outcome's window
    let actor = "p1a";
    let start = asleep
        .log
        .iter()
        .position(|l| l.starts_with("|move|") && l[6..].starts_with(actor));
    eprintln!("start = {start:?}");
    if let Some(s) = start {
        let next = asleep.log.iter().skip(s + 1).position(|l| {
            l.starts_with("|move|") || l.starts_with("|upkeep") || l.starts_with("|turn|")
        });
        eprintln!("window ends after {next:?} lines; window = {:?}",
            &asleep.log[s + 1..s + 1 + next.unwrap_or(0)]);
    }
}

/// the wake-up turn. `conditions.rs:159` decrements to 0, CURES,
/// and returns Undef, so the move proceeds with status None and Sleep Talk
/// fails at onTry. The mask reads Slp at the decision and refuses nothing:
/// a measured FALSE NEGATIVE the commit message calls "untouched by
/// construction".
#[test]
fn wake_turn_sleep_talk_is_dead_and_deliberately_unmasked() {
    let (dex, b) = setup();
    let me = b.active_id(0).unwrap();
    let mut w = b.clone();
    w.set_log_enabled(true);
    w.set_status(&dex, me, "slp", Some(me), EffectHandle::None, true);
    w.poke_mut(me).status_state.set_int(DK::Time, 1); // wakes on its own move
    let refused = reasons(dominated_actions(&w, &dex, 0), &dex, "sleeptalk");
    eprintln!("mask on the wake turn: {refused:?}");
    let st = dex.moves.id("sleeptalk").unwrap();
    let pp_before = w.poke(me).get_move_slot(st).unwrap().pp;
    let foe = w.active_id(1).unwrap();
    let foe_hp = w.poke(foe).hp;
    w.log.clear();
    w.choose(&dex, 0, "move sleeptalk").unwrap();
    w.choose(&dex, 1, "move splash").unwrap();
    eprintln!("--- WAKE-TURN SLEEP TALK LOG ---");
    for l in &w.log {
        eprintln!("{l}");
    }
    eprintln!(
        "pp {} -> {}   status {:?}   foe hp {} -> {}",
        pp_before,
        w.poke(me).get_move_slot(st).unwrap().pp,
        w.poke(me).status,
        foe_hp,
        w.poke(foe).hp
    );
    assert!(refused.is_empty(), "mask says nothing on the wake turn");
    assert_eq!(w.poke(me).status, Status::None, "it woke up");
}

/// where `noop_census`'s 208 "engine confirmed" for this rule come
/// from. `read_outcome` sets `failed = true` on ANY `-fail`/`-immune`/`-miss`/
/// `cant` line in the window without checking whose it is, and a foe that is
/// asleep/paralysed emits its `|cant|` with no `|move|` of its own — i.e.
/// inside our window. The confirmation is about the FOE, not our move.
#[test]
fn census_engine_confirmed_comes_from_the_foes_cant_line() {
    let (dex, b) = setup();
    let foe = b.active_id(1).unwrap();
    let mut w = b.clone();
    w.set_log_enabled(true);
    // we are AWAKE (rule fires); the foe is asleep and will emit |cant|
    w.set_status(&dex, foe, "slp", Some(foe), EffectHandle::None, true);
    w.poke_mut(foe).status_state.set_int(DK::Time, 3);
    // make sure we move first
    w.poke_mut(w.active_id(0).unwrap()).boosts[4] = 6;
    let refused = reasons(dominated_actions(&w, &dex, 0), &dex, "sleeptalk");
    eprintln!("mask: {refused:?}");
    w.log.clear();
    w.choose(&dex, 0, "move sleeptalk").unwrap();
    w.choose(&dex, 1, "move splash").unwrap();
    eprintln!("--- AWAKE SLEEP TALK, ASLEEP FOE ---");
    for l in &w.log {
        eprintln!("{l}");
    }
    let actor = "p1a";
    let start = w.log.iter().position(|l| l.starts_with("|move|") && l[6..].starts_with(actor));
    let mut failed = false;
    if let Some(s) = start {
        for line in w.log.iter().skip(s + 1) {
            if line.starts_with("|move|") || line.starts_with("|upkeep") || line.starts_with("|turn|") {
                break;
            }
            if line.contains("[from]") { continue; }
            let tag = line.split('|').filter(|x| !x.is_empty()).next().unwrap_or("");
            eprintln!("  window line: {line}  tag={tag}");
            if matches!(tag, "-fail" | "-immune" | "-miss" | "cant") { failed = true; }
        }
    }
    eprintln!("read_outcome would report failed={failed}  (i.e. 'engine confirmed')");
    assert!(refused.len() == 1, "the rule fires here");
}
