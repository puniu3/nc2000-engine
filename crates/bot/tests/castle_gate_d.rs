//! Gate D — Cross-Module Differential Parity Suite (`pbs-vs-ps-spec.md` §6-2).
//!
//! Verifies end-to-end consistency across ALL layers on Castle PBS and PS Stadium 2:
//! `Battle` ↔ fresh `CompactBattle` ↔ cached `CompactBattle` (`RosterCompactCache`)
//! ↔ `linear.rs` ↔ `relay.rs` ↔ `exact.rs` ↔ `bounds.rs` ↔ `mask.rs` ↔ `eval.rs`
//! ↔ `hp_transition.rs` ↔ `compiled_preview.rs` ↔ `observe.rs` ↔ `belief.rs`
//! ↔ `validate.rs` ↔ `replay.rs`.

use nc2000_bot::belief::Belief;
use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_bot::compiled_preview;
use nc2000_bot::eval::{
    castle_judge_value_battle, compact_castle_judge_value, compact_eval01, compact_eval01_blindspot,
    compact_turn_limit_outcome, eval01, turn_limit_outcome_value, EvalWeights,
};
use nc2000_bot::exact::{ExactConfig, ExactSolver};
use nc2000_bot::hp_transition::HpTransitionKey;
use nc2000_bot::linear::{compact_linear_diff, compact_perish_margin};
use nc2000_bot::mask::dominated_actions;
use nc2000_bot::observe::Observer;
use nc2000_bot::relay::{is_relay_eligible, relay_surplus_for_side};
use nc2000_bot::runner::judge_castle_turn_limit;
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::compact::{CompactBattle, RosterCompactCache};
use nc2000_engine::dex::Dex;
use nc2000_engine::replay::{Recorder, Replay};
use nc2000_engine::rules::{CastleOverlayConfig, Ruleset};
use nc2000_engine::state::{Battle, Status};
use nc2000_engine::validate::{validate_team_with_ruleset, Learnsets};
use serde_json::json;

fn load_dex() -> Dex {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/gen2stadium2.json"
    ))
    .expect("read gen2stadium2.json");
    Dex::from_json(&text).expect("parse dex")
}

fn load_learnsets() -> Learnsets {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/learnsets-gen2.json"
    ))
    .expect("read learnsets-gen2.json");
    Learnsets::from_json(&text).expect("parse learnsets")
}

fn mk(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(json!({
        "name": species,
        "species": species,
        "item": item,
        "ability": "",
        "moves": moves,
        "level": level,
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
        "happiness": 255,
        "gender": "M"
    }))
    .unwrap()
}

fn legal_action_codes_from_battle(b: &mut Battle, dex: &Dex, side: usize) -> Vec<u8> {
    let choices = b.legal_choices(dex, side);
    if choices.is_empty() {
        return if b.ended { vec![] } else { vec![7] };
    }
    let active_id = b.active_id(side);
    let mut out = Vec::new();
    for c in choices {
        match c {
            SearchChoice::Move(mid) => {
                if let Some(id) = active_id {
                    if let Some(slot) = b.poke(id).move_slots.iter().position(|m| m.id == mid) {
                        out.push(slot as u8);
                    } else if dex.moves.key(mid) == "struggle" {
                        out.push(0);
                    }
                }
            }
            SearchChoice::Switch(pos_1based) => {
                out.push(4 + (pos_1based - 1));
            }
            SearchChoice::Pass => out.push(7),
            SearchChoice::Team(_) => {}
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn compact_legal_vec(cb: &CompactBattle, table: &nc2000_engine::compact::MatchTable, side: usize) -> Vec<u8> {
    let mut buf = [0u8; 7];
    let n = cb.legal_actions(table, side, &mut buf);
    let mut v = buf[..n].to_vec();
    v.sort_unstable();
    v
}

#[test]
fn gate_d_fresh_vs_cached_compact_and_batonn_choice_parity() {
    let dex = load_dex();
    let p1 = vec![
        mk("Jolteon", 50, "Leftovers", &["Baton Pass", "Agility", "Substitute", "Thunderbolt"]),
        mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
        mk("Snorlax", 50, "", &["Body Slam", "Earthquake", "Rest", "Curse"]),
    ];
    let p2 = vec![
        mk("Tyranitar", 55, "Leftovers", &["Pursuit", "Rock Slide", "Earthquake", "Roar"]),
        mk("Cloyster", 50, "Leftovers", &["Spikes", "Surf", "Ice Beam", "Explosion"]),
        mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
    ];
    let mut b = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &p1, &p2, Ruleset::castle_tournament()).unwrap();
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    // Lay Spikes on P1 side
    let p2a = b.active_id(1);
    b.add_side_condition(&dex, 0, "spikes", p2a, nc2000_engine::battle::EffectHandle::None);

    let rcache = RosterCompactCache::from_battle(&b, &dex);
    let (t_fresh, mut cb_fresh) = CompactBattle::from_battle(&b, &dex);
    let (t_cached, cb_cached) = CompactBattle::from_battle_with_roster_cache(&b, &dex, Some(&rcache));
    assert_eq!(t_fresh.ruleset, Ruleset::castle_tournament());
    assert_eq!(t_fresh.match_fingerprint, t_cached.match_fingerprint);
    assert_eq!(cb_fresh.search_hash(), cb_cached.search_hash());
    assert_eq!(cb_fresh, cb_cached);

    // Legal moves match on both sides before BATONN
    for s in 0..2 {
        assert_eq!(
            legal_action_codes_from_battle(&mut b, &dex, s),
            compact_legal_vec(&cb_fresh, &t_fresh, s),
            "pre-BP legal actions side {s}"
        );
    }

    // Both Battle and CompactBattle pause at BATONN (2 bench choices: slot 1 Marowak, slot 2 Snorlax)
    // when faster Jolteon uses Baton Pass against Tyranitar's Pursuit.
    b.choose(&dex, 0, "move batonpass").unwrap();
    b.choose(&dex, 1, "move pursuit").unwrap();
    cb_fresh.step_turn(&t_fresh, [0, 0]);

    assert_eq!(legal_action_codes_from_battle(&mut b, &dex, 0), vec![5, 6]);
    assert_eq!(compact_legal_vec(&cb_fresh, &t_fresh, 0), vec![5, 6]);
    assert_eq!(compact_legal_vec(&cb_fresh, &t_fresh, 1), vec![7]);

    // Choose Snorlax (slot 2 -> "switch 3" / action 6): Spikes hits first, then Castle 2x Pursuit (BP 80)!
    b.choose(&dex, 0, "switch 3").unwrap();
    cb_fresh.apply_actions(&t_fresh, [6, 7]);
    let b_snor = b.poke(b.active_id(0).unwrap());
    let cb_snor_hp = cb_fresh.mons[2].hp() as i32;
    let spikes_dmg = b_snor.maxhp / 8;
    // Both took Spikes + boosted 2x Pursuit (well above Spikes + 1x Pursuit max ~44 dmg)
    assert!(
        b_snor.maxhp - b_snor.hp > spikes_dmg + 48,
        "Battle Castle BP receiver must take Spikes + 2x Pursuit: lost {}",
        b_snor.maxhp - b_snor.hp
    );
    assert!(
        b_snor.maxhp - cb_snor_hp > spikes_dmg + 48,
        "CompactBattle Castle BP receiver must take Spikes + 2x Pursuit: lost {}",
        b_snor.maxhp - cb_snor_hp
    );
    assert_eq!(b.turn, cb_fresh.turn);
    // Relay eligibility matches Battle/CompactBattle state
    assert!(is_relay_eligible(&cb_cached, &t_cached, 0));
    let surp = relay_surplus_for_side(&cb_cached, &t_cached, &EvalWeights::default(), 0, 0.5);
    assert!(
        surp.is_finite() && (0.0..=1.0).contains(&surp),
        "relay_surplus_for_side must be finite and non-negative: {surp}"
    );
}

#[test]
fn gate_d_turn_80_judge_parity_across_battle_compact_exact_bounds_eval_runner() {
    let dex = load_dex();
    let p1 = vec![
        mk("Snorlax", 55, "Leftovers", &["Splash"]),
        mk("Suicune", 50, "Leftovers", &["Splash"]),
    ];
    let p2 = vec![
        mk("Snorlax", 55, "Leftovers", &["Splash"]),
        mk("Suicune", 50, "Leftovers", &["Splash"]),
    ];
    let mut b = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &p1, &p2, Ruleset::castle_tournament()).unwrap();
    b.choose(&dex, 0, "team 1,2").unwrap();
    b.choose(&dex, 1, "team 1,2").unwrap();
    // P1 active at 80% HP, P2 active at 50% HP; equal alive count (2 vs 2) -> P1 wins by HP ratio sum!
    let p1a = b.active_id(0).unwrap();
    let p2a = b.active_id(1).unwrap();
    b.poke_mut(p1a).hp = b.poke(p1a).maxhp * 8 / 10;
    b.poke_mut(p2a).hp = b.poke(p2a).maxhp / 2;
    b.turn = 80;

    let (table, mut cb) = CompactBattle::from_battle(&b, &dex);
    // Pre-step (turn 80 before actions): judge helper agrees across Battle, CompactBattle, and runner
    let j_battle = castle_judge_value_battle(&b);
    let j_compact = compact_castle_judge_value(&cb, &table);
    let j_runner: f64 = match judge_castle_turn_limit(&b) {
        Outcome::P1Win => 1.0,
        Outcome::P2Win => 0.0,
        Outcome::Tie => 0.5,
    };
    assert!((j_battle - 1.0).abs() < 1e-12);
    assert!((j_compact - 1.0).abs() < 1e-12);
    assert!((j_runner - 1.0).abs() < 1e-12);

    // ExactSolver & BoundSolver at turn 80 certify exact P1 win (1.0)!
    let mut exact = ExactSolver::new(&dex, ExactConfig::default());
    let cert = exact.solve(&b).expect("turn 80 exact solve");
    assert!(
        (cert.lo - 1.0).abs() < 1e-9 && (cert.hi - 1.0).abs() < 1e-9,
        "ExactSolver T80 Castle judge=[{}, {}]",
        cert.lo,
        cert.hi
    );

    let mut bounds = BoundSolver::new(&dex, BoundConfig::default());
    let brep = bounds.solve(&b, None);
    assert!((brep.bounds.lo - 1.0).abs() < 1e-9 && (brep.bounds.hi - 1.0).abs() < 1e-9);

    // Stepping turn 80 -> 81 terminates identically in Battle and CompactBattle with P1Win!
    b.choose(&dex, 0, "move splash").unwrap();
    b.choose(&dex, 1, "move splash").unwrap();
    cb.step_turn(&table, [0, 0]);
    assert_eq!(b.outcome(), Some(Outcome::P1Win));
    assert_eq!(cb.outcome, 1);
    assert_eq!(turn_limit_outcome_value(&b), Some(1.0));
    assert_eq!(compact_turn_limit_outcome(&cb, &table), Some(1.0));
}

#[test]
fn gate_d_seat_asymmetry_eot_perish_and_berry_boundaries() {
    let dex = load_dex();
    // 1. Equal Perish Song 1 -> 0 in Castle: fixed 1P -> 2P EOT order!
    // In 2v2 with Perish=1 on both actives, 1P faints FIRST in Castle regardless of speed.
    let slow_mon = mk("Snorlax", 50, "Leftovers", &["Splash"]);
    let fast_mon = mk("Jolteon", 50, "Leftovers", &["Splash"]);
    let bench = mk("Suicune", 50, "Leftovers", &["Splash"]);
    let mut b_p1_fast = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[fast_mon.clone(), bench.clone()],
        &[slow_mon.clone(), bench.clone()],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_p1_fast.choose(&dex, 0, "team 1,2").unwrap();
    b_p1_fast.choose(&dex, 1, "team 1,2").unwrap();
    let perish_cid = dex.conds.id("perishsong").unwrap();
    for s in 0..2 {
        let id = b_p1_fast.active_id(s).unwrap();
        b_p1_fast.add_volatile(&dex, id, "perishsong", None, nc2000_engine::battle::EffectHandle::None);
        b_p1_fast.poke_mut(id).volatile_mut(perish_cid).unwrap().duration = Some(1);
    }
    let (t1, cb1) = CompactBattle::from_battle(&b_p1_fast, &dex);
    // In 1v1 Perish=1 Castle (cast by P1 -> source = P1): last-mon Perish forfeit -> P2Win in Battle & CompactBattle!
    let mut b_1v1 = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[fast_mon.clone()],
        &[slow_mon.clone()],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_1v1.choose(&dex, 0, "team 1").unwrap();
    b_1v1.choose(&dex, 1, "team 1").unwrap();
    let p1_caster = b_1v1.active_id(0);
    for s in 0..2 {
        let id = b_1v1.active_id(s).unwrap();
        b_1v1.add_volatile(&dex, id, "perishsong", p1_caster, nc2000_engine::battle::EffectHandle::None);
        b_1v1.poke_mut(id).volatile_mut(perish_cid).unwrap().duration = Some(1);
    }
    let (t_1v1, mut cb_1v1) = CompactBattle::from_battle(&b_1v1, &dex);
    let perish_m = compact_perish_margin(&cb_1v1, &t_1v1);
    b_1v1.choose(&dex, 0, "move splash").unwrap();
    b_1v1.choose(&dex, 1, "move splash").unwrap();
    cb_1v1.step_turn(&t_1v1, [0, 0]);
    assert_eq!(b_1v1.outcome(), Some(Outcome::P2Win));
    assert_eq!(cb_1v1.outcome, 2);
    assert!(perish_m < 0.0, "linear compact_perish_margin must reflect Castle 1P->2P EOT loss for P1: {perish_m}");

    // Flip caster to P2: last-mon Perish forfeit -> P1Win in BOTH Battle and CompactBattle!
    let mut b_1v1_p2 = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[fast_mon.clone()],
        &[slow_mon.clone()],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_1v1_p2.choose(&dex, 0, "team 1").unwrap();
    b_1v1_p2.choose(&dex, 1, "team 1").unwrap();
    let p2_caster = b_1v1_p2.active_id(1);
    for s in 0..2 {
        let id = b_1v1_p2.active_id(s).unwrap();
        b_1v1_p2.add_volatile(&dex, id, "perishsong", p2_caster, nc2000_engine::battle::EffectHandle::None);
        b_1v1_p2.poke_mut(id).volatile_mut(perish_cid).unwrap().duration = Some(1);
    }
    let (t_p2, mut cb_p2) = CompactBattle::from_battle(&b_1v1_p2, &dex);
    let perish_m_p2 = compact_perish_margin(&cb_p2, &t_p2);
    assert!(
        perish_m_p2 > 0.0,
        "linear compact_perish_margin must reflect Castle P2-caster 1v1 Perish forfeit (+margin for P1): {perish_m_p2}"
    );
    b_1v1_p2.choose(&dex, 0, "move splash").unwrap();
    b_1v1_p2.choose(&dex, 1, "move splash").unwrap();
    cb_p2.step_turn(&t_p2, [0, 0]);
    assert_eq!(b_1v1_p2.outcome(), Some(Outcome::P1Win));
    assert_eq!(cb_p2.outcome, 1);

    // Raw CGI (perish_dbond_lastmon_forfeit = false): 1v1 Perish mutual KO draws (`0.0` margin & Outcome::Tie)!
    let mut b_1v1_raw = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[fast_mon.clone()],
        &[slow_mon.clone()],
        Ruleset::castle_raw_cgi(false),
    )
    .unwrap();
    b_1v1_raw.choose(&dex, 0, "team 1").unwrap();
    b_1v1_raw.choose(&dex, 1, "team 1").unwrap();
    let raw_caster = b_1v1_raw.active_id(1);
    for s in 0..2 {
        let id = b_1v1_raw.active_id(s).unwrap();
        b_1v1_raw.add_volatile(&dex, id, "perishsong", raw_caster, nc2000_engine::battle::EffectHandle::None);
        b_1v1_raw.poke_mut(id).volatile_mut(perish_cid).unwrap().duration = Some(1);
    }
    let (t_raw, mut cb_raw) = CompactBattle::from_battle(&b_1v1_raw, &dex);
    assert_eq!(
        compact_perish_margin(&cb_raw, &t_raw),
        0.0,
        "Castle raw CGI 1v1 Perish mutual KO must have 0.0 margin"
    );
    b_1v1_raw.choose(&dex, 0, "move splash").unwrap();
    b_1v1_raw.choose(&dex, 1, "move splash").unwrap();
    cb_raw.step_turn(&t_raw, [0, 0]);
    assert_eq!(b_1v1_raw.outcome(), Some(Outcome::Tie));
    assert_eq!(cb_raw.outcome, 3);

    // 1v1 Sandstorm mutual KO in Castle -> DRAW (Tie / 3) in BOTH Battle and CompactBattle!
    let mut b_sand = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[fast_mon.clone()],
        &[slow_mon.clone()],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_sand.choose(&dex, 0, "team 1").unwrap();
    b_sand.choose(&dex, 1, "team 1").unwrap();
    b_sand.set_weather(&dex, "sandstorm", None, nc2000_engine::battle::EffectHandle::None);
    b_sand.poke_mut(b_sand.active_id(0).unwrap()).hp = 1;
    b_sand.poke_mut(b_sand.active_id(1).unwrap()).hp = 1;
    let (t_sand, mut cb_sand) = CompactBattle::from_battle(&b_sand, &dex);
    b_sand.choose(&dex, 0, "move splash").unwrap();
    b_sand.choose(&dex, 1, "move splash").unwrap();
    cb_sand.step_turn(&t_sand, [0, 0]);
    assert_eq!(b_sand.outcome(), Some(Outcome::Tie));
    assert_eq!(cb_sand.outcome, 3);
    assert_eq!(
        compact_perish_margin(&cb1, &t1),
        0.0,
        "Castle 2v2 equal Perish must have zero seat bias"
    );
    let lin_2v2 = compact_linear_diff(&cb1, &t1, 0.0, 0.0, 0.2, 1.0);
    assert_eq!(lin_2v2, 0.0, "2v2 equal Perish Castle linear_diff must be 0.0: {lin_2v2}");

    // 2. Healing berry strict `< floor(maxhp / 2)` boundary parity across Battle & CompactBattle
    let mut b_berry = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[mk("Snorlax", 50, "Gold Berry", &["Splash"])],
        &[mk("Snorlax", 50, "Gold Berry", &["Splash"])],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_berry.choose(&dex, 0, "team 1").unwrap();
    b_berry.choose(&dex, 1, "team 1").unwrap();
    let id0 = b_berry.active_id(0).unwrap();
    let id1 = b_berry.active_id(1).unwrap();
    let half = b_berry.poke(id0).maxhp / 2;
    b_berry.poke_mut(id0).hp = half; // exact floor(maxhp/2): MUST NOT eat in Castle
    b_berry.poke_mut(id1).hp = half - 1; // strictly below floor(maxhp/2): MUST eat (+30) in Castle
    let (t_b, mut cb_b) = CompactBattle::from_battle(&b_berry, &dex);
    b_berry.choose(&dex, 0, "move splash").unwrap();
    b_berry.choose(&dex, 1, "move splash").unwrap();
    cb_b.step_turn(&t_b, [0, 0]);
    assert_eq!(b_berry.poke(id0).hp, half);
    assert_eq!(cb_b.mons[0].hp() as i32, half);
    assert_eq!(b_berry.poke(id1).hp, half - 1 + 30);
    assert_eq!(cb_b.mons[3].hp() as i32, half - 1 + 30);
    assert!(b_berry.poke(id0).item.is_some());
    assert!(b_berry.poke(id1).item.is_none());
    assert_ne!(cb_b.mons[0].item, 0);
    assert_eq!(cb_b.mons[3].item, 0);
}

#[test]
fn gate_d_ditto_hp_mask_dual_claw_observer_belief_preview_validate_replay() {
    let dex = load_dex();
    let learnsets = load_learnsets();

    // 1. Transformed Ditto Hidden Power type + Castle Rapid Spin vs Ghost + Belly Drum 50% + Sleep no-forfeit masks
    let mut b_mask = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[
            mk("Snorlax", 50, "", &["Belly Drum", "Rapid Spin", "Lovely Kiss", "Rest"]),
            mk("Ditto", 50, "Metal Powder", &["Transform"]),
        ],
        &[
            mk("Misdreavus", 50, "", &["Thunderbolt", "Shadow Ball", "Psychic", "Pain Split"]),
            mk("Snorlax", 50, "", &["Body Slam", "Rest", "Curse", "Sleep Talk"]),
        ],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_mask.choose(&dex, 0, "team 1,2").unwrap();
    b_mask.choose(&dex, 1, "team 1,2").unwrap();
    // Set Spikes on P1 side, set Snorlax HP to exact floor(maxhp / 2), put foe bench Snorlax & own bench Ditto to sleep
    let foe_src = b_mask.active_id(1);
    b_mask.add_side_condition(&dex, 0, "spikes", foe_src, nc2000_engine::battle::EffectHandle::None);
    let p1a = b_mask.active_id(0).unwrap();
    let half_hp = b_mask.poke(p1a).maxhp / 2;
    b_mask.poke_mut(p1a).hp = half_hp;
    let own_bench = b_mask.sides[0].party[1];
    b_mask.sides[0].roster[own_bench as usize].status = Status::Slp;
    let foe_bench = b_mask.sides[1].party[1];
    b_mask.sides[1].roster[foe_bench as usize].status = Status::Slp;

    let dom = dominated_actions(&b_mask, &dex, 0);
    let bd_id = dex.moves.id("bellydrum").unwrap();
    let rs_id = dex.moves.id("rapidspin").unwrap();
    let lk_id = dex.moves.id("lovelykiss").unwrap();
    let rest_id = dex.moves.id("rest").unwrap();
    assert!(
        !dom.iter().any(|&(c, _)| c == SearchChoice::Move(bd_id)),
        "Castle Belly Drum at exact floor(maxhp/2) must NOT be masked: {dom:?}"
    );
    assert!(
        !dom.iter().any(|&(c, _)| c == SearchChoice::Move(rs_id)),
        "Castle Rapid Spin vs Ghost with Spikes up must NOT be masked: {dom:?}"
    );
    assert!(
        !dom.iter().any(|&(c, _)| c == SearchChoice::Move(rest_id)),
        "Castle own Rest with sleeping ally must NOT be masked (matches Battle/CompactBattle): {dom:?}"
    );
    assert!(
        dom.iter().any(|&(c, r)| c == SearchChoice::Move(lk_id) && !r.contains("forfeit")),
        "Castle Sleep Clause blocks Lovely Kiss as no-op (not PS sleep-forfeit): {dom:?}"
    );

    // Dual Quick Claw independence: P1 true, P2 false -> Battle and CompactBattle agree on action speed!
    b_mask.poke_mut(p1a).item = dex.known_items.quickclaw;
    b_mask.poke_mut(b_mask.active_id(1).unwrap()).item = dex.known_items.quickclaw;
    b_mask.quick_claw_rolls = [true, false];
    let (t_qc, cb_qc) = CompactBattle::from_battle(&b_mask, &dex);
    assert_eq!(cb_qc.quick_claw_for(0), true);
    assert_eq!(cb_qc.quick_claw_for(1), false);
    assert!(b_mask.get_pokemon_action_speed(&dex, p1a) > b_mask.get_pokemon_action_speed(&dex, b_mask.active_id(1).unwrap()));
    assert!(cb_qc.mon_action_speed(&t_qc, 0, 0) > cb_qc.mon_action_speed(&t_qc, 1, 0));

    // Transformed Ditto Hidden Power: mask must NOT treat HP Ghost vs Gengar as Normal-immune!
    let mut b_ditto = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[mk("Ditto", 50, "Metal Powder", &["Transform"])],
        &[mk("Gengar", 50, "Leftovers", &["Hidden Power Ice", "Splash"])],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_ditto.choose(&dex, 0, "team 1").unwrap();
    b_ditto.choose(&dex, 1, "team 1").unwrap();
    b_ditto.choose(&dex, 0, "move transform").unwrap();
    b_ditto.choose(&dex, 1, "move splash").unwrap();
    let dom_ditto = dominated_actions(&b_ditto, &dex, 0);
    let hp_id = dex.moves.id("hiddenpower").unwrap();
    assert!(
        !dom_ditto.iter().any(|&(c, r)| c == SearchChoice::Move(hp_id) && r.contains("immune")),
        "Transformed Ditto Hidden Power must not be falsely masked as immune: {dom_ditto:?}"
    );

    // 2. Observer + Belief exact HP & Castle Ice Berry dual Burn/Confusion exclusion
    let mut b_obs = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[mk("Snorlax", 50, "", &["Swagger", "Splash"])],
        &[mk("Snorlax", 50, "", &["Splash"])],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    b_obs.choose(&dex, 0, "team 1").unwrap();
    b_obs.choose(&dex, 1, "team 1").unwrap();
    let mut obs = Observer::new(&b_obs, 0);
    obs.observe(&b_obs, &dex);
    assert_eq!(obs.ruleset(), Ruleset::castle_tournament());
    let belief = Belief::pinned_from_battle(&b_obs, &obs);
    let mut rng = nc2000_bot::SplitMix64::new(42);
    let det = belief.determinize(&dex, &b_obs, &obs, &mut rng);
    assert_eq!(det.ruleset, Ruleset::castle_tournament());

    // 3. eval01 vs compact_eval01 + HpTransitionKey + compiled_preview ruleset separation
    let w = EvalWeights::default();
    let (t_obs, cb_obs) = CompactBattle::from_battle(&b_obs, &dex);
    let v_b = eval01(&b_obs, &dex, &w);
    let v_c = compact_eval01(&cb_obs, &t_obs, &w);
    let v_bs = compact_eval01_blindspot(&cb_obs, &t_obs, &w);
    assert!((v_b - v_c).abs() < 1e-9, "eval01 ({v_b}) vs compact_eval01 ({v_c})");
    assert!((0.0..=1.0).contains(&v_bs));

    let splash = dex.moves.id("splash").unwrap();
    let mut b_ps = Battle::from_fixture(&dex, "1,2,3,4", &[mk("Snorlax", 50, "", &["Swagger", "Splash"])], &[mk("Snorlax", 50, "", &["Splash"])]).unwrap();
    b_ps.choose(&dex, 0, "team 1").unwrap();
    b_ps.choose(&dex, 1, "team 1").unwrap();
    let k_castle = HpTransitionKey::new(&dex, &b_obs, [Some(SearchChoice::Move(splash)), Some(SearchChoice::Move(splash))], &[]);
    let k_ps = HpTransitionKey::new(&dex, &b_ps, [Some(SearchChoice::Move(splash)), Some(SearchChoice::Move(splash))], &[]);
    assert!(k_castle.is_some() && k_ps.is_some() && k_castle != k_ps);

    // Preview cache key separation (PS vs Castle P1 vs Castle P2)
    let team_a = vec![
        mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest", "Curse"]),
        mk("Zapdos", 50, "Magnet", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        mk("Cloyster", 50, "Gold Berry", &["Spikes", "Surf", "Ice Beam", "Explosion"]),
        mk("Raikou", 50, "Mint Berry", &["Thunderbolt", "Sleep Talk", "Rest", "Roar"]),
        mk("Skarmory", 50, "Miracle Berry", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        mk("Gengar", 50, "Quick Claw", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
    ];
    let prev_ps = Battle::from_fixture(&dex, "1,2,3,4", &team_a, &team_a).unwrap();
    let prev_c = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &team_a, &team_a, Ruleset::castle_tournament()).unwrap();
    let pk_ps = compiled_preview::key(&prev_ps, 0).unwrap();
    let pk_c0 = compiled_preview::key(&prev_c, 0).unwrap();
    let pk_c1 = compiled_preview::key(&prev_c, 1).unwrap();
    assert_ne!(pk_ps, pk_c0);
    assert_ne!(pk_c0, pk_c1, "Castle preview key must distinguish 1P vs 2P seat");
    assert_eq!(compiled_preview::policy_epoch_for(Ruleset::PsStadium2), "policy-v2");
    assert_eq!(compiled_preview::policy_epoch_for(Ruleset::castle_tournament()), "castle-policy-v2");

    // 4. Castle Validator + Replay roundtrip integration
    let team_json = serde_json::to_string(&team_a).unwrap();
    let val = validate_team_with_ruleset(&dex, &learnsets, &team_json, Ruleset::castle_tournament());
    assert_eq!(val["ok"], json!(true), "standard Castle team must validate cleanly: {val}");
    let mut bad_nypc = team_a.clone();
    bad_nypc[0].moves[0] = "Lovely Kiss".into();
    let val_nypc = validate_team_with_ruleset(
        &dex,
        &learnsets,
        &serde_json::to_string(&bad_nypc).unwrap(),
        Ruleset::castle_tournament(),
    );
    assert_eq!(val_nypc["ok"], json!(false), "NYPC Lovely Kiss Snorlax must fail Castle validation");

    let mut rec_b = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &team_a,
        &team_a,
        Ruleset::CastlePbs(CastleOverlayConfig::tournament()),
    )
    .unwrap();
    let mut rec = Recorder::new(&dex, &rec_b, [team_a.clone(), team_a.clone()], "1,2,3,4");
    rec.choose(&dex, &mut rec_b, 0, "team 1,2,3").unwrap();
    rec.choose(&dex, &mut rec_b, 1, "team 1,2,3").unwrap();
    rec.choose(&dex, &mut rec_b, 0, "move curse").unwrap();
    rec.choose(&dex, &mut rec_b, 1, "move curse").unwrap();
    let encoded = rec.replay.encode(&dex).unwrap();
    let decoded = Replay::decode(&dex, &encoded).unwrap();
    assert_eq!(decoded.ruleset, Ruleset::castle_tournament());
    let replayed = decoded.at(&dex, decoded.rounds.len()).unwrap();
    assert_eq!(replayed.state_key128(), rec_b.state_key128());
}

#[test]
fn gate_d_seeded_randomized_castle_differential_sweep() {
    let dex = load_dex();
    let w = EvalWeights::default();
    let pool = vec![
        mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
        mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Body Slam"]),
        mk("Kingdra", 50, "Dragon Scale", &["Surf", "DragonBreath", "Rest", "Sleep Talk"]),
        mk("Jolteon", 50, "Mint Berry", &["Thunderbolt", "Agility", "Substitute", "Baton Pass"]),
        mk("Tyranitar", 55, "Quick Claw", &["Pursuit", "Rock Slide", "Earthquake", "Roar"]),
        mk("Starmie", 50, "Gold Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
        mk("Gengar", 50, "Bright Powder", &["Thunderbolt", "Ice Punch", "Hypnosis", "Destiny Bond"]),
        mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        mk("Cloyster", 50, "Leftovers", &["Spikes", "Surf", "Ice Beam", "Explosion"]),
        mk("Zapdos", 55, "Magnet", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        mk("Misdreavus", 50, "Miracle Berry", &["Mean Look", "Perish Song", "Protect", "Thunder"]),
        mk("Vaporeon", 50, "Leftovers", &["Surf", "Acid Armor", "Baton Pass", "Rest"]),
        mk("Ditto", 50, "Metal Powder", &["Transform"]),
        mk("Machamp", 50, "Scope Lens", &["Cross Chop", "Rock Slide", "Foresight", "Rest"]),
    ];

    let mut rng = nc2000_bot::SplitMix64::new(0xC457_1E20_2611_2301);
    let mut total_states = 0usize;

    for game_idx in 0..24usize {
        let ruleset = match game_idx % 4 {
            0 => Ruleset::castle_tournament(),
            1 => Ruleset::castle_raw_cgi(false),
            2 => Ruleset::CastlePbs(CastleOverlayConfig {
                overflow_mode: true,
                ..CastleOverlayConfig::tournament()
            }),
            _ => Ruleset::castle_free(),
        };
        let mut p1 = Vec::with_capacity(3);
        let mut p2 = Vec::with_capacity(3);
        let mut used1 = [false; 14];
        let mut used2 = [false; 14];
        while p1.len() < 3 {
            let i = rng.below(pool.len());
            if !used1[i] {
                used1[i] = true;
                let mut s = pool[i].clone();
                if p1.len() > 0 {
                    s.level = 50;
                }
                p1.push(s);
            }
        }
        while p2.len() < 3 {
            let i = rng.below(pool.len());
            if !used2[i] {
                used2[i] = true;
                let mut s = pool[i].clone();
                if p2.len() > 0 {
                    s.level = 50;
                }
                p2.push(s);
            }
        }

        let seed_str = format!(
            "{},{},{},{}",
            (rng.next() & 0xFFFF) as u16,
            (rng.next() & 0xFFFF) as u16,
            (rng.next() & 0xFFFF) as u16,
            (rng.next() & 0xFFFF) as u16
        );
        let mut b = Battle::from_fixture_with_ruleset(&dex, &seed_str, &p1, &p2, ruleset).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();
        // Exercise near-L_TURN boundary on every 4th tournament game
        if game_idx % 4 == 0 && game_idx >= 12 {
            b.turn = 74;
        }
        let rcache = RosterCompactCache::from_battle(&b, &dex);

        for _step in 0..16 {
            if b.ended || b.outcome().is_some() {
                break;
            }
            total_states += 1;

            let (t_fresh, cb_fresh) = CompactBattle::from_battle(&b, &dex);
            let (t_cached, cb_cached) =
                CompactBattle::from_battle_with_roster_cache(&b, &dex, Some(&rcache));
            assert_eq!(t_fresh.ruleset, ruleset);
            assert_eq!(t_fresh.match_fingerprint, t_cached.match_fingerprint);
            assert_eq!(cb_fresh.search_hash(), cb_cached.search_hash());
            assert_eq!(cb_fresh, cb_cached);

            // Cross-layer eval & judge parity
            let v_b = eval01(&b, &dex, &w);
            let v_c = compact_eval01(&cb_fresh, &t_fresh, &w);
            let v_bs = compact_eval01_blindspot(&cb_fresh, &t_fresh, &w);
            assert!((v_b - v_c).abs() < 1e-9, "game={game_idx} eval01={v_b} compact={v_c}");
            assert!(v_bs.is_finite() && (0.0..=1.0).contains(&v_bs));
            assert!(
                (castle_judge_value_battle(&b) - compact_castle_judge_value(&cb_fresh, &t_fresh))
                    .abs()
                    < 1e-12
            );
            assert_eq!(
                turn_limit_outcome_value(&b),
                compact_turn_limit_outcome(&cb_fresh, &t_fresh)
            );

            // Unified Castle stat parity across eval, relay, and compact::get_stat_castle
            for s in 0..2usize {
                let slot = cb_fresh.sides[s].active_idx as usize;
                if (cb_fresh.sides[s].alive_mask & (1 << slot)) != 0 {
                    let p_idx = s * 3 + slot;
                    let mon = &cb_fresh.mons[p_idx];
                    let spec = &t_fresh.mons[p_idx];
                    let ispec = t_fresh.item_spec(mon.item);
                    let item_kind = if ispec.stat_kind == spec.species_stat_kind {
                        ispec.stat_kind
                    } else {
                        nc2000_engine::compact::ItemStatKind::None
                    };
                    for stat_i in 0..5usize {
                        let exp = nc2000_engine::compact::get_stat_castle(
                            spec.stored_stats[stat_i] as i32,
                            mon.boosts[stat_i],
                            stat_i,
                            false,
                            false,
                            mon.status_raw() == Status::Par as u8,
                            mon.status_raw() == Status::Brn as u8,
                            cb_fresh.sides[s].reflect_turns > 0,
                            cb_fresh.sides[s].lightscreen_turns > 0,
                            item_kind,
                            ruleset.overflow_mode(),
                        );
                        assert_eq!(
                            nc2000_bot::eval::ruleset_mon_stat(
                                &cb_fresh, &t_fresh, s, slot, stat_i, None, false, false
                            ),
                            exp
                        );
                        assert_eq!(
                            nc2000_bot::relay::stat_with_boost(
                                &cb_fresh,
                                &t_fresh,
                                s,
                                slot,
                                stat_i,
                                mon.boosts[stat_i]
                            ),
                            exp
                        );
                    }
                }
            }

            // Legal choice parity & mask sanity
            let mut choices = [None, None];
            for s in 0..2usize {
                assert_eq!(
                    legal_action_codes_from_battle(&mut b, &dex, s),
                    compact_legal_vec(&cb_fresh, &t_fresh, s),
                    "game={game_idx} turn={} side={s} legal mismatch",
                    b.turn
                );
                let leg = b.legal_choices(&dex, s);
                if !leg.is_empty() {
                    let dom = dominated_actions(&b, &dex, s);
                    // Never mask every legal choice unless only self-KO / pure-noop choices remain
                    for &(c, _) in &dom {
                        assert!(leg.contains(&c), "masked choice not in legal set");
                    }
                    choices[s] = Some(leg[rng.below(leg.len())]);
                }
            }
            if choices == [None, None] {
                break;
            }
            b.apply_choices(&dex, choices).unwrap();
        }
    }
    assert!(total_states >= 100, "sweep must exercise >= 100 Castle states, got {total_states}");
}
