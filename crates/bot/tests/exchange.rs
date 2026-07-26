//! The M17 exchange scheme (Phase B) and the confusion term.
//!
//! Two claims are under test. The first is *inertness*: the shipped defaults
//! must read exactly as they did before, because `exchange_v2` reaches
//! `race_margin` too (the race term is the 1×1 case of the same function) and
//! the race term ships at 3.0. Every assertion below that compares "off" with
//! "on" is therefore also a regression net for the shipped eval.
//!
//! The second is that each new channel actually carries what it claims:
//!
//! - the pair matrix's game value replaces Phase A's maximin/minimax midpoint,
//!   and stays inside the interval those two bound (a mathematical property of
//!   any zero-sum matrix — if it ever escapes, the solver call is wrong);
//! - a mon that is not on the field is charged for getting there (the switch
//!   turn plus Spikes), which is the only channel through which switching has
//!   ever reached `eval01`;
//! - the residual clock accelerates with the gen-2 toxic counter instead of
//!   ticking a flat 1/8, so "Toxic just landed" and "Toxic is on turn six" stop
//!   being the same position;
//! - a confused attacker loses half its offense;
//! - the additive confusion penalty scales with the clock the engine keeps, and
//!   is zero on the last hindered turn (the engine lets that move through).

use conformance::load_dex;
use nc2000_bot::eval::{self, exchange_matrix, pair_margin};
use nc2000_bot::EvalWeights;
use nc2000_engine::battle::{EffectHandle, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, PokeId, Status, DK};

fn set(json: &str) -> PokemonSet {
    serde_json::from_str(json).unwrap()
}

fn snorlax() -> PokemonSet {
    set(r#"{"name":"Snorlax","species":"Snorlax","item":"Leftovers","ability":"No Ability","moves":["Body Slam","Curse","Earthquake","Double-Edge"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#)
}
fn zapdos() -> PokemonSet {
    set(r#"{"name":"Zapdos","species":"Zapdos","item":"Leftovers","ability":"No Ability","moves":["Thunderbolt","Drill Peck","Hidden Power Ice","Thunder Wave"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"N"}"#)
}
fn marowak() -> PokemonSet {
    set(r#"{"name":"Marowak","species":"Marowak","item":"Thick Club","ability":"No Ability","moves":["Earthquake","Rock Slide","Hidden Power Bug","Swords Dance"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":30,"atk":26,"def":26,"spa":30,"spd":30,"spe":30},"gender":"M"}"#)
}
fn starmie() -> PokemonSet {
    set(r#"{"name":"Starmie","species":"Starmie","item":"Leftovers","ability":"No Ability","moves":["Surf","Thunderbolt","Ice Beam","Recover"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"N"}"#)
}
fn machamp() -> PokemonSet {
    set(r#"{"name":"Machamp","species":"Machamp","item":"Leftovers","ability":"No Ability","moves":["Cross Chop","Earthquake","Rock Slide","Body Slam"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#)
}

fn start(dex: &Dex, p1: &[PokemonSet], p2: &[PokemonSet]) -> Battle {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", p1, p2).unwrap();
    b.set_log_enabled(false);
    b.choose(dex, 0, "team 1,2,3").unwrap();
    b.choose(dex, 1, "team 1,2,3").unwrap();
    b
}

/// A plain 3v3 with no hazards, no residual and no confusion: the position on
/// which Phase B must agree with Phase A pair-for-pair.
fn plain(dex: &Dex) -> Battle {
    start(dex, &[snorlax(), marowak(), starmie()], &[zapdos(), machamp(), starmie()])
}

fn v1() -> EvalWeights {
    EvalWeights { exchange: 1.0, ..EvalWeights::default() }
}
fn v2() -> EvalWeights {
    EvalWeights { exchange: 1.0, exchange_v2: true, ..EvalWeights::default() }
}

fn matrix(b: &Battle, dex: &Dex, w: &EvalWeights) -> (usize, usize, Vec<f64>) {
    let (rows, cols, cells) = exchange_matrix(b, dex, w).expect("both sides alive");
    (rows.len(), cols.len(), cells)
}

fn maximin_minimax(k0: usize, k1: usize, cells: &[f64]) -> (f64, f64) {
    let maximin = (0..k0)
        .map(|i| cells[i * k1..(i + 1) * k1].iter().cloned().fold(f64::INFINITY, f64::min))
        .fold(f64::NEG_INFINITY, f64::max);
    let minimax = (0..k1)
        .map(|j| (0..k0).map(|i| cells[i * k1 + j]).fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::INFINITY, f64::min);
    (maximin, minimax)
}

// ---- 1. inertness of the shipped configuration ---------------------------

#[test]
fn shipped_defaults_leave_the_scheme_off() {
    let d = EvalWeights::default();
    assert_eq!(d.exchange, 0.0, "the exchange term ships inert");
    assert_eq!(d.confusion, 0.0, "the confusion term ships inert");
    assert!(!d.exchange_v2, "Phase B ships off — it changes the race term too");
    // `exchange` at 0.0 means the term is not even consulted, so flipping the
    // Phase B flag alone cannot move `eval01` off the shipped default.
    let dex = load_dex();
    let b = plain(&dex);
    let on = EvalWeights { exchange_v2: true, ..EvalWeights::default() };
    // The race term is the one path Phase B can still reach at exchange 0.0,
    // and it is dormant outside last-mon-vs-last-mon, which `plain` is not.
    assert_eq!(eval::eval01(&b, &dex, &d), eval::eval01(&b, &dex, &on));
}

#[test]
fn phase_b_agrees_with_phase_a_where_nothing_new_applies() {
    let dex = load_dex();
    let b = plain(&dex);
    let (a0, d0) = (b.active_id(0).unwrap(), b.active_id(1).unwrap());
    // Both on the field, no residual, no confusion: identical arithmetic.
    assert_eq!(
        pair_margin(&b, &dex, &v1(), a0, d0),
        pair_margin(&b, &dex, &v2(), a0, d0),
        "on-field pair with no new mechanics must be bit-identical"
    );
    // The on-field CELL of the matrix likewise (row 0, col 0 are the actives).
    let (k0, k1, c1) = matrix(&b, &dex, &v1());
    let (_, _, c2) = matrix(&b, &dex, &v2());
    assert_eq!(c1[0], c2[0], "the active-vs-active cell owes no entry cost");
    assert_eq!((k0, k1), (3, 3));
}

// ---- 2. the game value replaces the midpoint ----------------------------

#[test]
fn game_value_stays_inside_the_maximin_minimax_interval() {
    let dex = load_dex();
    for b in [plain(&dex), start(&dex, &[marowak(), starmie(), snorlax()], &[machamp(), zapdos(), starmie()])] {
        let w = v2();
        let (k0, k1, cells) = matrix(&b, &dex, &w);
        let (maximin, minimax) = maximin_minimax(k0, k1, &cells);
        let value = eval::exchange_margin(&b, &dex, &w);
        // RM+ is iterative, so the interval check carries its convergence
        // tolerance: at 128 sweeps on a 3×3 the observed residual is ~1e-4.
        assert!(
            value >= maximin - 1e-3 && value <= minimax + 1e-3,
            "game value {value} escaped [{maximin}, {minimax}] — the solver call is wrong"
        );
    }
}

#[test]
fn game_value_is_the_cell_itself_at_one_by_one() {
    let dex = load_dex();
    let mut b = plain(&dex);
    // Reduce both sides to their active mon: the matrix is 1×1, so the value,
    // the midpoint, and `race_margin` are all the same number.
    for s in 0..2 {
        let keep = b.active_id(s).unwrap().slot;
        for slot in b.sides[s].party.clone() {
            if slot != keep {
                b.poke_mut(PokeId { side: s as u8, slot }).hp = 0;
                b.poke_mut(PokeId { side: s as u8, slot }).fainted = true;
            }
        }
    }
    for w in [v1(), v2()] {
        let (k0, k1, cells) = matrix(&b, &dex, &w);
        assert_eq!((k0, k1), (1, 1));
        let value = eval::exchange_margin(&b, &dex, &w);
        assert!((value - cells[0]).abs() < 1e-3, "1×1 value must be the cell ({value})");
        assert!(
            (value - eval::race_margin(&b, &dex, &w)).abs() < 1e-3,
            "the exchange term is a strict generalization of the race term"
        );
    }
}

// ---- 3. entry costs: switching is not free ------------------------------

#[test]
fn spikes_charges_the_bench_in_phase_b_only() {
    let dex = load_dex();
    let mut b = plain(&dex);
    let src = b.active_id(1).unwrap();
    b.add_side_condition(&dex, 0, "spikes", Some(src), EffectHandle::None);
    let (k0, k1, clean) = matrix(&plain(&dex), &dex, &v2());
    let (_, _, hazard) = matrix(&b, &dex, &v2());
    // Row 0 is side 0's active — it has already paid — so it must not move;
    // every benched row is worth strictly less than it was without Spikes.
    for j in 0..k1 {
        assert_eq!(clean[j], hazard[j], "the mon already in owes nothing");
    }
    let mut moved = 0;
    for i in 1..k0 {
        for j in 0..k1 {
            let (a, h) = (clean[i * k1 + j], hazard[i * k1 + j]);
            assert!(h <= a + 1e-9, "Spikes cannot improve a benched matchup ({a} -> {h})");
            if h < a - 1e-9 {
                moved += 1;
            }
        }
    }
    assert!(moved > 0, "Spikes must cost the bench something");
    // Phase A is blind to all of it: same matrix with and without the hazard.
    let (_, _, a_clean) = matrix(&plain(&dex), &dex, &v1());
    let (_, _, a_hazard) = matrix(&b, &dex, &v1());
    assert_eq!(a_clean, a_hazard, "Phase A cannot see entry costs — that is the gap");
}

#[test]
fn a_mon_that_dies_on_entry_loses_its_pairings() {
    let dex = load_dex();
    let mut b = plain(&dex);
    let src = b.active_id(1).unwrap();
    b.add_side_condition(&dex, 0, "spikes", Some(src), EffectHandle::None);
    // Put a benched mon below the Spikes tick (1/8 of max HP at one layer).
    let bench = *b.sides[0]
        .party
        .iter()
        .find(|&&s| Some(s) != b.active_id(0).map(|i| i.slot))
        .unwrap();
    let id = PokeId { side: 0, slot: bench };
    let tick = b.poke(id).maxhp / 8;
    b.poke_mut(id).hp = tick.max(1);
    let (_, k1, cells) = matrix(&b, &dex, &v2());
    let row = b.sides[0].party.iter().position(|&s| s == bench).unwrap();
    for j in 0..k1 {
        assert_eq!(
            cells[row * k1 + j], -1.0,
            "a mon that faints on the hazard cannot win any pairing"
        );
    }
}

// ---- 4. the residual clock follows the engine's toxic counter -----------

#[test]
fn toxic_counter_accelerates_the_clock_in_phase_b_only() {
    let dex = load_dex();
    // Side 1's active is badly poisoned and half dead; side 0 cannot race it
    // down quickly, so the position turns on how fast the poison does.
    let base = |counter: i64, hp_num: i32, hp_den: i32| -> Battle {
        let mut b = plain(&dex);
        let foe = b.active_id(1).unwrap();
        b.poke_mut(foe).status = Status::Tox;
        b.refresh_poke_mask(&dex, foe);
        let hp = b.poke(foe).maxhp * hp_num / hp_den;
        b.poke_mut(foe).hp = hp.max(1);
        // Side 0's attacker cannot contribute: with no usable move the pair
        // turns entirely on how fast the poison runs, which is the channel
        // under test. (Zeroing PP is the cheapest way to isolate it.)
        let me = b.active_id(0).unwrap();
        for i in 0..b.poke(me).move_slots.len() {
            b.poke_mut(me).move_slots[i].pp = 0;
        }
        b.add_volatile(&dex, foe, "residualdmg", Some(foe), EffectHandle::None);
        let rd = dex.conds_id("residualdmg").unwrap();
        b.poke_mut(foe).volatile_mut(rd).unwrap().set_int(DK::Counter, counter);
        b
    };
    // Scan the poisoned mon's HP rather than betting on one number: the claim
    // is directional (an advanced counter can only help the un-poisoned side)
    // and must hold everywhere, with a strict gain somewhere.
    let mut strict = 0;
    for num in 1..=8 {
        let (fresh, late) = (base(0, num, 12), base(6, num, 12));
        let (a0, d0) = (fresh.active_id(0).unwrap(), fresh.active_id(1).unwrap());
        // Phase A ticks a flat 1/8 and cannot tell the two apart at all.
        assert_eq!(
            pair_margin(&fresh, &dex, &v1(), a0, d0),
            pair_margin(&late, &dex, &v1(), a0, d0),
            "the flat tick is blind to the counter — that is the measured bias"
        );
        // Phase B reads the engine's `floor(maxhp/16) × counter` with the
        // counter climbing every turn, so the late clock kills sooner.
        let (p_fresh, p_late) = (
            pair_margin(&fresh, &dex, &v2(), a0, d0),
            pair_margin(&late, &dex, &v2(), a0, d0),
        );
        assert!(
            p_late >= p_fresh - 1e-12,
            "an advanced toxic counter cannot favour its owner ({p_fresh} -> {p_late} at {num}/12 HP)"
        );
        if p_late > p_fresh + 1e-12 {
            strict += 1;
        }
    }
    assert!(strict > 0, "the accelerating clock must change some position's value");
}

// ---- 5. confusion: lost offense, and a clock-shaped penalty ------------

#[test]
fn confusion_halves_the_attacker_in_phase_b() {
    let dex = load_dex();
    let mut b = plain(&dex);
    let me = b.active_id(0).unwrap();
    let foe = b.active_id(1).unwrap();
    // A race short enough for the term to claim anything at all.
    b.poke_mut(foe).hp = b.poke(foe).maxhp / 4;
    b.poke_mut(me).hp = b.poke(me).maxhp / 4;
    let clear = pair_margin(&b, &dex, &v2(), me, foe);
    b.add_volatile(&dex, me, "confusion", Some(foe), EffectHandle::None);
    let cf = dex.conds_id("confusion").unwrap();
    b.poke_mut(me).volatile_mut(cf).unwrap().set_int(DK::Time, 4);
    let confused = pair_margin(&b, &dex, &v2(), me, foe);
    assert!(
        confused < clear,
        "a confused attacker must fare worse in the exchange ({clear} -> {confused})"
    );
    // Phase A has no channel for it.
    let mut a = plain(&dex);
    a.poke_mut(foe).hp = a.poke(foe).maxhp / 4;
    a.poke_mut(me).hp = a.poke(me).maxhp / 4;
    let a_clear = pair_margin(&a, &dex, &v1(), me, foe);
    a.add_volatile(&dex, me, "confusion", Some(foe), EffectHandle::None);
    a.poke_mut(me).volatile_mut(cf).unwrap().set_int(DK::Time, 4);
    assert_eq!(a_clear, pair_margin(&a, &dex, &v1(), me, foe), "Phase A is confusion-blind");
}

#[test]
fn the_confusion_penalty_tracks_the_engine_clock() {
    let dex = load_dex();
    let w = EvalWeights { confusion: 1.0, ..EvalWeights::default() };
    let cf = dex.conds_id("confusion").unwrap();
    let with_clock = |t: i64| -> f64 {
        let mut b = plain(&dex);
        let me = b.active_id(0).unwrap();
        let foe = b.active_id(1).unwrap();
        b.add_volatile(&dex, me, "confusion", Some(foe), EffectHandle::None);
        b.poke_mut(me).volatile_mut(cf).unwrap().set_int(DK::Time, t);
        eval::eval01(&b, &dex, &w)
    };
    let dex_ref = &dex;
    let clean = eval::eval01(&plain(dex_ref), dex_ref, &w);
    // A clock of 1 means the very next move goes through unhindered
    // (`conditions.rs` decrements first and removes the volatile at 0), so
    // there is nothing left to charge for.
    assert!((with_clock(1) - clean).abs() < 1e-12, "the last confused turn costs nothing");
    // Beyond that the penalty grows with the clock, monotonically.
    let (t2, t3, t5) = (with_clock(2), with_clock(3), with_clock(5));
    assert!(t2 < clean, "a live confusion clock must cost side 0 something");
    assert!(t3 < t2 && t5 < t3, "longer clock, larger penalty ({t2} {t3} {t5})");
    // And the shipped default (weight 0.0) charges nothing at any clock.
    let off = EvalWeights::default();
    let mut b = plain(&dex);
    let me = b.active_id(0).unwrap();
    let foe = b.active_id(1).unwrap();
    let before = eval::eval01(&b, &dex, &off);
    b.add_volatile(&dex, me, "confusion", Some(foe), EffectHandle::None);
    b.poke_mut(me).volatile_mut(cf).unwrap().set_int(DK::Time, 5);
    assert_eq!(before, eval::eval01(&b, &dex, &off), "confusion ships inert");
}

// ---- 6. the candidate preset says what it means ------------------------

#[test]
fn exchange_scheme_moves_the_absorbed_terms_out() {
    let full = EvalWeights::exchange_scheme(0.75, 0.0);
    assert_eq!(full.exchange, 0.75);
    assert!(full.exchange_v2);
    assert_eq!(full.race, 0.0, "the matrix generalizes the race term — do not pay twice");
    for (name, v) in [
        ("brn", full.brn),
        ("par", full.par),
        ("slp", full.slp),
        ("frz", full.frz),
        ("psn", full.psn),
        ("tox", full.tox),
        ("substitute", full.substitute),
        ("spikes", full.spikes),
    ] {
        assert_eq!(v, 0.0, "{name} is the exchange's input now, not its own term");
    }
    // The material terms are what is left when the exchange declines to claim.
    let d = EvalWeights::default();
    assert_eq!((full.hp, full.alive, full.pp, full.threat), (d.hp, d.alive, d.pp, d.threat));
    // damp = 1.0 keeps them, which is the double-counting end of the sweep.
    let kept = EvalWeights::exchange_scheme(0.75, 1.0);
    assert_eq!((kept.tox, kept.spikes, kept.substitute), (d.tox, d.spikes, d.substitute));
}
