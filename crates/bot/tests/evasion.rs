//! Regression: the bot's threat/eval must track the gen-2 accuracy×evasion
//! stage so a Double-Team / Baton-Pass evasion wall registers as danger (and
//! phazing as relief) instead of being invisible.
//!
//! The reported weakness (sohehe-vs-puniu3 replay: Jolteon stacks Double Team
//! to +6, Baton Passes the evasion, and the bot dithers) traced to
//! `expected_hit_fraction` using only *base* move accuracy — a +6-evasion foe
//! looked exactly as hittable as an unboosted one, so the search saw no danger
//! and never valued phazing. The fix folds `Battle::hit_probability` (the real
//! gen-2 accuracy roll) into the threat feature.
//!
//! `couple_evasion: false` in `EvalWeights` reproduces the pre-fix eval, so the
//! tests below assert the *contrast* directly (blind before, correct after) and
//! do not depend on any external old-eval build.

use conformance::load_dex;
use nc2000_bot::eval::{self, best_hit_fraction, expected_hit_fraction};
use nc2000_bot::mcts::Playout;
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{Agent, EvalWeights, RmAgent, RmConfig};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::{toid, Dex, MoveId};
use nc2000_engine::state::Battle;

fn set(json: &str) -> PokemonSet {
    serde_json::from_str(json).unwrap()
}

/// Crobat (bot lead) carries a phazer (Whirlwind) + attacks; opponent Snorlax
/// stands in for the Baton-Pass recipient whose evasion we stack.
fn crobat_phazer() -> PokemonSet {
    set(r#"{"name":"Crobat","species":"Crobat","item":"Leftovers","ability":"No Ability","moves":["Whirlwind","Sludge Bomb","Wing Attack","Swift"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#)
}
fn snorlax() -> PokemonSet {
    set(r#"{"name":"Snorlax","species":"Snorlax","item":"Leftovers","ability":"No Ability","moves":["Body Slam","Curse","Rest","Earthquake"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"M"}"#)
}
fn porygon2() -> PokemonSet {
    set(r#"{"name":"Porygon2","species":"Porygon2","item":"Mint Berry","ability":"No Ability","moves":["Recover","Thunderbolt","Ice Beam","Curse"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31},"gender":"N"}"#)
}
fn marowak() -> PokemonSet {
    set(r#"{"name":"Marowak","species":"Marowak","item":"Thick Club","ability":"No Ability","moves":["Earthquake","Rock Slide","Hidden Power Bug","Swords Dance"],"level":50,"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"ivs":{"hp":30,"atk":26,"def":26,"spa":30,"spd":30,"spe":30},"gender":"M"}"#)
}

fn start(dex: &Dex, p1: &[PokemonSet], p2: &[PokemonSet]) -> Battle {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", p1, p2).unwrap();
    b.set_log_enabled(false);
    b.choose(dex, 0, "team 1,2,3").unwrap();
    b.choose(dex, 1, "team 1,2,3").unwrap();
    b
}

fn mid(dex: &Dex, name: &str) -> MoveId {
    dex.moves.id(&toid(name)).unwrap()
}

fn weights(couple: bool) -> EvalWeights {
    EvalWeights { couple_evasion: couple, ..EvalWeights::default() }
}

/// The evasion-wall position: bot Crobat active, opponent Snorlax active with
/// `eva`/`atk` boost stages (a Double-Team + Baton-Passed sweeper). 3v3 so the
/// eval is not material-saturated.
fn wall(dex: &Dex, eva: i8, atk: i8) -> Battle {
    let bot = vec![crobat_phazer(), porygon2(), marowak()];
    let opp = vec![snorlax(), porygon2(), marowak()];
    let mut b = start(dex, &bot, &opp);
    let snor = b.active_id(1).unwrap();
    b.poke_mut(snor).boosts[6] = eva;
    b.poke_mut(snor).boosts[0] = atk;
    b
}

// ---- 1. the engine accessor applies the gen-2 evasion stage --------------

#[test]
fn hit_probability_applies_evasion_stage() {
    let dex = load_dex();
    let b = wall(&dex, 0, 0);
    let cro = b.active_id(0).unwrap();
    let snor = b.active_id(1).unwrap();
    let sludge = mid(&dex, "sludgebomb"); // 100% base accuracy
    let swift = mid(&dex, "swift"); // never-miss

    // At 0 evasion a 100%-accuracy move never misses; a never-miss move is 1.0.
    let mut z = b.clone();
    z.poke_mut(snor).boosts[6] = 0;
    assert!((z.hit_probability(&dex, cro, snor, sludge) - 1.0).abs() < 1e-9);
    assert!((z.hit_probability(&dex, cro, snor, swift) - 1.0).abs() < 1e-9);

    // Hit chance falls monotonically as evasion climbs, reaching ~84/256 at +6.
    let mut prev = 1.01;
    for eva in [0i8, 2, 4, 6] {
        let mut e = b.clone();
        e.poke_mut(snor).boosts[6] = eva;
        let p = e.hit_probability(&dex, cro, snor, sludge);
        assert!(p < prev, "hit prob must fall with evasion (eva={eva})");
        prev = p;
    }
    let mut e6 = b.clone();
    e6.poke_mut(snor).boosts[6] = 6;
    assert!(
        (e6.hit_probability(&dex, cro, snor, sludge) - (84.0 / 256.0)).abs() < 5e-3,
        "gen-2 +6 evasion collapses a 100% move to 84/256"
    );
    // Never-miss ignores evasion entirely.
    assert!((e6.hit_probability(&dex, cro, snor, swift) - 1.0).abs() < 1e-9);
}

// ---- 2. the defect + contrast: the threat feature was blind to evasion ----

#[test]
fn threat_feature_blind_before_fix_collapses_after() {
    let dex = load_dex();
    let b0 = wall(&dex, 0, 4);
    let b6 = wall(&dex, 6, 4);
    let (cro0, snor0) = (b0.active_id(0).unwrap(), b0.active_id(1).unwrap());
    let (cro6, snor6) = (b6.active_id(0).unwrap(), b6.active_id(1).unwrap());

    // Pre-fix (base accuracy only): the +6-evasion wall looks EXACTLY as
    // hittable as the unboosted mon — the eval sees no danger. This is the bug.
    let old0 = best_hit_fraction(&b0, &dex, cro0, snor0, false);
    let old6 = best_hit_fraction(&b6, &dex, cro6, snor6, false);
    assert!((old0 - old6).abs() < 1e-9, "pre-fix threat is blind to evasion");

    // Post-fix: the same +6 wall collapses the bot's best hit fraction to well
    // under half — the eval now feels the danger.
    let new0 = best_hit_fraction(&b0, &dex, cro0, snor0, true);
    let new6 = best_hit_fraction(&b6, &dex, cro6, snor6, true);
    assert!((new0 - old0).abs() < 1e-9, "0-evasion unchanged by the fix");
    assert!(new6 < 0.5 * new0, "fix collapses threat vs +6 evasion ({new6} vs {new0})");
}

#[test]
fn rollout_policy_stops_throwing_attacks_at_the_wall() {
    // The heavy rollout ranks moves by `expected_hit_fraction`. Pre-fix it
    // preferred the strong STAB attack even against a +6-evasion wall (it
    // "believed" the attack lands); post-fix the never-miss move outranks the
    // attack that now whiffs 2/3 of the time.
    let dex = load_dex();
    let b = wall(&dex, 6, 4);
    let cro = b.active_id(0).unwrap();
    let snor = b.active_id(1).unwrap();
    let sludge = mid(&dex, "sludgebomb");
    let swift = mid(&dex, "swift");

    let sludge_old = expected_hit_fraction(&b, &dex, cro, snor, sludge, false);
    let swift_old = expected_hit_fraction(&b, &dex, cro, snor, swift, false);
    assert!(sludge_old > swift_old, "pre-fix: whiffing STAB outranks never-miss");

    let sludge_new = expected_hit_fraction(&b, &dex, cro, snor, sludge, true);
    let swift_new = expected_hit_fraction(&b, &dex, cro, snor, swift, true);
    assert!(swift_new > sludge_new, "post-fix: never-miss outranks the whiffing STAB");
}

// ---- 3. the eval feels danger, and phazing value emerges -----------------

#[test]
fn coupled_eval_feels_danger_and_rewards_phazing() {
    let dex = load_dex();
    // A non-saturated operating point (moderate atk boost) so the sigmoid is
    // responsive: measure the bot's (side-0) win estimate.
    let b6 = wall(&dex, 6, 2);
    let danger_new = eval::eval01(&b6, &dex, &weights(true));
    let danger_old = eval::eval01(&b6, &dex, &weights(false));
    // The physically-correct eval rates the +6-evasion position as strictly
    // more dangerous than the evasion-blind eval does.
    assert!(danger_new < danger_old, "coupled eval feels more danger ({danger_new} vs {danger_old})");

    // Phazing value emerges: resetting the opponent's evasion (what Roar /
    // Whirlwind achieve) recovers the bot's win estimate, and it recovers
    // MORE under the coupled eval than the evasion-blind one — the threat
    // channel is doing real work on top of the linear boost term.
    let mut reset = b6.clone();
    let snor = reset.active_id(1).unwrap();
    reset.poke_mut(snor).boosts[6] = 0;
    let recover_new = eval::eval01(&reset, &dex, &weights(true)) - danger_new;
    let recover_old = eval::eval01(&reset, &dex, &weights(false)) - danger_old;
    assert!(recover_new > 0.05, "phazing must produce a real eval jump ({recover_new})");
    assert!(recover_new > recover_old, "coupling adds phaze value ({recover_new} vs {recover_old})");
}

// ---- 4. the search discovers phazing instead of dithering ----------------

fn skuct_pick(dex: &Dex, b: &Battle, iters: u32, seed: u64) -> SearchChoice {
    let choices = b.clone().legal_choices(dex, 0);
    let cfg = RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        playout: Playout::Heavy { eps: 0.2, turns: 8, weights: weights(true) },
        ..Default::default()
    };
    RmAgent::new(cfg, seed).choose(b, dex, 0, &choices)
}

#[test]
fn fixed_search_phazes_the_evasion_wall() {
    // With the fix the bot, facing a +6-evasion Baton-Pass sweeper and holding
    // Whirlwind, resets the evasion rather than chipping into misses. Deter-
    // ministic per seed; asserted over several so it is not a lucky draw.
    let dex = load_dex();
    let whirlwind = mid(&dex, "whirlwind");
    let b = wall(&dex, 6, 4);
    let phazes = [1u64, 7, 42]
        .into_iter()
        .filter(|&s| skuct_pick(&dex, &b, 1500, s) == SearchChoice::Move(whirlwind))
        .count();
    assert!(phazes >= 2, "fixed search should phaze in a majority of seeds (got {phazes}/3)");
}
