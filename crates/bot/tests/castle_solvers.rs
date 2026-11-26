use nc2000_bot::bounds::{BoundCache, BoundConfig, BoundSolver};
use nc2000_bot::compiled_preview;
use nc2000_bot::endgame::{EndgameConfig, EndgameSession};
use nc2000_bot::eval::{
    castle_judge_value_battle, compact_eval01, compact_turn_limit_outcome, eval01,
    turn_limit_outcome_value, EvalWeights,
};
use nc2000_bot::exact::{ExactConfig, ExactSolver};
use nc2000_bot::hp_transition::HpTransitionKey;
use nc2000_bot::linear::{
    compact_perish_margin, survival_curve_4, toeplitz_coupled_race, DmgKernel1D,
};
use nc2000_bot::mask::{dominated_actions, public_dominated_actions};
use nc2000_bot::mcts::{MctsAgent, MctsConfig, Playout};
use nc2000_bot::relay::{apply_relay_leaf_lift, is_relay_eligible, relay_surplus_for_side};
use nc2000_bot::stall::{classify_two_sided_heal, ClassifyError};
use nc2000_bot::Agent;
use nc2000_engine::battle::{EffectHandle, PokemonSet, SearchChoice};
use nc2000_engine::compact::{vol_bit, CompactBattle, CompactWeather};
use nc2000_engine::dex::Dex;
use nc2000_engine::rules::{CastleOverlayConfig, Ruleset};
use nc2000_engine::state::{Battle, Gender, Status};

fn make_set(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
    make_set_ext(species, level, item, moves, "M", [30; 6])
}

fn make_set_ext(
    species: &str,
    level: u8,
    item: &str,
    moves: &[&str],
    gender: &str,
    ivs: [u8; 6],
) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": species,
        "species": species,
        "item": item,
        "ability": "No Ability",
        "moves": moves,
        "level": level,
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {
            "hp": ivs[0], "atk": ivs[1], "def": ivs[2],
            "spa": ivs[3], "spd": ivs[4], "spe": ivs[5]
        },
        "happiness": 255,
        "gender": gender
    }))
    .unwrap()
}

fn start_battle(
    dex: &Dex,
    ruleset: Ruleset,
    p1: &[PokemonSet],
    p2: &[PokemonSet],
) -> Battle {
    let mut b = Battle::from_fixture_with_ruleset(dex, "1,2,3,4", p1, p2, ruleset).unwrap();
    b.set_log_enabled(false);
    let pick1 = (1..=p1.len())
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let pick2 = (1..=p2.len())
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    b.choose(dex, 0, &format!("team {pick1}")).unwrap();
    b.choose(dex, 1, &format!("team {pick2}")).unwrap();
    b
}

fn mv(dex: &Dex, key: &str) -> SearchChoice {
    SearchChoice::Move(dex.moves.id(key).unwrap())
}

#[test]
fn ps_1v1_mutual_residual_ko_faster_wins_and_castle_sandstorm_draws() {
    let dex = conformance::load_dex();
    let fast_noheal = make_set("Zapdos", 55, "No Item", &["Splash"]);
    let slow_noheal = make_set("Snorlax", 50, "No Item", &["Splash"]);

    // 1. Non-healing Toeplitz path: PS faster wins & strictly antisymmetric; Castle Sandstorm draws.
    for (ruleset, expect_ps) in [
        (Ruleset::PsStadium2, true),
        (Ruleset::castle_tournament(), false),
    ] {
        let b01 = start_battle(&dex, ruleset, &[fast_noheal.clone()], &[slow_noheal.clone()]);
        let (t01, mut cb01) = CompactBattle::from_battle(&b01, &dex);
        cb01.weather_kind = CompactWeather::Sandstorm;
        cb01.weather_turns = 5;
        cb01.mons[0].hp_pool[0] = 1;
        cb01.mons[3].hp_pool[0] = 1;

        let b10 = start_battle(&dex, ruleset, &[slow_noheal.clone()], &[fast_noheal.clone()]);
        let (t10, mut cb10) = CompactBattle::from_battle(&b10, &dex);
        cb10.weather_kind = CompactWeather::Sandstorm;
        cb10.weather_turns = 5;
        cb10.mons[0].hp_pool[0] = 1;
        cb10.mons[3].hp_pool[0] = 1;

        let r01 = toeplitz_coupled_race(&cb01, &t01, 0, 0, 1, 0);
        let r10 = toeplitz_coupled_race(&cb10, &t10, 0, 0, 1, 0);
        if expect_ps {
            assert!(r01 > 0.99, "PS 1v1 Sandstorm faster side must win, got {r01}");
            assert!((r01 + r10).abs() < 1e-12, "PS antisymmetry: {r01} vs {r10}");
        } else {
            assert!(
                r01.abs() < 1e-6 && r10.abs() < 1e-6,
                "Castle 1v1 Sandstorm mutual KO must draw (0.0 margin), got {r01}, {r10}"
            );
        }
    }

    // 2. Bellman path (both have Rest, both guaranteed asleep at 1 HP in Sandstorm -> mutual EOT KO).
    let fast_heal = make_set("Zapdos", 55, "No Item", &["Splash", "Rest"]);
    let slow_heal = make_set("Snorlax", 50, "No Item", &["Splash", "Rest"]);
    for (ruleset, expect_ps) in [
        (Ruleset::PsStadium2, true),
        (Ruleset::castle_tournament(), false),
    ] {
        let b01 = start_battle(&dex, ruleset, &[fast_heal.clone()], &[slow_heal.clone()]);
        let (t01, mut cb01) = CompactBattle::from_battle(&b01, &dex);
        cb01.weather_kind = CompactWeather::Sandstorm;
        cb01.weather_turns = 5;
        cb01.mons[0].hp_pool[0] = 1;
        cb01.mons[3].hp_pool[0] = 1;
        cb01.mons[0].set_status(Status::Slp as u8, 3);
        cb01.mons[3].set_status(Status::Slp as u8, 3);

        let b10 = start_battle(&dex, ruleset, &[slow_heal.clone()], &[fast_heal.clone()]);
        let (t10, mut cb10) = CompactBattle::from_battle(&b10, &dex);
        cb10.weather_kind = CompactWeather::Sandstorm;
        cb10.weather_turns = 5;
        cb10.mons[0].hp_pool[0] = 1;
        cb10.mons[3].hp_pool[0] = 1;
        cb10.mons[0].set_status(Status::Slp as u8, 3);
        cb10.mons[3].set_status(Status::Slp as u8, 3);

        let r01 = toeplitz_coupled_race(&cb01, &t01, 0, 0, 1, 0);
        let r10 = toeplitz_coupled_race(&cb10, &t10, 0, 0, 1, 0);
        if expect_ps {
            assert!(
                r01 > 0.99,
                "PS Bellman 1v1 mutual residual KO must award win to faster side, got {r01}"
            );
            assert!(
                (r01 + r10).abs() < 1e-5,
                "PS Bellman antisymmetry violated: {r01} vs {r10}"
            );
        } else {
            assert!(
                r01.abs() < 1e-5 && r10.abs() < 1e-5,
                "Castle Bellman 1v1 Sandstorm mutual KO must draw, got {r01}, {r10}"
            );
        }
    }
}

#[test]
fn linear_and_relay_drain_half_blocked_by_substitute() {
    let dex = conformance::load_dex();
    let drain_att = make_set("Exeggutor", 55, "No Item", &["Giga Drain"]);
    let sub_def = make_set("Starmie", 50, "No Item", &["Ice Beam", "Recover"]);
    let b = start_battle(&dex, Ruleset::PsStadium2, &[drain_att], &[sub_def]);
    let (table, mut cb) = CompactBattle::from_battle(&b, &dex);

    let k_no_sub = DmgKernel1D::for_move(&cb, &table, 0, 0, 1, 0, 0);
    assert!(k_no_sub.max_dmg > 0, "Giga Drain deals damage without Sub");

    cb.mons[3].vol_flags |= vol_bit::SUBSTITUTE;
    cb.mons[3].hp_pool[1] = 35;

    let k_sub = DmgKernel1D::for_move(&cb, &table, 0, 0, 1, 0, 0);
    assert_eq!(k_sub.max_dmg, 0, "Giga Drain must deal 0 vs active Sub");
    assert_eq!(k_sub.mean_dmg, 0.0);
    assert_eq!(
        DmgKernel1D::best_attack(&cb, &table, 0, 0, 1, 0).max_dmg,
        0
    );
    assert_eq!(
        DmgKernel1D::best_attack_grid(&cb, &table, 0, 0, 1, 0).max_dmg,
        0
    );
    let r = toeplitz_coupled_race(&cb, &table, 0, 0, 1, 0);
    assert!(
        r < -0.5,
        "Giga Drain-only attacker stuck against Sub Starmie must lose race, got {r}"
    );

    // Relay solver: Sub + Swords Dance + BP passer vs Giga Drain-only defender vs Razor Leaf defender.
    let passer = make_set(
        "Scizor",
        50,
        "No Item",
        &["Substitute", "Swords Dance", "Baton Pass", "Steel Wing"],
    );
    let recv = make_set("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide"]);
    let foe_drain = make_set("Venusaur", 55, "No Item", &["Giga Drain"]);
    let foe_razor = make_set("Venusaur", 55, "No Item", &["Razor Leaf"]);
    let foe_bench = make_set("Blissey", 50, "No Item", &["Soft-Boiled", "Ice Beam"]);

    let b_drain = start_battle(
        &dex,
        Ruleset::PsStadium2,
        &[passer.clone(), recv.clone()],
        &[foe_drain, foe_bench.clone()],
    );
    let b_razor = start_battle(
        &dex,
        Ruleset::PsStadium2,
        &[passer, recv],
        &[foe_razor, foe_bench],
    );
    let (t_d, mut cb_d) = CompactBattle::from_battle(&b_drain, &dex);
    let (t_r, mut cb_r) = CompactBattle::from_battle(&b_razor, &dex);
    cb_d.mons[0].vol_flags |= vol_bit::SUBSTITUTE;
    cb_d.mons[0].hp_pool[1] = 35;
    cb_r.mons[0].vol_flags |= vol_bit::SUBSTITUTE;
    cb_r.mons[0].hp_pool[1] = 35;

    let w = EvalWeights::default();
    let s_drain = relay_surplus_for_side(&cb_d, &t_d, &w, 0, 0.5);
    let s_razor = relay_surplus_for_side(&cb_r, &t_r, &w, 0, 0.5);
    assert!(
        s_drain >= s_razor,
        "Relay behind Sub vs Giga Drain-only ({s_drain}) must be at least as strong as vs Razor Leaf ({s_razor})"
    );
}

#[test]
fn linear_toxic_residualdmg_counter_and_castle_burn_isolation() {
    let dex = conformance::load_dex();
    let a = make_set("Snorlax", 55, "No Item", &["Splash"]);
    let b_mon = make_set("Snorlax", 55, "No Item", &["Splash"]);

    // 1. Fresh Toxic (`cnt = 0x10 | 0`): first tick is 1/16, NOT 2/16!
    let battle_ps = start_battle(&dex, Ruleset::PsStadium2, &[a.clone()], &[b_mon.clone()]);
    let (t_ps, mut cb_ps) = CompactBattle::from_battle(&battle_ps, &dex);
    let maxhp = t_ps.mons[3].maxhp;
    let tick1 = (maxhp / 16).max(1);
    let tick2 = ((maxhp / 16) * 2).max(1);
    cb_ps.mons[3].set_status(Status::Tox as u8, 0x10 | 0);
    // HP strictly between 1/16 and 2/16 -> must survive turn 1 (1/16), faint turn 2 (2/16).
    let start_hp = tick1 + 2;
    assert!(start_hp < tick2);
    let s = survival_curve_4(&cb_ps, &t_ps, 0, 1, start_hp);
    assert!(
        (s[1] - 1.0).abs() < 1e-6,
        "First Toxic tick at c=0 must deal 1/16 ({tick1}), surviving from {start_hp}, got S(1)={}",
        s[1]
    );
    assert!(
        s[2] < 1e-6,
        "Second Toxic tick at c=2 ({tick2}) must KO remaining HP, got S(2)={}",
        s[2]
    );

    // Switched-out bench Toxic (`cnt = 0x00`, residualdmg cleared) converts to regular Psn (1/8)
    // on switch-in (`ToxSwitchIn`), so at `start_hp < tick2` it faints on Turn 1.
    cb_ps.mons[3].set_status(Status::Tox as u8, 0x00);
    let s_bench_tox = survival_curve_4(&cb_ps, &t_ps, 0, 1, start_hp);
    assert!(
        s_bench_tox[1] < 1e-6,
        "Switched-out bench Toxic (cnt=0x00) converts to Psn (1/8) and KOs at {start_hp}, got S(1)={}",
        s_bench_tox[1]
    );

    // 2. Castle Burn ignores passed `residualdmg` bit (0x10 | 6), PS applies 7/16!
    let battle_cs = start_battle(&dex, Ruleset::castle_tournament(), &[a], &[b_mon]);
    let (t_cs, mut cb_cs) = CompactBattle::from_battle(&battle_cs, &dex);
    cb_ps.mons[3].set_status(Status::Brn as u8, 0x10 | 6);
    cb_cs.mons[3].set_status(Status::Brn as u8, 0x10 | 6);
    let brn_1_8 = (maxhp / 8).max(1);
    let hp_mid = brn_1_8 + 5; // survives 1/8 burn, faints to 7/16 residualdmg!
    let s_ps = survival_curve_4(&cb_ps, &t_ps, 0, 1, hp_mid);
    let s_cs = survival_curve_4(&cb_cs, &t_cs, 0, 1, hp_mid);
    assert!(
        s_ps[1] < 1e-6,
        "PS Burn with residualdmg c=6 must deal 7/16 and KO at {hp_mid}, got {}",
        s_ps[1]
    );
    assert!(
        (s_cs[1] - 1.0).abs() < 1e-6,
        "Castle Burn ignores residualdmg and deals 1/8 ({brn_1_8}), surviving at {hp_mid}, got {}",
        s_cs[1]
    );
}

#[test]
fn linear_brightpowder_exact_minus_20_over_256_and_castle_accuracy_table() {
    let dex = conformance::load_dex();
    let att = make_set("Zapdos", 55, "No Item", &["Thunder", "Thunderbolt"]);
    let def_bp = make_set("Snorlax", 55, "Bright Powder", &["Splash"]);

    let b_ps = start_battle(&dex, Ruleset::PsStadium2, &[att.clone()], &[def_bp.clone()]);
    let (t_ps, mut cb_ps) = CompactBattle::from_battle(&b_ps, &dex);
    let k_thunder = DmgKernel1D::for_move(&cb_ps, &t_ps, 0, 0, 1, 0, 0);
    let p_miss = k_thunder
        .pts
        .iter()
        .take(k_thunder.len)
        .find(|&&(dmg, _)| dmg == 0)
        .map_or(0.0, |&(_, p)| p);
    // Thunder 70%: floor(70 * 255 / 100) = 178; BrightPowder subtracts exact 20 -> 158/256.
    let expected_miss = 1.0 - (158.0 / 256.0);
    assert!(
        (p_miss - expected_miss).abs() < 1e-5,
        "BrightPowder must subtract exact 20/256 (miss={expected_miss}), got {p_miss}"
    );

    // Castle exact 3/8 vs PS GB LUT (36%) at -5 accuracy stage on 100% move (Thunderbolt, slot 1).
    let def_plain = make_set("Snorlax", 55, "No Item", &["Splash"]);
    let b_ps2 = start_battle(&dex, Ruleset::PsStadium2, &[att.clone()], &[def_plain.clone()]);
    let (t_ps2, mut cb_ps2) = CompactBattle::from_battle(&b_ps2, &dex);
    let b_cs2 = start_battle(&dex, Ruleset::castle_tournament(), &[att], &[def_plain]);
    let (t_cs2, mut cb_cs2) = CompactBattle::from_battle(&b_cs2, &dex);
    cb_ps2.mons[0].boosts[5] = -5;
    cb_cs2.mons[0].boosts[5] = -5;
    let miss_ps = DmgKernel1D::for_move(&cb_ps2, &t_ps2, 0, 0, 1, 0, 1)
        .pts
        .iter()
        .find(|&&(d, _)| d == 0)
        .map_or(0.0, |&(_, p)| p);
    let miss_cs = DmgKernel1D::for_move(&cb_cs2, &t_cs2, 0, 0, 1, 0, 1)
        .pts
        .iter()
        .find(|&&(d, _)| d == 0)
        .map_or(0.0, |&(_, p)| p);
    // PS: floor(255 * 36 / 100) = 91 -> 91/256; Castle: floor(255 * 3 / 8) = 95 -> 95/256.
    assert!((miss_ps - (1.0 - 91.0 / 256.0)).abs() < 1e-5, "PS -5 acc miss={miss_ps}");
    assert!((miss_cs - (1.0 - 95.0 / 256.0)).abs() < 1e-5, "Castle -5 acc miss={miss_cs}");
    // Prevent unused mut warning
    cb_ps.turn = 1;
}

#[test]
fn linear_castle_mechanics_crit_explosion_neg_stage_berry_recoil_speed_sleep_and_seat_asymmetry() {
    let dex = conformance::load_dex();

    // 1. Crit rate 17/256 (Castle) > 1/16 (PS) on identical clean attack.
    let lax = make_set("Snorlax", 50, "No Item", &["Body Slam"]);
    let skarm = make_set("Skarmory", 50, "No Item", &["Splash"]);
    let (t_ps, cb_ps) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[lax.clone()], &[skarm.clone()]),
        &dex,
    );
    let (t_cs, cb_cs) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[lax], &[skarm]),
        &dex,
    );
    let k_ps = DmgKernel1D::for_move(&cb_ps, &t_ps, 0, 0, 1, 0, 0);
    let k_cs = DmgKernel1D::for_move(&cb_cs, &t_cs, 0, 0, 1, 0, 0);
    assert!(
        k_cs.mean_dmg > k_ps.mean_dmg,
        "Castle 17/256 crit rate must yield higher mean damage than PS 1/16 ({} vs {})",
        k_cs.mean_dmg,
        k_ps.mean_dmg
    );

    // 2. Explosion Def/2 BEFORE >=256 /4 (Probe 12E2): Chansey Explosion vs Cloyster Lv60 (Def >= 256).
    let chansey = make_set("Chansey", 55, "No Item", &["Explosion"]);
    let cloyster = make_set("Cloyster", 60, "No Item", &["Splash"]);
    let (t_ex_ps, cb_ex_ps) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[chansey.clone()], &[cloyster.clone()]),
        &dex,
    );
    let (t_ex_cs, cb_ex_cs) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[chansey], &[cloyster]),
        &dex,
    );
    // Verify Cloyster Def >= 256 so the branch actually distinguishes!
    assert!(t_ex_ps.mons[3].stored_stats[1] >= 256);
    let ex_ps = DmgKernel1D::for_move(&cb_ex_ps, &t_ex_ps, 0, 0, 1, 0, 0);
    let ex_cs = DmgKernel1D::for_move(&cb_ex_cs, &t_ex_cs, 0, 0, 1, 0, 0);
    assert!(
        ex_cs.mean_dmg > ex_ps.mean_dmg + 1.0,
        "Castle Def/2-before-/4 Explosion ({}) must strictly exceed PS ({})",
        ex_cs.mean_dmg,
        ex_ps.mean_dmg
    );

    // 3. Negative stat stage exact 2/3 vs PS 66% (Probe 12B).
    let blissey_att = make_set_ext("Blissey", 5, "No Item", &["Explosion"], "F", [0; 6]);
    let blissey_def = make_set_ext("Blissey", 10, "No Item", &["Splash"], "F", [0; 6]);
    let (t_ns_ps, mut cb_ns_ps) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[blissey_att.clone()], &[blissey_def.clone()]),
        &dex,
    );
    let (t_ns_cs, mut cb_ns_cs) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[blissey_att], &[blissey_def]),
        &dex,
    );
    cb_ns_ps.mons[0].boosts[0] = -1;
    cb_ns_cs.mons[0].boosts[0] = -1;
    let ns_ps = DmgKernel1D::for_move(&cb_ns_ps, &t_ns_ps, 0, 0, 1, 0, 0);
    let ns_cs = DmgKernel1D::for_move(&cb_ns_cs, &t_ns_cs, 0, 0, 1, 0, 0);
    assert!(
        ns_cs.mean_dmg > ns_ps.mean_dmg,
        "Castle -1 exact 2/3 ({}) must exceed PS 66% ({})",
        ns_cs.mean_dmg,
        ns_ps.mean_dmg
    );

    // 4. Healing berry threshold: Castle strict `< floor(maxhp/2)` vs PS `<= floor(maxhp/2)`.
    let hitter = make_set("Alakazam", 50, "No Item", &["Seismic Toss"]); // exact 50 dmg
    let berry_def = make_set("Starmie", 50, "Gold Berry", &["Splash"]);
    let (t_gb_ps, cb_gb_ps) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[hitter.clone()], &[berry_def.clone()]),
        &dex,
    );
    let (t_gb_cs, cb_gb_cs) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[hitter], &[berry_def]),
        &dex,
    );
    let mx = t_gb_ps.mons[3].maxhp;
    // Start at `floor(mx/2) + 50`: after 1 Seismic Toss, HP is EXACTLY `floor(mx/2)`.
    // PS triggers Gold Berry (+30) so defender survives 3 hits longer than Castle (no trigger on hit 1).
    let start_exact = mx / 2 + 50;
    let sp = survival_curve_4(&cb_gb_ps, &t_gb_ps, 0, 1, start_exact);
    let sc = survival_curve_4(&cb_gb_cs, &t_gb_cs, 0, 1, start_exact);
    // At step 3 (3 * 50 = 150 dmg): starting from mx/2 + 50 (Starmie Lv50 maxhp=167 -> 83+50=133):
    // PS: 133 -> 83 -> +30 = 113 -> 63 -> 13 (survives step 3 = 1.0!).
    // Castle: 133 -> 83 (83 < 83 false!) -> 33 -> 0 (faints step 3 = 0.0!).
    assert!(
        sp[3] > sc[3] + 0.5,
        "PS Gold Berry must trigger at exact floor(maxhp/2) (PS={}, Castle={})",
        sp[3],
        sc[3]
    );

    // 5. Recoil vs Sub (0 recoil in Castle vs 1 HP recoil in PS) & 1v1 recoil double-KO overlay.
    let de_att = make_set("Snorlax", 55, "No Item", &["Double-Edge"]);
    let sub_foe = make_set("Snorlax", 50, "No Item", &["Splash", "Rest"]);
    let (t_rec_ps, mut cb_rec_ps) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[de_att.clone()], &[sub_foe.clone()]),
        &dex,
    );
    let (t_rec_ct, mut cb_rec_ct) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[de_att.clone()], &[sub_foe.clone()]),
        &dex,
    );
    let (t_rec_cr, mut cb_rec_cr) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_raw_cgi(false), &[de_att], &[sub_foe]),
        &dex,
    );
    // 1v1 double-KO via Double-Edge: attacker at 1 HP, slower defender at 10 HP (no Sub).
    cb_rec_ps.mons[0].hp_pool[0] = 1;
    cb_rec_ps.mons[3].hp_pool[0] = 10;
    cb_rec_ct.mons[0].hp_pool[0] = 1;
    cb_rec_ct.mons[3].hp_pool[0] = 10;
    cb_rec_cr.mons[0].hp_pool[0] = 1;
    cb_rec_cr.mons[3].hp_pool[0] = 10;
    let r_rec_ps = toeplitz_coupled_race(&cb_rec_ps, &t_rec_ps, 0, 0, 1, 0);
    let r_rec_ct = toeplitz_coupled_race(&cb_rec_ct, &t_rec_ct, 0, 0, 1, 0);
    let r_rec_cr = toeplitz_coupled_race(&cb_rec_cr, &t_rec_cr, 0, 0, 1, 0);
    assert!(r_rec_ps > 0.9, "PS 1v1 recoil KO skips recoil -> attacker wins, got {r_rec_ps}");
    assert!(r_rec_ct > 0.9, "Castle tournament recoil_1v1_attacker_wins -> attacker wins, got {r_rec_ct}");
    assert!(
        r_rec_cr < r_rec_ct - 0.3,
        "Castle raw CGI recoil double-KO draws (0.0 margin vs tournament {r_rec_ct}), got {r_rec_cr}"
    );

    // 6. Non-truncated Castle speed (Probe 5B/12H): Paralyzed Spe 123 (30.75) vs Paralyzed Spe 121 (30.25).
    // Lv50 Gengar max Spe = 162; adjust Spe IVs so two Gengars have Spe differing by 2 inside the same /4 bucket!
    let g_hi = make_set_ext("Alakazam", 50, "No Item", &["Psychic"], "M", [30, 30, 30, 30, 30, 26]);
    let g_lo = make_set_ext("Alakazam", 50, "No Item", &["Psychic"], "M", [30, 30, 30, 30, 30, 24]);
    let (t_sp_ps, mut cb_sp_ps) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[g_hi.clone()], &[g_lo.clone()]),
        &dex,
    );
    let (t_sp_cs, mut cb_sp_cs) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[g_hi], &[g_lo]),
        &dex,
    );
    let s0_raw = t_sp_ps.mons[0].stored_stats[4];
    let s1_raw = t_sp_ps.mons[3].stored_stats[4];
    assert!(s0_raw > s1_raw && s0_raw / 4 == s1_raw / 4, "need same /4 bucket: {s0_raw} vs {s1_raw}");
    for cb in [&mut cb_sp_ps, &mut cb_sp_cs] {
        cb.mons[0].set_status(Status::Par as u8, 0);
        cb.mons[0].vol_flags |= vol_bit::PAR_SPEED_DROP;
        cb.mons[3].set_status(Status::Par as u8, 0);
        cb.mons[3].vol_flags |= vol_bit::PAR_SPEED_DROP;
        cb.mons[0].hp_pool[0] = 1;
        cb.mons[3].hp_pool[0] = 1;
    }
    let r_sp_ps = toeplitz_coupled_race(&cb_sp_ps, &t_sp_ps, 0, 0, 1, 0);
    let r_sp_cs = toeplitz_coupled_race(&cb_sp_cs, &t_sp_cs, 0, 0, 1, 0);
    assert!(r_sp_ps.abs() < 1e-6, "PS truncates both paralyzed speeds -> tie (0.0), got {r_sp_ps}");
    assert!(r_sp_cs > 0.5, "Castle exact rational speed makes higher Spe win, got {r_sp_cs}");

    // 7. 4-turn Castle sleep table distinguishes `status_counter = 5` (4 skipped turns) vs `4` (3 skipped turns).
    let slp_mon = make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Rest"]);
    let foe_mon = make_set("Machamp", 50, "No Item", &["Cross Chop"]);
    let (t_slp_cs, mut cb_slp_cs) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[slp_mon], &[foe_mon]),
        &dex,
    );
    cb_slp_cs.mons[0].hp_pool[0] = t_slp_cs.mons[0].maxhp;
    cb_slp_cs.mons[0].set_status(Status::Slp as u8, 4);
    let r_slp4 = toeplitz_coupled_race(&cb_slp_cs, &t_slp_cs, 0, 0, 1, 0);
    cb_slp_cs.mons[0].set_status(Status::Slp as u8, 5);
    let r_slp5 = toeplitz_coupled_race(&cb_slp_cs, &t_slp_cs, 0, 0, 1, 0);
    assert!(
        r_slp5 < r_slp4,
        "Castle 4-turn sleep (counter=5, {r_slp5}) must be worse than 3-turn sleep (counter=4, {r_slp4})"
    );

    // 8. Equal-Perish 1v1 adjudication: PS faster wins (antisymmetric); Castle honors
    //    `perish_dbond_lastmon_forfeit` caster bit (`turn_extra_flags` bit 2/3) in 1v1,
    //    draws (`0.0`) under `raw_cgi` or ambiguous caster, and has zero seat bias in non-1v1.
    let p_fast = make_set("Zapdos", 55, "No Item", &["Splash"]);
    let p_slow = make_set("Snorlax", 50, "No Item", &["Splash"]);
    let (t_per_ps01, mut cb_per_ps01) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[p_fast.clone()], &[p_slow.clone()]),
        &dex,
    );
    let (t_per_ps10, mut cb_per_ps10) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::PsStadium2, &[p_slow.clone()], &[p_fast.clone()]),
        &dex,
    );
    let (t_per_cs01, mut cb_per_cs01) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_tournament(), &[p_fast.clone()], &[p_slow.clone()]),
        &dex,
    );
    let (t_per_raw, mut cb_per_raw) = CompactBattle::from_battle(
        &start_battle(&dex, Ruleset::castle_raw_cgi(false), &[p_fast.clone()], &[p_slow.clone()]),
        &dex,
    );
    let (t_per_2v2, mut cb_per_2v2) = CompactBattle::from_battle(
        &start_battle(
            &dex,
            Ruleset::castle_tournament(),
            &[p_fast.clone(), p_slow.clone()],
            &[p_slow.clone(), p_fast.clone()],
        ),
        &dex,
    );
    for cb in [
        &mut cb_per_ps01,
        &mut cb_per_ps10,
        &mut cb_per_cs01,
        &mut cb_per_raw,
        &mut cb_per_2v2,
    ] {
        cb.mons[0].vol_flags |= vol_bit::PERISH_SONG;
        cb.mons[0].set_perish_turns(1);
        cb.mons[3].vol_flags |= vol_bit::PERISH_SONG;
        cb.mons[3].set_perish_turns(1);
    }
    let m_ps01 = compact_perish_margin(&cb_per_ps01, &t_per_ps01);
    let m_ps10 = compact_perish_margin(&cb_per_ps10, &t_per_ps10);
    assert!(m_ps01 > 0.9 && (m_ps01 + m_ps10).abs() < 1e-12, "PS perish antisymmetric: {m_ps01} vs {m_ps10}");

    // Castle 1v1: no caster bit -> draw (0.0); P1 caster -> -1.0; P2 caster -> +1.0; raw CGI -> 0.0.
    assert_eq!(compact_perish_margin(&cb_per_cs01, &t_per_cs01), 0.0);
    cb_per_cs01.turn_extra_flags = 1 << 2; // P1 cast Perish Song
    assert!(
        compact_perish_margin(&cb_per_cs01, &t_per_cs01) < -0.9,
        "Castle 1v1 P1-caster Perish must forfeit P1"
    );
    cb_per_cs01.turn_extra_flags = 1 << 3; // P2 cast Perish Song
    assert!(
        compact_perish_margin(&cb_per_cs01, &t_per_cs01) > 0.9,
        "Castle 1v1 P2-caster Perish must forfeit P2 (+margin for P1)"
    );
    cb_per_raw.turn_extra_flags = 1 << 2;
    assert_eq!(
        compact_perish_margin(&cb_per_raw, &t_per_raw),
        0.0,
        "Castle raw CGI without perish_dbond_lastmon_forfeit must draw (0.0)"
    );
    cb_per_2v2.turn_extra_flags = 1 << 2;
    cb_per_2v2.mons[0].vol_flags |= vol_bit::TRAPPED_OR_LOCKED;
    cb_per_2v2.mons[3].vol_flags |= vol_bit::TRAPPED_OR_LOCKED;
    assert_eq!(
        compact_perish_margin(&cb_per_2v2, &t_per_2v2),
        0.0,
        "Castle non-1v1 equal Perish must have zero seat bias"
    );
}

#[test]
fn relay_bp_receiver_after_switch_in_burn_poison_and_castle_pursuit_and_turn_limit() {
    let dex = conformance::load_dex();
    let passer = make_set(
        "Jolteon",
        55,
        "No Item",
        &["Agility", "Substitute", "Baton Pass", "Thunderbolt"],
    );
    let receiver = make_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide"]);
    let foe = make_set("Snorlax", 50, "No Item", &["Body Slam"]);

    let b = start_battle(
        &dex,
        Ruleset::PsStadium2,
        &[passer.clone(), receiver.clone()],
        &[foe.clone()],
    );
    let (table, mut cb) = CompactBattle::from_battle(&b, &dex);
    let recv_maxhp = table.mons[1].maxhp;
    let brn_slip = (recv_maxhp / 8).max(1);

    // Burned receiver with HP == floor(maxhp/8) faints on Baton Pass switch-in (no Spikes needed).
    cb.mons[1].set_status(Status::Brn as u8, 0);
    cb.mons[1].hp_pool[0] = brn_slip;
    assert!(
        !is_relay_eligible(&cb, &table, 0),
        "Receiver fainting to AfterSwitchInSelf burn slip ({brn_slip}) must make relay ineligible"
    );

    // 1 HP above burn slip -> survives entry -> eligible!
    cb.mons[1].hp_pool[0] = brn_slip + 1;
    assert!(
        is_relay_eligible(&cb, &table, 0),
        "Receiver surviving AfterSwitchInSelf burn slip must be relay-eligible"
    );

    // Add 1 layer of Spikes (maxhp/8) + Poison (maxhp/8) -> lethal at 2 * slip!
    cb.sides[0].side_flags = 1;
    cb.mons[1].set_status(Status::Psn as u8, 0);
    cb.mons[1].hp_pool[0] = brn_slip * 2;
    assert!(
        !is_relay_eligible(&cb, &table, 0),
        "Receiver fainting to Spikes + AfterSwitchInSelf poison must make relay ineligible"
    );
    cb.mons[1].hp_pool[0] = brn_slip * 2 + 1;
    assert!(is_relay_eligible(&cb, &table, 0));

    // Castle 2x Pursuit after fast Baton Pass (Probe 12C) & L_TURN=80 damping.
    let alakazam = make_set("Alakazam", 50, "No Item", &["Psychic", "Thunder Punch"]);
    let pursuiter = make_set("Tyranitar", 50, "No Item", &["Pursuit"]);
    let b_ps_p = start_battle(
        &dex,
        Ruleset::PsStadium2,
        &[passer.clone(), alakazam.clone()],
        &[pursuiter.clone()],
    );
    let b_cs_p = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[passer, alakazam],
        &[pursuiter],
    );
    let (t_ps_p, mut cb_ps_p) = CompactBattle::from_battle(&b_ps_p, &dex);
    let (t_cs_p, mut cb_cs_p) = CompactBattle::from_battle(&b_cs_p, &dex);
    cb_ps_p.mons[0].boosts[4] = 2;
    cb_cs_p.mons[0].boosts[4] = 2;
    let w = EvalWeights::default();
    let surp_ps = relay_surplus_for_side(&cb_ps_p, &t_ps_p, &w, 0, 0.5);
    let surp_cs = relay_surplus_for_side(&cb_cs_p, &t_cs_p, &w, 0, 0.5);
    assert!(
        surp_cs <= surp_ps,
        "Castle 2x Pursuit on fast BP receiver ({surp_cs}) must not exceed PS ({surp_ps})"
    );

    // Bot 80T adaptation is abolished: relay leaf lift does not damp by turn number.
    let v_base = 0.42;
    cb_cs_p.turn = 10;
    let lift_early = apply_relay_leaf_lift(&cb_cs_p, &t_cs_p, &w, v_base);
    cb_cs_p.turn = 80;
    let lift_late = apply_relay_leaf_lift(&cb_cs_p, &t_cs_p, &w, v_base);
    assert!(
        (lift_early - lift_late).abs() < 1e-12,
        "Relay leaf lift must be turn-invariant after abolishing 80T adaptation ({lift_early} vs {lift_late})"
    );
}

#[test]
fn mask_transformed_ditto_hp_and_all_castle_mask_rules() {
    let dex = conformance::load_dex();

    // 1. Transformed Ditto Hidden Power type: full-world masks immune copied type; public does NOT.
    let ditto = make_set("Ditto", 50, "No Item", &["Hidden Power", "Tackle"]);
    let skarm = make_set("Skarmory", 50, "No Item", &["Drill Peck"]);
    let mut b_ditto = start_battle(&dex, Ruleset::PsStadium2, &[ditto], &[skarm]);
    let att = b_ditto.active_id(0).unwrap();
    b_ditto.poke_mut(att).transformed = true;
    b_ditto.poke_mut(att).hp_type = dex.known_types.ground;
    let hp_choice = mv(&dex, "hiddenpower");
    let dom = dominated_actions(&b_ditto, &dex, 0);
    let pub_dom = public_dominated_actions(&b_ditto, &dex, 0);
    assert!(
        dom.iter().any(|&(c, _)| c == hp_choice),
        "Full-world mask must exclude transformed Ditto Ground Hidden Power vs Flying Skarmory"
    );
    assert!(
        !pub_dom.iter().any(|&(c, _)| c == hp_choice),
        "Public mask must keep transformed Ditto Hidden Power unmasked (hidden DVs)"
    );

    // 2. Castle Sleep Clause: foe sleep move & own Rest are NO-OPs (never forfeits!).
    let sleeper = make_set("Exeggutor", 50, "No Item", &["Sleep Powder", "Psychic", "Rest", "Reflect"]);
    let ally = make_set("Snorlax", 50, "No Item", &["Body Slam"]);
    let foe1 = make_set("Starmie", 50, "No Item", &["Surf"]);
    let foe2 = make_set("Blissey", 50, "No Item", &["Ice Beam"]);

    let mut b_slp_ps = start_battle(
        &dex,
        Ruleset::PsStadium2,
        &[sleeper.clone(), ally.clone()],
        &[foe1.clone(), foe2.clone()],
    );
    let mut b_slp_cs = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[sleeper, ally],
        &[foe1, foe2],
    );
    for b in [&mut b_slp_ps, &mut b_slp_cs] {
        // Put foe bench (slot 1) asleep and appeared; poison active foe so wait_exempt == true!
        let f_bench = b.sides[1].party[1] as usize;
        b.sides[1].roster[f_bench].status = Status::Slp;
        b.sides[1].roster[f_bench].previously_switched_in = 1;
        let f_act = b.active_id(1).unwrap();
        b.poke_mut(f_act).status = Status::Psn;
    }
    let sp_choice = mv(&dex, "sleeppowder");
    let ps_dom = dominated_actions(&b_slp_ps, &dex, 0);
    let cs_dom_wait = dominated_actions(&b_slp_cs, &dex, 0);
    assert!(
        ps_dom
            .iter()
            .any(|&(c, why)| c == sp_choice && why.contains("forfeit")),
        "PS Sleep Clause must mask Sleep Powder as forfeit even under wait_exempt"
    );
    assert!(
        !cs_dom_wait.iter().any(|&(c, _)| c == sp_choice),
        "Castle Sleep Clause is a no-op (not forfeit), so wait_exempt keeps Sleep Powder unmasked"
    );
    // Clear foe poison -> wait_exempt == false -> Castle masks Sleep Powder as no-op (NOT forfeit)!
    let f_act = b_slp_cs.active_id(1).unwrap();
    b_slp_cs.poke_mut(f_act).status = Status::None;
    let cs_dom = dominated_actions(&b_slp_cs, &dex, 0);
    assert!(
        cs_dom
            .iter()
            .any(|&(c, why)| c == sp_choice && !why.contains("forfeit")),
        "Castle Sleep Clause must mask Sleep Powder as no-op when not wait-exempt: {cs_dom:?}"
    );
    // Own bench ally asleep -> Castle still permits self-inflicted Rest (matches Battle/CompactBattle)!
    let own_bench = b_slp_cs.sides[0].party[1] as usize;
    b_slp_cs.sides[0].roster[own_bench].status = Status::Slp;
    let my_act = b_slp_cs.active_id(0).unwrap();
    b_slp_cs.poke_mut(my_act).hp -= 30; // damaged so full-HP heal rule does NOT fire
    let rest_choice = mv(&dex, "rest");
    let cs_rest_dom = dominated_actions(&b_slp_cs, &dex, 0);
    assert!(
        !cs_rest_dom.iter().any(|&(c, _)| c == rest_choice),
        "Castle Sleep Clause only blocks foe sleep, NOT own Rest while an ally is asleep: {cs_rest_dom:?}"
    );

    // 3. Castle Rapid Spin vs Ghost (Probe 12D2): NOT masked with Spikes/Leech Seed; masked with Mean Look only.
    let spinner = make_set("Starmie", 50, "No Item", &["Rapid Spin", "Surf"]);
    let ghost = make_set("Gengar", 50, "No Item", &["Thunderbolt"]);
    let mut b_rs_ps = start_battle(&dex, Ruleset::PsStadium2, &[spinner.clone()], &[ghost.clone()]);
    let mut b_rs_cs = start_battle(&dex, Ruleset::castle_tournament(), &[spinner], &[ghost]);
    let att_ps = b_rs_ps.active_id(0).unwrap();
    let att_cs = b_rs_cs.active_id(0).unwrap();
    b_rs_ps.add_side_condition(&dex, 0, "spikes", Some(att_ps), EffectHandle::None);
    b_rs_cs.add_side_condition(&dex, 0, "spikes", Some(att_cs), EffectHandle::None);
    let rs_choice = mv(&dex, "rapidspin");
    assert!(
        dominated_actions(&b_rs_ps, &dex, 0)
            .iter()
            .any(|&(c, _)| c == rs_choice),
        "PS masks Rapid Spin into lone Ghost even with Spikes"
    );
    assert!(
        !dominated_actions(&b_rs_cs, &dex, 0)
            .iter()
            .any(|&(c, _)| c == rs_choice),
        "Castle must NOT mask Rapid Spin into Ghost when Spikes are up"
    );
    assert!(
        !public_dominated_actions(&b_rs_cs, &dex, 0)
            .iter()
            .any(|&(c, _)| c == rs_choice),
        "Castle public mask must NOT exclude Rapid Spin into Ghost when Spikes are up"
    );
    // Remove Spikes, set trapped (Mean Look) only -> Castle MUST mask Rapid Spin (Probe 12D2).
    b_rs_cs.sides[0].side_conditions.clear();
    b_rs_cs.poke_mut(att_cs).trapped = true;
    assert!(
        dominated_actions(&b_rs_cs, &dex, 0)
            .iter()
            .any(|&(c, _)| c == rs_choice),
        "Castle Rapid Spin does not clear Mean Look -> masked vs Ghost when no Spikes/Leech/Bind"
    );
    // Add Leech Seed -> Castle must NOT mask Rapid Spin.
    b_rs_cs.add_volatile(&dex, att_cs, "leechseed", Some(att_cs), EffectHandle::None);
    assert!(
        !dominated_actions(&b_rs_cs, &dex, 0)
            .iter()
            .any(|&(c, _)| c == rs_choice),
        "Castle Rapid Spin clears Leech Seed vs Ghost -> must NOT be masked"
    );

    // 4. Castle Belly Drum exact half HP threshold (`hp == floor(maxhp / 2)` succeeds in Castle!).
    let drummer = make_set("Snorlax", 55, "No Item", &["Belly Drum", "Body Slam"]);
    let target = make_set("Steelix", 50, "No Item", &["Earthquake"]);
    let mut b_bd_ps = start_battle(&dex, Ruleset::PsStadium2, &[drummer.clone()], &[target.clone()]);
    let mut b_bd_cs = start_battle(&dex, Ruleset::castle_tournament(), &[drummer], &[target]);
    let bd_choice = mv(&dex, "bellydrum");
    for b in [&mut b_bd_ps, &mut b_bd_cs] {
        let a = b.active_id(0).unwrap();
        let half = b.poke(a).maxhp / 2;
        b.poke_mut(a).hp = half;
    }
    assert!(
        dominated_actions(&b_bd_ps, &dex, 0)
            .iter()
            .any(|&(c, _)| c == bd_choice),
        "PS masks Belly Drum at exact hp == floor(maxhp/2)"
    );
    assert!(
        !dominated_actions(&b_bd_cs, &dex, 0)
            .iter()
            .any(|&(c, _)| c == bd_choice),
        "Castle allows Belly Drum at exact hp == floor(maxhp/2)"
    );
    let a = b_bd_cs.active_id(0).unwrap();
    b_bd_cs.poke_mut(a).hp -= 1;
    assert!(
        dominated_actions(&b_bd_cs, &dex, 0)
            .iter()
            .any(|&(c, _)| c == bd_choice),
        "Castle masks Belly Drum when hp < floor(maxhp/2)"
    );

    // 5. Sketch vs Sub (blocked in PS, bypasses in Castle) & Ghost Curse vs Sub (blocked in both).
    let smeargle = make_set("Gengar", 50, "No Item", &["Sketch", "Curse", "Thunderbolt"]);
    let sub_target = make_set("Snorlax", 50, "No Item", &["Body Slam"]);
    let mut b_sk_ps = start_battle(&dex, Ruleset::PsStadium2, &[smeargle.clone()], &[sub_target.clone()]);
    let mut b_sk_cs = start_battle(&dex, Ruleset::castle_tournament(), &[smeargle], &[sub_target]);
    for b in [&mut b_sk_ps, &mut b_sk_cs] {
        let def = b.active_id(1).unwrap();
        b.add_volatile(&dex, def, "substitute", Some(def), EffectHandle::None);
    }
    let sketch_c = mv(&dex, "sketch");
    let curse_c = mv(&dex, "curse");
    assert!(
        dominated_actions(&b_sk_ps, &dex, 0).iter().any(|&(c, _)| c == sketch_c),
        "PS blocks Sketch vs Substitute"
    );
    assert!(
        !dominated_actions(&b_sk_cs, &dex, 0).iter().any(|&(c, _)| c == sketch_c),
        "Castle allows Sketch through Substitute (Probe 12D1)"
    );
    assert!(
        dominated_actions(&b_sk_ps, &dex, 0).iter().any(|&(c, _)| c == curse_c),
        "PS blocks Ghost Curse vs Substitute"
    );
    assert!(
        dominated_actions(&b_sk_cs, &dex, 0).iter().any(|&(c, _)| c == curse_c),
        "Castle blocks Ghost Curse vs Substitute"
    );

    // 6. Last-mon Perish Song & Destiny Bond fail as harmless no-ops in BOTH Battle & CompactBattle
    //    (Probe 11A2), so `wait_exempt` MUST preserve them as safe wait moves even under tournament overlay,
    //    while non-wait-exempt states mask them as no-ops (not forfeits).
    let db_user = make_set("Gengar", 50, "No Item", &["Destiny Bond", "Perish Song", "Reflect"]);
    let f1 = make_set("Snorlax", 50, "No Item", &["Body Slam"]);
    let f2 = make_set("Zapdos", 50, "No Item", &["Thunderbolt"]);
    let mut b_tour = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[db_user.clone()],
        &[f1.clone(), f2.clone()],
    );
    let mut b_raw = start_battle(
        &dex,
        Ruleset::castle_raw_cgi(false),
        &[db_user],
        &[f1, f2],
    );
    // Poison foe so wait_exempt == true: safe wait moves MUST stay unmasked in both rulesets!
    for b in [&mut b_tour, &mut b_raw] {
        let def = b.active_id(1).unwrap();
        b.poke_mut(def).status = Status::Psn;
    }
    let db_c = mv(&dex, "destinybond");
    let ps_c = mv(&dex, "perishsong");
    assert!(
        !dominated_actions(&b_tour, &dex, 0)
            .iter()
            .any(|&(c, _)| c == db_c || c == ps_c),
        "Castle tournament wait_exempt must preserve last-mon Destiny Bond / Perish Song as safe wait moves"
    );
    assert!(
        !public_dominated_actions(&b_tour, &dex, 0)
            .iter()
            .any(|&(c, _)| c == db_c || c == ps_c),
        "Castle tournament public wait_exempt must preserve last-mon Destiny Bond / Perish Song"
    );
    assert!(
        !dominated_actions(&b_raw, &dex, 0)
            .iter()
            .any(|&(c, _)| c == db_c || c == ps_c),
        "Castle raw CGI wait_exempt must preserve last-mon Destiny Bond / Perish Song"
    );
    // Clear foe poison -> wait_exempt == false -> masked as no-op (NOT forfeit) in both!
    for b in [&mut b_tour, &mut b_raw] {
        let def = b.active_id(1).unwrap();
        b.poke_mut(def).status = Status::None;
        let dom = dominated_actions(b, &dex, 0);
        assert!(
            dom.iter().any(|&(c, why)| c == db_c && !why.contains("forfeit"))
                && dom.iter().any(|&(c, why)| c == ps_c && !why.contains("forfeit")),
            "Last-mon Destiny Bond / Perish Song must be masked as harmless no-op when not wait_exempt: {dom:?}"
        );
    }
}

#[test]
fn turn_limit_80_judge_and_cache_separation_across_solvers_and_compiled_preview() {
    let dex = conformance::load_dex();
    let p1_a = make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Rest"]);
    let p1_b = make_set("Starmie", 50, "No Item", &["Surf", "Recover"]);
    let p2_a = make_set("Blissey", 55, "Leftovers", &["Ice Beam", "Soft-Boiled"]);
    let p2_b = make_set("Zapdos", 50, "No Item", &["Thunderbolt", "Rest"]);

    // 1. Probe 12A1/12A2 exact HP-ratio sum judge (seat-unbiased!).
    let mut b_cs = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[p1_a.clone(), p1_b.clone()],
        &[p2_a.clone(), p2_b.clone()],
    );
    b_cs.turn = 81; // past L_TURN = 80
    // P1: 25% + 100% = 1.25; P2: 50% + 50% = 1.00 -> P1 wins (1.0).
    b_cs.sides[0].roster[0].hp = b_cs.sides[0].roster[0].maxhp / 4;
    b_cs.sides[0].roster[1].hp = b_cs.sides[0].roster[1].maxhp;
    b_cs.sides[1].roster[0].hp = b_cs.sides[1].roster[0].maxhp / 2;
    b_cs.sides[1].roster[1].hp = b_cs.sides[1].roster[1].maxhp / 2;

    assert_eq!(castle_judge_value_battle(&b_cs), 1.0);
    assert_eq!(turn_limit_outcome_value(&b_cs), Some(1.0));
    let w = EvalWeights::default();
    assert_eq!(eval01(&b_cs, &dex, &w), 1.0);

    let (t_cs, cb_cs) = CompactBattle::from_battle(&b_cs, &dex);
    assert_eq!(compact_turn_limit_outcome(&cb_cs, &t_cs), Some(1.0));
    assert_eq!(compact_eval01(&cb_cs, &t_cs, &w), 1.0);

    // Exact solver & BoundSolver agree immediately at turn = 81.
    let mut ex_solver = ExactSolver::new(&dex, ExactConfig::default());
    let ex_cert = ex_solver.solve(&b_cs).expect("terminal exact value");
    assert!((ex_cert.lo - 1.0).abs() < 1e-9 && (ex_cert.hi - 1.0).abs() < 1e-9);
    let mut bsolver = BoundSolver::new(&dex, BoundConfig::default());
    let brep = bsolver.solve(&b_cs, None);
    assert!((brep.bounds.lo - 1.0).abs() < 1e-9 && (brep.bounds.hi - 1.0).abs() < 1e-9);

    // Swap teams -> P2 wins (0.0) with zero seat bias!
    let mut b_cs_swapped = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[p2_a.clone(), p2_b.clone()],
        &[p1_a.clone(), p1_b.clone()],
    );
    b_cs_swapped.turn = 81;
    b_cs_swapped.sides[0].roster[0].hp = b_cs_swapped.sides[0].roster[0].maxhp / 2;
    b_cs_swapped.sides[0].roster[1].hp = b_cs_swapped.sides[0].roster[1].maxhp / 2;
    b_cs_swapped.sides[1].roster[0].hp = b_cs_swapped.sides[1].roster[0].maxhp / 4;
    b_cs_swapped.sides[1].roster[1].hp = b_cs_swapped.sides[1].roster[1].maxhp;
    assert_eq!(turn_limit_outcome_value(&b_cs_swapped), Some(0.0));

    // Alive count beats higher HP-ratio sum (Probe 12A3): 2 mons at 20% each (0.40) beat 1 mon at 100% (1.00).
    b_cs.sides[0].roster[0].hp = b_cs.sides[0].roster[0].maxhp / 5;
    b_cs.sides[0].roster[1].hp = b_cs.sides[0].roster[1].maxhp / 5;
    b_cs.sides[1].roster[0].hp = b_cs.sides[1].roster[0].maxhp;
    b_cs.sides[1].roster[1].hp = 0;
    b_cs.sides[1].roster[1].fainted = true;
    b_cs.sides[1].pokemon_left = 1;
    assert_eq!(castle_judge_value_battle(&b_cs), 1.0);

    // Bot 80T proximity adaptation is abolished: compact_eval01 is invariant to turn < lim.
    let mut b_prox = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[p1_a.clone()],
        &[p2_a.clone()],
    );
    b_prox.sides[0].roster[0].hp = b_prox.sides[0].roster[0].maxhp;
    b_prox.sides[1].roster[0].hp = b_prox.sides[1].roster[0].maxhp / 2;
    let (t_pr, mut cb_pr) = CompactBattle::from_battle(&b_prox, &dex);
    cb_pr.turn = 10;
    let v_early = compact_eval01(&cb_pr, &t_pr, &w);
    cb_pr.turn = 78;
    let v_late = compact_eval01(&cb_pr, &t_pr, &w);
    assert!(
        (v_late - v_early).abs() < 1e-12,
        "Castle eval must NOT inject pre-limit 80T proximity shaping ({v_late} vs {v_early})"
    );

    // MCTS chooses cleanly at turn = 80 without panicking or assuming 1000-turn cap.
    b_prox.turn = 80;
    let legal = b_prox.clone().legal_choices(&dex, 0);
    let mut mcts = MctsAgent::new(
        MctsConfig {
            iterations: 32,
            playout: Playout::Heavy {
                eps: 0.1,
                turns: 8,
                weights: EvalWeights::default(),
            },
            ..MctsConfig::default()
        },
        7,
    );
    let picked = mcts.choose(&b_prox, &dex, 0, &legal);
    assert!(
        legal.contains(&picked),
        "MCTS at Castle L_TURN=80 must return a legal action: {picked:?}"
    );

    // 2. Stall solver respects Castle `turn_limit = Some(80)` (does NOT wait for turn 1000).
    let mut b_stall = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[p1_a.clone()],
        &[p2_a.clone()],
    );
    b_stall.turn = 81;
    assert_eq!(
        classify_two_sided_heal(&b_stall, &dex),
        Err(ClassifyError::TurnPastLimit { turn: 81 }),
        "stall classifier must enforce Castle turn_limit = Some(80)"
    );

    // 3. HpTransitionKey separates PS vs Castle, accepts Castle pseudo-weather, and guards L_TURN.
    let hp_a = make_set("Snorlax", 55, "No Item", &["Body Slam"]);
    let hp_b = make_set("Blissey", 55, "No Item", &["Ice Beam"]);
    let b_hp_ps = start_battle(&dex, Ruleset::PsStadium2, &[hp_a.clone()], &[hp_b.clone()]);
    let mut b_hp_cs =
        start_battle(&dex, Ruleset::castle_tournament(), &[hp_a], &[hp_b]);
    let hp_choices = [Some(mv(&dex, "bodyslam")), Some(mv(&dex, "icebeam"))];
    let k_hp_ps = HpTransitionKey::new(&dex, &b_hp_ps, hp_choices, &[]).expect("PS key");
    let k_hp_cs = HpTransitionKey::new(&dex, &b_hp_cs, hp_choices, &[]).expect("Castle key");
    assert_ne!(k_hp_ps, k_hp_cs, "HpTransitionKey must separate PS vs Castle");
    b_hp_cs.turn = 80;
    assert!(
        HpTransitionKey::new(&dex, &b_hp_cs, hp_choices, &[]).is_none(),
        "HpTransitionKey must refuse Castle states at/past L_TURN=80"
    );

    // 4. BoundCache & EndgameSession separate PS vs Castle cleanly.
    let mut bcache = BoundCache::default();
    let mut b_ps_t81 = start_battle(&dex, Ruleset::PsStadium2, &[p1_a.clone()], &[p2_a.clone()]);
    b_ps_t81.turn = 81;
    let mut b_cs_t81 =
        start_battle(&dex, Ruleset::castle_tournament(), &[p1_a.clone()], &[p2_a.clone()]);
    b_cs_t81.turn = 81;
    b_cs_t81.sides[0].roster[0].hp = b_cs_t81.sides[0].roster[0].maxhp;
    b_cs_t81.sides[1].roster[0].hp = b_cs_t81.sides[1].roster[0].maxhp / 2;
    let mut s_c1 = BoundSolver::from_cache(&dex, BoundConfig::default(), bcache);
    let r_c1 = s_c1.solve(&b_cs_t81, None);
    assert!((r_c1.bounds.lo - 1.0).abs() < 1e-9);
    let bcache = s_c1.into_cache();
    // Switching BoundCache to PS battle at turn 81 must NOT return Castle's 1.0 bound!
    let mut s_p1 = BoundSolver::from_cache(
        &dex,
        BoundConfig {
            work_budget: 2,
            ..BoundConfig::default()
        },
        bcache,
    );
    let r_p1 = s_p1.solve(&b_ps_t81, None);
    assert!(r_p1.bounds.lo < 0.99, "BoundCache must reset on ruleset switch");

    let es = EndgameSession::new(EndgameConfig::default(), 0.0, 1000.0).unwrap();
    assert!(!es.eligible(&b_cs_t81), "EndgameSession must treat Castle turn > 80 as ineligible");
    assert!(es.eligible(&b_ps_t81), "EndgameSession must treat PS turn 81 as eligible");

    // 5. compiled_preview: PS backward compatibility + Castle key separation & policy_epoch_for.
    assert_eq!(compiled_preview::POLICY_EPOCH, "policy-v3");
    assert_eq!(compiled_preview::CASTLE_POLICY_EPOCH, "castle-policy-v3");
    assert_eq!(
        compiled_preview::policy_epoch_for(Ruleset::PsStadium2),
        "policy-v3"
    );
    assert_eq!(
        compiled_preview::policy_epoch_for(Ruleset::castle_tournament()),
        "castle-policy-v3"
    );

    let mut b_prev_ps = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[p1_a.clone(), p1_b.clone()],
        &[p2_a.clone(), p2_b.clone()],
        Ruleset::PsStadium2,
    )
    .unwrap();
    let k_ps0 = compiled_preview::key(&b_prev_ps, 0).unwrap();
    assert!(
        k_ps0.starts_with("[[") && !k_ps0.contains("castle-policy-v3"),
        "PS key format preserved: {k_ps0}"
    );
    // Opponent gender change must NOT alter legacy PS key!
    b_prev_ps.sides[1].roster[1].gender = Gender::F;
    assert_eq!(compiled_preview::key(&b_prev_ps, 0).unwrap(), k_ps0);

    let mut b_prev_cs = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &[p1_a.clone(), p1_b.clone()],
        &[p2_a.clone(), p2_b.clone()],
        Ruleset::castle_tournament(),
    )
    .unwrap();
    let k_cs0 = compiled_preview::key(&b_prev_cs, 0).unwrap();
    let k_cs1 = compiled_preview::key(&b_prev_cs, 1).unwrap();
    assert!(
        k_cs0.contains("\"epoch\":\"castle-policy-v3\""),
        "Castle key must carry castle-policy-v3 epoch: {k_cs0}"
    );
    assert_ne!(k_cs0, k_ps0, "Castle preview key must differ from PS key");
    assert_ne!(k_cs0, k_cs1, "Castle preview key must encode side");

    // Opponent gender distinguishes Castle preview key.
    let mut b_prev_cs_g = b_prev_cs.clone();
    b_prev_cs_g.sides[1].roster[1].gender = Gender::F;
    assert_ne!(
        compiled_preview::key(&b_prev_cs_g, 0).unwrap(),
        k_cs0,
        "Castle preview key must encode opponent gender"
    );

    // Opponent item presence (`preview_has_item`) distinguishes Castle preview key.
    let k_cs_noitem = compiled_preview::castle_key(&b_prev_cs, 0, Some(&[false, false])).unwrap();
    let k_cs_withitem = compiled_preview::castle_key(&b_prev_cs, 0, Some(&[true, false])).unwrap();
    assert_ne!(
        k_cs_noitem, k_cs_withitem,
        "Castle preview key must encode opponent preview_has_item flags"
    );

    // Turn limit & overflow mode distinguish Castle preview keys.
    b_prev_cs.ruleset = Ruleset::castle_free();
    let k_cs_free = compiled_preview::key(&b_prev_cs, 0).unwrap();
    b_prev_cs.ruleset = Ruleset::CastlePbs(CastleOverlayConfig {
        overflow_mode: true,
        ..CastleOverlayConfig::tournament()
    });
    let k_cs_ovfl = compiled_preview::key(&b_prev_cs, 0).unwrap();
    assert_ne!(k_cs0, k_cs_free, "Castle preview key must encode turn_limit");
    assert_ne!(k_cs0, k_cs_ovfl, "Castle preview key must encode overflow_mode");
    assert_eq!(
        compiled_preview::castle_key_from_flags(&b_prev_cs, 0, &[true, false]),
        compiled_preview::castle_key(&b_prev_cs, 0, Some(&[true, false]))
    );
}

#[test]
fn audit_fixes_qc_mask_stat_parity_dragon_items_foresight_2v1_lturn_sandstorm_bp_and_tableset() {
    let dex = conformance::load_dex();

    // 1. mask.rs Quick Claw item check: non-QC slower mon with `quick_claw_rolls = [true, false]`
    //    must NOT be treated as faster than foe (no false-positive full-HP Rest masking).
    let slow_rest = make_set("Snorlax", 50, "Leftovers", &["Rest", "Body Slam"]);
    let fast_foe = make_set("Zapdos", 55, "Leftovers", &["Thunderbolt"]);
    let mut b_qc = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[slow_rest],
        &[fast_foe],
    );
    b_qc.quick_claw_rolls = [true, false];
    let rest_c = mv(&dex, "rest");
    assert!(
        !dominated_actions(&b_qc, &dex, 0).iter().any(|&(c, _)| c == rest_c),
        "Non-QC Snorlax with quick_claw_rolls=[true,false] must NOT be treated as faster than Zapdos"
    );
    // Actually give Snorlax Quick Claw -> now it IS faster -> full-HP Rest is masked.
    let p1a = b_qc.active_id(0).unwrap();
    b_qc.poke_mut(p1a).item = dex.known_items.quickclaw;
    assert!(
        dominated_actions(&b_qc, &dex, 0).iter().any(|&(c, _)| c == rest_c),
        "Quick Claw Snorlax with quick_claw_rolls=[true,false] IS faster -> full-HP Rest masked"
    );
    // Both hold Quick Claw and both roll true -> speed tie -> NOT strictly faster -> unmasked.
    let p2a = b_qc.active_id(1).unwrap();
    b_qc.poke_mut(p2a).item = dex.known_items.quickclaw;
    b_qc.quick_claw_rolls = [true, true];
    assert!(
        !dominated_actions(&b_qc, &dex, 0).iter().any(|&(c, _)| c == rest_c),
        "Dual Quick Claw tie [true,true] must NOT claim strictly faster"
    );

    // 2. eval::ruleset_mon_stat & relay::stat_with_boost exact parity vs compact::get_stat_castle
    //    on Burned +6 Thick Club Marowak under both ovfl=false and ovfl=true.
    let wak = make_set("Marowak", 55, "Thick Club", &["Earthquake"]);
    let ditto = make_set("Ditto", 50, "Metal Powder", &["Transform"]);
    for ovfl in [false, true] {
        let rs = Ruleset::CastlePbs(CastleOverlayConfig {
            overflow_mode: ovfl,
            ..CastleOverlayConfig::tournament()
        });
        let b_st = start_battle(&dex, rs, &[wak.clone()], &[ditto.clone()]);
        let (t_st, mut cb_st) = CompactBattle::from_battle(&b_st, &dex);
        cb_st.mons[0].set_status(Status::Brn as u8, 0);
        cb_st.mons[0].boosts[0] = 6;
        cb_st.sides[1].reflect_turns = 5;
        cb_st.sides[1].lightscreen_turns = 5;
        for (side, stat_i) in [(0, 0), (0, 1), (0, 4), (1, 1), (1, 3)] {
            for boost in -6i8..=6 {
                let p_idx = side * 3;
                let spec = &t_st.mons[p_idx];
                let mon = &cb_st.mons[p_idx];
                let ispec = t_st.item_spec(mon.item);
                let item_kind = if ispec.stat_kind == spec.species_stat_kind {
                    ispec.stat_kind
                } else {
                    nc2000_engine::compact::ItemStatKind::None
                };
                let expected = nc2000_engine::compact::get_stat_castle(
                    spec.stored_stats[stat_i] as i32,
                    boost,
                    stat_i,
                    false,
                    false,
                    mon.status_raw() == Status::Par as u8,
                    mon.status_raw() == Status::Brn as u8,
                    cb_st.sides[side].reflect_turns > 0,
                    cb_st.sides[side].lightscreen_turns > 0,
                    item_kind,
                    ovfl,
                );
                let e_val = nc2000_bot::eval::ruleset_mon_stat(
                    &cb_st,
                    &t_st,
                    side,
                    0,
                    stat_i,
                    Some(boost),
                    false,
                    false,
                );
                let r_val =
                    nc2000_bot::relay::stat_with_boost(&cb_st, &t_st, side, 0, stat_i, boost);
                assert_eq!(e_val, expected, "eval stat mismatch ovfl={ovfl} side={side} stat={stat_i} b={boost}");
                assert_eq!(r_val, expected, "relay stat mismatch ovfl={ovfl} side={side} stat={stat_i} b={boost}");
            }
        }
        // Explicitly verify Burned +6 Thick Club Marowak Atk:
        // ovfl=false: stored*4*2 > 999 -> clamped to 999 -> Burn /2 = 499!
        // ovfl=true: (stored*4*2 % 1024) >> 1 (distinct from 499!)
        let atk_eval = nc2000_bot::eval::ruleset_mon_stat(&cb_st, &t_st, 0, 0, 0, Some(6), false, false);
        if !ovfl {
            assert_eq!(atk_eval, 499, "Burned +6 Thick Club Marowak ovfl=false must clamp 999 then /2 = 499");
        } else {
            let stored = t_st.mons[0].stored_stats[0] as i32;
            let raw = ((stored * 8) % 1024) >> 1;
            assert_eq!(atk_eval, raw.max(1));
        }
    }

    // 3. Dragon Scale vs Dragon Fang inversion across PS and Castle (`eval.rs` & `linear.rs`).
    let king_scale = make_set("Kingdra", 55, "Dragon Scale", &["DragonBreath"]);
    let king_fang = make_set("Kingdra", 55, "Dragon Fang", &["DragonBreath"]);
    let king_none = make_set("Kingdra", 55, "No Item", &["DragonBreath"]);
    let target = make_set("Snorlax", 55, "No Item", &["Splash"]);
    let db_id = dex.moves.id("dragonbreath").unwrap();
    for (rs, scale_boosts, fang_boosts) in [
        (Ruleset::PsStadium2, false, true),
        (Ruleset::castle_tournament(), true, false),
    ] {
        let b_n = start_battle(&dex, rs, &[king_none.clone()], &[target.clone()]);
        let b_s = start_battle(&dex, rs, &[king_scale.clone()], &[target.clone()]);
        let b_f = start_battle(&dex, rs, &[king_fang.clone()], &[target.clone()]);
        let h_n = nc2000_bot::eval::expected_hit_fraction(&b_n, &dex, b_n.active_id(0).unwrap(), b_n.active_id(1).unwrap(), db_id);
        let h_s = nc2000_bot::eval::expected_hit_fraction(&b_s, &dex, b_s.active_id(0).unwrap(), b_s.active_id(1).unwrap(), db_id);
        let h_f = nc2000_bot::eval::expected_hit_fraction(&b_f, &dex, b_f.active_id(0).unwrap(), b_f.active_id(1).unwrap(), db_id);
        let (tn, cbn) = CompactBattle::from_battle(&b_n, &dex);
        let (ts, cbs) = CompactBattle::from_battle(&b_s, &dex);
        let (tf, cbf) = CompactBattle::from_battle(&b_f, &dex);
        let kn = DmgKernel1D::for_move(&cbn, &tn, 0, 0, 1, 0, 0).mean_dmg;
        let ks = DmgKernel1D::for_move(&cbs, &ts, 0, 0, 1, 0, 0).mean_dmg;
        let kf = DmgKernel1D::for_move(&cbf, &tf, 0, 0, 1, 0, 0).mean_dmg;
        if scale_boosts {
            assert!(h_s > h_n * 1.05 && ks > kn * 1.05, "Castle Dragon Scale must boost DragonBreath");
            assert!((h_f - h_n).abs() < 1e-12 && (kf - kn).abs() < 1e-6, "Castle Dragon Fang must NOT boost");
        } else if fang_boosts {
            assert!(h_f > h_n * 1.05 && kf > kn * 1.05, "PS Dragon Fang must boost DragonBreath");
            assert!((h_s - h_n).abs() < 1e-12 && (ks - kn).abs() < 1e-6, "PS Dragon Scale must NOT boost");
        }
    }

    // 4. Castle Foresight `acc < eva` both-zero rule on non-Ghost defender in `eval.rs` & `linear.rs`.
    let zap = make_set("Zapdos", 55, "No Item", &["Thunderbolt"]);
    let lax = make_set("Snorlax", 55, "No Item", &["Splash"]);
    let mut b_fore = start_battle(&dex, Ruleset::castle_tournament(), &[zap], &[lax]);
    let p1 = b_fore.active_id(0).unwrap();
    let p2 = b_fore.active_id(1).unwrap();
    b_fore.poke_mut(p1).boosts[5] = -2; // acc = -2
    b_fore.poke_mut(p2).boosts[6] = 1; // eva = +1 -> acc < eva!
    let tb_id = dex.moves.id("thunderbolt").unwrap();
    let h_no_fore = nc2000_bot::eval::expected_hit_fraction(&b_fore, &dex, p1, p2, tb_id);
    b_fore.add_volatile(&dex, p2, "foresight", Some(p1), EffectHandle::None);
    let h_with_fore = nc2000_bot::eval::expected_hit_fraction(&b_fore, &dex, p1, p2, tb_id);
    let (tf_cb, cb_fore) = CompactBattle::from_battle(&b_fore, &dex);
    let k_fore = DmgKernel1D::for_move(&cb_fore, &tf_cb, 0, 0, 1, 0, 0);
    let miss_fore = k_fore.pts[..k_fore.len].iter().find(|&&(d, _)| d == 0).map_or(0.0, |&(_, p)| p);
    assert!(
        h_with_fore > h_no_fore * 1.8 && miss_fore < 1e-6,
        "Castle Foresight with acc(-2) < eva(+1) must zero both stages (h_no={h_no_fore}, h_yes={h_with_fore}, miss={miss_fore})"
    );

    // 5. Sandstorm + AfterMoveSelf residual priority in `linear.rs::mutual_eot_payoff`.
    let fast_s = make_set("Zapdos", 55, "No Item", &["Splash"]);
    let slow_s = make_set("Snorlax", 50, "No Item", &["Splash"]);
    let b_sand_brn = start_battle(&dex, Ruleset::castle_free(), &[fast_s], &[slow_s]);
    let (t_sb, mut cb_sb) = CompactBattle::from_battle(&b_sand_brn, &dex);
    cb_sb.weather_kind = CompactWeather::Sandstorm;
    cb_sb.weather_turns = 5;
    cb_sb.mons[0].hp_pool[0] = 1;
    cb_sb.mons[3].hp_pool[0] = 1;
    // Slow Snorlax is Burned (AfterMoveSelf ticks before EOT Sandstorm) -> Snorlax faints first -> Zapdos wins (+1.0)!
    cb_sb.mons[3].set_status(Status::Brn as u8, 0);
    let r_sb = toeplitz_coupled_race(&cb_sb, &t_sb, 0, 0, 1, 0);
    assert!(
        r_sb > 0.99,
        "In Castle Sandstorm, Burned mon takes AfterMoveSelf before EOT Sandstorm and loses: {r_sb}"
    );
    // Both Burned -> faster Zapdos takes AfterMoveSelf first and loses (-1.0)!
    cb_sb.mons[0].set_status(Status::Brn as u8, 0);
    let r_both_brn = toeplitz_coupled_race(&cb_sb, &t_sb, 0, 0, 1, 0);
    assert!(
        r_both_brn < -0.99,
        "When both are Burned in Castle Sandstorm, faster mover takes AfterMoveSelf first and loses: {r_both_brn}"
    );

    // 6. Directed BP `alive_mask` preservation under explicit `castle-tournament` turn 80 + `best_midturn_bp_switch` Pursuit flag.
    let passer = make_set("Jolteon", 55, "No Item", &["Baton Pass", "Agility", "Substitute", "Thunderbolt"]);
    let recv_weak = make_set("Alakazam", 50, "No Item", &["Psychic"]);
    let recv_bulk = make_set("Skarmory", 50, "No Item", &["Drill Peck"]);
    let foe_p = make_set("Tyranitar", 55, "No Item", &["Pursuit"]);
    let foe_b = make_set("Snorlax", 50, "No Item", &["Splash"]);
    let b_t80_bp = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[passer.clone(), recv_weak.clone(), recv_bulk.clone()],
        &[foe_p.clone(), foe_b.clone()],
    );
    let (t_bp80, mut cb_bp80) = CompactBattle::from_battle(&b_t80_bp, &dex);
    cb_bp80.turn = 80;
    cb_bp80.mons[0].boosts[4] = 2;
    cb_bp80.mons[0].hp_pool[0] = t_bp80.mons[0].maxhp / 5;
    cb_bp80.mons[1].hp_pool[0] = t_bp80.mons[1].maxhp / 5;
    cb_bp80.mons[2].hp_pool[0] = t_bp80.mons[2].maxhp / 5;
    cb_bp80.mons[3].hp_pool[0] = t_bp80.mons[3].maxhp;
    cb_bp80.mons[4].hp_pool[0] = t_bp80.mons[4].maxhp;
    // Directly invoke `apply_compact_actions_with_bp_target` at turn 80 with directed BP code to slot 2 (Skarmory):
    // Side 0 has 3 alive mons vs Side 1's 2 alive mons, so Turn-80 judge MUST keep alive_mask == 0b111 -> P1Win!
    let mut cb_bp_test = cb_bp80;
    cb_bp_test.mons[0].hp_pool[0] = t_bp80.mons[0].maxhp;
    cb_bp_test.mons[2].hp_pool[0] = t_bp80.mons[2].maxhp;
    // Choose Splash-like foe (turn 80 ends at EOT with judge): use foe slot 1 (Splash) active so no KO happens.
    cb_bp_test.sides[1].active_idx = 1;
    let directed_bp = ((2u8 + 1) << 4) | 0; // directed BP move 0 -> recv slot 2
    nc2000_bot::mcts::apply_compact_actions_with_bp_target(&mut cb_bp_test, &t_bp80, [directed_bp, 0]);
    if cb_bp_test.sides[0].request == 2 {
        cb_bp_test.apply_actions(&t_bp80, [6, 7]);
    }
    assert_eq!(cb_bp_test.sides[0].alive_mask, 0b111, "Turn-80 directed BP must never lose bench alive_mask bits");
    assert_eq!(cb_bp_test.outcome, 1, "3v2 low-HP Side 0 must win Turn-80 alive-count judge (outcome=1)");

    // `best_midturn_bp_switch` honors `bp_switch_flags` incoming 2x Pursuit bit (`1 << (2 + side)`):
    let mut cb_mid = CompactBattle::from_battle(&b_t80_bp, &dex).1;
    cb_mid.sides[0].request = 2;
    cb_mid.bp_switch_flags = (1 << 0) | (1 << 2); // side 0 fast-BP paused before Tyranitar's Pursuit
    cb_mid.mons[1].hp_pool[0] = t_bp80.mons[1].maxhp / 2; // Alakazam in OHKO range of Castle 2x Pursuit
    cb_mid.mons[2].hp_pool[0] = t_bp80.mons[2].maxhp; // Skarmory resists Pursuit cleanly
    let pick_bp = nc2000_bot::relay::best_midturn_bp_switch(
        &cb_mid,
        &t_bp80,
        &EvalWeights::default(),
        0,
        &[5, 6],
    );
    assert_eq!(
        pick_bp,
        Some(6),
        "best_midturn_bp_switch with Castle incoming Pursuit flag must pick bulky Skarmory (6) over OHKO'd Alakazam (5)"
    );

    // 8. `TableSet` refuses Castle battles (preventing PS seat-symmetric tables from answering Castle preview).
    let root = conformance::fixture::repo_root();
    let pool = nc2000_bot::preview::load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let mut ts = nc2000_bot::TableSet::from_pool(&dex, &pool);
    let pair_text = std::fs::read_to_string(root.join("fixtures/preview-tables-test/pair-00-01.json")).unwrap();
    ts.add_pair_json(&pair_text).unwrap();
    let b_ps_prev = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &pool.teams[0].sets,
        &pool.teams[1].sets,
        Ruleset::PsStadium2,
    )
    .unwrap();
    let b_cs_prev = Battle::from_fixture_with_ruleset(
        &dex,
        "1,2,3,4",
        &pool.teams[0].sets,
        &pool.teams[1].sets,
        Ruleset::castle_tournament(),
    )
    .unwrap();
    assert!(ts.lookup(&b_ps_prev, 0).is_some(), "PS TableSet must serve matching PS preview");
    assert!(ts.lookup(&b_cs_prev, 0).is_none(), "PS TableSet must refuse Castle preview lookup");
    assert!(ts.side_index(&b_cs_prev, 0).is_none(), "PS TableSet must refuse Castle side_index");
    assert!(
        ts.set_index_with_ruleset(&dex, &pool.teams[0].sets, Ruleset::castle_tournament()).is_none(),
        "PS TableSet must refuse Castle set_index_with_ruleset"
    );

    // 9. SkuctSearch clamps `turn_cap` to Castle `L_TURN=80` and evaluates exact turn-limit judge.
    let mut b_sk = start_battle(
        &dex,
        Ruleset::castle_tournament(),
        &[make_set("Snorlax", 55, "Leftovers", &["Splash", "Rest"])],
        &[make_set("Snorlax", 55, "Leftovers", &["Splash"])],
    );
    b_sk.turn = 80;
    let p1s = b_sk.active_id(0).unwrap();
    let p2s = b_sk.active_id(1).unwrap();
    b_sk.poke_mut(p1s).hp = b_sk.poke(p1s).maxhp;
    b_sk.poke_mut(p2s).hp = b_sk.poke(p2s).maxhp / 3;
    let mut sk = nc2000_bot::smmcts::SkuctSearch::new(
        &b_sk,
        &dex,
        nc2000_bot::RmConfig {
            iterations: 32,
            ..nc2000_bot::RmConfig::default()
        },
        13,
    );
    sk.step(&dex, 32);
    let legal_sk = b_sk.clone().legal_choices(&dex, 0);
    let chosen_sk = sk.best(0).expect("SkuctSearch at Castle turn 80 must pick a move");
    assert!(legal_sk.contains(&chosen_sk), "SkuctSearch at Castle turn 80 must pick a legal move");
    let best_mean = sk.means(0).into_iter().fold(0.0f64, f64::max);
    assert!(
        best_mean > 0.95,
        "SkuctSearch at Castle turn 80 must observe turn-limit judge win (>0.95), got {best_mean}"
    );
    assert!(
        sk.mean_depth() <= 1.5,
        "SkuctSearch at Castle turn 80 must clamp turn_cap to L_TURN=80 (depth={})",
        sk.mean_depth()
    );
}

#[test]
fn castle_unlimited_default_no_hidden_cap_and_solver_cycle_safety() {
    let dex = conformance::load_dex();

    // 1. Default Castle config and generic aliases resolve to unlimited (`castle-free`, `turn_limit: None`).
    assert_eq!(CastleOverlayConfig::default(), CastleOverlayConfig::free());
    assert_eq!(CastleOverlayConfig::default().turn_limit, None);
    for alias in ["castle", "castle-pbs", "pbs", "pbs-nc2000", "castle-free", "pbs-free"] {
        assert_eq!(
            Ruleset::parse_id(alias),
            Some(Ruleset::castle_free()),
            "alias `{alias}` must resolve to castle-free"
        );
    }
    assert_eq!(
        Ruleset::parse_id("castle-tournament"),
        Some(Ruleset::castle_tournament())
    );
    assert_eq!(Ruleset::castle_free().canonical_id(), "castle-free");
    assert_eq!(Ruleset::castle_free().turn_limit(), None);

    // 2. Truly unlimited: no hidden 80-turn or 1000-turn adjudication in Battle, CompactBattle, or eval.
    let gengar = make_set("Gengar", 50, "No Item", &["Splash"]);
    let mut b = start_battle(&dex, Ruleset::castle_free(), &[gengar.clone()], &[gengar]);
    for &t in &[80u16, 81, 1000, 1001, 1076, u16::MAX] {
        b.turn = t;
        assert_eq!(turn_limit_outcome_value(&b), None, "castle-free must not adjudicate at turn {t}");
        let (table, cb) = CompactBattle::from_battle(&b, &dex);
        assert_eq!(
            compact_turn_limit_outcome(&cb, &table),
            None,
            "compact castle-free must not adjudicate at turn {t}"
        );
    }

    // 3. `Battle::end_turn` saturates at `u16::MAX` without panicking or wrapping to 0.
    b.turn = u16::MAX;
    b.choose(&dex, 0, "move splash").unwrap();
    b.choose(&dex, 1, "move splash").unwrap();
    assert_eq!(b.turn, u16::MAX);
    assert_eq!(b.outcome(), None);

    // 4. Unlimited-turn cycle safety in `ExactSolver`, `BoundSolver`, and `stall.rs`:
    //    2v2 Splash-only mirror under `castle-free`: switching costs 0 PP (true infinite cycle)
    //    and Splash deals 0 damage, so no turn-limit adjudication can ever resolve the game.
    let g0 = make_set("Gengar", 50, "No Item", &["Splash"]);
    let b0 = make_set("Blissey", 50, "No Item", &["Splash"]);
    let mut b_inf = start_battle(
        &dex,
        Ruleset::castle_free(),
        &[g0.clone(), b0.clone()],
        &[g0, b0],
    );

    // Normal turn: ExactSolver & BoundSolver must terminate cleanly with full-uncertainty [0, 1].
    let mut ex = ExactSolver::new(
        &dex,
        ExactConfig {
            horizons: &[1, 2],
            work_budget: 50_000,
            ..ExactConfig::default()
        },
    );
    let cert = ex.solve(&b_inf).expect("ExactSolver must terminate on unlimited cycle");
    assert!((cert.lo - 0.0).abs() < 1e-9 && (cert.hi - 1.0).abs() < 1e-9);

    let mut bs = BoundSolver::new(
        &dex,
        BoundConfig {
            work_budget: 200,
            node_budget: 200,
            ..BoundConfig::default()
        },
    );
    let rep = bs.solve(&b_inf, None);
    assert!((rep.bounds.lo - 0.0).abs() < 1e-9 && (rep.bounds.hi - 1.0).abs() < 1e-9);

    // Near-saturated (`u16::MAX - 1`) and saturated (`u16::MAX`) turns: ExactSolver & BoundSolver
    // must terminate cleanly without cycling when `turn` stops strictly increasing.
    for &sat_turn in &[u16::MAX - 1, u16::MAX] {
        b_inf.turn = sat_turn;
        let cert_max = ex
            .solve(&b_inf)
            .expect("ExactSolver must guard u16::MAX turn saturation under turn_limit: None");
        assert!((cert_max.lo - 0.0).abs() < 1e-9 && (cert_max.hi - 1.0).abs() < 1e-9);
    }

    b_inf.turn = u16::MAX;
    let mut bs_max = BoundSolver::new(
        &dex,
        BoundConfig {
            work_budget: 100,
            node_budget: 100,
            ..BoundConfig::default()
        },
    );
    let rep_max = bs_max.solve(&b_inf, None);
    assert!((rep_max.bounds.lo - 0.0).abs() < 1e-9 && (rep_max.bounds.hi - 1.0).abs() < 1e-9);

    // Stall classifier accepts castle-free healing endgames without fabricating a turn-80 limit.
    let healer = make_set("Blissey", 50, "Leftovers", &["Soft-Boiled", "Ice Beam"]);
    let mut b_heal = start_battle(&dex, Ruleset::castle_free(), &[healer.clone()], &[healer]);
    b_heal.turn = 120;
    assert!(
        classify_two_sided_heal(&b_heal, &dex).is_ok(),
        "stall classifier must accept castle-free beyond turn 80"
    );
}
