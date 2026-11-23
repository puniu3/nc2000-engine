//! Tactical positional regression tests for the Setup-Relay simultaneous-move
//! Bellman subgame solver and equilibrium-gated rollout continuation (`relay.rs`).

use nc2000_bot::belief::Belief;
use nc2000_bot::blind::BlindSearch;
use nc2000_bot::observe::Observer;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_bot::LeafSettlementMode;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::json;

fn make_set(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(json!({
        "name": species,
        "species": species,
        "item": item,
        "ability": "No Ability",
        "moves": moves,
        "level": level,
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
        "happiness": 255,
        "gender": "M"
    }))
    .unwrap()
}

fn eval_root_pinned(
    dex: &Dex,
    battle: &Battle,
    side: usize,
    settlement: LeafSettlementMode,
    iters: u32,
    seed: u64,
) -> (f64, Vec<(String, f64, f64, f64)>) {
    let mut obs = Observer::new(battle, side);
    obs.observe(battle, dex);
    let belief = Belief::pinned_from_battle(battle, &obs);
    let cfg = RmConfig {
        iterations: iters,
        c: 0.4,
        rule: SelRule::Ucb,
        settlement,
        ..RmConfig::default()
    };
    let mut search = BlindSearch::new(battle, dex, cfg, side, seed);
    search.prune_dominated();
    search.step(dex, &belief, &obs, iters);

    let acts = search.actions();
    let visits = search.visits();
    let means = search.means();
    let total_v: u64 = visits.iter().map(|&n| n as u64).sum::<u64>().max(1);
    let eq = search.bayesian_equilibrium();
    let eq_val = eq
        .as_ref()
        .map(|e| {
            e.policy
                .iter()
                .zip(&e.action_values)
                .map(|(&p, &v)| p * v)
                .sum::<f64>()
        })
        .unwrap_or_else(|| means.iter().copied().fold(0.0, f64::max));

    let mut rows = Vec::with_capacity(acts.len());
    for i in 0..acts.len() {
        let inp = acts[i].to_input(dex);
        let frac = (visits[i] as f64) / (total_v as f64);
        let eq_q = eq
            .as_ref()
            .and_then(|e| e.action_values.get(i).copied())
            .unwrap_or(means[i]);
        let mix = eq
            .as_ref()
            .and_then(|e| e.policy.get(i).copied())
            .unwrap_or(frac);
        rows.push((inp, frac, eq_q, mix));
    }
    (eq_val, rows)
}

const SEEDS: [u64; 3] = [11, 23, 37];
const A1_SEEDS: [u64; 24] = [
    11, 23, 37, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121,
];

fn eval_root_pinned_avg(
    dex: &Dex,
    battle: &Battle,
    side: usize,
    settlement: LeafSettlementMode,
    iters: u32,
    seeds: &[u64],
) -> (f64, Vec<(String, f64, f64, f64)>) {
    let n = seeds.len() as f64;
    let mut avg_val = 0.0;
    let mut avg_rows: Vec<(String, f64, f64, f64)> = Vec::new();
    for (si, &seed) in seeds.iter().enumerate() {
        let (v, rows) = eval_root_pinned(dex, battle, side, settlement, iters, seed);
        avg_val += v / n;
        if si == 0 {
            avg_rows = rows
                .into_iter()
                .map(|(inp, f, q, m)| (inp, f / n, q / n, m / n))
                .collect();
        } else {
            for (inp, f, q, m) in rows {
                if let Some(slot) = avg_rows.iter_mut().find(|r| r.0 == inp) {
                    slot.1 += f / n;
                    slot.2 += q / n;
                    slot.3 += m / n;
                }
            }
        }
    }
    (avg_val, avg_rows)
}

fn find_stat(rows: &[(String, f64, f64, f64)], input: &str) -> (f64, f64, f64) {
    rows.iter()
        .find(|(inp, _, _, _)| inp == input)
        .map(|&(_, f, q, m)| (f, q, m))
        .unwrap_or((0.0, 0.0, 0.0))
}

#[test]
fn relay_bellman_proactive_pressure_evasion_deadline_and_premature_pass_guard() {
    let dex = super::common::dex();
    let iters = 27_000u32;

    // 1. Test A1: Suicune vs Jolteon [Sub, Agility, BP, Tbolt] + Thick Club Marowak
    let a1_p0 = vec![
        make_set("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Toxic"]),
        make_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest", "Curse"]),
    ];
    let a1_p1 = vec![
        make_set("Jolteon", 50, "Leftovers", &["Thunderbolt", "Substitute", "Agility", "Baton Pass"]),
        make_set("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
        make_set("Suicune", 50, "Leftovers", &["Surf", "Rest", "Sleep Talk", "Ice Beam"]),
    ];
    let mut b_a1 = Battle::from_fixture(&dex, "1,2,3,4", &a1_p0, &a1_p1).unwrap();
    b_a1.set_log_enabled(false);
    b_a1.choose(&dex, 0, "team 1,2,3").unwrap();
    b_a1.choose(&dex, 1, "team 1,2,3").unwrap();

    let (_, p1_no_relay) = eval_root_pinned_avg(
        &dex,
        &b_a1,
        1,
        LeafSettlementMode::BlindSpotLeafNoRelay,
        iters,
        &A1_SEEDS,
    );
    let (nr_sub_f, nr_sub_q, nr_sub_m) = find_stat(&p1_no_relay, "move substitute");
    let (_, nr_agi_q, _) = find_stat(&p1_no_relay, "move agility");
    let (_, p1_rows) = eval_root_pinned_avg(
        &dex,
        &b_a1,
        1,
        LeafSettlementMode::BlindSpotLeaf,
        iters,
        &A1_SEEDS,
    );
    let (sub_f, sub_q, sub_m) = find_stat(&p1_rows, "move substitute");
    let (agi_f, agi_q, agi_m) = find_stat(&p1_rows, "move agility");
    let (_, _, bp_m) = find_stat(&p1_rows, "move batonpass");
    let (_, tb_q, tb_m) = find_stat(&p1_rows, "move thunderbolt");
    // Under exact Gen 2 AfterSwitchInSelf (1/8 poison entry slip on Baton Pass) and explicit
    // BATONN receiver choice, Substitute (blocking Toxic + Surf) strictly beats bare Agility
    // and raw Thunderbolt, while Relay lifts Substitute visit/mix/Q (and Agility Q) over NoRelay
    // and allocates >85% mix across Relay setup/BP plays instead of raw Thunderbolt.
    assert!(
        sub_f + agi_f > 0.58
            && sub_f > nr_sub_f + 0.02
            && sub_m > nr_sub_m + 0.04
            && sub_q > nr_sub_q
            && agi_q > nr_agi_q
            && sub_q > tb_q
            && sub_q > agi_q
            && sub_m + agi_m + bp_m > 0.85
            && tb_m < 0.05,
        "Jolteon must weight Substitute/Relay with relay solver active: relay={p1_rows:?}, no_relay={p1_no_relay:?}"
    );

    let (_, p0_cand) = eval_root_pinned_avg(
        &dex,
        &b_a1,
        0,
        LeafSettlementMode::BlindSpotLeaf,
        iters,
        &A1_SEEDS,
    );
    let (surf_c, surf_qc, _) = find_stat(&p0_cand, "move surf");
    let (ib_c, ib_qc, _) = find_stat(&p0_cand, "move icebeam");
    let (_, tox_qc, _) = find_stat(&p0_cand, "move toxic");
    let (_, sw3_qc, sw3_mc) = find_stat(&p0_cand, "switch 3");
    assert!(
        surf_c + ib_c > 0.65
            && surf_qc.max(ib_qc) > tox_qc
            && surf_qc.max(ib_qc) > sw3_qc
            && sw3_mc > 0.15,
        "Suicune must maintain top-Q Sub-breaking pressure while keeping Snorlax pivot in Relay mix: cand={p0_cand:?}"
    );

    // 2. Test A2: Snorlax + Raikou [Roar] vs Zapdos [Double Team, Sub, BP] + Marowak
    let mut raikou_hp_ice = make_set("Raikou", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]);
    raikou_hp_ice.ivs.as_mut().unwrap().insert("def".into(), 26);
    let a2_p0 = vec![
        make_set("Snorlax", 50, "Leftovers", &["Body Slam", "Rest", "Curse", "Sleep Talk"]),
        raikou_hp_ice,
        make_set("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
    ];
    let a2_p1 = vec![
        make_set("Zapdos", 50, "Leftovers", &["Thunderbolt", "Double Team", "Substitute", "Baton Pass"]),
        make_set("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
        make_set("Suicune", 50, "Leftovers", &["Surf", "Rest", "Sleep Talk", "Ice Beam"]),
    ];
    let mut b_a2_0 = Battle::from_fixture(&dex, "1,2,3,4", &a2_p0, &a2_p1).unwrap();
    b_a2_0.set_log_enabled(false);
    b_a2_0.choose(&dex, 0, "team 1,2,3").unwrap();
    b_a2_0.choose(&dex, 1, "team 1,2,3").unwrap();

    let mut b_a2_1 = b_a2_0.clone();
    b_a2_1.choose(&dex, 0, "move curse").unwrap();
    b_a2_1.choose(&dex, 1, "move doubleteam").unwrap();
    b_a2_1.poke_mut(b_a2_1.active_id(0).unwrap()).boosts = [0; 7];

    let mut b_a2_2 = b_a2_1.clone();
    b_a2_2.choose(&dex, 0, "move curse").unwrap();
    b_a2_2.choose(&dex, 1, "move doubleteam").unwrap();
    b_a2_2.poke_mut(b_a2_2.active_id(0).unwrap()).boosts = [0; 7];

    let (v0, r_eva0) = eval_root_pinned_avg(
        &dex,
        &b_a2_0,
        0,
        LeafSettlementMode::BlindSpotLeaf,
        iters,
        &SEEDS,
    );
    let (v1, _) = eval_root_pinned_avg(
        &dex,
        &b_a2_1,
        0,
        LeafSettlementMode::BlindSpotLeaf,
        iters,
        &SEEDS,
    );
    let (v2, _) = eval_root_pinned_avg(
        &dex,
        &b_a2_2,
        0,
        LeafSettlementMode::BlindSpotLeaf,
        iters,
        &SEEDS,
    );
    let (_, bs_q, _) = find_stat(&r_eva0, "move bodyslam");
    let (_, rai_q, rai_m) = find_stat(&r_eva0, "switch 2");
    assert!(
        rai_q >= bs_q - 0.025 || rai_m > 0.18,
        "Pivot(Raikou) must enter primary equilibrium at eva=0: raikou_q={rai_q:.4}, bs_q={bs_q:.4}, raikou_mix={rai_m:.3}"
    );
    assert!(
        v0 > v1 + 0.04 && v1 > v2 + 0.04,
        "P0 root value must strictly decay as Zapdos evasion climbs 0->1->2: [{v0:.4}, {v1:.4}, {v2:.4}]"
    );

    // 3. Test A3: Winning Thunderbolt gate closure & Behind-Sub Agility before Baton Pass
    let a3_win_p0 = vec![
        make_set("Starmie", 50, "Leftovers", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
        make_set("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Ice Beam", "Toxic"]),
        make_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
    ];
    let a3_win_p1 = vec![
        make_set("Jolteon", 55, "Leftovers", &["Thunderbolt", "Substitute", "Agility", "Baton Pass"]),
        make_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
        make_set("Suicune", 50, "Leftovers", &["Surf", "Rest", "Sleep Talk", "Ice Beam"]),
    ];
    let mut b_a3_win = Battle::from_fixture(&dex, "1,2,3,4", &a3_win_p0, &a3_win_p1).unwrap();
    b_a3_win.set_log_enabled(false);
    b_a3_win.choose(&dex, 0, "team 1,2,3").unwrap();
    b_a3_win.choose(&dex, 1, "team 1,2,3").unwrap();
    let (_, r_win) = eval_root_pinned_avg(
        &dex,
        &b_a3_win,
        1,
        LeafSettlementMode::BlindSpotLeaf,
        iters,
        &SEEDS,
    );
    let (tb_f, _, tb_m) = find_stat(&r_win, "move thunderbolt");
    assert!(
        tb_f > 0.90 && tb_m > 0.90,
        "Winning Jolteon must click Thunderbolt without wasted setup: {r_win:?}"
    );

    let a3_sub_p0 = vec![
        make_set("Porygon2", 50, "Leftovers", &["Tackle", "Recover", "Thunder Wave", "Sharpen"]),
        make_set("Starmie", 55, "Mystic Water", &["Surf", "Psychic", "Thunderbolt", "Recover"]),
        make_set("Quagsire", 50, "Leftovers", &["Sludge Bomb", "Body Slam", "Rest", "Amnesia"]),
    ];
    let a3_sub_p1 = vec![
        make_set("Jolteon", 55, "Leftovers", &["Thunderbolt", "Agility", "Substitute", "Baton Pass"]),
        make_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
        make_set("Forretress", 50, "Leftovers", &["Spikes", "Rapid Spin", "Pin Missile", "Reflect"]),
    ];
    let mut b_a3_sub = Battle::from_fixture(&dex, "1,2,3,4", &a3_sub_p0, &a3_sub_p1).unwrap();
    b_a3_sub.set_log_enabled(false);
    b_a3_sub.choose(&dex, 0, "team 1,2,3").unwrap();
    b_a3_sub.choose(&dex, 1, "team 1,2,3").unwrap();
    b_a3_sub.choose(&dex, 0, "move thunderwave").unwrap();
    b_a3_sub.choose(&dex, 1, "move substitute").unwrap();
    let (_, r_sub) = eval_root_pinned_avg(
        &dex,
        &b_a3_sub,
        1,
        LeafSettlementMode::BlindSpotLeaf,
        iters,
        &SEEDS,
    );
    let (bp_f, bp_q, _) = find_stat(&r_sub, "move batonpass");
    let (ag_f, ag_q, _) = find_stat(&r_sub, "move agility");
    assert!(
        ag_f > 0.90 && ag_q > bp_q + 0.15 && bp_f < 0.05,
        "Jolteon behind Sub facing fast Water bench check must stack Agility before Baton Pass: {r_sub:?}"
    );
}
