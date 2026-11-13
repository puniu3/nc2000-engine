//! Verification suite for the 5-pillar rollout-free ISMCTS + analytical solver architecture:
//! 1. ISMCTS visible-state aggregation across determinized hidden worlds with zero psychic leaks.
//! 2. Semantic HP threshold-only splitting, online Welford HP distribution, and Markovian full-HP confluence.
//! 3. Rollout-free analytical leaf settlement (`SolverQuiescence`): strict antisymmetry, forced-switch
//!    game-theoretic resolution, multi-mon Bellman-Toeplitz exchange, and Counter/Mirror Coat modeling.
//! 4. 3-rank action pruning (`Unmarked` / `Marked` soft gate / `NoOp` hard prune) & switch-read recovery.
//! 5. Provably safe state diet & tactical choices on `report-4296`.

use std::sync::Arc;

use conformance::fixture::repo_root;
use nc2000_bot::{
    eval::{compact_eval_solver_leaf, compact_eval_solver_leaf_dist, eval_solver_leaf, EvalWeights},
    import::ProtocolAgent,
    position::PositionSpec,
    preview::MetaPool,
    smmcts::{
        classify_actions_public_root, classify_actions_sim, key_for_test, ActionRank, MonHpDist,
        RmConfig,
    },
    Belief, BlindSearch, Observer,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    compact::CompactBattle,
    dex::Dex,
    smallvec::SmallVec,
    state::Battle,
};

use super::common;

// The abandoned bundle remains opt-in for its regression tests, never the product default.
fn experimental_solver_config() -> RmConfig {
    RmConfig {
        collapse_sleep_actions: true,
        key_no_damage: true,
        settlement: nc2000_bot::accounting::LeafSettlementMode::SolverQuiescence,
        ismcts_visible_tree: true,
        semantic_hp_thresholds: true,
        three_rank_pruning: true,
        ..RmConfig::legacy_heavy_rollout()
    }
}

#[test]
fn default_search_uses_eight_turn_rollout_without_experimental_bundle() {
    let cfg = RmConfig::default();
    assert_eq!(
        format!("{cfg:?}"),
        format!("{:?}", RmConfig::legacy_heavy_rollout())
    );
    assert!(!cfg.ismcts_visible_tree && !cfg.semantic_hp_thresholds && !cfg.three_rank_pruning);
    assert_eq!(
        cfg.settlement,
        nc2000_bot::accounting::LeafSettlementMode::BlindSpotLeaf
    );
    assert!(matches!(
        cfg.playout,
        nc2000_bot::mcts::Playout::Heavy { turns: 8, .. }
    ));

    let dex = common::dex();
    // Recover-only mirrors cannot end early; the actual rollout must advance all eight turns.
    let team = vec![make_set("Starmie", "Leftovers", &["Recover"], 50)];
    let mut battle = Battle::from_fixture(dex, "1,2,3,4", &team, &team).unwrap();
    let picks = [0, 1].map(|side| battle.legal_choices(dex, side).into_iter().next());
    battle.apply_choices(dex, picks).unwrap();
    let turn = battle.turn;
    let value = nc2000_bot::mcts::playout_value(
        &mut battle,
        dex,
        &cfg.playout,
        turn + cfg.horizon,
        &mut nc2000_bot::SplitMix64::new(777),
        cfg.rollout_rules(),
    );
    assert!(value.is_finite() && (0.0..=1.0).contains(&value));
    let end = nc2000_bot::mcts::last_sim_end();
    assert_eq!(end.outcome, 0);
    assert_eq!(end.turn, turn + 8);
}

fn make_set(species: &str, item: &str, moves: &[&str], level: u8) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": species,
        "species": species,
        "item": item,
        "ability": "",
        "level": level,
        "moves": moves,
        "happiness": 255,
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30}
    }))
    .unwrap()
}

fn swap_sides_battle(dex: &Dex, s0: &[PokemonSet], s1: &[PokemonSet], seed: u64) -> (Battle, Battle) {
    let mut b01 = Battle::from_fixture(dex, &format!("1,2,3,{seed}"), s0, s1).unwrap();
    let mut b10 = Battle::from_fixture(dex, &format!("1,2,3,{seed}"), s1, s0).unwrap();
    let p01 = [0, 1].map(|s| b01.legal_choices(dex, s).into_iter().next());
    let p10 = [0, 1].map(|s| b10.legal_choices(dex, s).into_iter().next());
    b01.apply_choices(dex, p01).unwrap();
    b10.apply_choices(dex, p10).unwrap();
    (b01, b10)
}

#[test]
fn solver_leaf_is_strictly_antisymmetric_across_1v1_multimon_wobbuffet_and_forced_switches() {
    let dex = common::dex();
    let w = EvalWeights::default();

    // 1. Multi-mon (3v3) with Spikes, stat boosts, and damaged HP
    let t_a = vec![
        make_set("Snorlax", "Leftovers", &["Curse", "Body Slam", "Earthquake", "Rest"], 55),
        make_set("Zapdos", "Mint Berry", &["Thunder", "Hidden Power Ice", "Whirlwind", "Rest"], 50),
        make_set("Cloyster", "Leftovers", &["Spikes", "Surf", "Ice Beam", "Explosion"], 50),
    ];
    let t_b = vec![
        make_set("Marowak", "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Hidden Power Bug"], 50),
        make_set("Starmie", "Miracle Berry", &["Surf", "Psychic", "Thunderbolt", "Recover"], 50),
        make_set("Skarmory", "Leftovers", &["Drill Peck", "Curse", "Whirlwind", "Rest"], 55),
    ];
    let (mut b01, mut b10) = swap_sides_battle(dex, &t_a, &t_b, 17);
    let a0 = b01.active_id(0).unwrap();
    let a1 = b01.active_id(1).unwrap();
    b01.poke_mut(a0).hp = 140;
    b01.poke_mut(a0).boosts[0] = 1;
    b01.poke_mut(a1).hp = 95;

    let b0 = b10.active_id(0).unwrap();
    let b1 = b10.active_id(1).unwrap();
    b10.poke_mut(b0).hp = 95;
    b10.poke_mut(b1).hp = 140;
    b10.poke_mut(b1).boosts[0] = 1;

    let v01 = eval_solver_leaf(&b01, dex, &w);
    let v10 = eval_solver_leaf(&b10, dex, &w);
    assert!(
        (v01 + v10 - 1.0).abs() < 1e-6,
        "3v3 antisymmetry drift: v01={v01}, v10={v10}, sum={}",
        v01 + v10
    );

    // Verify 3-point HP distribution quadrature preserves strict antisymmetry too
    let (t01, cb01) = CompactBattle::from_battle(&b01, dex);
    let (t10, cb10) = CompactBattle::from_battle(&b10, dex);
    let vd01 = compact_eval_solver_leaf_dist(
        &cb01,
        &t01,
        &w,
        [142.5, 93.0],
        [9.5, 7.2],
        [(120, 165), (78, 110)],
    );
    let vd10 = compact_eval_solver_leaf_dist(
        &cb10,
        &t10,
        &w,
        [93.0, 142.5],
        [7.2, 9.5],
        [(78, 110), (120, 165)],
    );
    assert!(
        (vd01 + vd10 - 1.0).abs() < 1e-6,
        "3-point quadrature antisymmetry drift: vd01={vd01}, vd10={vd10}"
    );

    // 2. 1v1 Counter / Mirror Coat (Wobbuffet) antisymmetry & non-zero threat recognition
    let wob = vec![make_set("Wobbuffet", "Leftovers", &["Counter", "Mirror Coat", "Safeguard", "Destiny Bond"], 50)];
    let sta = vec![make_set("Starmie", "", &["Surf", "Return", "Recover", "Thunder Wave"], 50)];
    let (mut w01, mut w10) = swap_sides_battle(dex, &wob, &sta, 29);
    let wa0 = w01.active_id(0).unwrap();
    let wa1 = w01.active_id(1).unwrap();
    w01.poke_mut(wa0).hp = 260;
    w01.poke_mut(wa1).hp = 70;
    let wb0 = w10.active_id(0).unwrap();
    let wb1 = w10.active_id(1).unwrap();
    w10.poke_mut(wb0).hp = 70;
    w10.poke_mut(wb1).hp = 260;

    let vw01 = eval_solver_leaf(&w01, dex, &w);
    let vw10 = eval_solver_leaf(&w10, dex, &w);
    assert!((vw01 + vw10 - 1.0).abs() < 1e-6);
    assert!(
        vw01 > 0.35,
        "healthy Wobbuffet vs low-HP Starmie must not be evaluated as helpless (got {vw01})"
    );

    // 3. Forced-switch (`request == 2`) game-theoretic resolution antisymmetry
    let (mut f01, mut f10) = swap_sides_battle(dex, &t_a, &t_b, 41);
    let (ft01, mut fcb01) = CompactBattle::from_battle(&f01, dex);
    let (ft10, mut fcb10) = CompactBattle::from_battle(&f10, dex);
    fcb01.mons[0].hp_pool[0] = 0;
    fcb01.sides[0].alive_mask &= !1;
    fcb01.sides[0].request = 2;
    fcb01.sides[1].request = 0;

    fcb10.mons[3].hp_pool[0] = 0;
    fcb10.sides[1].alive_mask &= !1;
    fcb10.sides[1].request = 2;
    fcb10.sides[0].request = 0;
    let _ = (&mut f01, &mut f10);

    let vf01 = compact_eval_solver_leaf(&fcb01, &ft01, &w);
    let vf10 = compact_eval_solver_leaf(&fcb10, &ft10, &w);
    assert!(
        (vf01 + vf10 - 1.0).abs() < 1e-6,
        "forced-switch antisymmetry drift: vf01={vf01}, vf10={vf10}"
    );
}

#[test]
fn semantic_hp_collapses_non_threshold_rolls_splits_cliffs_and_merges_full_hp_recovery() {
    let dex = common::dex();
    let cfg = experimental_solver_config();
    let s0 = vec![make_set("Snorlax", "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"], 55)];
    let s1 = vec![make_set("Zapdos", "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Whirlwind", "Rest"], 55)];
    let (mut b_roll_lo, _) = swap_sides_battle(dex, &s0, &s1, 99);
    let mut b_roll_hi = b_roll_lo.clone();
    let mut b_ko_cliff = b_roll_lo.clone();

    let id0_lo = b_roll_lo.active_id(0).unwrap();
    let maxhp = b_roll_lo.poke(id0_lo).maxhp;
    // Two non-critical damage rolls well above 50% HP and in the same 3HKO/4HKO band:
    b_roll_lo.poke_mut(id0_lo).hp = (maxhp * 72) / 100;
    let id0_hi = b_roll_hi.active_id(0).unwrap();
    b_roll_hi.poke_mut(id0_hi).hp = (maxhp * 78) / 100;
    // Deterministic 1HKO cliff (`<= min_dmg` of Thunderbolt):
    let id0_ko = b_ko_cliff.active_id(0).unwrap();
    b_ko_cliff.poke_mut(id0_ko).hp = 18;

    let k_lo = key_for_test(&cfg, dex, &mut b_roll_lo);
    let k_hi = key_for_test(&cfg, dex, &mut b_roll_hi);
    let k_ko = key_for_test(&cfg, dex, &mut b_ko_cliff);
    assert_eq!(
        k_lo, k_hi,
        "non-threshold damage rolls within the same KO/Sub/Heal/Flail band must collapse to one key"
    );
    assert_ne!(
        k_lo, k_ko,
        "crossing into 1HKO cliff must split into a distinct semantic key"
    );

    // Online Welford HP distribution records roll width and resets to exact 0 variance on full-HP Rest
    let mut dist = MonHpDist::default();
    dist.observe(((maxhp * 72) / 100) as i16, maxhp as i16);
    dist.observe(((maxhp * 78) / 100) as i16, maxhp as i16);
    dist.observe(((maxhp * 75) / 100) as i16, maxhp as i16);
    assert!(dist.std_dev() > 4.0);
    assert!(dist.min_hp < dist.max_hp);

    // Markovian recovery confluence at exact full HP (`maxhp`)
    dist.observe(maxhp as i16, maxhp as i16);
    assert_eq!(dist.std_dev(), 0.0);
    assert_eq!(dist.min_hp, maxhp as i16);
    assert_eq!(dist.max_hp, maxhp as i16);

    b_roll_lo.poke_mut(id0_lo).hp = maxhp;
    b_ko_cliff.poke_mut(id0_ko).hp = maxhp;
    assert_eq!(
        key_for_test(&cfg, dex, &mut b_roll_lo),
        key_for_test(&cfg, dex, &mut b_ko_cliff),
        "divergent prior HP paths recovering to full HP must merge via Markovity"
    );
}

#[test]
fn three_rank_pruning_hard_prunes_noop_and_soft_gates_marked_actions() {
    let dex = common::dex();
    let cfg = experimental_solver_config();
    let s0 = vec![
        make_set("Marowak", "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"], 50),
        make_set("Starmie", "Leftovers", &["Surf", "Thunderbolt", "Safeguard", "Recover"], 50),
    ];
    let s1 = vec![
        make_set("Zapdos", "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Reflect", "Rest"], 50),
        make_set("Raikou", "Leftovers", &["Thunderbolt", "Crunch", "Reflect", "Rest"], 50),
    ];
    let (mut b, _) = swap_sides_battle(dex, &s0, &s1, 55);

    // Marowak vs Flying-type Zapdos at full HP:
    // - `Earthquake` into Flying Zapdos is a switch-read -> `ActionRank::Marked`
    // - `Rest` at 100% HP is premature recovery -> `ActionRank::Marked`
    // - `Rock Slide` and `Swords Dance` are `ActionRank::Unmarked`
    let root_acts = b.legal_choices(dex, 0);
    let root_ranks = classify_actions_public_root(&b, dex, 0, &root_acts, &cfg);
    let eq_id = dex.moves.id("earthquake").unwrap();
    let rs_id = dex.moves.id("rockslide").unwrap();
    let rest_id = dex.moves.id("rest").unwrap();
    for (act, rank) in root_acts.iter().zip(root_ranks.iter()) {
        if *act == SearchChoice::Move(eq_id) {
            assert_eq!(*rank, ActionRank::Marked, "Earthquake into Zapdos must be soft-gated Marked");
        } else if *act == SearchChoice::Move(rest_id) {
            assert_eq!(*rank, ActionRank::Marked, "Rest at 100% HP must be soft-gated Marked");
        } else if *act == SearchChoice::Move(rs_id) {
            assert_eq!(*rank, ActionRank::Unmarked, "Rock Slide into Zapdos must be Unmarked");
        }
    }

    // Interior `NoOp` hard-pruning: when Reflect is already active on Side 1, `Reflect` is stripped
    let src1 = b.active_id(1);
    b.add_side_condition(dex, 1, "reflect", src1, nc2000_engine::battle::EffectHandle::None);
    let mut s1_acts: SmallVec<[SearchChoice; 6]> = SmallVec::new();
    b.legal_choices_into(dex, 1, &mut s1_acts);
    let refl_move = SearchChoice::Move(dex.moves.id("reflect").unwrap());
    assert!(s1_acts.contains(&refl_move));
    let _ = classify_actions_sim(&b, dex, 1, &mut s1_acts, &cfg);
    assert!(
        !s1_acts.contains(&refl_move),
        "redundant Reflect while Reflect is already active must be hard-pruned as NoOp"
    );
}

#[test]
fn ismcts_aggregates_visible_states_deepens_search_and_nails_report_4296_tactics() {
    let dex = common::dex();
    let pool = common::team_pool_v2();

    // 1. Compare search depth & node aggregation: explicit experimental 5-pillar ISMCTS + solver vs legacy heavy rollout
    let own = pool.teams[0].sets.clone();
    let cand_a = pool.teams[1].sets.clone();
    let mut cand_b = cand_a.clone();
    // Ambiguous hidden bench set difference (never revealed on Turn 1)
    cand_b[5].happiness = Some(0);
    let meta = Arc::new(MetaPool {
        teams: vec![
            nc2000_bot::preview::MetaTeam {
                id: "cand-a".into(),
                sets: cand_a.clone(),
                weight: None,
            },
            nc2000_bot::preview::MetaTeam {
                id: "cand-b".into(),
                sets: cand_b.clone(),
                weight: None,
            },
        ],
    });

    let mut b_a = Battle::from_fixture(dex, "1,2,3,4", &own, &cand_a).unwrap();
    let mut b_b = Battle::from_fixture(dex, "1,2,3,4", &own, &cand_b).unwrap();
    for b in [&mut b_a, &mut b_b] {
        let p0 = b.legal_choices(dex, 0)[0];
        let p1 = b.legal_choices(dex, 1)[0];
        b.apply_choices(dex, [Some(p0), Some(p1)]).unwrap();
    }

    let obs_a = Observer::new(&b_a, 0);
    let mut bel_a = Belief::new(dex, &meta, &obs_a);
    bel_a.sync(dex, &obs_a);
    let obs_b = Observer::new(&b_b, 0);
    let mut bel_b = Belief::new(dex, &meta, &obs_b);
    bel_b.sync(dex, &obs_b);
    assert_eq!(bel_a.candidate_count(), 2);

    let iters = 2_400;
    let cfg_new = RmConfig {
        iterations: iters,
        c: 0.4,
        ..experimental_solver_config()
    };
    let cfg_leg = RmConfig {
        iterations: iters,
        c: 0.4,
        ..RmConfig::legacy_heavy_rollout()
    };

    let mut s_new_a = BlindSearch::new(&b_a, dex, cfg_new.clone(), 0, 777);
    s_new_a.step(dex, &bel_a, &obs_a, iters);

    let mut s_new_b = BlindSearch::new(&b_b, dex, cfg_new, 0, 777);
    s_new_b.step(dex, &bel_b, &obs_b, iters);

    // Zero hidden-state leak across ambiguous hidden candidates
    assert_eq!(s_new_a.visits(), s_new_b.visits());
    assert_eq!(s_new_a.means(), s_new_b.means());
    assert_eq!(s_new_a.best(), s_new_b.best());

    let mut s_leg = BlindSearch::new(&b_a, dex, cfg_leg, 0, 777);
    s_leg.step(dex, &bel_a, &obs_a, iters);

    assert!(
        s_new_a.mean_depth() > s_leg.mean_depth() + 0.75,
        "5-pillar ISMCTS must search substantially deeper than legacy rollout: new_mean={:.2} vs leg_mean={:.2}",
        s_new_a.mean_depth(),
        s_leg.mean_depth()
    );
    assert!(
        s_new_a.max_depth() > s_leg.max_depth(),
        "5-pillar ISMCTS max_depth={} must exceed legacy max_depth={}",
        s_new_a.max_depth(),
        s_leg.max_depth()
    );

    // 2. Tactical verification on `report-4296` (Turns 11, 24, 26)
    let root = repo_root();
    let dir = root.join("data/report-4296");
    let opp_sets: Vec<PokemonSet> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("opponent-team.json")).unwrap())
            .unwrap();
    let mpool = common::meta_pool_v0().clone();

    let run_report_turn = |turn: u16| -> (String, String) {
        let spec = PositionSpec::parse(
            &std::fs::read_to_string(dir.join(format!("turn-{turn}.json"))).unwrap(),
        )
        .unwrap();
        let cfg = RmConfig {
            iterations: 3_000,
            c: 0.4,
            ..experimental_solver_config()
        };
        let mut agent = ProtocolAgent::new(dex, 1, mpool.clone(), cfg, 61001);
        agent.pin_opponent(opp_sets.clone());
        agent.set_position(dex, &spec).unwrap();
        agent.step(dex, 3_000).unwrap();
        let s = agent.search().unwrap();
        let dbg = format!(
            "acts={:?} visits={:?} means={:?} eq={:?} mat={:?}",
            s.actions().iter().map(|a| a.to_input(dex)).collect::<Vec<_>>(),
            s.visits(),
            s.means(),
            s.bayesian_equilibrium().map(|e| (e.policy, e.action_values)),
            s.root_matrix()
                .into_iter()
                .map(|(i, c, n, m)| (i, c.to_input(dex), n, (m * 1000.0).round() / 1000.0))
                .collect::<Vec<_>>()
        );
        (agent.best(dex).unwrap(), dbg)
    };

    let (t11, d11) = run_report_turn(11);
    assert_eq!(
        t11,
        "move earthquake",
        "Turn 11: Marowak must click Earthquake to KO Shuckle before Rest ({d11})"
    );
    let (t24, d24) = run_report_turn(24);
    assert_ne!(
        t24,
        "move swordsdance",
        "Turn 24: poisoned low-HP Marowak must not click suicidal Swords Dance ({d24})"
    );
    let (t26, d26) = run_report_turn(26);
    assert_eq!(
        t26,
        "move thunderbolt",
        "Turn 26: must prefer Thunderbolt over Fire Punch against 13% HP Starmie ({d26})"
    );
}
