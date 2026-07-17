//! M8 table-consumption tests: the baked/counter agents must resolve live
//! rosters to pool teams, honor matrix orientation (row team a vs column
//! team b, transposed lookup when playing the other seat), and pick exactly
//! what the table says — a silent fallback to the inner agent would pass any
//! arena smoke test, so this pins the lookup path itself.

use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::{
    load_meta_pool, preview_actions, BakeCfg, MatrixEst, PairSolution, PairTable, SPACE_VERSION,
};
use nc2000_bot::{Agent, BakedPreviewAgent, CounterPickAgent, PreviewMode, RandomAgent, TableSet};
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::state::Battle;

// Action indices used below, all Max-Total-Level-legal for BOTH pool teams
// 0 (52/52/52/51/51/51) and 1 (55/55/50/50/50/50):
//   actions[3]  = [1,2,4]  (155 / — team 1: 160, so only used on side a)
//   actions[12] = [1,3,4]  (155 / 155)
//   actions[30] = [2,3,4]  (155 / 155)
/// Table over supports s0 = {actions[3], actions[12]},
/// s1 = {actions[12], actions[30]} with refine = [[0.9, 0.4], [0.6, 0.7]]
/// (row payoff): row BR vs col-argmax(1) is row 1; col BR vs row-argmax(0)
/// is col 1.
fn test_table(team_a: &str, team_b: &str) -> PairTable {
    let actions = preview_actions();
    let (s0, s1) = (vec![3usize, 12], vec![12usize, 30]);
    let refine = MatrixEst { rows: 2, cols: 2, n: vec![10; 4], v: vec![0.9, 0.4, 0.6, 0.7] };
    let mut p_a = vec![0.0; 60];
    p_a[12] = 1.0; // mixed = point mass on actions[12] for determinism
    let mut p_b = vec![0.0; 60];
    p_b[30] = 1.0;
    PairTable {
        team_a: team_a.into(),
        team_b: team_b.into(),
        actions,
        space_version: SPACE_VERSION,
        screen: MatrixEst::new(1, 1),
        support: [s0, s1],
        refine,
        sol: PairSolution {
            p_a,
            p_b,
            argmax_a: 3,
            argmax_b: 30,
            value: 0.6,
            guarantee_mixed_a: 0.6,
            guarantee_argmax_a: 0.4,
            guarantee_mixed_b: 0.4,
            guarantee_argmax_b: 0.3,
        },
        cfg: BakeCfg {
            screen_games: 0,
            refine_games: 10,
            support: 2,
            skuct_iters: 0,
            advisor_iters: 0,
            advisor_runs: 0,
            eps: 0.2,
            max_turns: 300,
            seed: 0,
        },
        secs: 0.0,
    }
}

fn pick(agent: &mut dyn Agent, battle: &Battle, side: usize) -> SearchChoice {
    let dex = load_dex();
    let mut b = battle.clone();
    let choices = b.legal_choices(&dex, side);
    agent.choose(&b, &dex, side, &choices)
}

#[test]
fn baked_and_counter_agents_honor_table_orientation() {
    let dex = load_dex();
    let pool = load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"));
    let dir = std::env::temp_dir().join(format!("nc2000-preview-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let table = test_table(&pool.teams[0].id, &pool.teams[1].id);
    std::fs::write(dir.join("pair-00-01.json"), serde_json::to_string(&table).unwrap()).unwrap();
    let tables: Arc<TableSet> = TableSet::load(&dex, &pool, &dir);
    assert_eq!(tables.len(), 1);
    let actions = preview_actions();

    // battle in table orientation (team 0 = P1/row) and transposed
    let fwd =
        Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[0].sets, &pool.teams[1].sets).unwrap();
    let rev =
        Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[1].sets, &pool.teams[0].sets).unwrap();

    let baked = |mode, seed| {
        BakedPreviewAgent::new(tables.clone(), Box::new(RandomAgent::new(seed)), mode, seed)
    };

    // argmax: row team picks argmax_a wherever it sits
    let mut a = baked(PreviewMode::Argmax, 1);
    assert_eq!(pick(&mut a, &fwd, 0), SearchChoice::Team(actions[3]), "row seat, P1");
    assert_eq!(pick(&mut a, &rev, 1), SearchChoice::Team(actions[3]), "row seat, P2");
    // ... and the column team picks argmax_b from either seat
    assert_eq!(pick(&mut a, &fwd, 1), SearchChoice::Team(actions[30]), "col seat, P2");
    assert_eq!(pick(&mut a, &rev, 0), SearchChoice::Team(actions[30]), "col seat, P1");

    // mixed point mass
    let mut m = baked(PreviewMode::Mixed, 7);
    assert_eq!(pick(&mut m, &fwd, 0), SearchChoice::Team(actions[12]));
    assert_eq!(pick(&mut m, &rev, 0), SearchChoice::Team(actions[30]));

    // counter vs argmax: row BR to col-argmax (col 1 → refine col [0.4, 0.7]) is
    // support row 1 = actions[12]; col BR to row-argmax (row 0 → 1-v = [0.1, 0.6])
    // is support col 1 = actions[30]
    let counter = |target| {
        CounterPickAgent::new(tables.clone(), Box::new(RandomAgent::new(2)), target)
    };
    let mut c = counter(PreviewMode::Argmax);
    assert_eq!(pick(&mut c, &fwd, 0), SearchChoice::Team(actions[12]), "row BR");
    assert_eq!(pick(&mut c, &rev, 0), SearchChoice::Team(actions[30]), "col BR, transposed");

    // counter vs mixed point masses: same BRs (p_b = col 1, p_a = row 1 → col BR
    // over 1-v[1][.] = [0.4, 0.3] is col 0 = actions[12])
    let mut c = counter(PreviewMode::Mixed);
    assert_eq!(pick(&mut c, &fwd, 0), SearchChoice::Team(actions[12]));
    assert_eq!(pick(&mut c, &fwd, 1), SearchChoice::Team(actions[12]));

    // unknown matchup (fixture teams absent from the pool) falls back to inner:
    // exercised implicitly — a pool battle between teams 2 and 3 has no table
    let other =
        Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[2].sets, &pool.teams[3].sets).unwrap();
    let mut a = baked(PreviewMode::Argmax, 3);
    let c = pick(&mut a, &other, 0);
    assert!(matches!(c, SearchChoice::Team(_)), "fallback still answers the request");

    std::fs::remove_dir_all(&dir).ok();
}

/// Stale-table detection (the 2026-07-17 Max Total Level preview fix): a
/// pre-fix file (space_version 0) must be REJECTED for a pair whose teams
/// have illegal 3-subsets (it was baked over the wrong action space) and
/// still ACCEPTED for a pair whose teams have every subset legal (the old
/// bake then coincides with the legal space). Rejected tables read as
/// missing — consumers fall back to live preview search.
#[test]
fn stale_pre_fix_tables_are_rejected() {
    let dex = load_dex();
    let pool = load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"));
    let actions = preview_actions();
    let affected = |i: usize| {
        let lv: Vec<u32> = pool.teams[i].sets.iter().map(|s| s.level as u32).collect();
        actions.iter().any(|t| t.iter().map(|&s| lv[s as usize - 1]).sum::<u32>() > 155)
    };
    // teams 0 and 1 are affected (the audit this fix responds to)
    assert!(affected(0) && affected(1));
    let (u1, u2) = {
        let mut it = (0..pool.teams.len()).filter(|&i| !affected(i));
        (it.next().expect("an unaffected team"), it.next().expect("two unaffected teams"))
    };

    // pre-fix file = current table with the version field stripped
    // (serde-defaults to 0), support/solution unchanged
    let strip = |tab: &PairTable| {
        let mut v: serde_json::Value = serde_json::to_value(tab).unwrap();
        v.as_object_mut().unwrap().remove("space_version");
        serde_json::from_value::<PairTable>(v).unwrap()
    };

    let mut set = TableSet::from_pool(&dex, &pool);
    let stale = strip(&test_table(&pool.teams[0].id, &pool.teams[1].id));
    assert_eq!(stale.space_version, 0);
    assert!(
        set.add_pair(stale).is_err(),
        "pre-fix table on an affected pair must be rejected as stale"
    );
    assert_eq!(set.len(), 0);

    let ok = strip(&test_table(&pool.teams[u1].id, &pool.teams[u2].id));
    set.add_pair(ok).expect("pre-fix table on an unaffected pair stays valid");
    assert_eq!(set.len(), 1);

    // current-version file with equilibrium mass on an illegal action:
    // rejected (defense in depth against a mislabeled file)
    let mut bad = test_table(&pool.teams[0].id, &pool.teams[1].id);
    bad.sol.p_b[0] = 0.5; // actions[0] = [1,2,3]: 160 for team 1
    assert!(
        set.add_pair(bad).is_err(),
        "current-version table with illegal-action mass must be rejected"
    );
}
