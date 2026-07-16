//! M8 table-consumption tests: the baked/counter agents must resolve live
//! rosters to pool teams, honor matrix orientation (row team a vs column
//! team b, transposed lookup when playing the other seat), and pick exactly
//! what the table says — a silent fallback to the inner agent would pass any
//! arena smoke test, so this pins the lookup path itself.

use std::sync::Arc;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::{
    load_meta_pool, preview_actions, BakeCfg, MatrixEst, PairSolution, PairTable,
};
use nc2000_bot::{Agent, BakedPreviewAgent, CounterPickAgent, PreviewMode, RandomAgent, TableSet};
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::state::Battle;

/// Table over supports s0 = {actions[0], actions[3]}, s1 = {actions[0], actions[6]}
/// with refine = [[0.9, 0.4], [0.6, 0.7]] (row payoff):
/// row BR vs col-argmax(1) is row 1; col BR vs row-argmax(0) is col 1.
fn test_table(team_a: &str, team_b: &str) -> PairTable {
    let actions = preview_actions();
    let (s0, s1) = (vec![0usize, 3], vec![0usize, 6]);
    let refine = MatrixEst { rows: 2, cols: 2, n: vec![10; 4], v: vec![0.9, 0.4, 0.6, 0.7] };
    let mut p_a = vec![0.0; 60];
    p_a[3] = 1.0; // mixed = point mass on actions[3] for determinism
    let mut p_b = vec![0.0; 60];
    p_b[6] = 1.0;
    PairTable {
        team_a: team_a.into(),
        team_b: team_b.into(),
        actions,
        screen: MatrixEst::new(1, 1),
        support: [s0, s1],
        refine,
        sol: PairSolution {
            p_a,
            p_b,
            argmax_a: 0,
            argmax_b: 6,
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
    assert_eq!(pick(&mut a, &fwd, 0), SearchChoice::Team(actions[0]), "row seat, P1");
    assert_eq!(pick(&mut a, &rev, 1), SearchChoice::Team(actions[0]), "row seat, P2");
    // ... and the column team picks argmax_b from either seat
    assert_eq!(pick(&mut a, &fwd, 1), SearchChoice::Team(actions[6]), "col seat, P2");
    assert_eq!(pick(&mut a, &rev, 0), SearchChoice::Team(actions[6]), "col seat, P1");

    // mixed point mass
    let mut m = baked(PreviewMode::Mixed, 7);
    assert_eq!(pick(&mut m, &fwd, 0), SearchChoice::Team(actions[3]));
    assert_eq!(pick(&mut m, &rev, 0), SearchChoice::Team(actions[6]));

    // counter vs argmax: row BR to col-argmax (col 1 → refine col [0.4, 0.7]) is
    // support row 1 = actions[3]; col BR to row-argmax (row 0 → 1-v = [0.1, 0.6])
    // is support col 1 = actions[6]
    let counter = |target| {
        CounterPickAgent::new(tables.clone(), Box::new(RandomAgent::new(2)), target)
    };
    let mut c = counter(PreviewMode::Argmax);
    assert_eq!(pick(&mut c, &fwd, 0), SearchChoice::Team(actions[3]), "row BR");
    assert_eq!(pick(&mut c, &rev, 0), SearchChoice::Team(actions[6]), "col BR, transposed");

    // counter vs mixed point masses: same BRs (p_b = col 1, p_a = row 1 → col BR
    // over 1-v[1][.] = [0.4, 0.3] is col 0 = actions[0])
    let mut c = counter(PreviewMode::Mixed);
    assert_eq!(pick(&mut c, &fwd, 0), SearchChoice::Team(actions[3]));
    assert_eq!(pick(&mut c, &fwd, 1), SearchChoice::Team(actions[0]));

    // unknown matchup (fixture teams absent from the pool) falls back to inner:
    // exercised implicitly — a pool battle between teams 2 and 3 has no table
    let other =
        Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[2].sets, &pool.teams[3].sets).unwrap();
    let mut a = baked(PreviewMode::Argmax, 3);
    let c = pick(&mut a, &other, 0);
    assert!(matches!(c, SearchChoice::Team(_)), "fallback still answers the request");

    std::fs::remove_dir_all(&dir).ok();
}
