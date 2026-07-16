//! M8 meta pool smoke test: every team in data/meta-pool-v0/meta-pool.json
//! loads as engine `PokemonSet`s and plays random full games to completion
//! against its neighbor in the ranking.

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;

#[derive(serde::Deserialize)]
struct MetaPool {
    teams: Vec<MetaTeam>,
}

#[derive(serde::Deserialize)]
struct MetaTeam {
    id: String,
    sets: Vec<PokemonSet>,
}

struct TestRng(u64);
impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn pick(&mut self, len: usize) -> usize {
        (self.next() % len as u64) as usize
    }
}

#[test]
fn meta_pool_teams_play_to_completion() {
    let path = repo_root().join("data/meta-pool-v0/meta-pool.json");
    let pool: MetaPool =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert!(pool.teams.len() >= 30, "pool unexpectedly small: {}", pool.teams.len());
    let dex = load_dex();

    for (i, team) in pool.teams.iter().enumerate() {
        assert_eq!(team.sets.len(), 6, "{}: want 6 sets", team.id);
        let opp = &pool.teams[(i + 1) % pool.teams.len()];
        let mut battle =
            Battle::from_fixture(&dex, "1,2,3,4", &team.sets, &opp.sets).unwrap();
        battle.set_log_enabled(false);
        battle.reseed(0x00C0_FFEE ^ (i as u64));
        let mut rng = TestRng(0xFEED_FACE ^ (i as u64) << 24);
        let mut steps = 0u32;
        while battle.outcome().is_none() {
            steps += 1;
            assert!(steps < 10_000, "{} vs {}: no termination", team.id, opp.id);
            let picks = [0usize, 1].map(|side_n| {
                let legal = battle.legal_choices(&dex, side_n);
                if legal.is_empty() {
                    None
                } else {
                    Some(legal[rng.pick(legal.len())])
                }
            });
            assert!(
                picks.iter().any(|p| p.is_some()),
                "{} vs {}: battle running but nobody owes a choice",
                team.id,
                opp.id,
            );
            battle.apply_choices(&dex, picks).unwrap();
        }
    }
}
