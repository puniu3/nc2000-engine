//! Every fixture in both corpora parses against the schema, its choices parse
//! against the choice grammar, and its seeds parse against the PRNG.

use conformance::fixture::{corpus_files, repo_root, Fixture};
use nc2000_engine::choice::Choice;
use nc2000_engine::prng::Prng;

fn corpora() -> Vec<std::path::PathBuf> {
    let root = repo_root();
    let mut files = corpus_files(&root.join("fixtures/corpus-v1/puredata"));
    files.extend(corpus_files(&root.join("fixtures/corpus-v1/full")));
    files
}

#[test]
fn all_fixtures_parse_and_are_wellformed() {
    let files = corpora();
    assert_eq!(files.len(), 60, "expected 60 fixtures");
    for path in files {
        let fx = Fixture::load(&path).unwrap();
        assert_eq!(fx.p1team.len(), 6);
        assert_eq!(fx.p2team.len(), 6);
        assert!(fx.snapshots.len() >= 2, "{path:?}: too few snapshots");
        assert!(!fx.choices.is_empty());
        assert!(
            Prng::from_seed_str(&fx.seed).is_some(),
            "{path:?}: battle seed unparseable: {}",
            fx.seed
        );
        for snap in &fx.snapshots {
            assert!(
                Prng::from_seed_str(&snap.prng_seed).is_some(),
                "{path:?}: snapshot seed unparseable: {}",
                snap.prng_seed
            );
            assert_eq!(snap.sides.len(), 2);
            for side in &snap.sides {
                // 6 registered before team preview resolves; PS truncates
                // side.pokemon to the 3 picked once the battle starts.
                assert!(
                    side.pokemon.len() == 6 || side.pokemon.len() == 3,
                    "{path:?}: unexpected party size {}",
                    side.pokemon.len()
                );
            }
        }
        for line in &fx.choices {
            Choice::parse(&line.choice)
                .unwrap_or_else(|e| panic!("{path:?}: line {}: {e}", line.index));
        }
        // NC2000: exactly 3 picked per side → 3 mons should ever act per side.
        let last = fx.snapshots.last().unwrap();
        for side in &last.sides {
            assert!(side.pokemon_left <= 3, "{path:?}: picked team size violated");
        }
    }
}

#[test]
fn gen2_semantics_hold_in_fixtures() {
    // Teams are validator-canonical: gen2 = 'No Ability' everywhere.
    for path in corpora() {
        let fx = Fixture::load(&path).unwrap();
        for set in fx.p1team.iter().chain(fx.p2team.iter()) {
            assert_eq!(set.ability, "No Ability", "{path:?}: {}", set.name);
            assert!((50..=55).contains(&set.level), "{path:?}: level rule");
        }
    }
}
