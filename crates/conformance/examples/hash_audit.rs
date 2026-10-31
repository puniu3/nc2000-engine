//! Compare the exact stable and transient hash streams of two engine builds.
//! Emit deterministic JSON for every request boundary in the frozen fixtures.
use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::{load_dex, ps_reference_battle};
use nc2000_engine::state::Battle;
use serde_json::json;

fn keys(battle: &Battle) -> Vec<String> {
    let mut out = vec![battle.state_key().to_string(), battle.state_key128().to_string()];
    for buckets in [None, Some(1), Some(16), Some(256)] {
        for omit in [false, true] {
            out.push(battle.search_key(buckets, omit).to_string());
        }
    }
    out
}

fn main() {
    let dex = load_dex();
    let mut states = 0;
    let mut fixtures = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&repo_root().join("fixtures/corpus-v1").join(corpus)) {
            let fixture = Fixture::load(&path).unwrap();
            let mut battle = ps_reference_battle(&dex, &fixture.seed, &fixture.p1team, &fixture.p2team).unwrap();
            let mut rows = vec![keys(&battle)];
            for choice in &fixture.choices {
                battle.choose(&dex, usize::from(choice.side != "p1"), &choice.choice).unwrap();
                rows.push(keys(&battle));
            }
            states += rows.len();
            fixtures.push(json!({"corpus":corpus,"index":fixture.meta.index,
                "keys":rows,"final_state":battle.essence(&dex),"log":battle.log}));
        }
    }
    println!("{}", json!({"schema":"nc2000-hash-audit-v1","states":states,"fixtures":fixtures}));
}
