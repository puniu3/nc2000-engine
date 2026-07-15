//! THE conformance test: replay every golden fixture on the Rust engine and
//! require bit-exact snapshot parity (state + prng seed at every snapshot).

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::{load_dex, replay};

fn run_corpus(name: &str) {
    let dex = load_dex();
    let files = corpus_files(&repo_root().join("fixtures/corpus-v1").join(name));
    assert!(!files.is_empty());
    let mut failures = Vec::new();
    for path in files {
        let fx = Fixture::load(&path).unwrap();
        if let Err(e) = replay(&dex, &fx) {
            failures.push(format!("{path:?}: {e:?}"));
        }
    }
    assert!(failures.is_empty(), "{} fixtures diverged:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn puredata_corpus_replays_bit_exact() {
    run_corpus("puredata");
}

#[test]
fn full_corpus_replays_bit_exact() {
    run_corpus("full");
}
