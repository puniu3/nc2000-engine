//! Conformance harness: replays golden fixtures (generated from the PS
//! reference by `tools/gen-fixtures.js`) against the Rust engine and reports
//! the first divergence per battle.

pub mod compare;
pub mod fixture;

use fixture::Fixture;
use nc2000_engine::battle::EngineError;
use nc2000_engine::dex::Dex;

/// Replays one fixture on the Rust engine, checking every snapshot.
///
/// Returns Ok(()) on full parity, the first `Divergence` as an error string,
/// or `EngineError::Unimplemented` while the port is incomplete.
pub fn replay(_dex: &Dex, fx: &Fixture) -> Result<(), EngineError> {
    // Milestone 1 wiring (activate as Battle::from_fixture lands):
    //   let mut battle = Battle::from_fixture(dex, &fx.seed, &fx.p1team, &fx.p2team)?;
    //   compare snapshot 0; then for each choice line, apply and compare on
    //   log growth, per the snapshot contract in fixture.rs.
    let _ = fx;
    Err(EngineError::Unimplemented("replay: engine port has not reached milestone 1"))
}

pub fn load_dex() -> Dex {
    let path = fixture::repo_root().join("data/gen2stadium2.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    Dex::from_json(&json).expect("dex JSON must parse")
}
