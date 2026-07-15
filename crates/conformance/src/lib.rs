//! Conformance harness: replays golden fixtures (generated from the PS
//! reference by `tools/gen-fixtures.js`) against the Rust engine and reports
//! the first divergence per battle.

pub mod compare;
pub mod fixture;

use compare::{first_diff, Divergence};
use fixture::{Fixture, Snapshot};
use nc2000_engine::battle::EngineError;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

#[derive(Debug)]
pub enum ReplayError {
    Engine(String),
    Diverged(Box<Divergence>),
    /// The engine produced fewer/more snapshots than the fixture.
    SnapshotCount { expected: usize, actual: usize },
}

/// Compare one snapshot (state essence + log window) against the fixture.
fn check_snapshot(
    dex: &Dex,
    battle: &Battle,
    snap: &Snapshot,
    index: usize,
    log_from: usize,
) -> Result<(), Box<Divergence>> {
    let actual = battle.essence(dex);
    let expected = serde_json::json!({
        "turn": snap.turn,
        "prngSeed": snap.prng_seed,
        "requestState": snap.request_state,
        "field": {
            "weather": snap.field.weather,
            "weatherState": snap.field.weather_state,
            "pseudoWeather": snap.field.pseudo_weather,
        },
        "sides": snap.sides.iter().map(|s| serde_json::json!({
            "pokemonLeft": s.pokemon_left,
            "sideConditions": s.side_conditions,
            "active": s.active,
            "pokemon": s.pokemon.iter().map(|p| serde_json::json!({
                "ident": p.ident,
                "species": p.species,
                "hp": p.hp,
                "maxhp": p.maxhp,
                "fainted": p.fainted,
                "status": p.status,
                "statusState": p.status_state,
                "boosts": p.boosts,
                "item": p.item,
                "lastItem": p.last_item,
                "itemState": p.item_state,
                "moves": p.moves.iter().map(|m| serde_json::json!({
                    "id": m.id, "pp": m.pp, "disabled": m.disabled,
                })).collect::<Vec<_>>(),
                "volatiles": p.volatiles,
                "types": p.types,
                "transformed": p.transformed,
                "active": p.active,
                "position": p.position,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });

    // state diff
    if let Some((path, exp, act)) = first_diff(&expected, &actual, "") {
        return Err(Box::new(Divergence {
            snapshot_index: index,
            turn: snap.turn,
            path,
            expected: exp,
            actual: act,
            log_context: snap.log.clone(),
        }));
    }

    // log window diff
    let actual_log: Vec<&str> = battle.log[log_from..].iter().map(|s| s.as_str()).collect();
    for (i, expected_line) in snap.log.iter().enumerate() {
        let actual_line = actual_log.get(i).copied().unwrap_or("<missing>");
        if expected_line != actual_line {
            return Err(Box::new(Divergence {
                snapshot_index: index,
                turn: snap.turn,
                path: format!(".log[{i}]"),
                expected: expected_line.clone(),
                actual: actual_line.to_string(),
                log_context: snap.log.clone(),
            }));
        }
    }
    if actual_log.len() != snap.log.len() {
        return Err(Box::new(Divergence {
            snapshot_index: index,
            turn: snap.turn,
            path: ".log.len".into(),
            expected: snap.log.len().to_string(),
            actual: format!(
                "{} (extra: {:?})",
                actual_log.len(),
                &actual_log[snap.log.len().min(actual_log.len())..]
            ),
            log_context: snap.log.clone(),
        }));
    }
    Ok(())
}

/// Replays one fixture on the Rust engine, checking every snapshot.
pub fn replay(dex: &Dex, fx: &Fixture) -> Result<(), ReplayError> {
    let mut battle = Battle::from_fixture(dex, &fx.seed, &fx.p1team, &fx.p2team)
        .map_err(|e| ReplayError::Engine(format!("{e:?}")))?;

    let mut snap_idx = 0;
    let mut log_pos = 0;

    // snapshot 0: right after both players are set
    if let Some(snap) = fx.snapshots.first() {
        check_snapshot(dex, &battle, snap, 0, log_pos).map_err(ReplayError::Diverged)?;
        log_pos = battle.log.len();
        snap_idx = 1;
    }

    for line in &fx.choices {
        let side_n = if line.side == "p1" { 0 } else { 1 };
        let before_len = battle.log.len();
        match battle.choose(dex, side_n, &line.choice) {
            Ok(()) => {}
            Err(EngineError::Unimplemented(what)) => {
                return Err(ReplayError::Engine(format!("unimplemented: {what}")));
            }
            Err(e) => return Err(ReplayError::Engine(format!("choice {:?}: {e:?}", line.choice))),
        }
        if battle.log.len() > before_len {
            let Some(snap) = fx.snapshots.get(snap_idx) else {
                return Err(ReplayError::SnapshotCount {
                    expected: fx.snapshots.len(),
                    actual: snap_idx + 1,
                });
            };
            check_snapshot(dex, &battle, snap, snap_idx, log_pos).map_err(ReplayError::Diverged)?;
            log_pos = battle.log.len();
            snap_idx += 1;
        }
    }
    if snap_idx != fx.snapshots.len() {
        return Err(ReplayError::SnapshotCount { expected: fx.snapshots.len(), actual: snap_idx });
    }
    Ok(())
}

pub fn load_dex() -> Dex {
    let path = fixture::repo_root().join("data/gen2stadium2.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    Dex::from_json(&json).expect("dex JSON must parse")
}
