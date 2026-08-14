//! The solver report: everything a study screen shows about one position.
//!
//! Assembled here rather than in the wasm bridge so the browser and the CLI
//! (`examples/solve_position.rs`) read the same numbers — the parity twin
//! rule this repo already applies to the benchmarks.
//!
//! Three things are reported beyond the score, because a score alone teaches
//! nothing about *why*:
//!
//! - the **root matrix**: the search's estimate per (our action, their
//!   action) pair. The per-action means everyone quotes are this matrix
//!   marginalized over what the opponent actually got played, which is
//!   exactly the information a human needs and the marginal destroys;
//! - the **damage table**: engine-truth damage, computed through
//!   `get_damage_synthetic`, so screens, boosts, items and type effects are
//!   the real ones rather than a calculator's re-derivation;
//! - a **searched line**: one continuation the tree actually visited.
//!
//! Every number that depends on a hidden field is labelled as such. The
//! opponent's damage output, its switch targets and the assumed line all
//! rest on an imputed set; presenting them as facts would teach the user to
//! trust a guess.

use nc2000_engine::battle::moveexec::{get_active_move, get_damage_synthetic};
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::{Category, Dex, MoveId};
use nc2000_engine::state::{Battle, PokeId};
use serde_json::{json, Value};

use crate::blind::BlindSearch;
use crate::import::ProtocolAgent;
use crate::observe::Observer;

pub const SCHEMA: &str = "nc2000-analysis-v1";

/// Gen 2 damage variance: the top roll times 217/255, floored.
const MIN_ROLL_NUM: f64 = 217.0;
const MIN_ROLL_DEN: f64 = 255.0;

/// How an action reads to a UI, without any dex lookup on the other side.
fn action_json(b: &Battle, dex: &Dex, side: usize, obs: Option<&Observer>, c: SearchChoice) -> Value {
    match c {
        SearchChoice::Move(id) => json!({
            "input": c.to_input(dex),
            "kind": "move",
            "move": dex.moves.key(id),
        }),
        SearchChoice::Switch(pos) => {
            let slot = b.sides[side].party.get(pos as usize - 1).copied();
            // A never-appeared opponent mon's identity is imputed, not known:
            // naming it here would hand the user a guess in the shape of a
            // fact. Its own side always knows.
            let known = match (slot, obs) {
                (Some(sl), Some(o)) => o.mons().get(sl as usize).map(|m| m.appeared).unwrap_or(false),
                (Some(_), None) => true,
                _ => false,
            };
            let species = slot
                .filter(|_| known)
                .map(|sl| dex.species.key(b.sides[side].roster[sl as usize].species).to_string());
            json!({
                "input": c.to_input(dex),
                "kind": "switch",
                "pos": pos,
                "species": species,
            })
        }
        SearchChoice::Team(slots) => json!({
            "input": c.to_input(dex),
            "kind": "team",
            "slots": slots,
        }),
        SearchChoice::Pass => json!({"input": c.to_input(dex), "kind": "pass"}),
    }
}

/// One attacker move against the current defender.
fn damage_row(
    b: &mut Battle,
    dex: &Dex,
    att: PokeId,
    def: PokeId,
    mid: MoveId,
    revealed: bool,
) -> Value {
    let ms = dex.move_static(mid);
    let mut fake = get_active_move(dex, mid);
    fake.no_damage_variance = true;
    fake.will_crit = Some(false);
    // Hidden Power's type and power come from callbacks `get_damage` never
    // runs; plant them from the attacker's rolled DVs, exactly as the ad-hoc
    // damage harness does.
    if dex.moves.key(mid) == "hiddenpower" {
        let a = b.poke(att);
        fake.move_type = a.hp_type;
        fake.base_move_type = a.hp_type;
        let special = matches!(
            dex.type_name(a.hp_type),
            "Fire" | "Water" | "Grass" | "Electric" | "Psychic" | "Ice" | "Dragon" | "Dark"
        );
        fake.category = if special { Category::Special } else { Category::Physical };
    }
    let max = get_damage_synthetic(b, dex, att, def, fake.clone()).unwrap_or(0.0) as i32;
    let min = ((max as f64) * MIN_ROLL_NUM / MIN_ROLL_DEN).floor() as i32;
    fake.will_crit = Some(true);
    let crit = get_damage_synthetic(b, dex, att, def, fake).unwrap_or(0.0) as i32;
    let d = b.poke(def);
    let (hp, maxhp) = (d.hp, d.maxhp);
    let ko = if max <= 0 {
        "never"
    } else if min >= hp {
        "always"
    } else if max >= hp {
        "possible"
    } else {
        "never"
    };
    json!({
        "move": dex.moves.key(mid),
        "revealed": revealed,
        "min": min,
        "max": max,
        "crit": crit,
        // `null` = never misses (PS `accuracy: true`), which is a different
        // statement from 100%.
        "accuracy": match ms.accuracy {
            nc2000_engine::dex::Accuracy::AlwaysHits => None,
            nc2000_engine::dex::Accuracy::Pct(p) => Some(p),
        },
        "hp": hp,
        "maxhp": maxhp,
        // Hits to KO: the guaranteed count (every roll minimum) and the
        // luckiest one. Equal ⇒ the count is not a roll at all.
        "hitsGuaranteed": if min > 0 { Some((hp + min - 1) / min) } else { None::<i32> },
        "hitsBest": if max > 0 { Some((hp + max - 1) / max) } else { None::<i32> },
        "ko": ko,
    })
}

/// Both actives' damage in both directions. `theirs` rests on the imputed
/// opponent set the searcher is currently holding, so each row says whether
/// the move was publicly revealed or assumed.
pub fn damage_table(battle: &Battle, dex: &Dex, side: usize, obs: Option<&Observer>) -> Value {
    let mut b = battle.clone();
    b.set_log_enabled(false);
    let (Some(mine), Some(theirs)) = (b.active_id(side), b.active_id(1 - side)) else {
        return json!({"mine": [], "theirs": []});
    };
    let my_moves: Vec<MoveId> = b.poke(mine).move_slots.iter().map(|m| m.id).collect();
    let their_moves: Vec<MoveId> = b.poke(theirs).move_slots.iter().map(|m| m.id).collect();
    let their_slot = theirs.slot as usize;
    let revealed: Vec<MoveId> = obs
        .and_then(|o| o.mons().get(their_slot))
        .map(|m| m.revealed_moves.clone())
        .unwrap_or_default();
    let attack: Vec<Value> =
        my_moves.iter().map(|&m| damage_row(&mut b, dex, mine, theirs, m, true)).collect();
    let defend: Vec<Value> = their_moves
        .iter()
        .map(|&m| {
            let known = revealed.iter().any(|&r| crate::observe::move_matches(dex, m, r));
            damage_row(&mut b, dex, theirs, mine, m, known)
        })
        .collect();
    json!({"mine": attack, "theirs": defend})
}

/// The full report for a decision point the agent has already installed.
/// `line_plies` = 0 skips the searched line (it costs one determinization
/// and a few engine steps, which is nothing next to the search, but a caller
/// stepping the search in slices does not want it on every tick).
pub fn report(agent: &ProtocolAgent, dex: &Dex, line_plies: usize, line_seed: u64) -> Value {
    let Some(search) = agent.search() else {
        return json!({"schema": SCHEMA, "error": "no position installed"});
    };
    let side = agent.side();
    let battle = agent.battle();
    let obs = agent.observer();

    let acts = search.actions();
    let visits = search.visits();
    let means = search.means();
    let dominated = search.dominated();
    let total: u32 = visits.iter().sum();
    let reasons: Vec<(SearchChoice, &'static str)> = battle
        .map(|b| {
            let mut bb = b.clone();
            bb.set_log_enabled(false);
            crate::smmcts::dominated_actions(&bb, dex, side)
        })
        .unwrap_or_default();

    let mut order: Vec<usize> = (0..acts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(visits[i]));

    let actions: Vec<Value> = order
        .iter()
        .map(|&i| {
            let mut row = battle
                .map(|b| action_json(b, dex, side, None, acts[i]))
                .unwrap_or_else(|| json!({"input": acts[i].to_input(dex)}));
            row["visits"] = json!(visits[i]);
            row["mean"] = json!(means[i]);
            row["frac"] = json!(if total > 0 {
                visits[i] as f64 / total as f64
            } else if acts.is_empty() {
                0.0
            } else {
                1.0 / acts.len() as f64
            });
            row["dominated"] = json!(dominated.get(i).copied().unwrap_or(false));
            row["reason"] = json!(reasons
                .iter()
                .find(|(c, _)| *c == acts[i])
                .map(|(_, why)| *why));
            row
        })
        .collect();

    json!({
        "schema": SCHEMA,
        "side": side,
        "turn": battle.map(|b| b.turn).unwrap_or(0),
        "iterations": search.iterations(),
        "preview": search.is_preview(),
        "belief": serde_json::from_str::<Value>(&agent.belief_info()).unwrap_or(Value::Null),
        "actions": actions,
        "matrix": matrix_json(search, battle, dex, side, obs, &order),
        "damage": battle
            .map(|b| damage_table(b, dex, side, obs))
            .unwrap_or_else(|| json!({"mine": [], "theirs": []})),
        "line": if line_plies > 0 {
            line_json(agent, dex, line_seed, line_plies)
        } else {
            Value::Null
        },
    })
}

/// The root matrix as a dense rows-by-columns grid (rows follow the action
/// list's display order). Unsampled cells are `null`: UCB concentrates, so
/// most of a wide matrix is genuinely unmeasured, and a zero there would
/// read as a verdict.
fn matrix_json(
    search: &BlindSearch,
    battle: Option<&Battle>,
    dex: &Dex,
    side: usize,
    obs: Option<&Observer>,
    order: &[usize],
) -> Value {
    let cells = search.root_matrix();
    let mut cols: Vec<SearchChoice> = Vec::new();
    for &(_, c, _, _) in &cells {
        if !cols.contains(&c) {
            cols.push(c);
        }
    }
    // busiest opponent replies first
    let weight = |c: SearchChoice| -> u32 {
        cells.iter().filter(|(_, cc, _, _)| *cc == c).map(|(_, _, n, _)| *n).sum()
    };
    cols.sort_by_key(|&c| std::cmp::Reverse(weight(c)));

    let grid: Vec<Vec<Value>> = order
        .iter()
        .map(|&row| {
            cols.iter()
                .map(|&c| {
                    match cells.iter().find(|(a, cc, _, _)| *a == row && *cc == c) {
                        Some(&(_, _, n, mean)) => json!({"n": n, "mean": mean}),
                        None => Value::Null,
                    }
                })
                .collect()
        })
        .collect();

    json!({
        "cols": cols
            .iter()
            .map(|&c| match battle {
                Some(b) => action_json(b, dex, 1 - side, obs, c),
                None => json!({"input": c.to_input(dex)}),
            })
            .collect::<Vec<Value>>(),
        "cells": grid,
    })
}

fn line_json(agent: &ProtocolAgent, dex: &Dex, seed: u64, plies: usize) -> Value {
    let (Some(search), Some(belief), Some(obs)) =
        (agent.search(), agent.belief(), agent.observer())
    else {
        return Value::Null;
    };
    let line = search.principal_line(dex, belief, obs, seed, plies);
    json!({
        "assumed": line
            .assumed
            .iter()
            .enumerate()
            .map(|(slot, (species, moves))| json!({
                "slot": slot,
                "species": dex.species.key(*species),
                "moves": moves.iter().map(|&m| dex.moves.key(m)).collect::<Vec<_>>(),
                // A mon that has appeared has a public species; its moves may
                // still be assumed. One that has not is assumed entirely.
                "appeared": obs.mons().get(slot).map(|m| m.appeared).unwrap_or(false),
            }))
            .collect::<Vec<Value>>(),
        "steps": line
            .steps
            .iter()
            .map(|s| json!({
                "mine": s.mine.map(|c| c.to_input(dex)),
                "theirs": s.theirs.map(|c| c.to_input(dex)),
                "log": s.log,
                "outcome": s.outcome.map(|o| match o {
                    nc2000_engine::battle::Outcome::P1Win => "p1",
                    nc2000_engine::battle::Outcome::P2Win => "p2",
                    nc2000_engine::battle::Outcome::Tie => "tie",
                }),
            }))
            .collect::<Vec<Value>>(),
    })
}
