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
//!   action) pair, plus the two summaries a single number cannot carry —
//!   what an action is worth against an opponent playing the matrix's
//!   equilibrium mixture, and what it is worth against their best reply.
//!   The raw per-action mean is neither: it averages over the opponent
//!   actions UCB happened to explore, so it sits ABOVE the worst case and
//!   flatters every option that a rare reply punishes;
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
use crate::smmcts::solve_rm_plus;

pub const SCHEMA: &str = "nc2000-analysis-v1";

/// A cell sampled fewer times than this is not evidence about a worst case:
/// the minimum of six noisy estimates is biased low, and one thin cell would
/// own the answer. Such cells still show in the matrix (with their count) —
/// they are just not allowed to define `worst`.
const MIN_CELL: u32 = 20;

/// RM+ sweeps over the sampled matrix. The matrix is tiny (single-digit
/// dimensions) so this is microseconds, and the same solver the baked
/// preview tables and the RM root use.
const SOLVE_SWEEPS: u32 = 2000;

/// Playouts per ply of the searched line, as a share of the analysis that
/// asked for it and a ceiling on that share. Each ply is a full-information
/// search of one determinized state — far cheaper than the root's blind
/// search — so a tenth of the budget buys a continuation whose every step
/// was actually chosen. Tying it to the budget matters at both ends: a quick
/// look should not spend ten times its own cost on the story underneath it,
/// and a deep one should not illustrate itself with a shallow guess.
const LINE_ITERS_MAX: u32 = 3000;
const LINE_ITERS_MIN: u32 = 200;

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

    let m = MatrixSummary::build(search, &means, &order);

    let actions: Vec<Value> = order
        .iter()
        .enumerate()
        .map(|(rank, &i)| {
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
            // What the action is worth if they answer it with the mixture
            // the matrix says they should, and if they answer it with the
            // single reply that hurts it most. The first is the value of
            // choosing it in a simultaneous game; the second is the floor.
            row["equity"] = json!(m.equity.get(rank).copied());
            row["worst"] = json!(m.worst.get(rank).copied().flatten());
            row["mix"] = json!(m.mix_mine.get(rank).copied());
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
        // The position's own value, and the mixture each side should play to
        // hold it. In a simultaneous game a single best move is the special
        // case, not the rule — this is where that becomes visible.
        "equilibrium": {
            "value": m.value,
            "theirs": m.mix_theirs,
        },
        "matrix": matrix_json(search, battle, dex, side, obs, &m),
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

/// The sampled root matrix, summarized once and read by everything that
/// needs it: the grid the screen draws, the per-action equity and floor, and
/// the equilibrium mixture.
struct MatrixSummary {
    /// Opponent replies, busiest first.
    cols: Vec<SearchChoice>,
    /// `cells[row][col]` — `None` where the pair was never sampled. Rows
    /// follow the display order handed in, not the action order.
    cells: Vec<Vec<Option<(u32, f64)>>>,
    /// Per row, value against the opponent's equilibrium mixture.
    equity: Vec<f64>,
    /// Per row, the worst sampled reply (`None` when every cell in the row
    /// is too thin to be evidence).
    worst: Vec<Option<f64>>,
    /// Equilibrium mixtures, ours and theirs.
    mix_mine: Vec<f64>,
    mix_theirs: Vec<f64>,
    /// The game value of the sampled matrix.
    value: f64,
}

impl MatrixSummary {
    fn build(search: &BlindSearch, means: &[f64], order: &[usize]) -> MatrixSummary {
        let sampled = search.root_matrix();
        let mut cols: Vec<SearchChoice> = Vec::new();
        for &(_, c, _, _) in &sampled {
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        let weight = |c: SearchChoice| -> u32 {
            sampled.iter().filter(|(_, cc, _, _)| *cc == c).map(|(_, _, n, _)| *n).sum()
        };
        cols.sort_by_key(|&c| std::cmp::Reverse(weight(c)));

        let cells: Vec<Vec<Option<(u32, f64)>>> = order
            .iter()
            .map(|&row| {
                cols.iter()
                    .map(|&c| {
                        sampled
                            .iter()
                            .find(|(a, cc, _, _)| *a == row && *cc == c)
                            .map(|&(_, _, n, mean)| (n, mean))
                    })
                    .collect()
            })
            .collect();

        // Only cells the search actually sampled enough times are evidence.
        // Everything thinner is replaced by the row's own average — the one
        // stand-in that adds no claim about the reply it stands for — and a
        // column with no evidence anywhere is dropped rather than handed to
        // the solver as a reply the opponent might have.
        let thick = |c: Option<(u32, f64)>| c.filter(|(n, _)| *n >= MIN_CELL).map(|(_, m)| m);
        let keep: Vec<usize> =
            (0..cols.len()).filter(|&c| cells.iter().any(|row| thick(row[c]).is_some())).collect();
        let (rows, ncols) = (order.len(), keep.len());
        if rows == 0 || ncols == 0 {
            return MatrixSummary {
                cols,
                cells,
                equity: vec![0.5; rows],
                worst: vec![None; rows],
                mix_mine: vec![1.0 / rows.max(1) as f64; rows],
                mix_theirs: vec![],
                value: 0.5,
            };
        }
        let mut flat = vec![0.0f64; rows * ncols];
        for (r, &i) in order.iter().enumerate() {
            for (c, &col) in keep.iter().enumerate() {
                flat[r * ncols + c] = thick(cells[r][col]).unwrap_or(means[i]);
            }
        }
        let (mix_mine, mix_kept) = solve_rm_plus(&flat, [rows, ncols], SOLVE_SWEEPS);
        let equity: Vec<f64> = (0..rows)
            .map(|r| (0..ncols).map(|c| flat[r * ncols + c] * mix_kept[c]).sum())
            .collect();
        let value: f64 = (0..rows).map(|r| equity[r] * mix_mine[r]).sum();
        // The floor is the minimum of the SAME row the equity averages over,
        // so it can never sit above it — and it is reported only for a row
        // that has some evidence to floor.
        let worst: Vec<Option<f64>> = (0..rows)
            .map(|r| {
                keep.iter().any(|&c| thick(cells[r][c]).is_some()).then(|| {
                    (0..ncols).map(|c| flat[r * ncols + c]).fold(f64::INFINITY, f64::min)
                })
            })
            .collect();
        // The mixture is reported over the columns as displayed, so a dropped
        // column reads as the zero it is.
        let mut mix_theirs = vec![0.0; cols.len()];
        for (c, &col) in keep.iter().enumerate() {
            mix_theirs[col] = mix_kept[c];
        }
        MatrixSummary { cols, cells, equity, worst, mix_mine, mix_theirs, value }
    }
}

/// The grid the screen draws. Unsampled cells are `null`: UCB concentrates,
/// so most of a wide matrix is genuinely unmeasured, and a zero there would
/// read as a verdict. Each column also carries how often the reply was even
/// LEGAL — in blind play a move exists only in the candidate teams that
/// carry it, and a column sampled rarely because it is rarely available says
/// something quite different from one the search kept declining.
fn matrix_json(
    search: &BlindSearch,
    battle: Option<&Battle>,
    dex: &Dex,
    side: usize,
    obs: Option<&Observer>,
    m: &MatrixSummary,
) -> Value {
    let iters = search.iterations().max(1) as f64;
    let avail = search.root_replies();
    json!({
        "cols": m
            .cols
            .iter()
            .map(|&c| {
                let mut col = match battle {
                    Some(b) => action_json(b, dex, 1 - side, obs, c),
                    None => json!({"input": c.to_input(dex)}),
                };
                let n = avail.iter().find(|(x, _)| *x == c).map(|(_, n)| *n).unwrap_or(0);
                col["available"] = json!(n as f64 / iters);
                col
            })
            .collect::<Vec<Value>>(),
        "cells": m
            .cells
            .iter()
            .map(|row| {
                Value::Array(
                    row.iter()
                        .map(|c| match c {
                            Some((n, mean)) => json!({"n": n, "mean": mean}),
                            None => Value::Null,
                        })
                        .collect(),
                )
            })
            .collect::<Vec<Value>>(),
    })
}

fn line_json(agent: &ProtocolAgent, dex: &Dex, seed: u64, plies: usize) -> Value {
    let (Some(search), Some(belief), Some(obs)) =
        (agent.search(), agent.belief(), agent.observer())
    else {
        return Value::Null;
    };
    let iters = (search.iterations() / 10).clamp(LINE_ITERS_MIN, LINE_ITERS_MAX);
    let line = search.principal_line(dex, belief, obs, seed, plies, iters, search.best());
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
                "mineTarget": s.mine_target.map(|sp| dex.species.key(sp)),
                "theirsTarget": s.theirs_target.map(|sp| dex.species.key(sp)),
                "iterations": s.iterations,
                // How likely the shown outcome was, among this step's chance
                // events. A line that keeps taking 30% branches is a story;
                // this is how the reader tells the two apart.
                "prob": s.prob,
                "effects": s
                    .effects
                    .iter()
                    .map(|e| json!({
                        "side": e.side,
                        "mine": e.side == agent.side(),
                        "species": dex.species.key(e.species),
                        "hpBefore": e.hp_before,
                        "hpAfter": e.hp_after,
                        "maxhp": e.maxhp,
                        "statusBefore": e.status_before.as_str(),
                        "statusAfter": e.status_after.as_str(),
                        "active": e.active,
                        "switchedIn": e.switched_in,
                    }))
                    .collect::<Vec<Value>>(),
                "outcome": s.outcome.map(|o| match o {
                    nc2000_engine::battle::Outcome::P1Win => "p1",
                    nc2000_engine::battle::Outcome::P2Win => "p2",
                    nc2000_engine::battle::Outcome::Tie => "tie",
                }),
            }))
            .collect::<Vec<Value>>(),
    })
}
