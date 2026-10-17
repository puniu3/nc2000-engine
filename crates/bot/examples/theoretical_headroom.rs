use nc2000_bot::{
    bounds::{BoundConfig, BoundSolver},
    player::action_input,
};
use nc2000_engine::battle::enumerate::enumerate_step_partial;
use nc2000_engine::{
    battle::{Outcome, SearchChoice},
    dex::Dex,
    state::{Battle, Pokemon, RequestKind, Status, BOOST_NAMES},
};
use serde_json::{json, Value};
use std::{
    io::Write,
    time::{Duration, Instant},
};
fn status_str(s: Status) -> &'static str {
    match s {
        Status::None => "",
        other => other.as_str(),
    }
}

fn request_str(r: Option<RequestKind>) -> &'static str {
    match r {
        Some(RequestKind::TeamPreview) => "teampreview",
        Some(RequestKind::Move) => "move",
        Some(RequestKind::Switch) => "switch",
        Some(RequestKind::Wait) | None => "",
    }
}

fn poke_json(dex: &Dex, p: &Pokemon) -> serde_json::Value {
    let mut boosts = serde_json::Map::new();
    for (&b, name) in p.boosts.iter().zip(BOOST_NAMES) {
        boosts.insert(name.to_string(), serde_json::json!(b));
    }
    let moves: Vec<serde_json::Value> = p
        .move_slots
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": dex.moves.key(s.id),
                "name": dex.move_static(s.id).name,
                "pp": s.pp,
                "maxpp": s.maxpp,
                "disabled": s.disabled,
            })
        })
        .collect();
    let types: Vec<&str> = p.types.iter().map(|t| dex.type_name(t)).collect();
    let volatiles: Vec<&str> = p
        .volatiles
        .iter()
        .map(|(id, _)| dex.conds_key(*id))
        .collect();
    serde_json::json!({
        "species": dex.species.get(p.species).name,
        "name": p.name.as_str(),
        "level": p.level,
        "gender": p.gender.as_str(),
        "hp": p.hp.max(0),
        "maxhp": p.maxhp,
        "fainted": p.fainted,
        "status": status_str(p.status),
        "boosts": boosts,
        "moves": moves,
        "item": p.item.map(|it| dex.items.get(it).name.clone()),
        "types": types,
        "volatiles": volatiles,
        "trapped": p.trapped,
    })
}

fn state_view_json(battle: &Battle, dex: &Dex) -> serde_json::Value {
    let sides: Vec<serde_json::Value> = (0..2)
        .map(|n| {
            let side = &battle.sides[n];
            // party in display order (preview: 6 mons; battle: the picked 3)
            let party: Vec<serde_json::Value> = side
                .party
                .iter()
                .map(|&slot| poke_json(dex, &side.roster[slot as usize]))
                .collect();
            // display index of the active mon, if any
            let active = side
                .active
                .and_then(|slot| side.party.iter().position(|&x| x == slot));
            let conditions: Vec<&str> = side
                .side_conditions
                .iter()
                .map(|(id, _)| dex.conds_key(*id))
                .collect();
            serde_json::json!({
                "name": side.name,
                "active": active,
                "party": party,
                "pokemonLeft": side.pokemon_left,
                "sideConditions": conditions,
                "request": request_str(side.request_state()),
            })
        })
        .collect();
    let field = serde_json::json!({
        "weather": battle.field.weather.map(|w| dex.conds_key(w)),
        "pseudoWeather": battle
            .field
            .pseudo_weather
            .iter()
            .map(|(id, _)| dex.conds_key(*id))
            .collect::<Vec<_>>(),
    });
    serde_json::json!({
        "turn": battle.turn,
        "sides": sides,
        "field": field,
        "outcome": battle.outcome().map(|o| match o {
            Outcome::P1Win => "p1",
            Outcome::P2Win => "p2",
            Outcome::Tie => "tie",
        }),
    })
}

// ------------------------------------------------------------------ misc

/// Validates a `nc2000-fork-v1` document and returns what a player needs:
/// `{label, info, turn, botSide, arms: [{input, label}]}`. Arm inputs come
/// back in the form `Battle.applyChoice` takes.
fn normalized(mut v: Value) -> Value {
    for s in v["sides"].as_array_mut().unwrap() {
        for p in s["party"].as_array_mut().unwrap() {
            for m in p["moves"].as_array_mut().unwrap() {
                if m["id"].as_str().unwrap().starts_with("hiddenpower") {
                    m["id"] = json!("hiddenpower");
                    m["name"] = json!("Hidden Power");
                }
            }
        }
    }
    v
}
fn input_equal(a: &str, b: &str) -> bool {
    a == b || (a == "move hiddenpower" && b.starts_with("move hiddenpower"))
}
fn picks(dex: &Dex, b: &mut Battle, row: &Value) -> [Option<SearchChoice>; 2] {
    let legal = [0, 1].map(|s| b.legal_choices(dex, s));
    let mut out = [None, None];
    for p in row["picks"].as_array().unwrap() {
        let s = p[0].as_u64().unwrap() as usize;
        out[s] = Some(
            *legal[s]
                .iter()
                .find(|&&a| input_equal(&action_input(dex, a), p[1].as_str().unwrap()))
                .unwrap(),
        );
    }
    out
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    let dex = conformance::load_dex();
    let base = std::path::Path::new("findings/deployed-endurance-v1");
    let mut out = std::fs::File::create_new(&args[2]).unwrap();
    let mut audited = 0;
    let mut roots = 0;
    let mut cases = Vec::new();
    let mut root4 = None;
    for g in 0..8 {
        let id = format!("g{g}");
        let all: Vec<Value> = std::fs::read_to_string(base.join(format!("games/{id}.jsonl")))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect();
        let head = &all[0];
        let teams: [Vec<nc2000_engine::battle::PokemonSet>; 2] =
            serde_json::from_value(head["teams"].clone()).unwrap();
        let battle = Battle::from_fixture(
            &dex,
            head["job"]["seed"].as_str().unwrap(),
            &teams[0],
            &teams[1],
        )
        .unwrap();
        let rows: Vec<Value> = all
            .into_iter()
            .filter(|r| r["type"] == "decision")
            .collect();
        if g == 4 {
            let mut b = battle.clone();
            for r in rows.iter().take(12) {
                let p = picks(&dex, &mut b, r);
                b.apply_choices(&dex, p).unwrap();
            }
            root4 = Some(b);
        }
        let final_game: Value =
            serde_json::from_slice(&std::fs::read(base.join(format!("games/{id}.json"))).unwrap())
                .unwrap();
        cases.push((id, battle, rows, final_game["outcome"].clone()));
    }
    let mut files: Vec<_> = std::fs::read_dir(base.join("continuations"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "json")
                && !p
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .contains("process.")
        })
        .collect();
    files.sort();
    for path in files {
        let case: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut b = root4.as_ref().unwrap().clone();
        b.reseed(case["chanceSeed"].as_u64().unwrap());
        b.set_log_enabled(false);
        b.set_log_enabled(true);
        cases.push((
            path.file_stem().unwrap().to_str().unwrap().to_string(),
            b,
            case["trace"].as_array().unwrap().clone(),
            case["outcome"].clone(),
        ));
    }
    assert_eq!(cases.len(), 40);
    for (id, mut b, rows, outcome) in cases {
        for (d, row) in rows.iter().enumerate() {
            if normalized(state_view_json(&b, &dex)) != normalized(row["state"].clone()) {
                std::fs::write("findings/theoretical-headroom-v1/parity-mismatch.json",json!({"id":id,"decision":d,"native":state_view_json(&b,&dex),"deployed":row["state"]}).to_string()).unwrap();
                panic!("state mismatch {id}/{d}");
            }
            if let Some(seed) = row["seed"].as_str() {
                assert_eq!(b.prng.seed_str(), seed);
            }
            audited += 1;
            let chosen = picks(&dex, &mut b, row);
            if mode == "control" && id == "g0" && b.turn == 20 {
                let mut control = b.clone();
                let active = control.active_id(1).unwrap();
                control.poke_mut(active).hp = 1;
                let legal = [0, 1].map(|side| control.legal_choices(&dex, side));
                let mut matrix = Vec::new();
                for &own in &legal[1] {
                    let mut values = Vec::new();
                    for &foe in &legal[0] {
                        let e =
                            enumerate_step_partial(&dex, &control, [Some(foe), Some(own)], 4096);
                        let mut lo = 0.;
                        let mut unknown = e.pending.iter().map(|p| p.mass).sum::<f64>();
                        for leaf in e.leaves {
                            match leaf.battle.outcome() {
                                Some(Outcome::P2Win) => lo += leaf.prob,
                                None => unknown += leaf.prob,
                                _ => {}
                            }
                        }
                        values.push([lo, lo + unknown]);
                    }
                    matrix.push(values);
                }
                let names = legal[1]
                    .iter()
                    .map(|&a| action_input(&dex, a))
                    .collect::<Vec<_>>();
                let good = names.iter().position(|a| a == "move surf").unwrap();
                let bad = names.iter().position(|a| a == "move sleeptalk").unwrap();
                let lower = matrix[good].iter().map(|x| x[0]).fold(1., f64::min);
                let upper = matrix[bad].iter().map(|x| x[1]).fold(1., f64::min);
                assert!(lower - upper > 0.99);
                writeln!(out,"{}",json!({"synthetic_not_evidence_of_bot_error":true,"origin":"g0 turn20","mutation":"P2 Suicune HP set to 1","intentionally_bad_action":"move sleeptalk","better_action":"move surf","win_gap_lower":lower-upper,"actions":names,"opponent_actions":legal[0].iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>(),"matrix":matrix})).unwrap();
                roots += 1;
            }

            let narrow = d + 3 >= rows.len() || b.sides.iter().all(|s| s.pokemon_left == 1);
            let endgame = b.turn > 0
                && b.is_idle_move_request()
                && if mode == "focus" {
                    (id == "g4" && [10, 53].contains(&b.turn))
                        || ((id == "g4-d12-baseline-4" || id == "g4-d12-baseline-12")
                            && b.turn == 500)
                } else if mode == "wide" {
                    (id.len() == 2 && !narrow)
                        || (id == "g4-d12-baseline-4" && b.turn == 500)
                        || (id.contains("baseline-12") && b.turn == 500)
                } else {
                    narrow
                };
            if mode != "audit" && mode != "control" && endgame {
                let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
                let actions = legal
                    .each_ref()
                    .map(|v| v.iter().map(|&a| action_input(&dex, a)).collect::<Vec<_>>());
                let selected =
                    [0, 1].map(|s| chosen[s].and_then(|a| legal[s].iter().position(|&c| c == a)));
                let t = Instant::now();
                let mut cells = Vec::new();
                let mut work = 0;
                for &a in &legal[0] {
                    for &c in &legal[1] {
                        let e = enumerate_step_partial(&dex, &b, [Some(a), Some(c)], 4096);
                        work += e.runs;
                        let mut win = [0., 0.];
                        let mut tie = 0.;
                        let mut unresolved = e.pending.iter().map(|p| p.mass).sum::<f64>();
                        for l in e.leaves {
                            match l.battle.outcome() {
                                Some(Outcome::P1Win) => win[0] += l.prob,
                                Some(Outcome::P2Win) => win[1] += l.prob,
                                Some(Outcome::Tie) => tie += l.prob,
                                None => unresolved += l.prob,
                            }
                        }
                        assert!((win.iter().sum::<f64>() + tie + unresolved - 1.).abs() < 1e-8);
                        cells.push(json!({"win":win,"tie":tie,"unknown":unresolved}));
                    }
                }
                let mut record = json!({"id":id,"decision":d,"turn":b.turn,"actions":actions,"selected":selected,"cells":cells,"runs":work,"ms":t.elapsed().as_secs_f64()*1000.});
                if mode == "solve" || mode == "wide" || mode == "focus" {
                    let targets: Vec<_> = (0..2)
                        .map(|s| {
                            (
                                s,
                                (0..legal[s].len())
                                    .map(|i| f64::from(Some(i) == selected[s]))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    let mut solver = BoundSolver::new(
                        &dex,
                        BoundConfig {
                            time_budget: Some(Duration::from_millis(if mode == "focus" {
                                10000
                            } else {
                                1000
                            })),
                            work_budget: if mode == "focus" { 200000 } else { 20000 },
                            node_budget: if mode == "focus" { 100000 } else { 10000 },
                            cell_cap: 256,
                            retain_eager_prefix: true,
                            ..Default::default()
                        },
                    );
                    let r = solver.solve_policy_losses(&b, &targets, 0.001);
                    let m = solver.root_payoffs(&b).unwrap();
                    record["bounds"] = json!({"value":[r.bounds.lo,r.bounds.hi],"cells":m.cells.iter().map(|v|[v.lo,v.hi]).collect::<Vec<_>>(),"losses":targets.iter().map(|(s,p)|{let v=m.policy_loss(*s,p);[v.lo,v.hi]}).collect::<Vec<_>>(),"stop":format!("{:?}",r.stop)});
                }
                writeln!(out, "{record}").unwrap();
                out.flush().unwrap();
                roots += 1;
                if roots % 10 == 0 {
                    eprintln!("roots={roots} {id}/{d}");
                }
            }
            b.apply_choices(&dex, chosen).unwrap();
        }
        assert!(b.outcome().is_some());
        assert_eq!(state_view_json(&b, &dex)["outcome"], outcome);
    }
    eprintln!("audited={audited} roots={roots}");
}
