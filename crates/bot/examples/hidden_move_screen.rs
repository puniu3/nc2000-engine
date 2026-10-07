use nc2000_bot::{
    exact::{solve_matrix_full, MatrixSolution},
    preview::load_meta_pool,
    smmcts::SelRule,
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, Outcome, PokemonSet, SearchChoice},
    state::Battle,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

fn table(dex: &nc2000_engine::dex::Dex, mut b: Battle, side: usize) -> Value {
    let own = b.legal_choices(dex, side);
    let replies = b.legal_choices(dex, 1 - side);
    let mut lower = vec![vec![0.; replies.len()]; own.len()];
    let mut upper = lower.clone();
    let mut cells = Vec::new();
    for (coin, weight) in [(false, 196. / 256.), (true, 60. / 256.)] {
        b.quick_claw_roll = coin;
        b.update_all_speeds(dex);
        for (i, &a) in own.iter().enumerate() {
            for (j, &reply) in replies.iter().enumerate() {
                let mut choices = [None, None];
                choices[side] = Some(a);
                choices[1 - side] = Some(reply);
                let enumeration = enumerate_step_partial(dex, &b, choices, 128);
                let mut mass = [0.; 4];
                for leaf in &enumeration.leaves {
                    let k = match leaf.battle.outcome() {
                        Some(Outcome::P1Win) => {
                            if side == 0 {
                                0
                            } else {
                                2
                            }
                        }
                        Some(Outcome::P2Win) => {
                            if side == 1 {
                                0
                            } else {
                                2
                            }
                        }
                        Some(Outcome::Tie) => 1,
                        None => 3,
                    };
                    mass[k] += leaf.prob;
                }
                let pending: f64 = enumeration.pending.iter().map(|p| p.mass).sum();
                assert!((mass.iter().sum::<f64>() + pending - 1.).abs() < 1e-8);
                let lo = mass[0] + mass[1] * 0.5;
                let hi = lo + mass[3] + pending;
                lower[i][j] += weight * lo;
                upper[i][j] += weight * hi;
                cells.push(json!({"coin":coin,"own":i,"reply":j,"mass":mass,"pending":pending,"runs":enumeration.runs}));
            }
        }
    }
    json!({"actions":own.iter().map(|a|a.to_input(dex)).collect::<Vec<_>>(),"replies":replies.iter().map(|a|a.to_input(dex)).collect::<Vec<_>>(),"lower":lower,"upper":upper,"cells":cells})
}
fn solution(s: MatrixSolution) -> Value {
    json!({"value":s.value,"gap":s.gap,"own":s.x,"opponent":s.y})
}
fn solve(tables: &[&Value]) -> Value {
    let n = tables[0]["actions"].as_array().unwrap().len();
    let m = tables
        .iter()
        .map(|t| t["replies"].as_array().unwrap().len())
        .sum();
    let mut results = serde_json::Map::new();
    for name in ["lower", "upper"] {
        let data: Vec<_> = (0..n)
            .flat_map(|i| {
                tables.iter().flat_map(move |t| {
                    t[name][i]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                })
            })
            .collect();
        results.insert(name.into(), solution(solve_matrix_full(&data, n, m)));
    }
    Value::Object(results)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus census output");
    let games: Vec<Value> = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let census: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut groups = BTreeMap::<(u64, usize, usize), Vec<(usize, &Value)>>::new();
    for (id, r) in census["rows"].as_array().unwrap().iter().enumerate() {
        groups
            .entry((
                r["game"].as_u64().unwrap(),
                r["decision"].as_u64().unwrap() as usize,
                r["side"].as_u64().unwrap() as usize,
            ))
            .or_default()
            .push((id, r));
    }
    let mut roots = Vec::new();
    let mut rows = Vec::new();
    for ((game_id, decision, side), variants) in groups {
        let game = games.iter().find(|g| g["game"] == game_id).unwrap();
        let original: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let seed = game["seed"].as_str().unwrap();
        let mut base = Battle::from_fixture(&dex, seed, &original[0], &original[1]).unwrap();
        base.set_log_enabled(true);
        let mut obs = Observer::new(&base, side);
        let mut history = Vec::new();
        for d in game["decisions"].as_array().unwrap().iter().take(decision) {
            assert_eq!(base.state_key128().to_string(), d["key"].as_str().unwrap());
            let acts = [0, 1].map(|s| base.legal_choices(&dex, s));
            let chosen = [0, 1].map(|s| d["choices"][s].as_u64().map(|i| acts[s][i as usize]));
            history.push((base.active_id(1 - side), chosen));
            base.apply_choices(&dex, chosen).unwrap();
            obs.observe(&base, &dex);
        }
        assert_eq!(
            base.state_key128().to_string(),
            variants[0].1["key"].as_str().unwrap()
        );
        let base_table = table(&dex, base.clone(), side);
        let base_solution = solve(&[&base_table]);
        let potential = base_solution["lower"]["value"].as_f64().unwrap()
            - base_solution["lower"]["gap"].as_f64().unwrap();
        let mut search = BlindSearch::new(
            &base,
            &dex,
            RmConfig {
                c: 0.4,
                rule: SelRule::Ucb,
                ..Default::default()
            },
            side,
            0x2026110301,
        );
        let belief = Belief::new(&dex, &pool, &obs);
        search.step(&dex, &belief, &obs, 27000);
        let best = search.best().unwrap().to_input(&dex);
        let chosen = base_table["actions"]
            .as_array()
            .unwrap()
            .iter()
            .position(|a| a == &best)
            .unwrap();
        let root_id = roots.len();
        roots.push(json!({"game":game_id,"decision":decision,"side":side,"key":base.state_key128().to_string(),"table":base_table,"solution":base_solution,"potential":potential,"variants":variants.len(),"screened":potential>0.01,"search":{"best":best,"chosen":chosen,"seed":0x2026110301u64,"visits":search.visits(),"means":search.means(),"rng":search.search_rng_state()}}));
        if potential > 0.01 {
            for (id, r) in variants {
                let slot = r["foe_slot"].as_u64().unwrap() as usize;
                let move_slot = r["move_slot"].as_u64().unwrap() as usize;
                let mut teams = original.clone();
                teams[1 - side][slot].moves[move_slot] = r["replacement"].as_str().unwrap().into();
                let mut alt = Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
                alt.set_log_enabled(true);
                let old =
                    SearchChoice::Move(dex.moves.id(r["original_move"].as_str().unwrap()).unwrap());
                let new =
                    SearchChoice::Move(dex.moves.id(r["replacement"].as_str().unwrap()).unwrap());
                for (active, choice) in &history {
                    let mut c = *choice;
                    if active.is_some_and(|a| a.slot as usize == slot) && c[1 - side] == Some(old) {
                        c[1 - side] = Some(new);
                    }
                    alt.apply_choices(&dex, c).unwrap();
                }
                assert_eq!(
                    alt.state_key128().to_string(),
                    r["alternate_key"].as_str().unwrap()
                );
                assert_eq!(alt.log, base.log);
                let alt_table = table(&dex, alt, side);
                assert_eq!(alt_table["actions"], base_table["actions"]);
                let pair = solve(&[&base_table, &alt_table]);
                let pure_upper = [&base_table, &alt_table]
                    .iter()
                    .flat_map(|t| {
                        t["upper"][chosen]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                    })
                    .fold(1., f64::min);
                let lower = pair["lower"]["value"].as_f64().unwrap()
                    - pair["lower"]["gap"].as_f64().unwrap();
                let regret = (lower - pure_upper - 1e-8).max(0.);
                if regret > 0. {
                    eprintln!("GAP game={game_id} decision={decision} side={side} replacement={} regret={regret}",r["replacement"]);
                }
                rows.push(json!({"census_id":id,"root_id":root_id,"table":alt_table,"solution":pair,"regret_lower":regret}));
            }
        }
        eprintln!("root={root_id} game={game_id} decision={decision} side={side} potential={potential} screened_pairs={}",rows.len());
        std::fs::write(
            &args[3],
            serde_json::to_vec(
                &json!({"cap":128,"threshold":0.01,"iterations":27000,"roots":roots,"rows":rows}),
            )
            .unwrap(),
        )
        .unwrap();
    }
}
