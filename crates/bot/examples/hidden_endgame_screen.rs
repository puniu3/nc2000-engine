use nc2000_bot::{
    exact::solve_matrix_full, preview::load_meta_pool, rng::SplitMix64, smmcts::SelRule, Belief,
    BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, Outcome, PokemonSet},
    state::Battle,
};
use serde_json::{json, Value};
use std::{io::BufRead, path::Path};

fn solve(types: &[Vec<Vec<f64>>]) -> Value {
    let n = types[0].len();
    let response = |x: &[f64]| -> Vec<usize> {
        types
            .iter()
            .map(|t| {
                (0..t[0].len())
                    .min_by(|&a, &b| {
                        let va: f64 = (0..n).map(|i| x[i] * t[i][a]).sum();
                        let vb: f64 = (0..n).map(|i| x[i] * t[i][b]).sum();
                        va.total_cmp(&vb)
                    })
                    .unwrap()
            })
            .collect()
    };
    let column = |r: &[usize]| -> Vec<f64> {
        (0..n)
            .map(|i| types.iter().zip(r).map(|(t, &j)| t[i][j]).sum::<f64>() / types.len() as f64)
            .collect()
    };
    let mut replies = vec![response(&vec![1. / n as f64; n])];
    for _ in 0..256 {
        let cols: Vec<_> = replies.iter().map(|r| column(r)).collect();
        let matrix: Vec<_> = (0..n)
            .flat_map(|i| cols.iter().map(move |c| c[i]))
            .collect();
        let solution = solve_matrix_full(&matrix, n, cols.len());
        let next = response(&solution.x);
        let lo: f64 = solution
            .x
            .iter()
            .zip(column(&next))
            .map(|(p, v)| p * v)
            .sum();
        let hi = (0..n)
            .map(|i| {
                cols.iter()
                    .zip(&solution.y)
                    .map(|(c, p)| p * c[i])
                    .sum::<f64>()
            })
            .fold(0., f64::max);
        if hi - lo < 1e-9 || replies.contains(&next) {
            assert!(hi - lo < 1e-8);
            return json!({"bounds":[lo,hi],"own":solution.x,"opponent":solution.y,"replies":replies});
        }
        replies.push(next);
    }
    panic!("column generation did not converge")
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "corpus output");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut cases = Vec::new();
    let mut decisions = 0;
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        b.set_log_enabled(true);
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            decisions += 1;
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
            if b.is_idle_move_request() && b.sides.iter().all(|s| s.pokemon_left == 1) {
                for side in 0..2 {
                    if actions[side].len() > 1 {
                        let order = b
                            .state_key128()
                            .wrapping_mul(0x9e3779b97f4a7c15)
                            .wrapping_add(side as u128);
                        cases.push((
                            order,
                            game["game"].clone(),
                            decision,
                            side,
                            b.clone(),
                            observers[side].clone(),
                        ));
                    }
                }
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize])),
            )
            .unwrap();
            for obs in &mut observers {
                obs.observe(&b, &dex);
            }
        }
    }
    let available = cases.len();
    cases.sort_by_key(|c| c.0);
    cases.truncate(24);
    let mut rows = Vec::new();
    for (_, game, decision, side, mut b, obs) in cases {
        let belief = Belief::new(&dex, &pool, &obs);
        let legal = b.legal_choices(&dex, side);
        let mut types = Vec::new();
        let mut lowers = Vec::new();
        let mut uppers = Vec::new();
        let mut rng = SplitMix64::new(0x2026110201);
        for sample in 0..16 {
            let mut world = belief.determinize(&dex, &b, &obs, &mut rng);
            assert_eq!(world.legal_choices(&dex, side), legal);
            assert_eq!(world.sides[1 - side].pokemon_left, 1);
            let replies = world.legal_choices(&dex, 1 - side);
            let mut lower = vec![vec![0.; replies.len()]; legal.len()];
            let mut upper = lower.clone();
            let mut cells = Vec::new();
            let mut keys = Vec::new();
            for (coin, weight) in [(false, 196. / 256.), (true, 60. / 256.)] {
                world.quick_claw_roll = coin;
                world.update_all_speeds(&dex);
                keys.push(world.state_key128().to_string());
                for (i, &a) in legal.iter().enumerate() {
                    for (j, &reply) in replies.iter().enumerate() {
                        let mut choices = [None, None];
                        choices[side] = Some(a);
                        choices[1 - side] = Some(reply);
                        let result = enumerate_step_partial(&dex, &world, choices, 512);
                        let mut mass = [0.; 4];
                        for leaf in &result.leaves {
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
                        let pending: f64 = result.pending.iter().map(|p| p.mass).sum();
                        assert!((mass.iter().sum::<f64>() + pending - 1.).abs() < 1e-8);
                        let lo = mass[0] + 0.5 * mass[1];
                        let hi = lo + mass[3] + pending;
                        lower[i][j] += weight * lo;
                        upper[i][j] += weight * hi;
                        cells.push(json!({"coin":coin,"own":i,"reply":j,"mass":mass,"pending":pending,"runs":result.runs}));
                    }
                }
            }
            types.push(json!({"sample":sample,"keys":keys,"replies":replies.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"cells":cells,"lower":lower,"upper":upper}));
            lowers.push(lower);
            uppers.push(upper);
        }
        let low = solve(&lowers);
        let high = solve(&uppers);
        let pure = |tables: &[Vec<Vec<f64>>]| -> Vec<f64> {
            (0..legal.len())
                .map(|i| {
                    tables
                        .iter()
                        .map(|t| t[i].iter().copied().fold(1., f64::min))
                        .sum::<f64>()
                        / tables.len() as f64
                })
                .collect()
        };
        let pure_lower = pure(&lowers);
        let pure_upper = pure(&uppers);
        let mut search = BlindSearch::new(
            &b,
            &dex,
            RmConfig {
                c: 0.4,
                rule: SelRule::Ucb,
                ..Default::default()
            },
            side,
            0x2026110202,
        );
        search.step(&dex, &belief, &obs, 27000);
        let best = search.best().unwrap();
        let chosen = legal.iter().position(|&a| a == best).unwrap();
        let regret_lower = (low["bounds"][0].as_f64().unwrap() - pure_upper[chosen] - 1e-8).max(0.);
        eprintln!("game={game} decision={decision} side={side} bounds={:.6}..{:.6} regret_lower={regret_lower:.6}",low["bounds"][0].as_f64().unwrap(),high["bounds"][1].as_f64().unwrap());
        rows.push(json!({"game":game,"decision":decision,"side":side,"key":b.state_key128().to_string(),"turn":b.turn,"actions":legal.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"types":types,"lower_solution":low,"upper_solution":high,"pure_lower":pure_lower,"pure_upper":pure_upper,"search":{"seed":0x2026110202u64,"best":best.to_input(&dex),"chosen":chosen,"visits":search.visits(),"means":search.means(),"rng":search.search_rng_state()},"regret_lower":regret_lower}));
        std::fs::write(&args[2],serde_json::to_vec(&json!({"decisions":decisions,"available":available,"samples":16,"sample_seed":0x2026110201u64,"cap":512,"iterations":27000,"rows":rows})).unwrap()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(types: &[Vec<Vec<f64>>]) -> f64 {
        let s = solve(types);
        assert!(s["bounds"][1].as_f64().unwrap() - s["bounds"][0].as_f64().unwrap() < 1e-8);
        s["bounds"][0].as_f64().unwrap()
    }

    #[test]
    fn own_policy_is_common_and_opponent_reply_can_depend_on_type() {
        let common = vec![vec![vec![1.], vec![0.]], vec![vec![0.], vec![1.]]];
        assert!((value(&common) - 0.5).abs() < 1e-8);
        let informed_reply = vec![
            vec![vec![1., 0.], vec![1., 0.]],
            vec![vec![0., 1.], vec![0., 1.]],
        ];
        assert!(value(&informed_reply).abs() < 1e-8);
    }

    #[test]
    fn mixture_and_unequal_reply_counts_match_exhaustive_normal_form() {
        let types = vec![
            vec![vec![1., 0.], vec![0., 1.]],
            vec![vec![0.2, 0.8, 0.5], vec![0.7, 0.1, 0.4]],
        ];
        let mut matrix = Vec::new();
        for i in 0..2 {
            for a in 0..2 {
                for b in 0..3 {
                    matrix.push((types[0][i][a] + types[1][i][b]) / 2.);
                }
            }
        }
        let full = solve_matrix_full(&matrix, 2, 6);
        assert!((value(&types) - full.value).abs() < 1e-8);
    }
}
