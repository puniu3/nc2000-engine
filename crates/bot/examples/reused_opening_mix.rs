use nc2000_bot::{exact::solve_matrix_full, Observer};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::time::Instant;

mod aliased_witness;
mod numeric_witness;
mod reused_witness;
mod threshold_witness;

fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn mixture(lower: &[[f64; 2]]) -> (Vec<f64>, f64) {
    let matrix: Vec<_> = (0..2)
        .flat_map(|a| lower.iter().map(move |v| v[a]))
        .collect();
    let solution = solve_matrix_full(&matrix, 2, lower.len());
    let value = lower
        .iter()
        .map(|v| solution.x[0] * v[0] + solution.x[1] * v[1])
        .fold(1., f64::min);
    (solution.x, (value - 1e-8).max(0.))
}

fn dual(upper: &[(usize, Vec<f64>)], actions: usize) -> (Vec<(usize, f64)>, f64) {
    if upper.is_empty() {
        return (Vec::new(), 1.);
    }
    let matrix: Vec<_> = (0..actions)
        .flat_map(|a| upper.iter().map(move |(_, v)| v[a]))
        .collect();
    let solution = solve_matrix_full(&matrix, actions, upper.len());
    let value = (0..actions)
        .map(|a| {
            upper
                .iter()
                .zip(&solution.y)
                .map(|((_, v), q)| q * v[a])
                .sum::<f64>()
        })
        .fold(0., f64::max);
    (
        upper
            .iter()
            .zip(solution.y)
            .map(|((i, _), q)| (*i, q))
            .collect(),
        (value + 1e-8).min(1.),
    )
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 6, "corpus screen output budget-ms backend");
    let numeric = match args[5].as_str() {
        "numeric" | "reuse" | "audit" => true,
        "ledger" => false,
        _ => panic!("invalid backend"),
    };
    let budget_ms: u64 = args[4].parse().unwrap();
    let screen: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let case = &screen["case"];
    let side = case["side"].as_u64().unwrap() as usize;
    let slot = case["foe_slot"].as_u64().unwrap() as usize;
    let decision = case["decision"].as_u64().unwrap() as usize;
    let game: Value = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .find(|g| g["game"] == case["game"])
        .unwrap();
    let dex = conformance::load_dex();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let seed = game["seed"].as_str().unwrap();
    let setup = Instant::now();
    let mut base = Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
    let mut actions = Vec::new();
    for record in game["decisions"].as_array().unwrap().iter().take(decision) {
        assert_eq!(
            base.state_key128().to_string(),
            record["key"].as_str().unwrap()
        );
        let legal = [0, 1].map(|s| base.legal_choices(&dex, s));
        let choices = [0, 1].map(|s| record["choices"][s].as_u64().map(|i| legal[s][i as usize]));
        actions.push(choices);
        base.apply_choices(&dex, choices).unwrap();
    }
    let own_names: Vec<_> = base
        .legal_choices(&dex, side)
        .iter()
        .map(|a| a.to_input(&dex))
        .collect();
    let mut worlds = Vec::new();
    for row in screen["rows"].as_array().unwrap() {
        assert_eq!(row["validity"]["ok"], true);
        assert_eq!(row["prefix_checks"], decision + 1);
        assert!(row.get("mismatch").is_none());
        let request = &row["request"];
        let mut variant = teams.clone();
        let mut mon = serde_json::to_value(&variant[1 - side][slot]).unwrap();
        for (field, value) in request["patch"].as_object().unwrap() {
            mon[field] = value.clone();
        }
        variant[1 - side][slot] = serde_json::from_value(mon).unwrap();
        if let Some(other) = request["swapped_slot"].as_u64() {
            variant[1 - side][other as usize].item = teams[1 - side][slot].item.clone();
        }
        let mut battle = Battle::from_fixture(&dex, seed, &variant[0], &variant[1]).unwrap();
        battle.set_log_enabled(true);
        let mut obs = Observer::new(&battle, side);
        for &choices in &actions {
            battle.apply_choices(&dex, choices).unwrap();
            obs.observe(&battle, &dex);
        }
        assert_eq!(
            battle.state_key128().to_string(),
            row["root"]["key"].as_str().unwrap()
        );
        assert_eq!(
            digest(&serde_json::to_string(&battle.log).unwrap()),
            case["log_hash"].as_str().unwrap()
        );
        assert_eq!(
            digest(&format!("{:?}", obs.mons())),
            case["observer_hash"].as_str().unwrap()
        );
        assert_eq!(
            battle
                .legal_choices(&dex, side)
                .iter()
                .map(|a| a.to_input(&dex))
                .collect::<Vec<_>>(),
            own_names
        );
        battle.set_log_enabled(false);
        worlds.push(battle);
    }
    let setup_seconds = setup.elapsed().as_secs_f64();
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[3]).unwrap());
    writeln!(
        output,
        "{}",
        json!({"kind":"config","case":case,"worlds":worlds.len(),"own_actions":own_names,
        "policies":["move rockslide","move swordsdance"],"continuation":["move rockslide"],
        "opponent_opening":["move sleeptalk","move sleeptalk","move hiddenpowerice"],
        "budget_ms":budget_ms,"backend":args[5],"gap_tolerance":2e-5,"setup_seconds":setup_seconds})
    )
    .unwrap();
    let start = Instant::now();
    let n = worlds.len();
    let mut lower = vec![[0f64; 2]; n];
    let mut rock_done = vec![false; n];
    let mut upper = Vec::<(usize, Vec<f64>)>::new();
    let mut upper_done = vec![false; n];
    let mut swords = 0;
    let mut runs = 0;
    let mut calls = 0;
    let mut snapshots = Vec::new();
    let mut mix = vec![0., 1.];
    let mut lo = 0.;
    let mut hi = 1.;
    let mut adversary = Vec::new();
    let mut checkpoints = [0.5, 1., 5., 10., 30., 60., 120.].into_iter().peekable();
    loop {
        let elapsed = start.elapsed().as_secs_f64();
        while checkpoints.peek().is_some_and(|&t| elapsed >= t) {
            let threshold = checkpoints.next().unwrap();
            snapshots.push(json!({"checkpoint_seconds":threshold,"seconds":elapsed,"lower":lo,"upper":hi,
                "mix":mix,"swords":swords,"rocks":rock_done.iter().filter(|&&v|v).count(),"calls":calls,"runs":runs}));
        }
        if elapsed * 1000. >= budget_ms as f64 || hi - lo <= 2e-5 {
            break;
        }
        let (index, kind) = if swords < n {
            (swords, "swordsdance")
        } else {
            let candidate = (0..n)
                .filter(|&i| rock_done[i] && !upper_done[i] && lower[i][0].max(lower[i][1]) < hi)
                .min_by(|&a, &b| {
                    lower[a][0]
                        .max(lower[a][1])
                        .total_cmp(&lower[b][0].max(lower[b][1]))
                });
            if let Some(i) = candidate {
                (i, "upper")
            } else if let Some(i) = (0..n).filter(|&i| !rock_done[i]).min_by(|&a, &b| {
                let score = |i: usize| mix[0] * lower[i][0] + mix[1] * lower[i][1];
                score(a).total_cmp(&score(b)).then(a.cmp(&b))
            }) {
                (i, "rockslide")
            } else {
                break;
            }
        };
        let target = if kind == "upper" { 1 - side } else { side };
        let first = if kind == "upper" { "sleeptalk" } else { kind };
        let mut cfg = json!({"game":case["game"],"decision":case["decision"],"target_side":target,"policy_side":target,
            "first_choice":format!("move {first}"),"priority":[if kind=="upper" {"move hiddenpowerice"} else {"move rockslide"}],
            "rest_at_percent":null,"horizon":if kind=="upper" {3} else {2},"work_budget":200000,"cell_cap":10000,
            "guarded_keys":true,"support_certificate":true});
        let begin = Instant::now();
        if kind == "upper" {
            cfg["opening"] = json!(["move sleeptalk", "move sleeptalk", "move hiddenpowerice"]);
        }
        let ledger = if args[5] == "reuse" || args[5] == "audit" {
            reused_witness::solve(&dex, worlds[index].clone(), cfg, args[5] == "audit")
        } else if numeric {
            numeric_witness::solve(&dex, worlds[index].clone(), cfg)
        } else if kind == "upper" {
            aliased_witness::upper(&dex, worlds[index].clone(), cfg)
        } else {
            threshold_witness::solve(&dex, worlds[index].clone(), cfg, 1., "native")
        };
        let cell_seconds = begin.elapsed().as_secs_f64();
        runs += ledger["runs"].as_u64().unwrap();
        calls += 1;
        if kind == "upper" {
            let root = &ledger["nodes"][ledger["root"].as_u64().unwrap() as usize];
            let values = own_names
                .iter()
                .map(|name| {
                    let branch = root["branches"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|b| b["choices"][side] == *name)
                        .unwrap();
                    1. - branch["bounds"][0].as_f64().unwrap()
                })
                .collect();
            upper.push((index, values));
            upper_done[index] = true;
        } else {
            let action = usize::from(kind == "swordsdance");
            lower[index][action] = ledger["bounds"][0].as_f64().unwrap();
            if action == 1 {
                swords += 1;
            } else {
                rock_done[index] = true;
            }
        }
        if swords == n {
            (mix, lo) = mixture(&lower);
            (adversary, hi) = dual(&upper, own_names.len());
        }
        writeln!(
            output,
            "{}",
            json!({"kind":kind,"index":index,"id":screen["rows"][index]["request"]["id"],
            "seconds":start.elapsed().as_secs_f64(),"cell_seconds":cell_seconds,"ledger":ledger,
            "mix":mix,"adversary":adversary,"bounds":[lo,hi],"calls":calls,"runs":runs})
        )
        .unwrap();
        if calls % 100 == 0 || kind == "upper" {
            output.flush().unwrap();
            eprintln!(
                "{calls} {kind} {index} {:.3}s runs={runs} mix={mix:?} bounds=[{lo},{hi}]",
                start.elapsed().as_secs_f64()
            );
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    writeln!(
        output,
        "{}",
        json!({"kind":"result","seconds":seconds,"setup_seconds":setup_seconds,"mix":mix,
        "adversary":adversary,"bounds":[lo,hi],"calls":calls,"runs":runs,"snapshots":snapshots,
        "swords":swords,"rocks":rock_done.iter().filter(|&&v|v).count(),"uppers":upper.len(),
        "converged":hi-lo <= 2e-5})
    )
    .unwrap();
    output.flush().unwrap();
    eprintln!("finished {seconds:.3}s [{lo},{hi}] mix={mix:?}");
}
