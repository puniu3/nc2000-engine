use nc2000_bot::{
    exact::solve_matrix_full, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, Outcome, PokemonSet},
    state::Battle,
};
use serde_json::json;

fn set(species: &str, item: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(
        json!({"species":species,"name":species,"gender":"M","level":50,"item":item,"moves":moves,
        "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}}),
    )
    .unwrap()
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 2, "output");
    let dex = conformance::load_dex();
    let teams = [
        vec![set("Snorlax", "Quick Claw", &["Rest", "Swagger"])],
        vec![set("Machamp", "", &["Seismic Toss", "Cross Chop"])],
    ];
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
    let mut obs = Observer::new(&b, 0);
    let mut belief = Belief::pinned_from_battle(&b, &obs);
    b.choose(&dex, 0, "team 1").unwrap();
    b.choose(&dex, 1, "team 1").unwrap();
    b.turn = 1000;
    for (side, hp) in [(0, 40), (1, 1)] {
        let id = b.active_id(side).unwrap();
        b.poke_mut(id).hp = hp;
    }
    obs.observe(&b, &dex);
    belief.sync(&dex, &obs);
    let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
    assert_eq!(legal.each_ref().map(|a| a.len()), [2, 2]);
    let mut cells = Vec::new();
    let mut matrices = [vec![], vec![]];
    let mut runs = 0;
    for (coin, matrix) in matrices.iter_mut().enumerate() {
        b.quick_claw_roll = coin == 1;
        for &a in &legal[0] {
            for &r in &legal[1] {
                let enumeration = enumerate_step(&dex, &b, [Some(a), Some(r)], 100000).unwrap();
                runs += enumeration.runs;
                let mut outcomes = [0.; 3];
                for leaf in &enumeration.leaves {
                    let i = match leaf.battle.outcome().expect("terminal cell") {
                        Outcome::P1Win => 0,
                        Outcome::Tie => 1,
                        Outcome::P2Win => 2,
                    };
                    outcomes[i] += leaf.prob;
                }
                assert!((outcomes.iter().sum::<f64>() - 1.).abs() < 1e-10);
                let value = outcomes[0] + 0.5 * outcomes[1];
                matrix.push(value);
                cells.push(json!({"coin":coin,"own":a.to_input(&dex),"reply":r.to_input(&dex),"outcomes":outcomes,"value":value,"runs":enumeration.runs}));
            }
        }
    }
    let p = 60. / 256.;
    let hidden = matrices[0]
        .iter()
        .zip(&matrices[1])
        .map(|(a, b)| (1. - p) * a + p * b)
        .collect::<Vec<_>>();
    let informed = (0..2)
        .flat_map(|i| (0..2).flat_map(move |a| (0..2).map(move |b| (i, a, b))))
        .map(|(i, a, b)| (1. - p) * matrices[0][i * 2 + a] + p * matrices[1][i * 2 + b])
        .collect::<Vec<_>>();
    let h = solve_matrix_full(&hidden, 2, 2);
    let f = solve_matrix_full(&informed, 2, 4);
    let mut searches = Vec::new();
    b.quick_claw_roll = false;
    for enabled in [false, true] {
        for seed in 0..32 {
            let mut search = BlindSearch::new(
                &b,
                &dex,
                RmConfig {
                    c: 0.4,
                    unseen_claw_root: enabled,
                    rule: SelRule::Ucb,
                    ..Default::default()
                },
                0,
                0x2026102200 + seed,
            );
            search.step(&dex, &belief, &obs, 27000);
            searches.push(json!({"enabled":enabled,"seed":0x2026102200+seed,"best":search.best().unwrap().to_input(&dex),"visits":search.visits(),"means":search.means()}));
        }
    }
    let solution = |s: nc2000_bot::exact::MatrixSolution| json!({"value":s.value,"gap":s.gap,"own":s.x,"opponent":s.y});
    std::fs::write(&args[1],serde_json::to_vec_pretty(&json!({"teams":teams,"turn":b.turn,"hp":[40,1],"key":b.state_key128().to_string(),
        "actions":legal.map(|v|v.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),"claw_probability":p,"runs":runs,"cells":cells,
        "hidden_matrix":hidden,"informed_matrix":informed,"hidden_solution":solution(h),"informed_solution":solution(f),"searches":searches})).unwrap()).unwrap();
}
