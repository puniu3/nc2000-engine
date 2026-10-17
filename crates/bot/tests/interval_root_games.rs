//! End-to-end checks on a class of finite simultaneous stochastic games.
//! The baseline's actual continuation, not an oracle continuation, supplies Q.
use nc2000_bot::{
    exact::solve_matrix_full,
    rng::SplitMix64,
    root_policy::interval::{analyze, InformationModel, PayoffBounds},
};

struct Node {
    rows: usize,
    cols: usize,
    edges: Vec<(f64, f64, usize)>,
}
fn matrix(n: &Node, values: &[f64]) -> Vec<f64> {
    n.edges
        .iter()
        .map(|&(terminal, prob, child)| prob * terminal + (1. - prob) * values[child])
        .collect()
}
fn guarantee(a: &[f64], p: &[f64], cols: usize) -> f64 {
    (0..cols)
        .map(|j| {
            p.iter()
                .enumerate()
                .map(|(i, p)| p * a[i * cols + j])
                .sum::<f64>()
        })
        .fold(1., f64::min)
}
fn evaluate(nodes: &[Node], policies: &[Vec<f64>]) -> Vec<f64> {
    let mut v = vec![0.; nodes.len() + 1];
    for i in (0..nodes.len()).rev() {
        v[i] = guarantee(&matrix(&nodes[i], &v), &policies[i], nodes[i].cols);
    }
    v
}

#[test]
fn policy_iteration_improves_all_states_and_solves_finite_games() {
    let mut rng = SplitMix64::new(2026101702);
    let mut changes = 0;
    for _ in 0..200 {
        let count = 7;
        let nodes: Vec<_> = (0..count)
            .map(|i| {
                let rows = 2 + rng.below(3) as usize;
                let cols = 2 + rng.below(3) as usize;
                let edges = (0..rows * cols)
                    .map(|_| {
                        (
                            rng.below(11) as f64 / 10.,
                            rng.below(5) as f64 / 4.,
                            i + 1 + rng.below(count - i),
                        )
                    })
                    .collect();
                Node { rows, cols, edges }
            })
            .collect();
        let mut policies: Vec<_> = nodes
            .iter()
            .map(|n| {
                let mut p = vec![0.; n.rows];
                p[0] = 1.;
                p
            })
            .collect();
        let mut optimal = vec![0.; count + 1];
        for i in (0..count).rev() {
            let n = &nodes[i];
            optimal[i] = solve_matrix_full(&matrix(n, &optimal), n.rows, n.cols).value;
        }
        for _ in 0..count {
            let before = evaluate(&nodes, &policies);
            for (i, n) in nodes.iter().enumerate() {
                let q = matrix(n, &before);
                let r = analyze(
                    &vec![true; n.rows],
                    &policies[i],
                    &[PayoffBounds {
                        replies: n.cols,
                        lower: q.clone(),
                        upper: q,
                    }],
                    &InformationModel::WorstCase,
                    0.,
                )
                .unwrap();
                changes += usize::from(r.replace);
                policies[i] = r.selected;
            }
            let after = evaluate(&nodes, &policies);
            assert!(before.iter().zip(&after).all(|(a, b)| b + 1e-10 >= *a));
        }
        let actual = evaluate(&nodes, &policies);
        assert!(actual
            .iter()
            .zip(optimal)
            .all(|(v, opt)| (opt - v).abs() < 4e-7));
    }
    assert!(changes > 500);
}

#[test]
fn resource_waiting_requires_a_continuation_policy_not_just_one_mixed_move() {
    let attack_success = 0.91461181640625;
    let mut optimum = attack_success;
    for resources in 1..=32 {
        let q = vec![attack_success, 0., 0., optimum];
        let r = analyze(
            &[true, true],
            &[0., 1.],
            &[PayoffBounds {
                replies: 2,
                lower: q.clone(),
                upper: q.clone(),
            }],
            &InformationModel::WorstCase,
            0.,
        )
        .unwrap();
        optimum = guarantee(&q, &r.selected, 2);
        assert!((r.selected[0] - 1. / (resources + 1) as f64).abs() < 1e-9);
        assert!((optimum - attack_success / (resources + 1) as f64).abs() < 1e-9);
    }
    // A correct first distribution plus "always wait" after a successful
    // defensive reply has value zero. The continuation cell is zero, not V_6.
    let mixed_first = [1. / 8., 7. / 8.];
    assert_eq!(
        guarantee(&[attack_success, 0., 0., 0.], &mixed_first, 2),
        0.
    );
    // Reusing the root's 1/8 at every remaining resource count is also wrong.
    let mut stationary = attack_success / 8.;
    for _ in 0..7 {
        stationary = (attack_success / 8.).min(7. / 8. * stationary);
    }
    assert!(stationary < attack_success / 8. / 2.);
}
