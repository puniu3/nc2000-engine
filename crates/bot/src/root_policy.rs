use crate::exact::solve_matrix_full;

#[derive(Clone, serde::Serialize)]
pub struct RootCell {
    pub own: usize,
    pub reply: usize,
    pub samples: u32,
    pub mean: f64,
}

#[derive(serde::Serialize)]
pub struct RootType {
    pub samples: u32,
    pub replies: usize,
    pub cells: Vec<RootCell>,
}

pub struct MixedRoot {
    pub policy: Vec<f64>,
    pub estimated_advantage: f64,
}

struct TypeMatrix {
    weight: f64,
    replies: usize,
    values: Vec<f64>,
}

fn response_column(types: &[TypeMatrix], policy: &[f64]) -> Vec<f64> {
    let mut column = vec![0.; policy.len()];
    for t in types {
        let reply = (0..t.replies)
            .min_by(|&a, &b| {
                let value = |j| {
                    policy
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| p * t.values[i * t.replies + j])
                        .sum::<f64>()
                };
                value(a).total_cmp(&value(b))
            })
            .unwrap();
        for (i, value) in column.iter_mut().enumerate() {
            *value += t.weight * t.values[i * t.replies + reply];
        }
    }
    column
}

fn guarantee(types: &[TypeMatrix], policy: &[f64]) -> f64 {
    response_column(types, policy)
        .iter()
        .zip(policy)
        .map(|(v, p)| v * p)
        .sum()
}

fn best_pure(types: &[TypeMatrix], rows: usize) -> f64 {
    (0..rows)
        .map(|i| {
            types
                .iter()
                .map(|t| {
                    t.weight
                        * t.values[i * t.replies..(i + 1) * t.replies]
                            .iter()
                            .copied()
                            .fold(f64::INFINITY, f64::min)
                })
                .sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

fn solve_bayesian(types: &[TypeMatrix], rows: usize) -> Option<Vec<f64>> {
    let initial = vec![1. / rows as f64; rows];
    let mut columns = vec![response_column(types, &initial)];
    for _ in 0..128 {
        let matrix: Vec<f64> = (0..rows)
            .flat_map(|i| columns.iter().map(move |c| c[i]))
            .collect();
        let mut solution = solve_matrix_full(&matrix, rows, columns.len());
        if !solution.gap.is_finite()
            || solution.gap > 1e-8
            || solution.x.iter().any(|p| !p.is_finite() || *p < -1e-12)
            || (solution.x.iter().sum::<f64>() - 1.).abs() > 1e-8
        {
            return None;
        }
        solution.x.iter_mut().for_each(|p| *p = p.max(0.));
        let total = solution.x.iter().sum::<f64>();
        solution.x.iter_mut().for_each(|p| *p /= total);
        let response = response_column(types, &solution.x);
        let lower: f64 = response.iter().zip(&solution.x).map(|(v, p)| v * p).sum();
        let upper = solution.value + solution.gap / 2.;
        if upper - lower <= 1e-8 {
            return Some(solution.x);
        }
        if columns.contains(&response) {
            return None;
        }
        columns.push(response);
    }
    None
}

struct RootMatrices {
    actions: Vec<usize>,
    means: Vec<TypeMatrix>,
    lower: Vec<TypeMatrix>,
    upper: Vec<TypeMatrix>,
}

fn matrices(eligible: &[bool], iterations: u32, types: &[RootType]) -> Option<RootMatrices> {
    let actions: Vec<usize> = (0..eligible.len()).filter(|&i| eligible[i]).collect();
    let n = actions.len();
    if n < 2
        || iterations == 0
        || types.is_empty()
        || types.iter().any(|t| t.samples == 0 || t.replies == 0)
        || types.iter().map(|t| u64::from(t.samples)).sum::<u64>() != u64::from(iterations)
    {
        return None;
    }
    let total_cells = n * types.iter().map(|t| t.replies).sum::<usize>();
    let mut out = RootMatrices {
        actions,
        means: Vec::new(),
        lower: Vec::new(),
        upper: Vec::new(),
    };
    for t in types {
        let m = t.replies;
        let mut means = vec![0.5; n * m];
        let mut lower = vec![0.; n * m];
        let mut upper = vec![1.; n * m];
        let mut seen = vec![false; eligible.len() * m];
        for cell in &t.cells {
            if cell.own >= eligible.len()
                || cell.reply >= m
                || cell.samples == 0
                || cell.samples > t.samples
                || !(0.0..=1.0).contains(&cell.mean)
            {
                return None;
            }
            let index = cell.own * m + cell.reply;
            if std::mem::replace(&mut seen[index], true) {
                return None;
            }
            let Some(i) = out.actions.iter().position(|&own| own == cell.own) else {
                continue;
            };
            let radius = ((200. * total_cells as f64).ln() / (2. * f64::from(cell.samples))).sqrt();
            means[i * m + cell.reply] = cell.mean;
            lower[i * m + cell.reply] = (cell.mean - radius).max(0.);
            upper[i * m + cell.reply] = (cell.mean + radius).min(1.);
        }
        let weight = f64::from(t.samples) / f64::from(iterations);
        out.means.push(TypeMatrix {
            weight,
            replies: m,
            values: means,
        });
        out.lower.push(TypeMatrix {
            weight,
            replies: m,
            values: lower,
        });
        out.upper.push(TypeMatrix {
            weight,
            replies: m,
            values: upper,
        });
    }
    Some(out)
}

pub fn estimated_bayesian_mix_gain(
    eligible: &[bool],
    iterations: u32,
    types: &[RootType],
) -> Option<f64> {
    let matrices = matrices(eligible, iterations, types)?;
    let n = matrices.actions.len();
    let policy = solve_bayesian(&matrices.means, n)?;
    Some((guarantee(&matrices.means, &policy) - best_pure(&matrices.means, n)).max(0.))
}

#[derive(serde::Serialize)]
pub struct RootAnalysis {
    pub policy: Vec<f64>,
    pub estimated_guarantee: f64,
    pub estimated_best_pure: f64,
    pub lower_guarantee: f64,
    pub upper_best_pure: f64,
}

impl RootAnalysis {
    pub fn estimated_gain(&self) -> f64 {
        (self.estimated_guarantee - self.estimated_best_pure).max(0.)
    }

    pub fn margin(&self) -> f64 {
        self.lower_guarantee - self.upper_best_pure
    }
}

pub fn analyze_bayesian_root(
    eligible: &[bool],
    iterations: u32,
    types: &[RootType],
) -> Option<RootAnalysis> {
    let matrices = matrices(eligible, iterations, types)?;
    let n = matrices.actions.len();
    let mixed = solve_bayesian(&matrices.means, n)?;
    let mut policy = vec![0.; eligible.len()];
    for (i, &a) in matrices.actions.iter().enumerate() {
        policy[a] = mixed[i];
    }
    Some(RootAnalysis {
        policy,
        estimated_guarantee: guarantee(&matrices.means, &mixed),
        estimated_best_pure: best_pure(&matrices.means, n),
        lower_guarantee: guarantee(&matrices.lower, &mixed),
        upper_best_pure: best_pure(&matrices.upper, n),
    })
}

// Search continuations are nonstationary; these margins do not certify game values.
pub fn bayesian_mixed_root(
    eligible: &[bool],
    iterations: u32,
    types: &[RootType],
) -> Option<MixedRoot> {
    let analysis = analyze_bayesian_root(eligible, iterations, types)?;
    let estimated_advantage = analysis.margin();
    if estimated_advantage <= 0.02 {
        return None;
    }
    Some(MixedRoot {
        policy: analysis.policy,
        estimated_advantage,
    })
}

pub fn estimated_mix_gain(
    eligible: &[bool],
    iterations: u32,
    availability: &[u32],
    cells: &[RootCell],
) -> Option<f64> {
    if availability.len() < 2 || availability.iter().any(|&a| a != iterations) {
        return None;
    }
    estimated_bayesian_mix_gain(
        eligible,
        iterations,
        &[RootType {
            samples: iterations,
            replies: availability.len(),
            cells: cells.to_vec(),
        }],
    )
}

pub fn mixed_root(
    eligible: &[bool],
    iterations: u32,
    availability: &[u32],
    cells: &[RootCell],
) -> Option<MixedRoot> {
    if availability.len() < 2 || availability.iter().any(|&a| a != iterations) {
        return None;
    }
    bayesian_mixed_root(
        eligible,
        iterations,
        &[RootType {
            samples: iterations,
            replies: availability.len(),
            cells: cells.to_vec(),
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(values: &[f64], replies: usize, samples: u32) -> Vec<RootCell> {
        values
            .iter()
            .enumerate()
            .map(|(i, &mean)| RootCell {
                own: i / replies,
                reply: i % replies,
                samples,
                mean,
            })
            .collect()
    }

    #[test]
    fn crossed_replies_require_mixing_but_pure_saddles_do_not() {
        let mix = mixed_root(
            &[true, true],
            40000,
            &[40000, 40000],
            &cells(&[0., 1., 1., 0.], 2, 10000),
        )
        .unwrap();
        assert!((mix.policy[0] - 0.5).abs() < 1e-12);
        assert!((mix.policy[1] - 0.5).abs() < 1e-12);
        assert!(mix.estimated_advantage > 0.45);
        assert!(mixed_root(
            &[true, true],
            40000,
            &[40000, 40000],
            &cells(&[0.8, 0.9, 0.2, 0.3], 2, 10000)
        )
        .is_none());
    }

    #[test]
    fn absent_replies_and_insufficient_samples_cannot_enable_mixing() {
        let values = cells(&[0., 1., 1., 0.], 2, 10000);
        assert!(mixed_root(&[true, true], 40000, &[40000, 39999], &values).is_none());
        assert!(mixed_root(&[true, true], 0, &[0, 0], &values).is_none());
        assert!(mixed_root(&[true, true], 4, &[4, 4], &cells(&[0., 1., 1., 0.], 2, 1)).is_none());
        assert!(mixed_root(&[true, true], 40000, &[40000, 40000], &values[..3]).is_none());
    }

    #[test]
    fn promising_but_uncertain_matrices_request_more_evidence() {
        let crossed = cells(&[0., 1., 1., 0.], 2, 1);
        assert_eq!(
            estimated_mix_gain(&[true, true], 4, &[4, 4], &crossed),
            Some(0.5)
        );
        assert!(mixed_root(&[true, true], 4, &[4, 4], &crossed).is_none());
        assert_eq!(
            estimated_mix_gain(&[true, true], 4, &[4, 3], &crossed),
            None
        );
        assert_eq!(
            estimated_mix_gain(
                &[true, true],
                4,
                &[4, 4],
                &cells(&[0.8, 0.9, 0.2, 0.3], 2, 1)
            ),
            Some(0.)
        );
    }

    #[test]
    fn excluded_actions_receive_no_mass_even_when_they_dominate() {
        let mix = mixed_root(
            &[true, true, false],
            60000,
            &[60000, 60000],
            &cells(&[0., 1., 1., 0., 1., 1.], 2, 10000),
        )
        .unwrap();
        assert_eq!(mix.policy[2], 0.);
        assert!((mix.policy.iter().sum::<f64>() - 1.).abs() < 1e-12);
        assert!(mixed_root(
            &[false, false, true],
            60000,
            &[60000, 60000],
            &cells(&[0., 1., 1., 0., 1., 1.], 2, 10000)
        )
        .is_none());
    }

    fn typed(values: &[f64], replies: usize) -> RootType {
        RootType {
            samples: values.len() as u32 * 100000,
            replies,
            cells: cells(values, replies, 100000),
        }
    }

    #[test]
    fn privately_known_types_cannot_be_averaged_before_best_response() {
        let types = [typed(&[0., 0.5, 0., 1.], 2), typed(&[1., 0., 0.5, 0.5], 2)];
        let iterations = types.iter().map(|t| t.samples).sum();
        let naive = mixed_root(
            &[true, true],
            800000,
            &[800000, 800000],
            &cells(&[0.5, 0.25, 0.25, 0.75], 2, 200000),
        )
        .unwrap();
        assert!((naive.policy[0] - 2. / 3.).abs() < 1e-10);
        let m = matrices(&[true, true], iterations, &types).unwrap();
        assert!((guarantee(&m.means, &naive.policy) - 1. / 12.).abs() < 1e-10);
        assert!((best_pure(&m.means, 2) - 0.25).abs() < 1e-10);
        let correct = solve_bayesian(&m.means, 2).unwrap();
        assert!((correct[1] - 1.).abs() < 1e-10);
        assert!((guarantee(&m.means, &correct) - 0.25).abs() < 1e-10);
        assert_eq!(
            estimated_bayesian_mix_gain(&[true, true], iterations, &types),
            Some(0.)
        );
        assert!(bayesian_mixed_root(&[true, true], iterations, &types).is_none());
    }

    #[test]
    fn identical_types_and_different_legal_reply_sets_preserve_valid_mixtures() {
        let types = [
            typed(&[0., 1., 1., 0.], 2),
            typed(&[0., 1., 1., 1., 0., 1.], 3),
        ];
        let iterations = types.iter().map(|t| t.samples).sum();
        let mixed = bayesian_mixed_root(&[true, true], iterations, &types).unwrap();
        assert!((mixed.policy[0] - 0.5).abs() < 1e-10);
        assert!(mixed.estimated_advantage > 0.45);
        let missing = [
            typed(&[0., 1., 1., 0.], 2),
            RootType {
                samples: 400000,
                replies: 2,
                cells: vec![],
            },
        ];
        assert!(bayesian_mixed_root(&[true, true], 800000, &missing).is_none());
        assert!(bayesian_mixed_root(&[true, true], iterations - 1, &types).is_none());
    }

    fn expanded(types: &[TypeMatrix], rows: usize) -> Vec<f64> {
        let columns: usize = types.iter().map(|t| t.replies).product();
        (0..rows)
            .flat_map(|i| {
                (0..columns).map(move |mut column| {
                    types
                        .iter()
                        .map(|t| {
                            let reply = column % t.replies;
                            column /= t.replies;
                            t.weight * t.values[i * t.replies + reply]
                        })
                        .sum::<f64>()
                })
            })
            .collect()
    }

    #[test]
    fn response_oracle_matches_all_contingent_policies() {
        let mut rng = crate::rng::SplitMix64::new(2026111701);
        for trial in 0..600 {
            let rows = 2 + trial % 3;
            let types: Vec<TypeMatrix> = [2, 3, 2]
                .into_iter()
                .enumerate()
                .map(|(t, replies)| TypeMatrix {
                    replies,
                    weight: [0.2, 0.3, 0.5][t],
                    values: (0..rows * replies)
                        .map(|_| rng.below(5) as f64 / 4.)
                        .collect(),
                })
                .collect();
            let full = expanded(&types, rows);
            let reference = solve_matrix_full(&full, rows, 12);
            assert!(reference.gap <= 1e-8);
            let policy = solve_bayesian(&types, rows).unwrap_or_else(|| panic!("trial {trial}"));
            assert!(
                (guarantee(&types, &policy) - reference.value).abs() <= 1e-8,
                "trial {trial}"
            );
        }
    }

    #[test]
    fn margins_apply_to_the_returned_policy() {
        let mut rng = crate::rng::SplitMix64::new(2026111702);
        let mut accepted = 0;
        for _ in 0..300 {
            let types: Vec<RootType> = (0..3)
                .map(|_| {
                    let values: Vec<f64> = (0..9).map(|_| rng.below(5) as f64 / 4.).collect();
                    typed(&values, 3)
                })
                .collect();
            let iterations = types.iter().map(|t| t.samples).sum();
            if let Some(mixed) = bayesian_mixed_root(&[true; 3], iterations, &types) {
                let m = matrices(&[true; 3], iterations, &types).unwrap();
                let advantage = guarantee(&m.lower, &mixed.policy) - best_pure(&m.upper, 3);
                assert!(advantage > 0.02);
                assert!((advantage - mixed.estimated_advantage).abs() < 1e-12);
                accepted += 1;
            }
        }
        assert!(accepted > 20, "only {accepted} accepted mixtures");
    }
}
