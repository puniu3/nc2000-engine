use crate::exact::solve_matrix_full;

pub const MIN_MIX_MARGIN: f64 = 0.02;

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
    if estimated_advantage <= MIN_MIX_MARGIN {
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

#[derive(Clone, Debug)]
pub struct BayesianEquilibrium {
    pub policy: Vec<f64>,
    pub action_values: Vec<f64>,
    pub pure_guarantees: Vec<f64>,
    pub equilibrium_value: f64,
    pub best_pure_value: f64,
    pub best_pure_action: usize,
}

/// Solve the empirical-Bayes shrunk Bayesian root game across sampled determinized worlds:
/// `\max_{p \in \Delta} \sum_k w_k \min_j \sum_i p_i \hat{V}_k(i, j)`.
pub fn solve_bayesian_equilibrium(
    eligible: &[bool],
    global_visits: &[u32],
    global_means: &[f64],
    types: &[RootType],
) -> Option<BayesianEquilibrium> {
    solve_bayesian_equilibrium_with_priors(eligible, global_visits, global_means, types, &[])
}

pub fn solve_bayesian_equilibrium_with_priors(
    eligible: &[bool],
    global_visits: &[u32],
    global_means: &[f64],
    types: &[RootType],
    pooled_priors: &[Vec<RootCell>],
) -> Option<BayesianEquilibrium> {
    let k_total = eligible.len();
    if k_total == 0
        || global_visits.len() != k_total
        || global_means.len() != k_total
        || types.is_empty()
    {
        return None;
    }
    let all_elig: Vec<usize> = (0..k_total).filter(|&i| eligible[i]).collect();
    if all_elig.is_empty() {
        return None;
    }
    if all_elig.len() == 1 {
        let only = all_elig[0];
        let mut policy = vec![0.0; k_total];
        policy[only] = 1.0;
        let v = global_means[only];
        return Some(BayesianEquilibrium {
            policy,
            action_values: global_means.to_vec(),
            pure_guarantees: global_means.to_vec(),
            equilibrium_value: v,
            best_pure_value: v,
            best_pure_action: only,
        });
    }
    let max_v = all_elig
        .iter()
        .map(|&i| global_visits[i])
        .max()
        .unwrap_or(0);
    let min_own_v = (max_v / 32).clamp(4, 64).min(max_v);
    let mut actions: Vec<usize> = all_elig
        .iter()
        .copied()
        .filter(|&i| global_visits[i] >= min_own_v)
        .collect();
    if actions.len() < 2 {
        actions = all_elig.clone();
    }
    let n = actions.len();

    let mut raw_worlds: Vec<(f64, usize, Vec<f64>)> = Vec::with_capacity(types.len());
    let mut total_weight = 0.0_f64;

    for (t_idx, t) in types.iter().enumerate() {
        let m = t.replies;
        if t.samples == 0 || m == 0 || t.cells.is_empty() {
            continue;
        }
        let mut cell_n = vec![0u32; n * m];
        let mut cell_w = vec![0.0f64; n * m];
        let mut row_n = vec![0u32; n];
        let mut row_w = vec![0.0f64; n];
        let mut col_n = vec![0u32; m];
        let mut col_w = vec![0.0f64; m];
        let mut world_n = 0u32;
        let mut world_w = 0.0f64;

        for cell in &t.cells {
            if cell.reply >= m || cell.samples == 0 || !cell.mean.is_finite() {
                continue;
            }
            let Some(r) = actions.iter().position(|&own| own == cell.own) else {
                continue;
            };
            let s = cell.samples;
            let w = f64::from(s) * cell.mean.clamp(0.0, 1.0);
            cell_n[r * m + cell.reply] += s;
            cell_w[r * m + cell.reply] += w;
            row_n[r] += s;
            row_w[r] += w;
            col_n[cell.reply] += s;
            col_w[cell.reply] += w;
            world_n += s;
            world_w += w;
        }
        if world_n == 0 {
            continue;
        }
        let mut prior_n = vec![0u32; n * m];
        let mut prior_mean = vec![0.0f64; n * m];
        if let Some(priors) = pooled_priors.get(t_idx) {
            for cell in priors {
                if cell.reply >= m || cell.samples == 0 || !cell.mean.is_finite() {
                    continue;
                }
                if let Some(r) = actions.iter().position(|&own| own == cell.own) {
                    prior_n[r * m + cell.reply] = cell.samples;
                    prior_mean[r * m + cell.reply] = cell.mean.clamp(0.0, 1.0);
                }
            }
        }
        let world_mean = world_w / f64::from(world_n);
        let max_col_n = col_n.iter().copied().max().unwrap_or(0);
        let min_col_n = (world_n / (2 * m as u32))
            .max(max_col_n / 5)
            .clamp(4, 96)
            .min(max_col_n);
        let mut active_cols: Vec<usize> = (0..m).filter(|&j| col_n[j] >= min_col_n).collect();
        if active_cols.is_empty() {
            active_cols = (0..m).filter(|&j| col_n[j] > 0).collect();
        }
        let m_act = active_cols.len();
        let mut values = vec![0.5f64; n * m_act];
        const KAPPA_ROW: f64 = 16.0;
        const KAPPA_CELL: f64 = 64.0;
        for r in 0..n {
            let own = actions[r];
            let g_mean = global_means[own].clamp(0.0, 1.0);
            let mu_row = (row_w[r] + KAPPA_ROW * g_mean) / (f64::from(row_n[r]) + KAPPA_ROW);
            for (c_idx, &j) in active_cols.iter().enumerate() {
                let mu_col = if col_n[j] > 0 {
                    col_w[j] / f64::from(col_n[j])
                } else {
                    world_mean
                };
                let col_blunder_bonus = (mu_col - world_mean).max(0.0);
                let additive_prior = (mu_row + col_blunder_bonus).clamp(0.0, 1.0);
                let p_n = f64::from(prior_n[r * m + j]);
                let mu_prior = if p_n > 0.0 {
                    let pooled_val = prior_mean[r * m + j].clamp(0.0, 1.0);
                    (p_n * pooled_val + 8.0 * additive_prior) / (p_n + 8.0)
                } else {
                    additive_prior
                };
                let cn = f64::from(cell_n[r * m + j]);
                let cw = cell_w[r * m + j];
                values[r * m_act + c_idx] = (cw + KAPPA_CELL * mu_prior) / (cn + KAPPA_CELL);
            }
        }
        let w_k = f64::from(world_n);
        total_weight += w_k;
        raw_worlds.push((w_k, m_act, values));
    }

    if raw_worlds.is_empty() || total_weight <= 0.0 {
        return None;
    }

    let type_matrices: Vec<TypeMatrix> = raw_worlds
        .into_iter()
        .map(|(w_k, replies, values)| TypeMatrix {
            weight: w_k / total_weight,
            replies,
            values,
        })
        .collect();

    let mut pure_guarantees = global_means.to_vec();
    for (r, &own) in actions.iter().enumerate() {
        pure_guarantees[own] = type_matrices
            .iter()
            .map(|t| {
                t.weight
                    * t.values[r * t.replies..(r + 1) * t.replies]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>();
    }
    let best_pure_sub = (0..n)
        .max_by(|&a, &b| {
            pure_guarantees[actions[a]]
                .total_cmp(&pure_guarantees[actions[b]])
                .then_with(|| global_visits[actions[a]].cmp(&global_visits[actions[b]]))
        })
        .unwrap_or(0);
    let best_pure_action = actions[best_pure_sub];
    let best_pure_value = pure_guarantees[best_pure_action];

    let mut initial = vec![0.0; n];
    initial[best_pure_sub] = 1.0;
    let mut columns = vec![response_column(&type_matrices, &initial)];
    let uniform = vec![1.0 / n as f64; n];
    let u_col = response_column(&type_matrices, &uniform);
    if !columns.contains(&u_col) {
        columns.push(u_col);
    }

    let mut best_x = initial.clone();
    let mut best_y = vec![1.0; 1];
    let mut best_cols_len = 1usize;

    for _ in 0..64 {
        let c_len = columns.len();
        let matrix: Vec<f64> = (0..n)
            .flat_map(|i| columns.iter().map(move |c| c[i]))
            .collect();
        let mut solution = solve_matrix_full(&matrix, n, c_len);
        if !solution.gap.is_finite()
            || solution.gap > 1e-6
            || solution.x.iter().any(|p| !p.is_finite() || *p < -1e-9)
            || (solution.x.iter().sum::<f64>() - 1.0).abs() > 1e-6
        {
            break;
        }
        solution.x.iter_mut().for_each(|p| *p = p.max(0.0));
        let total = solution.x.iter().sum::<f64>();
        if total <= 0.0 {
            break;
        }
        solution.x.iter_mut().for_each(|p| *p /= total);
        best_x = solution.x.clone();
        best_y = solution.y.clone();
        best_cols_len = c_len;

        let response = response_column(&type_matrices, &solution.x);
        let lower: f64 = response.iter().zip(&solution.x).map(|(v, p)| v * p).sum();
        let upper = solution.value + solution.gap / 2.0;
        if upper - lower <= 1e-7 || columns.contains(&response) {
            break;
        }
        columns.push(response);
    }

    let mut action_values = pure_guarantees.clone();
    if best_y.len() == best_cols_len && best_cols_len <= columns.len() {
        let y_sum: f64 = best_y.iter().map(|&y| y.max(0.0)).sum();
        if y_sum > 0.0 {
            for (r, &own) in actions.iter().enumerate() {
                let eq_v: f64 = (0..best_cols_len)
                    .map(|c| (best_y[c].max(0.0) / y_sum) * columns[c][r])
                    .sum();
                action_values[own] = 0.5 * eq_v + 0.5 * pure_guarantees[own];
            }
        }
    }

    let max_p = best_x.iter().copied().fold(0.0_f64, f64::max);
    let mut purified_x: Vec<f64> = best_x
        .iter()
        .map(|&p| if p >= 0.05 * max_p { p } else { 0.0 })
        .collect();
    let p_sum: f64 = purified_x.iter().sum();
    if p_sum > 0.0 {
        purified_x.iter_mut().for_each(|p| *p /= p_sum);
    } else {
        purified_x = vec![0.0; n];
        purified_x[best_pure_sub] = 1.0;
    }

    let eq_guarantee = guarantee(&type_matrices, &purified_x);
    let mut policy = vec![0.0; k_total];
    if eq_guarantee > best_pure_value + 0.005 {
        for (r, &own) in actions.iter().enumerate() {
            policy[own] = purified_x[r];
        }
        Some(BayesianEquilibrium {
            policy,
            action_values,
            pure_guarantees,
            equilibrium_value: eq_guarantee,
            best_pure_value,
            best_pure_action,
        })
    } else {
        policy[best_pure_action] = 1.0;
        Some(BayesianEquilibrium {
            policy,
            action_values,
            pure_guarantees,
            equilibrium_value: best_pure_value,
            best_pure_value,
            best_pure_action,
        })
    }
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
                assert!(advantage > MIN_MIX_MARGIN);
                assert!((advantage - mixed.estimated_advantage).abs() < 1e-12);
                accepted += 1;
            }
        }
        assert!(accepted > 20, "only {accepted} accepted mixtures");
    }

    #[test]
    fn the_matrix_solver_resolves_small_game_value_gains() {
        let earthquake = 0.91461181640625;
        let destiny_bond = earthquake / 7.;
        let values = [
            1. - earthquake,
            1.,
            1.,
            1. - destiny_bond,
            1. - earthquake,
            1. - earthquake,
        ];
        let types = [typed(&values, 2)];
        let iterations = types[0].samples;
        let evidence = analyze_bayesian_root(&[true; 3], iterations, &types).unwrap();
        assert!(evidence.estimated_gain() > 0.016 && evidence.estimated_gain() < 0.017);
        assert!(evidence.margin() > 1e-8 && evidence.margin() < 0.02);
        assert!((evidence.policy[0] - 0.125).abs() < 1e-10);
        assert!((evidence.policy[1] - 0.875).abs() < 1e-10);
        assert_eq!(evidence.policy[2], 0.);
        let uncertain = [RootType {
            samples: 6,
            replies: 2,
            cells: cells(&values, 2, 1),
        }];
        assert!(bayesian_mixed_root(&[true; 3], 6, &uncertain).is_none());
    }

    #[test]
    fn a_positive_search_margin_does_not_certify_continuation_value() {
        let samples = [81855, 81854, 81854, 454729, 81854, 81854];
        let means = [
            0.08785046728971962,
            0.9985706257482835,
            1.,
            0.8442742820449103,
            0.08280597160798495,
            0.08400322525472183,
        ];
        let types = [RootType {
            samples: 864000,
            replies: 2,
            cells: samples
                .into_iter()
                .zip(means)
                .enumerate()
                .map(|(i, (samples, mean))| RootCell {
                    own: i / 2,
                    reply: i % 2,
                    samples,
                    mean,
                })
                .collect(),
        }];
        let analysis = analyze_bayesian_root(&[true; 3], 864000, &types).unwrap();
        assert!((analysis.margin() - 0.013157734486514139).abs() < 1e-10);
        let earthquake = 0.91461181640625;
        let oracle = [TypeMatrix {
            weight: 1.,
            replies: 2,
            values: vec![
                1. - earthquake,
                1.,
                1.,
                1. - earthquake / 7.,
                1. - earthquake,
                1. - earthquake,
            ],
        }];
        assert!(guarantee(&oracle, &analysis.policy) + 0.0028 < best_pure(&oracle, 3));
        assert!(bayesian_mixed_root(&[true; 3], 864000, &types).is_none());
    }

    #[test]
    fn bayesian_equilibrium_filters_exploration_blunders_and_mixes_on_cyclic_payoffs() {
        // Action 0 has a higher raw average only because of opponent exploration blunders
        // in column 1 (`0.95`), whereas against the opponent's dominant reply (column 0,
        // 1800 visits) Action 1 (`0.55`) strictly beats Action 0 (`0.25`).
        let pure_type = [RootType {
            samples: 2000,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 900, mean: 0.25 },
                RootCell { own: 0, reply: 1, samples: 100, mean: 0.95 },
                RootCell { own: 1, reply: 0, samples: 900, mean: 0.55 },
                RootCell { own: 1, reply: 1, samples: 100, mean: 0.55 },
            ],
        }];
        let eq = solve_bayesian_equilibrium(
            &[true, true],
            &[1000, 1000],
            &[0.32, 0.55],
            &pure_type,
        )
        .unwrap();
        assert_eq!(eq.best_pure_action, 1);
        assert_eq!(eq.policy, vec![0.0, 1.0]);

        // Matching-pennies world where both columns are actively played: solver returns 50/50 mix.
        let cyclic_type = [RootType {
            samples: 2000,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 500, mean: 0.80 },
                RootCell { own: 0, reply: 1, samples: 500, mean: 0.20 },
                RootCell { own: 1, reply: 0, samples: 500, mean: 0.20 },
                RootCell { own: 1, reply: 1, samples: 500, mean: 0.80 },
            ],
        }];
        let mix_eq = solve_bayesian_equilibrium(
            &[true, true],
            &[1000, 1000],
            &[0.50, 0.50],
            &cyclic_type,
        )
        .unwrap();
        assert!((mix_eq.policy[0] - 0.5).abs() < 1e-4);
        assert!((mix_eq.policy[1] - 0.5).abs() < 1e-4);
        assert!(mix_eq.equilibrium_value > mix_eq.best_pure_value + 0.15);
    }
}
