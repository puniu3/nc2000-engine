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

fn response_column_with_replies(types: &[TypeMatrix], policy: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut column = vec![0.; policy.len()];
    let mut replies = Vec::with_capacity(types.len());
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
        replies.push(reply);
        for (i, value) in column.iter_mut().enumerate() {
            *value += t.weight * t.values[i * t.replies + reply];
        }
    }
    (column, replies)
}

fn response_column(types: &[TypeMatrix], policy: &[f64]) -> Vec<f64> {
    response_column_with_replies(types, policy).0
}

fn guarantee(types: &[TypeMatrix], policy: &[f64]) -> f64 {
    response_column(types, policy)
        .iter()
        .zip(policy)
        .map(|(v, p)| v * p)
        .sum()
}

fn row_pure_guarantees(types: &[TypeMatrix], rows: usize) -> Vec<f64> {
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
        .collect()
}

fn best_pure(types: &[TypeMatrix], rows: usize) -> f64 {
    row_pure_guarantees(types, rows)
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max)
}

struct BayesianColumnSolution {
    x: Vec<f64>,
    y: Vec<f64>,
    columns: Vec<Vec<f64>>,
    col_replies: Vec<Vec<usize>>,
    used_cols: usize,
}

/// Shared column-generation LP solver for ex-interim Bayesian minimax games:
/// `\max_{p \in \Delta} \sum_k w_k \min_{j \in R_k} \sum_i p_i V_k(i, j)`.
fn solve_bayesian_columns(
    types: &[TypeMatrix],
    rows: usize,
    seed_pure: Option<usize>,
    require_exact: bool,
) -> Option<BayesianColumnSolution> {
    let uniform = vec![1.0 / rows as f64; rows];
    let mut columns = Vec::new();
    let mut col_replies = Vec::new();
    let mut best_x = if let Some(pure_idx) = seed_pure {
        let mut p = vec![0.0; rows];
        p[pure_idx] = 1.0;
        let (c0, r0) = response_column_with_replies(types, &p);
        columns.push(c0);
        col_replies.push(r0);
        let (u_col, u_rep) = response_column_with_replies(types, &uniform);
        if !columns.contains(&u_col) {
            columns.push(u_col);
            col_replies.push(u_rep);
        }
        p
    } else {
        let (u_col, u_rep) = response_column_with_replies(types, &uniform);
        columns.push(u_col);
        col_replies.push(u_rep);
        uniform
    };
    let mut best_y = vec![1.0; 1];
    let mut best_cols_len = 1usize;
    let mut any_valid = seed_pure.is_some();

    let max_iters = if require_exact { 128 } else { 64 };
    let gap_tol = if require_exact { 1e-8 } else { 1e-6 };
    let neg_tol = if require_exact { -1e-12 } else { -1e-9 };
    let sum_tol = if require_exact { 1e-8 } else { 1e-6 };
    let conv_tol = if require_exact { 1e-8 } else { 1e-7 };

    for _ in 0..max_iters {
        let c_len = columns.len();
        let matrix: Vec<f64> = (0..rows)
            .flat_map(|i| columns.iter().map(move |c| c[i]))
            .collect();
        let mut solution = solve_matrix_full(&matrix, rows, c_len);
        if !solution.gap.is_finite()
            || solution.gap > gap_tol
            || solution.x.iter().any(|p| !p.is_finite() || *p < neg_tol)
            || (solution.x.iter().sum::<f64>() - 1.0).abs() > sum_tol
        {
            if require_exact {
                return None;
            }
            break;
        }
        solution.x.iter_mut().for_each(|p| *p = p.max(0.0));
        let total = solution.x.iter().sum::<f64>();
        if total <= 0.0 {
            if require_exact {
                return None;
            }
            break;
        }
        solution.x.iter_mut().for_each(|p| *p /= total);
        best_x = solution.x.clone();
        best_y = solution.y.clone();
        best_cols_len = c_len;
        any_valid = true;

        let (response, reps) = response_column_with_replies(types, &solution.x);
        let lower: f64 = response.iter().zip(&solution.x).map(|(v, p)| v * p).sum();
        let upper = solution.value + solution.gap / 2.0;
        if upper - lower <= conv_tol {
            return Some(BayesianColumnSolution {
                x: best_x,
                y: best_y,
                columns,
                col_replies,
                used_cols: best_cols_len,
            });
        }
        if columns.contains(&response) {
            if require_exact {
                return None;
            }
            break;
        }
        columns.push(response);
        col_replies.push(reps);
    }

    if !require_exact && any_valid {
        Some(BayesianColumnSolution {
            x: best_x,
            y: best_y,
            columns,
            col_replies,
            used_cols: best_cols_len,
        })
    } else {
        None
    }
}

fn solve_bayesian(types: &[TypeMatrix], rows: usize) -> Option<Vec<f64>> {
    solve_bayesian_columns(types, rows, None, true).map(|s| s.x)
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

#[derive(Clone, Debug)]
pub struct BayesianEquilibrium {
    pub policy: Vec<f64>,
    pub action_values: Vec<f64>,
    pub pure_guarantees: Vec<f64>,
    pub equilibrium_value: f64,
    /// Conservative lower-confidence value of `policy` used to gate overriding
    /// UCB's top-visited choice at decision time.
    pub override_value: f64,
    pub best_pure_value: f64,
    pub best_pure_action: usize,
}

pub(crate) fn min_equilibrium_visits(max_visits: u32) -> u32 {
    (max_visits / 8).max(4).min(max_visits)
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
    let singleton_result = |only: usize| {
        let mut policy = vec![0.0; k_total];
        policy[only] = 1.0;
        let v = global_means[only];
        BayesianEquilibrium {
            policy,
            action_values: global_means.to_vec(),
            pure_guarantees: global_means.to_vec(),
            equilibrium_value: v,
            override_value: v,
            best_pure_value: v,
            best_pure_action: only,
        }
    };
    if all_elig.len() == 1 {
        return Some(singleton_result(all_elig[0]));
    }
    let max_v = all_elig
        .iter()
        .map(|&i| global_visits[i])
        .max()
        .unwrap_or(0);
    let min_own_v = min_equilibrium_visits(max_v);
    let mut actions: Vec<usize> = all_elig
        .iter()
        .copied()
        .filter(|&i| global_visits[i] >= min_own_v)
        .collect();
    if actions.len() == 1 {
        return Some(singleton_result(actions[0]));
    }
    if actions.is_empty() {
        actions = all_elig.clone();
    }
    let n = actions.len();

    let mut raw_worlds: Vec<(f64, usize, Vec<f64>, Vec<f64>)> = Vec::with_capacity(types.len());
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
        let mut lcb_values = vec![0.0f64; n * m_act];
        const KAPPA_ROW: f64 = 16.0;
        const KAPPA_COL: f64 = 8.0;
        const KAPPA_CELL: f64 = 64.0;
        const KAPPA_LCB: f64 = 8.0;
        for r in 0..n {
            let own = actions[r];
            let g_mean = global_means[own].clamp(0.0, 1.0);
            let mu_row = (row_w[r] + KAPPA_ROW * g_mean) / (f64::from(row_n[r]) + KAPPA_ROW);
            for (c_idx, &j) in active_cols.iter().enumerate() {
                let mu_col =
                    (col_w[j] + KAPPA_COL * world_mean) / (f64::from(col_n[j]) + KAPPA_COL);
                let col_delta = mu_col - world_mean;
                let additive_prior = (mu_row + col_delta).clamp(0.0, 1.0);
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

                // For overriding MCTS's visit winner, do not hide sparse negative cells
                // under heavy shrinkage and subtract a 1-sigma finite-sample bound.
                let lcb_est = (cw + KAPPA_LCB * mu_prior) / (cn + KAPPA_LCB);
                let eff_n = cn + 0.25 * p_n;
                let se = (lcb_est * (1.0 - lcb_est) / (eff_n + 1.0)
                    + 0.25 / (eff_n + 1.0).powi(2))
                .sqrt();
                lcb_values[r * m_act + c_idx] = (lcb_est - se).clamp(0.0, 1.0);
            }
        }
        let w_k = f64::from(world_n);
        total_weight += w_k;
        raw_worlds.push((w_k, m_act, values, lcb_values));
    }

    if raw_worlds.is_empty() || total_weight <= 0.0 {
        return None;
    }

    let mut type_matrices: Vec<TypeMatrix> = Vec::with_capacity(raw_worlds.len());
    let mut lcb_type_matrices: Vec<TypeMatrix> = Vec::with_capacity(raw_worlds.len());
    for (w_k, replies, values, lcb_values) in raw_worlds {
        let weight = w_k / total_weight;
        type_matrices.push(TypeMatrix {
            weight,
            replies,
            values,
        });
        lcb_type_matrices.push(TypeMatrix {
            weight,
            replies,
            values: lcb_values,
        });
    }

    let mut pure_guarantees = global_means.to_vec();
    for (r, (&own, pg)) in actions
        .iter()
        .zip(row_pure_guarantees(&type_matrices, n))
        .enumerate()
    {
        let _ = r;
        pure_guarantees[own] = pg;
    }
    let best_pure_sub = (0..n)
        .max_by(|&a, &b| {
            pure_guarantees[actions[a]]
                .total_cmp(&pure_guarantees[actions[b]])
                .then_with(|| global_visits[actions[a]].cmp(&global_visits[actions[b]]))
                .then_with(|| actions[b].cmp(&actions[a]))
        })
        .unwrap_or(0);
    let best_pure_action = actions[best_pure_sub];
    let best_pure_value = pure_guarantees[best_pure_action];

    let col_sol = solve_bayesian_columns(&type_matrices, n, Some(best_pure_sub), false)?;
    let BayesianColumnSolution {
        x: best_x,
        y: best_y,
        columns,
        col_replies: _,
        used_cols: best_cols_len,
    } = col_sol;

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
        let override_value = guarantee(&lcb_type_matrices, &purified_x);
        for (r, &own) in actions.iter().enumerate() {
            policy[own] = purified_x[r];
        }
        Some(BayesianEquilibrium {
            policy,
            action_values,
            pure_guarantees,
            equilibrium_value: eq_guarantee,
            override_value,
            best_pure_value,
            best_pure_action,
        })
    } else {
        policy[best_pure_action] = 1.0;
        let mut pure_policy = vec![0.0; n];
        pure_policy[best_pure_sub] = 1.0;
        let override_value = guarantee(&lcb_type_matrices, &pure_policy);
        Some(BayesianEquilibrium {
            policy,
            action_values,
            pure_guarantees,
            equilibrium_value: best_pure_value,
            override_value,
            best_pure_value,
            best_pure_action,
        })
    }
}

/// Extended Stage-2 Bayesian equilibrium result returning the exact per-type opponent equilibrium
/// distribution `y*(t, j)`, active columns, and unblocked search-guidance values for all eligible
/// own actions while keeping readout LCB gating strictly free of 1-ply prior pseudo-counts.
#[derive(Clone, Debug)]
pub struct Stage2BayesianEquilibrium {
    pub eq: BayesianEquilibrium,
    pub search_action_values: Vec<f64>,
    pub opp_eq_policy: Vec<Vec<f64>>,
    pub active_cols: Vec<Vec<usize>>,
    pub type_weights: Vec<f64>,
    /// Flat `k_total * types[t].replies` search-guidance cell estimates per input type `t`.
    pub search_v: Vec<Vec<f64>>,
}

/// Solve the `iter == 0` 1-ply Bayesian LP from read-only prior payoff matrices across canonical
/// root types. Returns `(p_0, opp_eq_policy_per_type, q_0)`.
pub fn solve_initial_bayesian_prior(
    eligible: &[bool],
    type_weights: &[f64],
    type_replies: &[usize],
    prior_cells: &[&[f64]],
) -> Option<(Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> {
    let k_total = eligible.len();
    let actions: Vec<usize> = (0..k_total).filter(|&i| eligible[i]).collect();
    if actions.is_empty() || type_weights.len() != type_replies.len() || type_weights.len() != prior_cells.len() {
        return None;
    }
    let n = actions.len();
    let mut tm_list: Vec<TypeMatrix> = Vec::new();
    let mut orig_map: Vec<usize> = Vec::new();
    let mut total_w = 0.0f64;
    for (t_idx, (&w, &m)) in type_weights.iter().zip(type_replies.iter()).enumerate() {
        if w <= 0.0 || m == 0 || prior_cells[t_idx].len() < k_total * m {
            continue;
        }
        total_w += w;
    }
    if total_w <= 0.0 {
        return None;
    }
    for (t_idx, (&w, &m)) in type_weights.iter().zip(type_replies.iter()).enumerate() {
        if w <= 0.0 || m == 0 || prior_cells[t_idx].len() < k_total * m {
            continue;
        }
        let mut vals = vec![0.5f64; n * m];
        for (r, &own) in actions.iter().enumerate() {
            for j in 0..m {
                vals[r * m + j] = prior_cells[t_idx][own * m + j].clamp(0.0, 1.0);
            }
        }
        tm_list.push(TypeMatrix {
            weight: w / total_w,
            replies: m,
            values: vals,
        });
        orig_map.push(t_idx);
    }
    if tm_list.is_empty() {
        return None;
    }
    let pg = row_pure_guarantees(&tm_list, n);
    let best_pure_sub = (0..n)
        .max_by(|&a, &b| pg[a].total_cmp(&pg[b]).then_with(|| actions[b].cmp(&actions[a])))
        .unwrap_or(0);
    let mut p0 = vec![0.0f64; k_total];
    let mut q0 = vec![0.5f64; k_total];
    let mut opp_pol: Vec<Vec<f64>> = type_replies.iter().map(|&m| vec![0.0; m]).collect();

    if n == 1 {
        p0[actions[0]] = 1.0;
        q0[actions[0]] = pg[0];
        for (tm_idx, tm) in tm_list.iter().enumerate() {
            let orig_t = orig_map[tm_idx];
            let best_j = (0..tm.replies)
                .min_by(|&a, &b| tm.values[a].total_cmp(&tm.values[b]).then_with(|| a.cmp(&b)))
                .unwrap_or(0);
            opp_pol[orig_t][best_j] = 1.0;
        }
        return Some((p0, opp_pol, q0));
    }

    let sol = solve_bayesian_columns(&tm_list, n, Some(best_pure_sub), false)?;
    for (r, &own) in actions.iter().enumerate() {
        p0[own] = sol.x[r];
    }
    let y_sum: f64 = sol.y.iter().take(sol.used_cols).map(|&y| y.max(0.0)).sum();
    if y_sum > 0.0 && sol.col_replies.len() >= sol.used_cols {
        for c in 0..sol.used_cols {
            let wc = sol.y[c].max(0.0) / y_sum;
            if wc <= 0.0 {
                continue;
            }
            for (tm_idx, &rep_j) in sol.col_replies[c].iter().enumerate() {
                let orig_t = orig_map[tm_idx];
                if rep_j < opp_pol[orig_t].len() {
                    opp_pol[orig_t][rep_j] += wc;
                }
            }
        }
    } else {
        let (_, reps) = response_column_with_replies(&tm_list, &sol.x);
        for (tm_idx, &rep_j) in reps.iter().enumerate() {
            let orig_t = orig_map[tm_idx];
            if rep_j < opp_pol[orig_t].len() {
                opp_pol[orig_t][rep_j] = 1.0;
            }
        }
    }
    for (r, &own) in actions.iter().enumerate() {
        let mut eq_v = 0.0f64;
        for (tm_idx, tm) in tm_list.iter().enumerate() {
            let orig_t = orig_map[tm_idx];
            let mut tv = 0.0f64;
            for j in 0..tm.replies {
                tv += opp_pol[orig_t][j] * tm.values[r * tm.replies + j];
            }
            eq_v += tm.weight * tv;
        }
        q0[own] = 0.5 * eq_v + 0.5 * pg[r];
    }
    Some((p0, opp_pol, q0))
}

/// Stage-2 Bayesian equilibrium solver with:
/// - Capped two-path LP row admission (`min_own_std <= 512` OR threat-covered rival `>= 32..128` with `cov >= 0.75`),
/// - Cross-type LCB effective-sample safety cap (`eff_n = cn + min(0.25 * p_n, cn + 8)`),
/// - Unblocked `opp_eq_policy` and `search_action_values` for all eligible actions (even on singleton LP),
/// - Strict isolation of optional 1-ply `prior_cells` to search-guidance (`search_v`) only.
pub fn solve_bayesian_equilibrium_stage2(
    eligible: &[bool],
    global_visits: &[u32],
    global_means: &[f64],
    types: &[RootType],
    pooled_priors: &[Vec<RootCell>],
    calibrated_priors: Option<&[&[f64]]>,
) -> Option<Stage2BayesianEquilibrium> {
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

    // Pass 1: Aggregate cell stats across ALL eligible rows per type so `active_cols` and
    // per-row active-column coverage `cov(i)` have zero circular dependency on row admission.
    struct TypeAgg {
        orig_t: usize,
        m: usize,
        world_n: u32,
        world_mean: f64,
        cell_n: Vec<u32>,
        cell_w: Vec<f64>,
        row_n: Vec<u32>,
        row_w: Vec<f64>,
        col_n: Vec<u32>,
        col_w: Vec<f64>,
        prior_n: Vec<u32>,
        prior_mean: Vec<f64>,
        active_cols: Vec<usize>,
    }

    let mut aggs: Vec<TypeAgg> = Vec::with_capacity(types.len());
    let mut total_all_weight = 0.0f64;
    for (t_idx, t) in types.iter().enumerate() {
        let m = t.replies;
        if t.samples == 0 || m == 0 || t.cells.is_empty() {
            continue;
        }
        let mut cell_n = vec![0u32; k_total * m];
        let mut cell_w = vec![0.0f64; k_total * m];
        let mut row_n = vec![0u32; k_total];
        let mut row_w = vec![0.0f64; k_total];
        let mut col_n = vec![0u32; m];
        let mut col_w = vec![0.0f64; m];
        let mut world_n = 0u32;
        let mut world_w = 0.0f64;

        for cell in &t.cells {
            if cell.own >= k_total
                || !eligible[cell.own]
                || cell.reply >= m
                || cell.samples == 0
                || !cell.mean.is_finite()
            {
                continue;
            }
            let s = cell.samples;
            let w = f64::from(s) * cell.mean.clamp(0.0, 1.0);
            cell_n[cell.own * m + cell.reply] += s;
            cell_w[cell.own * m + cell.reply] += w;
            row_n[cell.own] += s;
            row_w[cell.own] += w;
            col_n[cell.reply] += s;
            col_w[cell.reply] += w;
            world_n += s;
            world_w += w;
        }
        if world_n == 0 {
            continue;
        }
        let mut prior_n = vec![0u32; k_total * m];
        let mut prior_mean = vec![0.0f64; k_total * m];
        if let Some(priors) = pooled_priors.get(t_idx) {
            for cell in priors {
                if cell.own >= k_total
                    || !eligible[cell.own]
                    || cell.reply >= m
                    || cell.samples == 0
                    || !cell.mean.is_finite()
                {
                    continue;
                }
                prior_n[cell.own * m + cell.reply] = cell.samples;
                prior_mean[cell.own * m + cell.reply] = cell.mean.clamp(0.0, 1.0);
            }
        }
        let max_col_n = col_n.iter().copied().max().unwrap_or(0);
        let min_col_n = (world_n / (2 * m as u32))
            .max(max_col_n / 5)
            .clamp(4, 96)
            .min(max_col_n);
        let mut active_cols: Vec<usize> = (0..m).filter(|&j| col_n[j] >= min_col_n).collect();
        if active_cols.is_empty() {
            active_cols = (0..m).filter(|&j| col_n[j] > 0).collect();
        }
        if active_cols.is_empty() {
            continue;
        }
        total_all_weight += f64::from(world_n);
        aggs.push(TypeAgg {
            orig_t: t_idx,
            m,
            world_n,
            world_mean: world_w / f64::from(world_n),
            cell_n,
            cell_w,
            row_n,
            row_w,
            col_n,
            col_w,
            prior_n,
            prior_mean,
            active_cols,
        });
    }
    if aggs.is_empty() || total_all_weight <= 0.0 {
        return None;
    }

    // Pass 2: Safe two-path capped row admission (`min_own_std <= 512` OR threat-covered rival).
    let max_v = all_elig.iter().map(|&i| global_visits[i]).max().unwrap_or(0);
    let min_own_std = (max_v / 8).min(512).max(4).min(max_v);
    let min_own_threat = (max_v / 16).clamp(32, 128).min(max_v);

    let mut lp_actions: Vec<usize> = all_elig
        .iter()
        .copied()
        .filter(|&own| {
            let v = global_visits[own];
            if v >= min_own_std {
                return true;
            }
            if v < min_own_threat {
                return false;
            }
            let mut cov_w = 0.0f64;
            for agg in &aggs {
                let min_active = agg
                    .active_cols
                    .iter()
                    .map(|&j| agg.cell_n[own * agg.m + j])
                    .min()
                    .unwrap_or(0);
                if min_active >= 16 {
                    cov_w += f64::from(agg.world_n);
                }
            }
            (cov_w / total_all_weight) >= 0.75
        })
        .collect();
    if lp_actions.is_empty() {
        lp_actions = all_elig.clone();
    }

    // Pass 3: Build search-guidance cell matrices (ALL eligible × ALL replies, KAPPA_SEARCH = 12)
    // and strictly prior-free LP / LCB matrices (`lp_actions × active_cols`).
    const KAPPA_ROW: f64 = 16.0;
    const KAPPA_COL: f64 = 8.0;
    const KAPPA_CELL: f64 = 64.0;
    const KAPPA_LCB: f64 = 8.0;
    const KAPPA_SEARCH: f64 = 12.0;

    let n = lp_actions.len();
    let mut type_matrices: Vec<TypeMatrix> = Vec::with_capacity(aggs.len());
    let mut lcb_type_matrices: Vec<TypeMatrix> = Vec::with_capacity(aggs.len());
    let mut search_v: Vec<Vec<f64>> = types.iter().map(|t| vec![0.5; k_total * t.replies]).collect();
    let mut active_cols_out: Vec<Vec<usize>> = vec![Vec::new(); types.len()];
    let mut type_weights_out: Vec<f64> = vec![0.0; types.len()];

    for agg in &aggs {
        let weight = f64::from(agg.world_n) / total_all_weight;
        type_weights_out[agg.orig_t] = weight;
        active_cols_out[agg.orig_t] = agg.active_cols.clone();
        let m = agg.m;
        let m_act = agg.active_cols.len();
        let cal_prior = calibrated_priors
            .and_then(|cp| cp.get(agg.orig_t).copied())
            .filter(|sl| sl.len() >= k_total * m);

        // Search-guidance values over all eligible rows and all `0..m` replies.
        for &own in &all_elig {
            let g_mean = global_means[own].clamp(0.0, 1.0);
            let mu_row = (agg.row_w[own] + KAPPA_ROW * g_mean)
                / (f64::from(agg.row_n[own]) + KAPPA_ROW);
            for j in 0..m {
                let mu_col = (agg.col_w[j] + KAPPA_COL * agg.world_mean)
                    / (f64::from(agg.col_n[j]) + KAPPA_COL);
                let additive_prior = (mu_row + (mu_col - agg.world_mean)).clamp(0.0, 1.0);
                let p_n = f64::from(agg.prior_n[own * m + j]);
                let pooled_prior = if p_n > 0.0 {
                    let pv = agg.prior_mean[own * m + j].clamp(0.0, 1.0);
                    (p_n * pv + 8.0 * additive_prior) / (p_n + 8.0)
                } else {
                    additive_prior
                };
                let mu_search = if let Some(cp) = cal_prior {
                    let p_cal = cp[own * m + j].clamp(0.02, 0.98);
                    if p_n > 0.0 {
                        0.5 * pooled_prior + 0.5 * p_cal
                    } else {
                        p_cal
                    }
                } else {
                    pooled_prior
                };
                let cn = f64::from(agg.cell_n[own * m + j]);
                let cw = agg.cell_w[own * m + j];
                search_v[agg.orig_t][own * m + j] =
                    (cw + KAPPA_SEARCH * mu_search) / (cn + KAPPA_SEARCH);
            }
        }

        // Readout LP & LCB matrices (STRICTLY no 1-ply prior injection!).
        let mut values = vec![0.5f64; n * m_act];
        let mut lcb_values = vec![0.0f64; n * m_act];
        for (r, &own) in lp_actions.iter().enumerate() {
            let g_mean = global_means[own].clamp(0.0, 1.0);
            let mu_row = (agg.row_w[own] + KAPPA_ROW * g_mean)
                / (f64::from(agg.row_n[own]) + KAPPA_ROW);
            for (c_idx, &j) in agg.active_cols.iter().enumerate() {
                let mu_col = (agg.col_w[j] + KAPPA_COL * agg.world_mean)
                    / (f64::from(agg.col_n[j]) + KAPPA_COL);
                let additive_prior = (mu_row + (mu_col - agg.world_mean)).clamp(0.0, 1.0);
                let p_n = f64::from(agg.prior_n[own * m + j]);
                let mu_prior = if p_n > 0.0 {
                    let pv = agg.prior_mean[own * m + j].clamp(0.0, 1.0);
                    (p_n * pv + 8.0 * additive_prior) / (p_n + 8.0)
                } else {
                    additive_prior
                };
                let cn = f64::from(agg.cell_n[own * m + j]);
                let cw = agg.cell_w[own * m + j];
                values[r * m_act + c_idx] = (cw + KAPPA_CELL * mu_prior) / (cn + KAPPA_CELL);

                // Cross-type LCB safety cap: never let pooled prior fake high LCB confidence on cn == 0.
                let lcb_est = (cw + KAPPA_LCB * mu_prior) / (cn + KAPPA_LCB);
                let eff_n = cn + (0.25 * p_n).min(cn + 8.0);
                let se = (lcb_est * (1.0 - lcb_est) / (eff_n + 1.0)
                    + 0.25 / (eff_n + 1.0).powi(2))
                .sqrt();
                lcb_values[r * m_act + c_idx] = (lcb_est - se).clamp(0.0, 1.0);
            }
        }
        type_matrices.push(TypeMatrix { weight, replies: m_act, values });
        lcb_type_matrices.push(TypeMatrix { weight, replies: m_act, values: lcb_values });
    }

    let mut pure_guarantees = global_means.to_vec();
    for (&own, pg) in lp_actions.iter().zip(row_pure_guarantees(&type_matrices, n)) {
        pure_guarantees[own] = pg;
    }
    let best_pure_sub = (0..n)
        .max_by(|&a, &b| {
            pure_guarantees[lp_actions[a]]
                .total_cmp(&pure_guarantees[lp_actions[b]])
                .then_with(|| global_visits[lp_actions[a]].cmp(&global_visits[lp_actions[b]]))
                .then_with(|| lp_actions[b].cmp(&lp_actions[a]))
        })
        .unwrap_or(0);
    let best_pure_action = lp_actions[best_pure_sub];
    let best_pure_value = pure_guarantees[best_pure_action];

    let mut opp_eq_policy: Vec<Vec<f64>> = types.iter().map(|t| vec![0.0; t.replies]).collect();
    let eq = if n == 1 {
        let only = lp_actions[0];
        let mut policy = vec![0.0; k_total];
        policy[only] = 1.0;
        let mut pure_p = vec![1.0];
        let override_value = guarantee(&lcb_type_matrices, &pure_p);
        let _ = &mut pure_p;
        for (tm_idx, agg) in aggs.iter().enumerate() {
            let tm = &type_matrices[tm_idx];
            let best_c = (0..tm.replies)
                .min_by(|&a, &b| tm.values[a].total_cmp(&tm.values[b]).then_with(|| a.cmp(&b)))
                .unwrap_or(0);
            let orig_j = agg.active_cols[best_c];
            opp_eq_policy[agg.orig_t][orig_j] = 1.0;
        }
        BayesianEquilibrium {
            policy,
            action_values: pure_guarantees.clone(),
            pure_guarantees,
            equilibrium_value: best_pure_value,
            override_value,
            best_pure_value,
            best_pure_action: only,
        }
    } else {
        let col_sol = solve_bayesian_columns(&type_matrices, n, Some(best_pure_sub), false)?;
        let BayesianColumnSolution {
            x: best_x,
            y: best_y,
            columns,
            col_replies,
            used_cols: best_cols_len,
        } = col_sol;

        let mut action_values = pure_guarantees.clone();
        let y_sum: f64 = best_y.iter().take(best_cols_len).map(|&y| y.max(0.0)).sum();
        if best_y.len() >= best_cols_len && best_cols_len <= columns.len() && y_sum > 0.0 {
            for (r, &own) in lp_actions.iter().enumerate() {
                let eq_v: f64 = (0..best_cols_len)
                    .map(|c| (best_y[c].max(0.0) / y_sum) * columns[c][r])
                    .sum();
                action_values[own] = 0.5 * eq_v + 0.5 * pure_guarantees[own];
            }
            for c in 0..best_cols_len {
                let wc = best_y[c].max(0.0) / y_sum;
                if wc <= 0.0 {
                    continue;
                }
                for (tm_idx, &c_idx) in col_replies[c].iter().enumerate() {
                    let agg = &aggs[tm_idx];
                    let orig_j = agg.active_cols[c_idx];
                    opp_eq_policy[agg.orig_t][orig_j] += wc;
                }
            }
        } else {
            let (_, reps) = response_column_with_replies(&type_matrices, &best_x);
            for (tm_idx, &c_idx) in reps.iter().enumerate() {
                let agg = &aggs[tm_idx];
                let orig_j = agg.active_cols[c_idx];
                opp_eq_policy[agg.orig_t][orig_j] = 1.0;
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
            let override_value = guarantee(&lcb_type_matrices, &purified_x);
            for (r, &own) in lp_actions.iter().enumerate() {
                policy[own] = purified_x[r];
            }
            BayesianEquilibrium {
                policy,
                action_values,
                pure_guarantees,
                equilibrium_value: eq_guarantee,
                override_value,
                best_pure_value,
                best_pure_action,
            }
        } else {
            policy[best_pure_action] = 1.0;
            let mut pure_policy = vec![0.0; n];
            pure_policy[best_pure_sub] = 1.0;
            let override_value = guarantee(&lcb_type_matrices, &pure_policy);
            BayesianEquilibrium {
                policy,
                action_values,
                pure_guarantees,
                equilibrium_value: best_pure_value,
                override_value,
                best_pure_value,
                best_pure_action,
            }
        }
    };

    // Pass 4: Compute unblocked `search_action_values` for EVERY eligible action `own ∈ all_elig`!
    let mut search_action_values = global_means.to_vec();
    for &own in &all_elig {
        let mut eq_v = 0.0f64;
        let mut pg_v = 0.0f64;
        for agg in &aggs {
            let wt = type_weights_out[agg.orig_t];
            let m = agg.m;
            let sv = &search_v[agg.orig_t];
            let mut tv = 0.0f64;
            let mut tmin = f64::INFINITY;
            for &j in &agg.active_cols {
                let v = sv[own * m + j];
                tv += opp_eq_policy[agg.orig_t][j] * v;
                if v < tmin {
                    tmin = v;
                }
            }
            eq_v += wt * tv;
            pg_v += wt * tmin;
        }
        search_action_values[own] = 0.5 * eq_v + 0.5 * pg_v;
    }

    Some(Stage2BayesianEquilibrium {
        eq,
        search_action_values,
        opp_eq_policy,
        active_cols: active_cols_out,
        type_weights: type_weights_out,
        search_v,
    })
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

    fn single_type(samples: u32, replies: usize, cells: Vec<RootCell>) -> [RootType; 1] {
        [RootType {
            samples,
            replies,
            cells,
        }]
    }

    #[test]
    fn crossed_replies_require_mixing_but_pure_saddles_do_not() {
        let mix = bayesian_mixed_root(
            &[true, true],
            40000,
            &single_type(40000, 2, cells(&[0., 1., 1., 0.], 2, 10000)),
        )
        .unwrap();
        assert!((mix.policy[0] - 0.5).abs() < 1e-12);
        assert!((mix.policy[1] - 0.5).abs() < 1e-12);
        assert!(mix.estimated_advantage > 0.45);
        assert!(bayesian_mixed_root(
            &[true, true],
            40000,
            &single_type(40000, 2, cells(&[0.8, 0.9, 0.2, 0.3], 2, 10000)),
        )
        .is_none());
    }

    #[test]
    fn absent_replies_and_insufficient_samples_cannot_enable_mixing() {
        let values = cells(&[0., 1., 1., 0.], 2, 10000);
        assert!(bayesian_mixed_root(&[true, true], 39999, &single_type(40000, 2, values.clone())).is_none());
        assert!(bayesian_mixed_root(&[true, true], 0, &single_type(0, 2, values.clone())).is_none());
        assert!(bayesian_mixed_root(
            &[true, true],
            4,
            &single_type(4, 2, cells(&[0., 1., 1., 0.], 2, 1)),
        )
        .is_none());
        assert!(bayesian_mixed_root(
            &[true, true],
            40000,
            &single_type(40000, 1, values),
        )
        .is_none());
    }

    #[test]
    fn promising_but_uncertain_matrices_request_more_evidence() {
        let crossed = single_type(4, 2, cells(&[0., 1., 1., 0.], 2, 1));
        assert_eq!(
            estimated_bayesian_mix_gain(&[true, true], 4, &crossed),
            Some(0.5)
        );
        assert!(bayesian_mixed_root(&[true, true], 4, &crossed).is_none());
        assert_eq!(
            estimated_bayesian_mix_gain(&[true, true], 3, &crossed),
            None
        );
        assert_eq!(
            estimated_bayesian_mix_gain(
                &[true, true],
                4,
                &single_type(4, 2, cells(&[0.8, 0.9, 0.2, 0.3], 2, 1)),
            ),
            Some(0.)
        );
    }

    #[test]
    fn excluded_actions_receive_no_mass_even_when_they_dominate() {
        let t = single_type(60000, 2, cells(&[0., 1., 1., 0., 1., 1.], 2, 10000));
        let mix = bayesian_mixed_root(&[true, true, false], 60000, &t).unwrap();
        assert_eq!(mix.policy[2], 0.);
        assert!((mix.policy.iter().sum::<f64>() - 1.).abs() < 1e-12);
        assert!(bayesian_mixed_root(&[false, false, true], 60000, &t).is_none());
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
        let naive = bayesian_mixed_root(
            &[true, true],
            800000,
            &single_type(800000, 2, cells(&[0.5, 0.25, 0.25, 0.75], 2, 200000)),
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

    #[test]
    fn bayesian_equilibrium_conservative_shrinkage_tolerates_sparse_noise_and_avoids_genuine_trap() {
        // Under conservative Bayesian cell shrinkage (KAPPA_CELL=64.0, non-negative
        // col_blunder_bonus), a 3-sample noisy cell on a strong reply stays buffered toward
        // Action 1's empirical row mean rather than over-penalizing an exploratory action,
        // while a well-sampled genuine trap (500 samples at 0.20) is still decisively avoided.
        let sparse_noise = [RootType {
            samples: 8500,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 4000, mean: 0.76 },
                RootCell { own: 0, reply: 1, samples: 1600, mean: 0.72 },
                RootCell { own: 1, reply: 0, samples: 2897, mean: 0.80 },
                RootCell { own: 1, reply: 1, samples: 3, mean: 0.20 },
            ],
        }];
        let g0 = (4000.0 * 0.76 + 1600.0 * 0.72) / 5600.0;
        let g1 = (2897.0 * 0.80 + 3.0 * 0.20) / 2900.0;
        let eq = solve_bayesian_equilibrium(&[true, true], &[5600, 2900], &[g0, g1], &sparse_noise)
            .unwrap();
        assert_eq!(eq.best_pure_action, 1);

        // Genuine trap-avoidance: Action 0 has 5,000 visits and higher global mean,
        // but collapses on reply 1 (500 samples at 0.20), while Action 1 genuinely dominates
        // the worst-case across all columns (0.74 vs 0.72).
        let genuine_trap = [RootType {
            samples: 6000,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 4500, mean: 0.85 },
                RootCell { own: 0, reply: 1, samples: 500, mean: 0.20 },
                RootCell { own: 1, reply: 0, samples: 500, mean: 0.74 },
                RootCell { own: 1, reply: 1, samples: 500, mean: 0.72 },
            ],
        }];
        let trap_eq = solve_bayesian_equilibrium(
            &[true, true],
            &[5000, 1000],
            &[0.785, 0.73],
            &genuine_trap,
        )
        .unwrap();
        assert_eq!(trap_eq.best_pure_action, 1);
        assert_eq!(trap_eq.policy, vec![0.0, 1.0]);
        assert!(trap_eq.equilibrium_value > trap_eq.pure_guarantees[0] + MIN_MIX_MARGIN);
    }

    #[test]
    fn bayesian_equilibrium_rejects_sparse_low_visit_override_while_preserving_well_sampled_mix() {
        // Reproduces 3809 T13 structure: primary action has 3909 visits (mean 0.558),
        // while rejected switch only has 151 visits (< max_v / 8 = 488) and loses badly
        // against reply 1 (n=17, mean=0.290) and reply 2 (n=9, mean=0.467).
        let sparse_switch = [RootType {
            samples: 4060,
            replies: 3,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 2900, mean: 0.550 },
                RootCell { own: 0, reply: 1, samples: 600, mean: 0.580 },
                RootCell { own: 0, reply: 2, samples: 409, mean: 0.582 },
                RootCell { own: 1, reply: 0, samples: 119, mean: 0.617 },
                RootCell { own: 1, reply: 1, samples: 17, mean: 0.290 },
                RootCell { own: 1, reply: 2, samples: 9, mean: 0.467 },
            ],
        }];
        let eq = solve_bayesian_equilibrium(
            &[true, true],
            &[3909, 151],
            &[0.558, 0.562],
            &sparse_switch,
        )
        .unwrap();
        assert_eq!(eq.best_pure_action, 0);
        assert_eq!(eq.policy, vec![1.0, 0.0]);
        assert!(eq.override_value <= eq.pure_guarantees[0] + MIN_MIX_MARGIN);

        // Borderline under-sampled alternative (just above max_v/8) with few reply samples
        // must also fail the LCB override gate when worst-case evidence is weak.
        let weak_lcb = [RootType {
            samples: 4600,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 2000, mean: 0.56 },
                RootCell { own: 0, reply: 1, samples: 2000, mean: 0.56 },
                RootCell { own: 1, reply: 0, samples: 585, mean: 0.60 },
                RootCell { own: 1, reply: 1, samples: 15, mean: 0.54 },
            ],
        }];
        let weak_eq = solve_bayesian_equilibrium(
            &[true, true],
            &[4000, 600],
            &[0.56, 0.598],
            &weak_lcb,
        )
        .unwrap();
        assert!(
            weak_eq.override_value <= weak_eq.pure_guarantees[0] + MIN_MIX_MARGIN,
            "weakly sampled alternative must not pass LCB override gate: override={} pure0={}",
            weak_eq.override_value,
            weak_eq.pure_guarantees[0]
        );

        // Well-sampled genuine 2x2 RPS (3728 T17 structure): both actions exceed max_v / 8
        // and both replies are heavily sampled, so override_value clears MIN_MIX_MARGIN.
        let genuine_rps = [RootType {
            samples: 7848,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 3500, mean: 0.536 },
                RootCell { own: 0, reply: 1, samples: 1730, mean: 0.735 },
                RootCell { own: 1, reply: 0, samples: 1600, mean: 0.609 },
                RootCell { own: 1, reply: 1, samples: 1018, mean: 0.523 },
            ],
        }];
        let rps_eq = solve_bayesian_equilibrium(
            &[true, true],
            &[5230, 2618],
            &[0.602, 0.576],
            &genuine_rps,
        )
        .unwrap();
        assert!(rps_eq.policy[0] > 0.2 && rps_eq.policy[1] > 0.2);
        assert!(
            rps_eq.override_value > rps_eq.pure_guarantees[0] + MIN_MIX_MARGIN,
            "genuine RPS mix should pass LCB override gate: override={} pure0={}",
            rps_eq.override_value,
            rps_eq.pure_guarantees[0]
        );
    }

    #[test]
    fn solve_bayesian_equilibrium_flat_tie_prefers_lower_index() {
        let eq = solve_bayesian_equilibrium(
            &[true, true, true],
            &[100, 100, 100],
            &[1.0, 1.0, 1.0],
            &[RootType {
                samples: 300,
                replies: 2,
                cells: cells(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2, 50),
            }],
        )
        .unwrap();
        assert_eq!(eq.best_pure_action, 0);
        assert_eq!(eq.policy, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn stage2_bayesian_equilibrium_row_admission_lcb_cap_and_prior_isolation() {
        // 1. High-budget Turn-35 admission: max_v = 700_000 -> min_own_std capped at 512 (not 87_500!).
        //    Rival action 1 has 600 visits and forms a clean RPS equilibrium with action 0.
        let high_budget = [RootType {
            samples: 700_600,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 350_000, mean: 0.45 },
                RootCell { own: 0, reply: 1, samples: 350_000, mean: 0.75 },
                RootCell { own: 1, reply: 0, samples: 300, mean: 0.75 },
                RootCell { own: 1, reply: 1, samples: 300, mean: 0.45 },
            ],
        }];
        let s2_hi = solve_bayesian_equilibrium_stage2(
            &[true, true],
            &[700_000, 600],
            &[0.60, 0.60],
            &high_budget,
            &[],
            None,
        )
        .unwrap();
        assert!(
            s2_hi.eq.policy[0] > 0.3 && s2_hi.eq.policy[1] > 0.3,
            "capped min_own_std <= 512 must admit 600-visit RPS rival at 700k: {:?}",
            s2_hi.eq.policy
        );
        assert!(
            (s2_hi.opp_eq_policy[0].iter().sum::<f64>() - 1.0).abs() < 1e-9
                && s2_hi.opp_eq_policy[0][0] > 0.3
                && s2_hi.opp_eq_policy[0][1] > 0.3,
            "opp_eq_policy must capture mixed opponent threats: {:?}",
            s2_hi.opp_eq_policy[0]
        );

        // 2. 3809 T13 rejection: max_v = 3909, rival has 151 visits but reply 2 has only 9 (< 16)
        //    samples, so cov < 0.75 -> excluded from LP readout, singleton LP still populates
        //    opp_eq_policy and unblocked search_action_values for BOTH actions.
        let sparse_switch = [RootType {
            samples: 4060,
            replies: 3,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 2900, mean: 0.550 },
                RootCell { own: 0, reply: 1, samples: 600, mean: 0.580 },
                RootCell { own: 0, reply: 2, samples: 409, mean: 0.582 },
                RootCell { own: 1, reply: 0, samples: 119, mean: 0.617 },
                RootCell { own: 1, reply: 1, samples: 17, mean: 0.290 },
                RootCell { own: 1, reply: 2, samples: 9, mean: 0.467 },
            ],
        }];
        let s2_t13 = solve_bayesian_equilibrium_stage2(
            &[true, true],
            &[3909, 151],
            &[0.558, 0.562],
            &sparse_switch,
            &[],
            None,
        )
        .unwrap();
        assert_eq!(s2_t13.eq.policy, vec![1.0, 0.0]);
        assert!((s2_t13.opp_eq_policy[0].iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(
            s2_t13.search_action_values[1] < 0.55,
            "unblocked search_action_values must reflect dropped rival's reply-1 trap: {:?}",
            s2_t13.search_action_values
        );

        // 3. Path-2 threat-covered rival admission: max_v = 10_000 (min_own_std = 512,
        //    min_own_threat = 128), rival has 160 visits with >= 16 samples on every active col.
        let threat_covered = [RootType {
            samples: 10_160,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 5_000, mean: 0.40 },
                RootCell { own: 0, reply: 1, samples: 5_000, mean: 0.75 },
                RootCell { own: 1, reply: 0, samples: 80, mean: 0.72 },
                RootCell { own: 1, reply: 1, samples: 80, mean: 0.42 },
            ],
        }];
        let s2_cov = solve_bayesian_equilibrium_stage2(
            &[true, true],
            &[10_000, 160],
            &[0.575, 0.570],
            &threat_covered,
            &[],
            None,
        )
        .unwrap();
        assert!(
            s2_cov.eq.policy[1] > 0.25,
            "threat-covered rival with cov=1.0 and 160 >= 128 visits must enter LP: {:?}",
            s2_cov.eq.policy
        );

        // 4. Cross-type LCB safety cap: cn == 0 with huge pooled prior p_n == 2000 must still
        //    incur full finite-sample uncertainty (eff_n <= 8.0) and fail override gate.
        let pooled_spoof = [RootType {
            samples: 4600,
            replies: 2,
            cells: vec![
                RootCell { own: 0, reply: 0, samples: 2000, mean: 0.55 },
                RootCell { own: 0, reply: 1, samples: 2000, mean: 0.55 },
                RootCell { own: 1, reply: 0, samples: 600, mean: 0.61 },
            ],
        }];
        let huge_prior = [vec![RootCell {
            own: 1,
            reply: 1,
            samples: 2000,
            mean: 0.62,
        }]];
        let s2_lcb = solve_bayesian_equilibrium_stage2(
            &[true, true],
            &[4000, 600],
            &[0.55, 0.61],
            &pooled_spoof,
            &huge_prior,
            None,
        )
        .unwrap();
        assert!(
            s2_lcb.eq.override_value <= s2_lcb.eq.pure_guarantees[0] + MIN_MIX_MARGIN,
            "huge pooled prior on cn=0 must not spoof LCB override: override={} pure0={}",
            s2_lcb.eq.override_value,
            s2_lcb.eq.pure_guarantees[0]
        );

        // 5. Strict 1-ply calibrated_priors isolation: readout `eq` is bit-identical with or
        //    without `calibrated_priors`, while `search_v` and `search_action_values` update.
        let cal_mat = [0.40, 0.75, 0.90, 0.90];
        let cal_slices: [&[f64]; 1] = [&cal_mat];
        let s2_with_cal = solve_bayesian_equilibrium_stage2(
            &[true, true],
            &[10_000, 160],
            &[0.575, 0.570],
            &threat_covered,
            &[],
            Some(&cal_slices),
        )
        .unwrap();
        assert_eq!(s2_cov.eq.policy, s2_with_cal.eq.policy);
        assert_eq!(s2_cov.eq.override_value, s2_with_cal.eq.override_value);
        assert_eq!(s2_cov.eq.pure_guarantees, s2_with_cal.eq.pure_guarantees);
        assert!(
            s2_with_cal.search_action_values[1] > s2_cov.search_action_values[1] + 0.01,
            "calibrated_priors must guide search_action_values without touching readout LP"
        );
    }
}
