use crate::exact::solve_matrix_full;

pub struct RootCell {
    pub own: usize,
    pub reply: usize,
    pub samples: u32,
    pub mean: f64,
}

pub struct MixedRoot {
    pub policy: Vec<f64>,
    pub estimated_advantage: f64,
}

// Search continuations are nonstationary; these margins do not certify game values.
pub fn mixed_root(
    eligible: &[bool],
    iterations: u32,
    availability: &[u32],
    cells: &[RootCell],
) -> Option<MixedRoot> {
    let actions: Vec<usize> = (0..eligible.len()).filter(|&i| eligible[i]).collect();
    let n = actions.len();
    let m = availability.len();
    if iterations == 0 || n < 2 || m < 2 || availability.iter().any(|&a| a != iterations) {
        return None;
    }
    let mut lower = vec![0.; n * m];
    let mut upper = vec![1.; n * m];
    let mut means = vec![0.5; n * m];
    for (i, &own) in actions.iter().enumerate() {
        for j in 0..m {
            if let Some(cell) = cells.iter().find(|c| c.own == own && c.reply == j) {
                if cell.samples == 0 || !(0.0..=1.0).contains(&cell.mean) {
                    return None;
                }
                let radius =
                    ((200.0 * (n * m) as f64).ln() / (2. * f64::from(cell.samples))).sqrt();
                lower[i * m + j] = (cell.mean - radius).max(0.);
                upper[i * m + j] = (cell.mean + radius).min(1.);
                means[i * m + j] = cell.mean;
            }
        }
    }
    let pessimistic = solve_matrix_full(&lower, n, m);
    let guarantee = (0..m)
        .map(|j| {
            (0..n)
                .map(|i| pessimistic.x[i] * lower[i * m + j])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    let optimistic_pure = upper
        .chunks(m)
        .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
        .fold(f64::NEG_INFINITY, f64::max);
    let estimated_advantage = guarantee - optimistic_pure;
    if estimated_advantage <= 0.02 {
        return None;
    }
    let mixed = solve_matrix_full(&means, n, m);
    if mixed.gap > 1e-8 {
        return None;
    }
    let mut policy = vec![0.; eligible.len()];
    for (i, &a) in actions.iter().enumerate() {
        policy[a] = mixed.x[i];
    }
    Some(MixedRoot {
        policy,
        estimated_advantage,
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
}
