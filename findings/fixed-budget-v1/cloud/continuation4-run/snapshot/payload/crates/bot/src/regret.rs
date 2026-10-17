//! M17a regret-mining statistics.
//!
//! Search iterations inside one tree are adaptive and correlated.  The
//! uncertainty unit is therefore an independently seeded search, not an
//! iteration.  Discovery averages action values across search seeds;
//! confirmation compares candidate and played actions with paired seeds.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedRegret {
    pub mean: f64,
    pub ci95: Option<f64>,
    pub lower95: Option<f64>,
}

pub fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

/// Index of the highest seed-marginal action value.  Each inner vector is
/// one action's values across independent search seeds.
pub fn best_action(values: &[Vec<f64>]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, xs)| !xs.is_empty())
        .max_by(|(_, a), (_, b)| mean(a).total_cmp(&mean(b)))
        .map(|(i, _)| i)
}

/// Paired candidate-minus-played regret.  A small-sample Student-t critical
/// value is used because confirmation normally uses only 4--16 search
/// seeds.  `lower95` is not clipped: negative values correctly mean the
/// candidate has not been confirmed better than the played action.
pub fn paired_regret(candidate: &[f64], played: &[f64]) -> PairedRegret {
    assert_eq!(
        candidate.len(),
        played.len(),
        "paired sample count mismatch"
    );
    let delta: Vec<f64> = candidate.iter().zip(played).map(|(a, b)| a - b).collect();
    let m = mean(&delta);
    if delta.len() < 2 {
        return PairedRegret {
            mean: m,
            ci95: None,
            lower95: None,
        };
    }
    let ss = delta.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
    let se = (ss / (delta.len() - 1) as f64 / delta.len() as f64).sqrt();
    let ci = t95(delta.len()) * se;
    PairedRegret {
        mean: m,
        ci95: Some(ci),
        lower95: Some(m - ci),
    }
}

/// Two-sided 95% Student-t critical value by sample count.  Exact values
/// through n=30; the normal limit is adequate above it.
fn t95(n: usize) -> f64 {
    const T: [f64; 30] = [
        0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060,
        2.056, 2.052, 2.048, 2.045,
    ];
    if n <= 1 {
        0.0
    } else if n <= 30 {
        T[n - 1]
    } else {
        1.96
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_action_uses_equal_weight_per_search_seed() {
        let values = vec![vec![0.4, 0.6], vec![0.7, 0.7], vec![]];
        assert_eq!(best_action(&values), Some(1));
    }

    #[test]
    fn paired_regret_uses_differences_not_independent_errors() {
        let r = paired_regret(&[0.7, 0.8, 0.9], &[0.5, 0.6, 0.7]);
        assert!((r.mean - 0.2).abs() < 1e-12);
        assert!(r.ci95.unwrap() < 1e-12);
        assert!((r.lower95.unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn one_pair_has_no_fake_confidence_interval() {
        let r = paired_regret(&[0.8], &[0.5]);
        assert!((r.mean - 0.3).abs() < 1e-12);
        assert_eq!(r.ci95, None);
        assert_eq!(r.lower95, None);
    }
}
