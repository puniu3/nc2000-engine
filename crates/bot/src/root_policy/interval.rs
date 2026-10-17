//! Root policy improvement from externally justified payoff intervals.
//!
//! Unlike `RootCell` sample means, these inputs must enclose actual payoffs.
//! Rows are the SAME own actions in every type; each type includes ALL its
//! legal opponent replies. Missing cells must explicitly be [0, 1].
//! Opponents may know their type, but cannot see our private random draw.
//!
//! For hidden types, continuation payoffs must come from an admissible shared
//! information-set strategy. Independently optimal clairvoyant continuations
//! are not valid lower bounds for a blind player. This API does not manufacture
//! that missing evidence or infer type weights from adaptive sample counts.
//! Even with a known type, optimal-continuation payoffs only certify play that
//! follows that continuation. To improve an existing policy end to end, use
//! its fixed-continuation payoffs, or supply a controller realizing the bounds.
//!
//! An accepted replacement improves the worst-response guarantee, not payoff
//! against every fixed opponent. Unproved cases retain the complete baseline
//! distribution. No probability threshold removes rare but necessary actions.

use super::{best_pure, guarantee, solve_bayesian, TypeMatrix};

const PAD: f64 = 2e-8;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct PayoffBounds {
    pub replies: usize,
    /// Row-major, own-action perspective, normalized to [0, 1].
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum InformationModel {
    /// Externally specified probabilities; best response is taken WITHIN
    /// each private type before averaging. Weights must sum to one.
    KnownPrior(Vec<f64>),
    /// No distributional assumption: guarantee against every supplied type.
    WorstCase,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct IntervalAnalysis {
    pub candidate: Vec<f64>,
    pub candidate_value: [f64; 2],
    pub baseline_value: [f64; 2],
    /// Upper bound on the best pure guarantee among ELIGIBLE own actions.
    pub best_pure_upper: f64,
    pub improvement_lower: f64,
    /// Necessity of mixing is only relative to that same eligible action set.
    pub mixing_advantage_lower: f64,
    pub replace: bool,
    pub selected: Vec<f64>,
}

fn matrices(
    eligible: &[bool],
    types: &[PayoffBounds],
    model: &InformationModel,
) -> Result<(Vec<usize>, Vec<TypeMatrix>, Vec<TypeMatrix>), &'static str> {
    let actions: Vec<_> = (0..eligible.len()).filter(|&i| eligible[i]).collect();
    if actions.is_empty() || types.is_empty() {
        return Err("empty eligible actions or type family");
    }
    for t in types {
        if t.replies == 0
            || eligible.len().checked_mul(t.replies) != Some(t.lower.len())
            || t.lower.len() != t.upper.len()
            || t.lower.iter().zip(&t.upper).any(|(&l, &u)| {
                !l.is_finite() || !u.is_finite() || !(0.0..=u).contains(&l) || u > 1.
            })
        {
            return Err("incomplete or invalid payoff bounds");
        }
    }
    let build = |upper: bool| -> Result<Vec<TypeMatrix>, &'static str> {
        let values = |t: &PayoffBounds| {
            let raw = if upper { &t.upper } else { &t.lower };
            actions
                .iter()
                .flat_map(|&i| raw[i * t.replies..(i + 1) * t.replies].iter().copied())
                .collect()
        };
        match model {
            InformationModel::KnownPrior(weights) => {
                if weights.len() != types.len()
                    || weights.iter().any(|w| !w.is_finite() || *w < 0.)
                    || (weights.iter().sum::<f64>() - 1.).abs() > 1e-12
                {
                    return Err("invalid external type probabilities");
                }
                Ok(types
                    .iter()
                    .zip(weights)
                    .filter(|(_, w)| **w > 0.)
                    .map(|(t, &weight)| TypeMatrix {
                        weight,
                        replies: t.replies,
                        values: values(t),
                    })
                    .collect())
            }
            InformationModel::WorstCase => {
                let replies: usize = types.iter().map(|t| t.replies).sum();
                let values = actions
                    .iter()
                    .flat_map(|&i| {
                        types.iter().flat_map(move |t| {
                            let raw = if upper { &t.upper } else { &t.lower };
                            raw[i * t.replies..(i + 1) * t.replies].iter().copied()
                        })
                    })
                    .collect();
                Ok(vec![TypeMatrix {
                    weight: 1.,
                    replies,
                    values,
                }])
            }
        }
    };
    let lower = build(false)?;
    let upper = build(true)?;
    Ok((actions, lower, upper))
}

/// Solve the lower game, then verify its actual returned distribution against
/// every reply. `required_gain` is an explicit caller objective, not a hidden
/// 2% mixing cutoff. Set it to zero to accept any numerically resolved gain.
/// Unresolved optimization or malformed evidence returns an error; the caller
/// must retain its baseline. Correctly formed but inconclusive bounds return
/// `replace=false`, with the baseline copied unchanged into `selected`.
pub fn analyze(
    eligible: &[bool],
    baseline: &[f64],
    types: &[PayoffBounds],
    model: &InformationModel,
    required_gain: f64,
) -> Result<IntervalAnalysis, &'static str> {
    if !required_gain.is_finite()
        || required_gain < 0.
        || baseline.len() != eligible.len()
        || baseline
            .iter()
            .zip(eligible)
            .any(|(&p, &ok)| !p.is_finite() || p < 0. || (!ok && p != 0.))
        || (baseline.iter().sum::<f64>() - 1.).abs() > 1e-12
    {
        return Err("invalid baseline policy or required gain");
    }
    let (actions, lower, upper) = matrices(eligible, types, model)?;
    let p = solve_bayesian(&lower, actions.len()).ok_or("matrix optimization unresolved")?;
    let base: Vec<_> = actions.iter().map(|&i| baseline[i]).collect();
    let value = |q: &[f64]| {
        [
            (guarantee(&lower, q) - PAD).max(0.),
            (guarantee(&upper, q) + PAD).min(1.),
        ]
    };
    let candidate_value = value(&p);
    let baseline_value = value(&base);
    let best_pure_upper = (best_pure(&upper, actions.len()) + PAD).min(1.);
    let improvement_lower = candidate_value[0] - baseline_value[1];
    let replace = improvement_lower > required_gain;
    let mut candidate = vec![0.; eligible.len()];
    for (&i, &mass) in actions.iter().zip(&p) {
        candidate[i] = mass;
    }
    Ok(IntervalAnalysis {
        selected: if replace {
            candidate.clone()
        } else {
            baseline.to_vec()
        },
        candidate,
        candidate_value,
        baseline_value,
        best_pure_upper,
        improvement_lower,
        mixing_advantage_lower: candidate_value[0] - best_pure_upper,
        replace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn exact(values: &[f64], replies: usize) -> PayoffBounds {
        PayoffBounds {
            replies,
            lower: values.to_vec(),
            upper: values.to_vec(),
        }
    }
    fn run(t: PayoffBounds, base: &[f64]) -> IntervalAnalysis {
        analyze(
            &vec![true; base.len()],
            base,
            &[t],
            &InformationModel::WorstCase,
            0.,
        )
        .unwrap()
    }
    fn g(a: &[f64], p: &[f64]) -> f64 {
        (0..2)
            .map(|j| p[0] * a[j] + p[1] * a[2 + j])
            .fold(1., f64::min)
    }

    #[test]
    fn resource_waiting_class_and_rare_support_are_preserved() {
        for k in (1..=100).chain([999, 9999]) {
            let a = 0.9;
            let v = a / (k + 1) as f64;
            let r = run(exact(&[a, 0., 0., a / k as f64], 2), &[0., 1.]);
            assert!(r.replace);
            assert!((r.candidate[0] - 1. / (k + 1) as f64).abs() < 1e-9);
            assert!((r.candidate_value[0] - (v - PAD)).abs() < 1e-9);
            assert!(r.mixing_advantage_lower > 0.);
        }
    }
    #[test]
    fn counter_reply_outside_the_two_by_two_support_blocks_false_improvement() {
        let r = run(exact(&[1., 0., 0., 0., 1., 0.], 3), &[0., 1.]);
        assert!(!r.replace);
        assert_eq!(r.selected, vec![0., 1.]);
    }
    #[test]
    fn pure_saddle_and_already_good_mixed_baselines_are_retained() {
        let r = run(exact(&[0.8, 0.9, 0.2, 0.3], 2), &[1., 0.]);
        assert!(!r.replace);
        let r = run(exact(&[0.8, 0.9, 0.2, 0.3], 2), &[0., 1.]);
        assert!(r.replace);
        assert_eq!(r.selected, vec![1., 0.]);
        let r = run(exact(&[1., 0., 0., 1.], 2), &[0.5, 0.5]);
        assert!(!r.replace);
        assert_eq!(r.selected, vec![0.5, 0.5]);
    }
    #[test]
    fn three_way_cycle_is_solved_without_a_two_action_heuristic() {
        let r = run(
            exact(&[0.5, 0., 1., 1., 0.5, 0., 0., 1., 0.5], 3),
            &[1., 0., 0.],
        );
        assert!(r.replace);
        for p in r.selected {
            assert!((p - 1. / 3.).abs() < 1e-9);
        }
    }
    #[test]
    fn hidden_type_responses_precede_averaging_and_unknown_prior_is_robust() {
        let types = [exact(&[0., 0.5, 0., 1.], 2), exact(&[1., 0., 0.5, 0.5], 2)];
        let r = analyze(
            &[true, true],
            &[1., 0.],
            &types,
            &InformationModel::KnownPrior(vec![0.5, 0.5]),
            0.,
        )
        .unwrap();
        assert!(r.replace);
        assert_eq!(r.selected, vec![0., 1.]);
        assert!((r.candidate_value[0] - (0.25 - PAD)).abs() < 1e-9);
        let r = analyze(
            &[true, true],
            &[1., 0.],
            &types,
            &InformationModel::WorstCase,
            0.,
        )
        .unwrap();
        assert!(!r.replace);
        assert_eq!(r.candidate_value[0], 0.);
    }
    #[test]
    fn unresolved_and_malformed_evidence_do_not_authorize_a_change() {
        let r = run(
            PayoffBounds {
                replies: 2,
                lower: vec![0.; 4],
                upper: vec![1.; 4],
            },
            &[0.3, 0.7],
        );
        assert!(!r.replace);
        assert_eq!(r.selected, vec![0.3, 0.7]);
        assert!(analyze(
            &[true, true],
            &[0., 1.],
            &[exact(&[1., 0., 0.], 2)],
            &InformationModel::WorstCase,
            0.
        )
        .is_err());
        assert!(analyze(
            &[true, true],
            &[0., 1.],
            &[exact(&[1., 0., 0., 1.], 2)],
            &InformationModel::KnownPrior(vec![0.5]),
            0.
        )
        .is_err());
        assert!(analyze(
            &[true, true],
            &[f64::NAN, 1.],
            &[exact(&[1., 0., 0., 1.], 2)],
            &InformationModel::WorstCase,
            0.
        )
        .is_err());
    }
    #[test]
    fn accepted_intervals_improve_every_corner_game() {
        let mut rng = crate::rng::SplitMix64::new(2026101701);
        let mut accepted = 0;
        for _ in 0..2000 {
            let lower: Vec<_> = (0..4).map(|_| rng.below(9) as f64 / 10.).collect();
            let upper: Vec<_> = lower.iter().map(|l| l + 0.1).collect();
            let r = run(
                PayoffBounds {
                    replies: 2,
                    lower: lower.clone(),
                    upper: upper.clone(),
                },
                &[0., 1.],
            );
            if r.replace {
                accepted += 1;
                for mask in 0..16 {
                    let a: Vec<_> = (0..4)
                        .map(|i| {
                            if mask & (1 << i) == 0 {
                                lower[i]
                            } else {
                                upper[i]
                            }
                        })
                        .collect();
                    assert!(g(&a, &r.selected) - g(&a, &[0., 1.]) + 1e-12 >= r.improvement_lower);
                }
            }
        }
        assert!(accepted > 200);
    }
    #[test]
    fn permutation_and_excluded_actions_do_not_change_the_guarantee() {
        let r = analyze(
            &[true, false, true],
            &[0., 0., 1.],
            &[exact(&[1., 0., 1., 1., 0., 0.1], 2)],
            &InformationModel::WorstCase,
            0.,
        )
        .unwrap();
        assert!(r.replace);
        assert_eq!(r.selected[1], 0.);
        let s = run(exact(&[0.1, 0., 0., 1.], 2), &[1., 0.]);
        assert!((r.candidate_value[0] - s.candidate_value[0]).abs() < 1e-10);
    }
}
