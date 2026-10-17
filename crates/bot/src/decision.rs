//! Execution policies, separate from search diagnostics and random draws.
//! A distribution is conditional on this search's evidence/seed, not a
//! certificate of equilibrium or of the whole agent's win probability.

use crate::rng::SplitMix64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolicyMode {
    #[default]
    Legacy,
    /// Empirical visit sampling is an ablation, not a Nash solver.
    VisitSample,
    /// Existing Bayesian root estimator, retaining its evidence gate.
    EstimatedMix,
}

impl PolicyMode {
    pub fn parse(value: &str) -> Result<Self, &'static str> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "visit-sample" => Ok(Self::VisitSample),
            "estimated-mix" => Ok(Self::EstimatedMix),
            _ => Err("unknown action policy mode"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PolicyAction<T> {
    pub input: T,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct DecisionPolicy<T> {
    source: &'static str,
    actions: Vec<PolicyAction<T>>,
}

impl<T: Clone + PartialEq> DecisionPolicy<T> {
    /// Validate, merge aliases and normalize. No support truncation: rare
    /// positive actions remain playable. The caller supplies legal actions.
    pub fn new(source: &'static str, entries: impl IntoIterator<Item = (T, f64)>) -> Result<Self, &'static str> {
        let mut actions: Vec<PolicyAction<T>> = Vec::new();
        for (input, probability) in entries {
            if !probability.is_finite() || probability < 0. { return Err("invalid policy mass"); }
            if probability == 0. { continue; }
            if let Some(row) = actions.iter_mut().find(|row| row.input == input) {
                row.probability += probability;
            } else { actions.push(PolicyAction { input, probability }); }
        }
        let total: f64 = actions.iter().map(|row| row.probability).sum();
        if !total.is_finite() || total <= 0. { return Err("empty policy mass"); }
        for row in &mut actions { row.probability /= total; }
        Ok(Self { source, actions })
    }

    pub fn pure(source: &'static str, action: T) -> Self {
        Self { source, actions: vec![PolicyAction { input: action, probability: 1. }] }
    }

    pub fn actions(&self) -> &[PolicyAction<T>] { &self.actions }
    pub fn source(&self) -> &'static str { self.source }

    pub fn map<U: Clone + PartialEq>(&self, f: impl Fn(&T) -> U) -> DecisionPolicy<U> {
        DecisionPolicy::new(self.source, self.actions.iter().map(|row| (f(&row.input), row.probability)))
            .expect("mapping a valid policy preserves mass")
    }

    pub fn sample(&self, draw: f64) -> T {
        assert!(draw.is_finite() && (0.0..1.0).contains(&draw));
        let mut cumulative = 0.;
        for row in &self.actions {
            cumulative += row.probability;
            if draw < cumulative { return row.input.clone(); }
        }
        self.actions.last().expect("nonempty policy").input.clone()
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CommittedDecision<T> {
    pub input: T,
    pub policy: DecisionPolicy<T>,
}

/// Sampling has its own domain-separated RNG. Read-only policy inspection
/// never draws; commit is idempotent until the next decision's reset.
pub struct DecisionSampler<T> {
    rng: SplitMix64,
    committed: Option<CommittedDecision<T>>,
}

impl<T: Clone + PartialEq> DecisionSampler<T> {
    pub fn new(seed: u64) -> Self {
        Self { rng: SplitMix64::new(seed ^ 0x6465_6369_7369_6f6e), committed: None }
    }
    pub fn reset(&mut self) { self.committed = None; }
    pub fn committed(&self) -> Option<&CommittedDecision<T>> { self.committed.as_ref() }
    pub fn commit(&mut self, policy: DecisionPolicy<T>) -> &CommittedDecision<T> {
        self.committed.get_or_insert_with(|| CommittedDecision {
            input: policy.sample(self.rng.next_f64()), policy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_preserves_rare_support_and_merges_aliases() {
        let p = DecisionPolicy::new("test", [("attack", 0.0001), ("wait", 0.5), ("wait", 0.4999)]).unwrap();
        assert_eq!(p.actions.len(), 2);
        assert_eq!(p.sample(0.), "attack");
        assert_eq!(p.sample(0.0001), "wait");
        assert_eq!(p.sample(1. - f64::EPSILON), "wait");
        let counts = (0..10000).filter(|i| p.sample((*i as f64 + 0.5) / 10000.) == "attack").count();
        assert_eq!(counts, 1);
        assert!(DecisionPolicy::new("bad", [(0, f64::NAN)]).is_err());
        assert!(DecisionPolicy::new("bad", [(0, -0.1)]).is_err());
        assert!(DecisionPolicy::new("bad", [(0, 0.)]).is_err());
        assert_eq!(p.map(|_| "same").actions[0].probability, 1.);
    }

    #[test]
    fn commits_once_and_next_decision_uses_fresh_draw() {
        let p = DecisionPolicy::new("test", [(0, 0.125), (1, 0.875)]).unwrap();
        let mut a = DecisionSampler::new(42);
        let mut b = DecisionSampler::new(42);
        let mut counts = [0; 2];
        for _ in 0..1000 {
            let expected = b.commit(p.clone()).clone();
            assert_eq!(a.commit(p.clone()), &expected);
            // Even a different later proposal cannot redraw a committed move.
            assert_eq!(a.commit(DecisionPolicy::pure("other", 9)), &expected);
            counts[expected.input] += 1;
            a.reset(); b.reset();
        }
        assert!(counts[0] > 80 && counts[0] < 180, "{counts:?}");
    }
}
