use nc2000_engine::{battle::SearchChoice, fxhash::FxHashMap};

use crate::{rng::SplitMix64, smmcts::select_ucb_counts};

#[derive(Clone, PartialEq, Eq, Hash)]
struct History {
    own_action: Option<SearchChoice>,
    log: Vec<String>,
    actions: Vec<SearchChoice>,
}

pub struct ContinuationGroup {
    pub own_action: Option<SearchChoice>,
    pub log: Vec<String>,
    pub actions: Vec<SearchChoice>,
    pub visits: Vec<u32>,
    pub rewards: Vec<f64>,
}

pub struct FirstContinuation {
    side: usize,
    shared: bool,
    index: FxHashMap<History, usize>,
    groups: Vec<ContinuationGroup>,
}

impl FirstContinuation {
    pub(crate) fn new(side: usize, shared: bool) -> Self {
        assert!(side < 2);
        Self {
            side,
            shared,
            index: FxHashMap::default(),
            groups: Vec::new(),
        }
    }

    pub fn groups(&self) -> &[ContinuationGroup] {
        &self.groups
    }
    pub(crate) fn side(&self) -> usize {
        self.side
    }
    pub(crate) fn shared(&self) -> bool {
        self.shared
    }

    pub(crate) fn group(
        &mut self,
        own_action: Option<SearchChoice>,
        log: &[String],
        actions: &[SearchChoice],
    ) -> usize {
        let history = History {
            own_action,
            log: log.to_vec(),
            actions: actions.to_vec(),
        };
        if let Some(&i) = self.index.get(&history) {
            return i;
        }
        let i = self.groups.len();
        self.groups.push(ContinuationGroup {
            own_action,
            log: history.log.clone(),
            actions: history.actions.clone(),
            visits: vec![0; actions.len()],
            rewards: vec![0.; actions.len()],
        });
        self.index.insert(history, i);
        i
    }

    pub(crate) fn select(&mut self, group: usize, c: f64, rng: &mut SplitMix64) -> usize {
        let stats = &mut self.groups[group];
        select_ucb_counts(c, rng, &mut stats.visits, &stats.rewards)
    }

    pub(crate) fn record_selection(&mut self, group: usize, action: usize) {
        self.groups[group].visits[action] += 1;
    }

    pub(crate) fn observe(&mut self, group: usize, action: usize, reward0: f64) {
        self.groups[group].rewards[action] += if self.side == 0 {
            reward0
        } else {
            1. - reward0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histories_preserve_own_action_observations_and_legal_order() {
        let mut p = FirstContinuation::new(0, true);
        let actions = [SearchChoice::Switch(2), SearchChoice::Switch(3)];
        let log = vec!["|turn|5".to_owned()];
        let first = p.group(Some(actions[0]), &log, &actions);
        assert_eq!(first, p.group(Some(actions[0]), &log, &actions));
        assert_ne!(first, p.group(Some(actions[1]), &log, &actions));
        assert_ne!(
            first,
            p.group(Some(actions[0]), &["|turn|6".to_owned()], &actions)
        );
        assert_ne!(
            first,
            p.group(Some(actions[0]), &log, &[actions[1], actions[0]])
        );
    }

    #[test]
    fn shared_ucb_credits_each_selection_once_with_own_reward() {
        for side in 0..2 {
            let mut p = FirstContinuation::new(side, true);
            let group = p.group(
                None,
                &[],
                &[SearchChoice::Switch(2), SearchChoice::Switch(3)],
            );
            let mut rng = SplitMix64::new(123);
            for _ in 0..100 {
                let action = p.select(group, 0.4, &mut rng);
                p.observe(group, action, 0.25);
            }
            assert_eq!(p.groups[group].visits.iter().sum::<u32>(), 100);
            for (n, w) in p.groups[group].visits.iter().zip(&p.groups[group].rewards) {
                assert_eq!(*w, f64::from(*n) * if side == 0 { 0.25 } else { 0.75 });
            }
        }
    }
}
