use nc2000_engine::{battle::SearchChoice, fxhash::FxHashMap, state::RequestState};

use crate::{rng::SplitMix64, smmcts::select_ucb_counts};

#[derive(Clone, PartialEq, Eq, Hash)]
struct Transition {
    parent: usize,
    own_action: Option<SearchChoice>,
    request: RequestState,
    log: Vec<String>,
}

pub struct HistoryStep {
    pub parent: usize,
    pub own_action: Option<SearchChoice>,
    pub request: RequestState,
    pub log: Vec<String>,
    pub depth: usize,
}

pub struct HistoryGroup {
    pub history: usize,
    pub actions: Vec<SearchChoice>,
    pub visits: Vec<u32>,
    pub rewards: Vec<f64>,
}

pub struct HistoryContinuation {
    side: usize,
    shared: bool,
    transitions: FxHashMap<Transition, usize>,
    histories: Vec<HistoryStep>,
    index: FxHashMap<(usize, Vec<SearchChoice>), usize>,
    groups: Vec<HistoryGroup>,
}

impl HistoryContinuation {
    pub(crate) fn new(side: usize, shared: bool) -> Self {
        assert!(side < 2);
        Self {
            side,
            shared,
            transitions: FxHashMap::default(),
            histories: Vec::new(),
            index: FxHashMap::default(),
            groups: Vec::new(),
        }
    }

    pub fn groups(&self) -> &[HistoryGroup] {
        &self.groups
    }
    pub fn histories(&self) -> &[HistoryStep] {
        &self.histories
    }
    pub(crate) fn side(&self) -> usize {
        self.side
    }
    pub(crate) fn shared(&self) -> bool {
        self.shared
    }

    /// Zero is the root; positive history IDs index `histories()[id - 1]`.
    pub(crate) fn advance(
        &mut self,
        parent: usize,
        own_action: Option<SearchChoice>,
        request: RequestState,
        log: &[String],
    ) -> usize {
        assert!(parent <= self.histories.len());
        let transition = Transition {
            parent,
            own_action,
            request,
            log: log.to_vec(),
        };
        if let Some(&id) = self.transitions.get(&transition) {
            return id;
        }
        let depth = if parent == 0 {
            1
        } else {
            self.histories[parent - 1].depth + 1
        };
        self.histories.push(HistoryStep {
            parent,
            own_action,
            request,
            log: transition.log.clone(),
            depth,
        });
        let id = self.histories.len();
        self.transitions.insert(transition, id);
        id
    }

    pub(crate) fn group(&mut self, history: usize, actions: &[SearchChoice]) -> usize {
        assert!(history > 0 && history <= self.histories.len());
        let key = (history, actions.to_vec());
        if let Some(&id) = self.index.get(&key) {
            return id;
        }
        let id = self.groups.len();
        self.groups.push(HistoryGroup {
            history,
            actions: key.1.clone(),
            visits: vec![0; actions.len()],
            rewards: vec![0.; actions.len()],
        });
        self.index.insert(key, id);
        id
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
    fn prefixes_preserve_earlier_choices_requests_and_empty_boundaries() {
        let mut p = HistoryContinuation::new(0, true);
        let a = [SearchChoice::Switch(2), SearchChoice::Switch(3)];
        let log = vec!["|cant|p1a: Snorlax|par".to_owned()];
        let left = p.advance(0, Some(a[0]), RequestState::Move, &log);
        assert_eq!(left, p.advance(0, Some(a[0]), RequestState::Move, &log));
        let right = p.advance(0, Some(a[1]), RequestState::Move, &log);
        let left_next = p.advance(left, None, RequestState::Move, &[]);
        let right_next = p.advance(right, None, RequestState::Move, &[]);
        assert_ne!(p.group(left_next, &a), p.group(right_next, &a));
        assert_ne!(left_next, p.advance(left, None, RequestState::Switch, &[]));
        assert_ne!(
            left_next,
            p.advance(left_next, None, RequestState::Move, &[])
        );
        assert_eq!(p.histories[left_next - 1].depth, 2);
        assert_ne!(p.group(left_next, &a), p.group(left_next, &[a[1], a[0]]));
    }
}
