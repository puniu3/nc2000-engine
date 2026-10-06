use nc2000_engine::{
    battle::SearchChoice,
    fxhash::{FxHashMap, FxHasher},
    state::RequestState,
};

use crate::{rng::SplitMix64, smmcts::select_ucb_counts};
use std::hash::{Hash, Hasher};

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

#[derive(Default)]
struct CompactIndex {
    transition_heads: FxHashMap<u64, usize>,
    transition_next: Vec<usize>,
    group_heads: Vec<Option<usize>>,
    group_next: Vec<Option<usize>>,
}

pub struct HistoryContinuation {
    side: usize,
    shared: bool,
    compact: Option<CompactIndex>,
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
            compact: None,
            transitions: FxHashMap::default(),
            histories: Vec::new(),
            index: FxHashMap::default(),
            groups: Vec::new(),
        }
    }

    pub(crate) fn use_compact_index(&mut self) {
        assert!(self.histories.is_empty() && self.groups.is_empty());
        self.compact = Some(CompactIndex::default());
    }

    pub fn stored_log_bytes(&self) -> usize {
        self.histories
            .iter()
            .flat_map(|h| &h.log)
            .map(|s| s.len())
            .sum::<usize>()
            + self
                .transitions
                .keys()
                .flat_map(|h| &h.log)
                .map(|s| s.len())
                .sum::<usize>()
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
        if self.compact.is_some() {
            let mut hash = FxHasher::default();
            (parent, own_action, request, log).hash(&mut hash);
            return self.advance_compact(hash.finish(), parent, own_action, request, log);
        }
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

    fn advance_compact(
        &mut self,
        hash: u64,
        parent: usize,
        own_action: Option<SearchChoice>,
        request: RequestState,
        log: &[String],
    ) -> usize {
        let compact = self.compact.as_mut().unwrap();
        let head = compact.transition_heads.get(&hash).copied().unwrap_or(0);
        let mut id = head;
        while id != 0 {
            let h = &self.histories[id - 1];
            if h.parent == parent
                && h.own_action == own_action
                && h.request == request
                && h.log == log
            {
                return id;
            }
            id = compact.transition_next[id - 1];
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
            log: log.to_vec(),
            depth,
        });
        let id = self.histories.len();
        compact.transition_next.push(head);
        compact.group_heads.push(None);
        compact.transition_heads.insert(hash, id);
        id
    }

    pub(crate) fn group(&mut self, history: usize, actions: &[SearchChoice]) -> usize {
        assert!(history > 0 && history <= self.histories.len());
        if let Some(compact) = self.compact.as_mut() {
            let head = compact.group_heads[history - 1];
            let mut cursor = head;
            while let Some(id) = cursor {
                if self.groups[id].actions == actions {
                    return id;
                }
                cursor = compact.group_next[id];
            }
            let id = self.groups.len();
            self.groups.push(HistoryGroup {
                history,
                actions: actions.to_vec(),
                visits: vec![0; actions.len()],
                rewards: vec![0.; actions.len()],
            });
            compact.group_next.push(head);
            compact.group_heads[history - 1] = Some(id);
            return id;
        }
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
    fn compact_hash_collisions_preserve_full_history_equality_and_action_order() {
        let mut p = HistoryContinuation::new(0, true);
        p.use_compact_index();
        let actions = [SearchChoice::Switch(2), SearchChoice::Switch(3)];
        let log = vec!["|turn|5".to_owned()];
        let cases = [
            (0, Some(actions[0]), RequestState::Move, log.clone()),
            (0, Some(actions[1]), RequestState::Move, log.clone()),
            (0, Some(actions[0]), RequestState::Switch, log.clone()),
            (0, Some(actions[0]), RequestState::Move, vec![]),
            (1, Some(actions[0]), RequestState::Move, log.clone()),
        ];
        for (i, (parent, action, request, lines)) in cases.iter().enumerate() {
            assert_eq!(
                p.advance_compact(7, *parent, *action, *request, lines),
                i + 1
            );
        }
        for (i, (parent, action, request, lines)) in cases.iter().enumerate() {
            assert_eq!(
                p.advance_compact(7, *parent, *action, *request, lines),
                i + 1
            );
        }
        let first = p.group(1, &actions);
        let reversed = p.group(1, &[actions[1], actions[0]]);
        assert_ne!(first, reversed);
        assert_eq!(first, p.group(1, &actions));
        assert_eq!(reversed, p.group(1, &[actions[1], actions[0]]));
        assert!(p.transitions.is_empty() && p.index.is_empty());
    }

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
