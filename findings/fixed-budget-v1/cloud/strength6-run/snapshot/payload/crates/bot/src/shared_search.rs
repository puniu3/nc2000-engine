use std::hash::{Hash, Hasher};

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::{FxHashMap, FxHasher};
use nc2000_engine::state::Battle;

use crate::belief::Belief;
use crate::mcts::{outcome_reward, playout_value};
use crate::observe::Observer;
use crate::{RmConfig, SplitMix64};

struct Bandit {
    side: usize,
    actions: Vec<SearchChoice>,
    visits: Vec<u32>,
    rewards: Vec<f64>,
}

impl Bandit {
    fn new(side: usize, actions: Vec<SearchChoice>) -> Self {
        Self {
            side,
            visits: vec![0; actions.len()],
            rewards: vec![0.0; actions.len()],
            actions,
        }
    }

    fn pick(&mut self, rng: &mut SplitMix64, c: f64) -> usize {
        let untried: Vec<_> = self
            .visits
            .iter()
            .enumerate()
            .filter_map(|(i, &n)| (n == 0).then_some(i))
            .collect();
        let pick = if !untried.is_empty() {
            untried[rng.below(untried.len())]
        } else {
            let total = (self.visits.iter().sum::<u32>() as f64).ln();
            let mut best = 0;
            let mut score = f64::NEG_INFINITY;
            for (i, (&n, &w)) in self.visits.iter().zip(&self.rewards).enumerate() {
                let value = w / n as f64 + c * (total / n as f64).sqrt();
                if value > score {
                    score = value;
                    best = i;
                }
            }
            best
        };
        self.visits[pick] += 1;
        pick
    }
}

pub struct SharedSearch {
    base: Battle,
    side: usize,
    cfg: RmConfig,
    rng: SplitMix64,
    nodes: Vec<Bandit>,
    own: FxHashMap<u64, usize>,
    opponent: FxHashMap<u64, usize>,
    dominated: Vec<bool>,
    iterations: u32,
    depth_sum: u64,
}

fn own_key(battle: &Battle, side: usize, actions: &[SearchChoice], buckets: i64) -> u64 {
    let mut masked = battle.clone();
    let foe = &mut masked.sides[1 - side];
    let size = foe.party.len();
    let mut party = Vec::with_capacity(size);
    if let Some(active) = foe.active {
        party.push(active);
    }
    for (slot, mon) in foe.roster.iter().enumerate() {
        if mon.previously_switched_in > 0 && !party.contains(&(slot as u8)) {
            party.push(slot as u8);
        }
    }
    for slot in 0..foe.roster.len() as u8 {
        if party.len() == size {
            break;
        }
        if !party.contains(&slot) {
            party.push(slot);
        }
    }
    assert_eq!(party.len(), size);
    foe.party.clear();
    foe.party.extend(party.iter().copied());
    let mut outside = size as u8;
    for (slot, mon) in foe.roster.iter_mut().enumerate() {
        mon.position = if let Some(position) = party.iter().position(|&s| s as usize == slot) {
            position as u8
        } else {
            let position = outside;
            outside += 1;
            position
        };
        mon.speed = 0;
    }
    masked.quick_claw_roll = false;
    let mut hash = FxHasher::default();
    let key = if buckets > 0 {
        masked.state_key_bucketed(buckets)
    } else {
        masked.state_key()
    };
    key.hash(&mut hash);
    actions.hash(&mut hash);
    hash.finish()
}

impl SharedSearch {
    pub fn new(battle: &Battle, dex: &Dex, side: usize, cfg: RmConfig, seed: u64) -> Self {
        let mut base = battle.clone();
        base.set_log_enabled(false);
        let actions = base.legal_choices(dex, side);
        assert!(!actions.is_empty());
        let dominated = actions
            .iter()
            .map(|&action| {
                crate::smmcts::dominated_reason(&base, dex, side, action, cfg.mask_rules).is_some()
            })
            .collect();
        Self {
            base,
            side,
            cfg,
            rng: SplitMix64::new(seed),
            nodes: vec![Bandit::new(side, actions)],
            own: FxHashMap::default(),
            opponent: FxHashMap::default(),
            dominated,
            iterations: 0,
            depth_sum: 0,
        }
    }

    pub fn step(&mut self, dex: &Dex, belief: &Belief, observed: &Observer, iterations: u32) {
        let cap = self.base.turn.saturating_add(self.cfg.horizon);
        for _ in 0..iterations {
            let mut sim = belief.determinize(dex, &self.base, observed, &mut self.rng);
            let mut path = Vec::new();
            let mut depth = 0;
            let reward = loop {
                let choices = [sim.legal_choices(dex, 0), sim.legal_choices(dex, 1)];
                let mut ids = [None, None];
                let mut new_own = false;
                let mut new_foe = false;
                for side in 0..2 {
                    if choices[side].is_empty() {
                        continue;
                    }
                    if depth == 0 && side == self.side {
                        ids[side] = Some(0);
                        continue;
                    }
                    let key = if side == self.side {
                        own_key(&sim, side, &choices[side], self.cfg.hp_buckets)
                    } else if self.cfg.hp_buckets > 0 {
                        sim.state_key_bucketed(self.cfg.hp_buckets)
                    } else {
                        sim.state_key()
                    };
                    let table = if side == self.side {
                        &mut self.own
                    } else {
                        &mut self.opponent
                    };
                    let id = match table.get(&key) {
                        Some(&id) => id,
                        None => {
                            let id = self.nodes.len();
                            self.nodes.push(Bandit::new(side, choices[side].clone()));
                            table.insert(key, id);
                            if side == self.side {
                                new_own = true;
                            } else {
                                new_foe = true;
                            }
                            id
                        }
                    };
                    assert_eq!(
                        self.nodes[id].actions, choices[side],
                        "shared node changed legal actions"
                    );
                    ids[side] = Some(id);
                }
                if depth > 0 && (new_own || (choices[self.side].is_empty() && new_foe)) {
                    break playout_value(
                        &mut sim,
                        dex,
                        &self.cfg.playout,
                        cap,
                        &mut self.rng,
                        self.cfg.rollout_rules(),
                    );
                }
                let mut joint = [None, None];
                for side in 0..2 {
                    if let Some(id) = ids[side] {
                        let pick = self.nodes[id].pick(&mut self.rng, self.cfg.c);
                        joint[side] = Some(self.nodes[id].actions[pick]);
                        path.push((id, pick));
                    }
                }
                assert_ne!(joint, [None, None]);
                sim.apply_choices(dex, joint)
                    .expect("shared search chose an illegal action");
                depth += 1;
                if let Some(outcome) = sim.outcome() {
                    break outcome_reward(outcome);
                }
                if sim.turn > cap {
                    break playout_value(
                        &mut sim,
                        dex,
                        &self.cfg.playout,
                        cap,
                        &mut self.rng,
                        self.cfg.rollout_rules(),
                    );
                }
            };
            for (id, action) in path {
                let node = &mut self.nodes[id];
                node.rewards[action] += if node.side == 0 { reward } else { 1.0 - reward };
            }
            self.iterations += 1;
            self.depth_sum += depth;
        }
    }

    pub fn best(&self) -> SearchChoice {
        let root = &self.nodes[0];
        let pick = (0..root.actions.len())
            .filter(|&i| !self.dominated[i])
            .max_by_key(|&i| root.visits[i])
            .unwrap_or_else(|| {
                (0..root.actions.len())
                    .max_by_key(|&i| root.visits[i])
                    .unwrap()
            });
        root.actions[pick]
    }

    pub fn mean_depth(&self) -> f64 {
        self.depth_sum as f64 / self.iterations.max(1) as f64
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_policy(&self) -> Vec<(SearchChoice, u32, f64)> {
        let root = &self.nodes[0];
        root.actions
            .iter()
            .enumerate()
            .map(|(i, &action)| {
                (
                    action,
                    root.visits[i],
                    if root.visits[i] == 0 {
                        0.5
                    } else {
                        root.rewards[i] / root.visits[i] as f64
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn sampled_unrevealed_benches_share_own_statistics() {
        let dex = conformance::load_dex();
        let pool = crate::preview::load_meta_pool(
            &conformance::fixture::repo_root().join("data/meta-pool-v0/meta-pool.json"),
        );
        let mut battle =
            Battle::from_fixture(&dex, "2,3,5,7", &pool.teams[0].sets, &pool.teams[1].sets)
                .unwrap();
        let mut observed = Observer::new(&battle, 0);
        let picks = [
            Some(battle.legal_choices(&dex, 0)[0]),
            Some(battle.legal_choices(&dex, 1)[0]),
        ];
        battle.apply_choices(&dex, picks).unwrap();
        observed.observe(&battle, &dex);
        let belief = Belief::pinned(&dex, "foe", &pool.teams[1].sets, &observed);
        let mut own_keys = BTreeSet::new();
        let mut full_keys = BTreeSet::new();
        let mut rng = SplitMix64::new(43);
        for _ in 0..64 {
            let mut sample = belief.determinize(&dex, &battle, &observed, &mut rng);
            let actions = sample.legal_choices(&dex, 0);
            own_keys.insert(own_key(&sample, 0, &actions, 16));
            full_keys.insert(sample.state_key_bucketed(16));
        }
        assert_eq!(own_keys.len(), 1);
        assert!(full_keys.len() > 10);
        let mut search = SharedSearch::new(&battle, &dex, 0, RmConfig::default(), 47);
        search.step(&dex, &belief, &observed, 128);
        assert!(battle.legal_choices(&dex, 0).contains(&search.best()));
        assert_eq!(
            search.root_policy().iter().map(|row| row.1).sum::<u32>(),
            128
        );
    }
}
