use std::collections::{HashMap, VecDeque};

use nc2000_engine::battle::enumerate::{enumerate_step, ChanceLeaf, StepEnum};
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, PokeId, RequestState};

use crate::hp_transition::HpTransitionKey;

struct Leaf {
    battle: Battle,
    probability: f64,
    draws: usize,
    damage: [i32; 2],
}

struct Kernel {
    active: [PokeId; 2],
    required_hp: [i32; 2],
    maximum_hp: [i32; 2],
    leaves: Vec<Leaf>,
}

impl Kernel {
    fn new(base: &Battle, step: &StepEnum, terminal_regions: bool) -> Option<Self> {
        let active = [base.active_id(0)?, base.active_id(1)?];
        let mut required_hp = [1; 2];
        let mut maximum_hp = active.map(|id| base.poke(id).maxhp);
        let mut leaves = Vec::with_capacity(step.leaves.len());
        for leaf in &step.leaves {
            if !terminal_regions
                && (leaf.battle.outcome().is_some()
                    || leaf.battle.request_state != RequestState::Move
                    || leaf.battle.needs_choice() != [true, true])
            {
                return None;
            }
            let mut damage = [0; 2];
            for side in 0..2 {
                let id = active[side];
                let pokemon = leaf.battle.poke(id);
                if leaf.battle.active_id(side) != Some(id) {
                    return None;
                }
                if pokemon.fainted || pokemon.hp <= 0 {
                    if !terminal_regions || !pokemon.fainted || pokemon.hp != 0 {
                        return None;
                    }
                    maximum_hp[side] = maximum_hp[side].min(base.poke(id).hp);
                    continue;
                }
                damage[side] = base.poke(id).hp - pokemon.hp;
                if damage[side] < 0 {
                    return None;
                }
                required_hp[side] = required_hp[side].max(damage[side] + 1);
            }
            leaves.push(Leaf {
                battle: leaf.battle.clone(),
                probability: leaf.prob,
                draws: leaf.draws,
                damage,
            });
        }
        (!leaves.is_empty()).then_some(Self {
            active,
            required_hp,
            maximum_hp,
            leaves,
        })
    }

    fn translate(&self, target: &Battle) -> Option<StepEnum> {
        let hp = self.active.map(|id| target.poke(id).hp);
        if (0..2).any(|side| hp[side] < self.required_hp[side] || hp[side] > self.maximum_hp[side])
        {
            return None;
        }
        let leaves = self
            .leaves
            .iter()
            .map(|leaf| {
                let mut battle = leaf.battle.clone();
                for side in 0..2 {
                    if !battle.poke(self.active[side]).fainted {
                        battle.poke_mut(self.active[side]).hp = hp[side] - leaf.damage[side];
                    }
                }
                ChanceLeaf {
                    battle,
                    prob: leaf.probability,
                    draws: leaf.draws,
                }
            })
            .collect();
        Some(StepEnum {
            leaves,
            runs: 0,
            damage: Default::default(),
        })
    }
}

#[derive(Default, Clone, Debug)]
pub struct KernelStats {
    pub calls: usize,
    pub eligible: usize,
    pub hits: usize,
    pub rejected_hp: usize,
    pub rejected_source: usize,
    pub rejected_capacity: usize,
    pub evictions: usize,
    pub peak_leaves: usize,
    pub translated_leaves: usize,
}

pub struct HpAttackKernelCache {
    capacity: usize,
    leaves: usize,
    terminal_regions: bool,
    entries: HashMap<HpTransitionKey, VecDeque<Kernel>>,
    order: VecDeque<HpTransitionKey>,
    pub stats: KernelStats,
}

impl HpAttackKernelCache {
    pub fn new(leaf_capacity: usize) -> Self {
        Self {
            capacity: leaf_capacity,
            leaves: 0,
            terminal_regions: false,
            entries: HashMap::new(),
            order: VecDeque::new(),
            stats: KernelStats::default(),
        }
    }

    /// Terminal reuse requires unobservable damage history and preserves every leaf's KO pattern.
    pub fn with_terminal_regions(mut self) -> Self {
        assert_eq!(self.stats.calls, 0);
        self.terminal_regions = true;
        self
    }

    pub fn enumerate(
        &mut self,
        dex: &Dex,
        base: &Battle,
        choices: [Option<SearchChoice>; 2],
        cap: usize,
    ) -> Option<StepEnum> {
        self.stats.calls += 1;
        let key = (self.capacity > 0 && cap > 0)
            .then(|| HpTransitionKey::new(dex, base, choices, &[]))
            .flatten();
        self.stats.eligible += usize::from(key.is_some());
        if let Some(kernels) = key.as_ref().and_then(|key| self.entries.get(key)) {
            if let Some(step) = kernels.iter().find_map(|kernel| kernel.translate(base)) {
                self.stats.hits += 1;
                self.stats.translated_leaves += step.leaves.len();
                return Some(step);
            }
            self.stats.rejected_hp += 1;
        }
        let step = enumerate_step(dex, base, choices, cap)?;
        if let Some(key) =
            key.filter(|key| self.terminal_regions || !self.entries.contains_key(key))
        {
            if step.leaves.len() > self.capacity {
                self.stats.rejected_capacity += 1;
            } else if let Some(kernel) = Kernel::new(
                base,
                &step,
                self.terminal_regions && !base.damage_bookkeeping_observable(dex),
            ) {
                while self.leaves + kernel.leaves.len() > self.capacity {
                    let old = self.order.pop_front().unwrap();
                    let kernels = self.entries.get_mut(&old).unwrap();
                    self.leaves -= kernels.pop_front().unwrap().leaves.len();
                    if kernels.is_empty() {
                        self.entries.remove(&old);
                    }
                    self.stats.evictions += 1;
                }
                self.leaves += kernel.leaves.len();
                self.stats.peak_leaves = self.stats.peak_leaves.max(self.leaves);
                self.order.push_back(key.clone());
                self.entries.entry(key).or_default().push_back(kernel);
            } else {
                self.stats.rejected_source += 1;
            }
        }
        Some(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;
    use std::collections::BTreeMap;

    fn battle(dex: &Dex, side: usize) -> (Battle, [Option<SearchChoice>; 2]) {
        let mons = [
            ("Snorlax", "Seismic Toss", "Mint Berry"),
            ("Vaporeon", "Ice Beam", "Bright Powder"),
        ]
        .map(|(species, move_name, item)| {
            serde_json::from_value::<PokemonSet>(serde_json::json!({
                "name":species,"species":species,"moves":[move_name],"item":item,"level":50
            }))
            .unwrap()
        });
        let mut battle = Battle::from_fixture(
            dex,
            "1,2,3,4",
            std::slice::from_ref(&mons[side]),
            std::slice::from_ref(&mons[1 - side]),
        )
        .unwrap();
        let preview = [0, 1].map(|s| battle.legal_choices(dex, s).first().copied());
        battle.apply_choices(dex, preview).unwrap();
        let choices = [0, 1].map(|s| battle.legal_choices(dex, s).first().copied());
        (battle, choices)
    }

    fn probability_map(step: StepEnum) -> BTreeMap<u128, f64> {
        let mut map = BTreeMap::new();
        for leaf in step.leaves {
            *map.entry(leaf.battle.state_key128()).or_default() += leaf.prob;
        }
        map
    }

    fn equal(actual: StepEnum, expected: StepEnum) {
        let actual = probability_map(actual);
        let expected = probability_map(expected);
        assert_eq!(actual.len(), expected.len());
        assert!((actual.values().sum::<f64>() - 1.).abs() < 1e-10);
        for (key, mass) in actual {
            assert!((mass - expected[&key]).abs() < 1e-10);
        }
    }

    #[test]
    fn complete_distributions_match_across_hp_and_ko_boundaries() {
        let dex = conformance::load_dex();
        for side in 0..2 {
            let (base, choices) = battle(&dex, side);
            let mut cache = HpAttackKernelCache::new(10_000);
            equal(
                cache.enumerate(&dex, &base, choices, 100_000).unwrap(),
                enumerate_step(&dex, &base, choices, 100_000).unwrap(),
            );
            assert_eq!(cache.stats.rejected_source, 0);
            for hp in [
                [200, 180],
                [180, 160],
                [100, 100],
                [51, 51],
                [50, 50],
                [49, 49],
                [1, 1],
            ] {
                let mut target = base.clone();
                for s in 0..2 {
                    let id = target.active_id(s).unwrap();
                    target.poke_mut(id).hp = hp[s];
                }
                equal(
                    cache.enumerate(&dex, &target, choices, 100_000).unwrap(),
                    enumerate_step(&dex, &target, choices, 100_000).unwrap(),
                );
            }
            assert!(cache.stats.hits >= 2 && cache.stats.rejected_hp >= 3);
            let mut target = base.clone();
            let id = target.active_id(0).unwrap();
            target.poke_mut(id).hp -= 1;
            let cached = cache.enumerate(&dex, &target, choices, 1).unwrap();
            assert_eq!(cached.runs, 0);
            equal(
                cached,
                enumerate_step(&dex, &target, choices, 100_000).unwrap(),
            );
            assert!(cache.enumerate(&dex, &target, choices, 0).is_none());
        }
    }

    #[test]
    fn partial_caps_disabled_storage_changed_inputs_and_evictions() {
        let dex = conformance::load_dex();
        let (base, choices) = battle(&dex, 0);
        for capacity in [0, 1, 10_000] {
            let mut cache = HpAttackKernelCache::new(capacity);
            assert!(cache.enumerate(&dex, &base, choices, 1).is_none());
            assert_eq!(cache.leaves, 0);
            let count = enumerate_step(&dex, &base, choices, 100_000)
                .unwrap()
                .leaves
                .len();
            equal(
                cache.enumerate(&dex, &base, choices, 100_000).unwrap(),
                enumerate_step(&dex, &base, choices, 100_000).unwrap(),
            );
            assert!(cache.stats.peak_leaves <= capacity);
            if capacity == 0 {
                assert_eq!(cache.stats.eligible, 0);
            }
            if capacity == 1 {
                assert!(cache.stats.rejected_capacity > 0);
            }
            if capacity > count {
                cache.capacity = count;
                let mut changed = base.clone();
                changed.turn += 1;
                equal(
                    cache.enumerate(&dex, &changed, choices, 100_000).unwrap(),
                    enumerate_step(&dex, &changed, choices, 100_000).unwrap(),
                );
                assert_eq!(cache.stats.evictions, 1);
                let id = changed.active_id(0).unwrap();
                changed.poke_mut(id).move_slots[0].pp -= 1;
                equal(
                    cache.enumerate(&dex, &changed, choices, 100_000).unwrap(),
                    enumerate_step(&dex, &changed, choices, 100_000).unwrap(),
                );
                assert_eq!(cache.stats.hits, 0);
            }
        }
        let mut cache = HpAttackKernelCache::new(10_000);
        let mut terminal = base.clone();
        for s in 0..2 {
            let id = terminal.active_id(s).unwrap();
            terminal.poke_mut(id).hp = 1;
        }
        equal(
            cache.enumerate(&dex, &terminal, choices, 100_000).unwrap(),
            enumerate_step(&dex, &terminal, choices, 100_000).unwrap(),
        );
        assert_eq!(cache.stats.rejected_source, 1);
        assert_eq!(cache.leaves, 0);
        let mut excluded = base.clone();
        let id = excluded.active_id(0).unwrap();
        excluded.poke_mut(id).item = dex.items.id("leftovers");
        equal(
            cache.enumerate(&dex, &excluded, choices, 100_000).unwrap(),
            enumerate_step(&dex, &excluded, choices, 100_000).unwrap(),
        );
        assert_eq!(cache.stats.eligible, 1);
    }
    #[test]
    fn admitted_moves_items_and_statuses_preserve_probability_maps() {
        use nc2000_engine::state::{Status, DK};
        let dex = conformance::load_dex();
        let mut hits = 0;
        for side in 0..2 {
            for move_name in [
                "bodyslam",
                "hydropump",
                "icebeam",
                "fireblast",
                "thunderbolt",
                "thunder",
                "earthquake",
                "seismictoss",
                "tackle",
                "splash",
            ] {
                for (status, item) in [
                    (Status::None, "quickclaw"),
                    (Status::Par, "miracleberry"),
                    (Status::Slp, "mintberry"),
                    (Status::Frz, "brightpowder"),
                ] {
                    let (mut base, mut choices) = battle(&dex, side);
                    let id = base.active_id(side).unwrap();
                    let move_id = dex.moves.id(move_name).unwrap();
                    base.poke_mut(id).move_slots[0].id = move_id;
                    base.poke_mut(id).item = dex.items.id(item);
                    base.poke_mut(id).status = status;
                    if status == Status::Slp {
                        base.poke_mut(id).status_state.set_int(DK::Time, 3);
                        base.poke_mut(id).status_state.set_int(DK::StartTime, 3);
                    }
                    base.refresh_poke_mask(&dex, id);
                    choices[side] = Some(SearchChoice::Move(move_id));
                    assert!(HpTransitionKey::new(&dex, &base, choices, &[]).is_some());
                    let mut cache = HpAttackKernelCache::new(20_000);
                    equal(
                        cache.enumerate(&dex, &base, choices, 100_000).unwrap(),
                        enumerate_step(&dex, &base, choices, 100_000).unwrap(),
                    );
                    let mut target = base.clone();
                    for s in 0..2 {
                        let id = target.active_id(s).unwrap();
                        target.poke_mut(id).hp -= 10;
                    }
                    equal(
                        cache.enumerate(&dex, &target, choices, 100_000).unwrap(),
                        enumerate_step(&dex, &target, choices, 100_000).unwrap(),
                    );
                    hits += cache.stats.hits;
                }
            }
        }
        assert!(hits > 60);
    }
    fn semantic_map(dex: &Dex, step: StepEnum) -> BTreeMap<(u8, u128), f64> {
        let mut map = BTreeMap::new();
        for leaf in step.leaves {
            let b = leaf.battle;
            let damage = !b.damage_bookkeeping_observable(dex);
            let claw = !b.quick_claw_roll_observable(dex);
            let key = if claw {
                b.state_key128_without_quick_claw_roll(dex, damage)
            } else if damage {
                b.state_key128_without_damage_bookkeeping()
            } else {
                b.state_key128()
            };
            *map.entry((u8::from(damage) | u8::from(claw) << 1, key))
                .or_default() += leaf.prob;
        }
        map
    }

    #[test]
    fn terminal_regions_match_complete_kernels_in_both_orientations() {
        let dex = conformance::load_dex();
        for side in 0..2 {
            let (base, choices) = battle(&dex, side);
            let mut cache = HpAttackKernelCache::new(16384).with_terminal_regions();
            for hp0 in [200, 100, 70, 60, 55, 51, 50, 49, 40, 30, 20, 10, 1] {
                for hp1 in [180, 100, 51, 50, 49, 40, 20, 1] {
                    let mut target = base.clone();
                    for (s, hp) in [hp0, hp1].into_iter().enumerate() {
                        let id = target.active_id(s).unwrap();
                        target.poke_mut(id).hp = hp;
                    }
                    let actual = semantic_map(
                        &dex,
                        cache.enumerate(&dex, &target, choices, 100_000).unwrap(),
                    );
                    let expected = semantic_map(
                        &dex,
                        enumerate_step(&dex, &target, choices, 100_000).unwrap(),
                    );
                    assert_eq!(actual.len(), expected.len());
                    assert!((actual.values().sum::<f64>() - 1.).abs() < 1e-10);
                    for (key, mass) in actual {
                        assert!((mass - expected[&key]).abs() < 1e-10);
                    }
                    assert!(cache.leaves <= 16384);
                }
            }
            assert!(cache.stats.hits > 30 && cache.stats.rejected_hp > 0);
        }
    }
    #[test]
    fn terminal_regions_keep_counter_history_and_turn_cap_guards() {
        let dex = conformance::load_dex();
        let (mut base, choices) = battle(&dex, 0);
        for s in 0..2 {
            let id = base.active_id(s).unwrap();
            base.poke_mut(id).hp = 1;
        }
        let id = base.active_id(0).unwrap();
        base.poke_mut(id).base_move_slots[0].id = dex.moves.id("counter").unwrap();
        assert!(base.damage_bookkeeping_observable(&dex));
        let mut cache = HpAttackKernelCache::new(16384).with_terminal_regions();
        for _ in 0..2 {
            equal(
                cache.enumerate(&dex, &base, choices, 100_000).unwrap(),
                enumerate_step(&dex, &base, choices, 100_000).unwrap(),
            );
        }
        assert_eq!(cache.stats.hits, 0);
        assert_eq!(cache.leaves, 0);
        let (mut base, choices) = battle(&dex, 0);
        base.turn = 1000;
        let mut cache = HpAttackKernelCache::new(16384).with_terminal_regions();
        for decrement in [0, 1, 20] {
            let mut b = base.clone();
            for s in 0..2 {
                let id = b.active_id(s).unwrap();
                b.poke_mut(id).hp -= decrement;
            }
            let actual = semantic_map(&dex, cache.enumerate(&dex, &b, choices, 100_000).unwrap());
            let expected = semantic_map(&dex, enumerate_step(&dex, &b, choices, 100_000).unwrap());
            assert_eq!(actual.len(), expected.len());
            for (key, mass) in actual {
                assert!((mass - expected[&key]).abs() < 1e-10);
            }
        }
        assert!(cache.stats.hits > 0);
    }
}
