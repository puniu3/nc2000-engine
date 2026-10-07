use nc2000_engine::{
    battle::{
        enumerate::{enumerate_step_with_runner, run_scripted, ChanceLeaf},
        SearchChoice,
    },
    dex::Dex,
    prng::Draw,
    state::Battle,
};

struct Pending {
    script: Vec<usize>,
    mass: f64,
    known: Option<(Battle, Vec<Draw>)>,
}

pub struct Stream<'a> {
    dex: &'a Dex,
    base: &'a Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
    pending: Vec<Pending>,
    ready: std::vec::IntoIter<ChanceLeaf>,
    pub runs: usize,
    pub emitted: f64,
    pub exhausted: bool,
    order_side: Option<usize>,
}

impl<'a> Stream<'a> {
    pub fn new(
        dex: &'a Dex,
        base: &'a Battle,
        choices: [Option<SearchChoice>; 2],
        cap: usize,
        order_side: Option<usize>,
    ) -> Self {
        Self {
            dex,
            base,
            choices,
            cap,
            pending: vec![Pending {
                script: Vec::new(),
                mass: 1.,
                known: None,
            }],
            ready: Vec::new().into_iter(),
            runs: 0,
            emitted: 0.,
            exhausted: false,
            order_side,
        }
    }

    pub fn restrict(&mut self, remaining: usize) {
        self.cap = self.cap.min(self.runs.saturating_add(remaining));
    }

    pub fn next(&mut self) -> Option<ChanceLeaf> {
        loop {
            if let Some(leaf) = self.ready.next() {
                self.emitted += leaf.prob;
                assert!(self.emitted <= 1. + 1e-9);
                return Some(leaf);
            }
            if self.exhausted {
                return None;
            }
            let task = self.pending.pop()?;
            let (battle, trace) = if let Some(known) = task.known {
                known
            } else {
                if self.runs >= self.cap {
                    self.exhausted = true;
                    return None;
                }
                self.runs += 1;
                run_scripted(self.dex, self.base, self.choices, &task.script)
            };
            assert!(trace.len() >= task.script.len());
            assert!(trace
                .iter()
                .zip(&task.script)
                .all(|(draw, &c)| draw.chosen == c));
            if trace.len() == task.script.len() {
                self.ready = vec![ChanceLeaf {
                    battle,
                    prob: task.mass,
                    draws: trace.len(),
                }]
                .into_iter();
                continue;
            }
            let p = task.script.len();
            if trace[p].label.starts_with("dmgvar") {
                let prefix = task.script;
                let mut known = Some((battle, trace));
                let dex = self.dex;
                let base = self.base;
                let choices = self.choices;
                let mut executed = 0;
                let mut runner = |suffix: &[usize]| {
                    let (battle, trace, ran) = if suffix.is_empty() && known.is_some() {
                        let (b, t) = known.take().unwrap();
                        (b, t, false)
                    } else {
                        let mut script = prefix.clone();
                        script.extend_from_slice(suffix);
                        let (b, t) = run_scripted(dex, base, choices, &script);
                        executed += 1;
                        (b, t, true)
                    };
                    assert!(
                        trace.len() >= p && trace.iter().zip(&prefix).all(|(d, &c)| d.chosen == c)
                    );
                    (battle, trace[p..].to_vec(), ran)
                };
                let step = enumerate_step_with_runner(
                    dex,
                    base,
                    choices,
                    self.cap.saturating_sub(self.runs),
                    &mut runner,
                );
                self.runs += executed;
                if let Some(step) = step {
                    assert_eq!(step.runs, executed);
                    self.ready = step
                        .leaves
                        .into_iter()
                        .map(|mut leaf| {
                            leaf.prob *= task.mass;
                            leaf.draws += p;
                            leaf
                        })
                        .collect::<Vec<_>>()
                        .into_iter();
                } else {
                    self.exhausted = true;
                    return None;
                }
            } else {
                let chosen = trace[p].chosen;
                let counts = trace[p].counts.clone();
                assert_eq!(counts.iter().sum::<u64>(), Draw::TOTAL);
                let mut known = Some((battle, trace));
                let mut children = Vec::new();
                for (c, count) in counts
                    .into_iter()
                    .enumerate()
                    .filter(|(_, count)| *count > 0)
                {
                    let mut script = task.script.clone();
                    script.push(c);
                    let cached = if c == chosen {
                        known.take()
                    } else if self.order_side.is_some() && self.runs < self.cap {
                        self.runs += 1;
                        Some(run_scripted(self.dex, self.base, self.choices, &script))
                    } else {
                        None
                    };
                    let mass = task.mass * count as f64 / Draw::TOTAL as f64;
                    let score = if let Some(side) = self.order_side {
                        cached.as_ref().map_or(0., |(b, _)| {
                            let value = nc2000_bot::eval::eval01(b, self.dex, &Default::default());
                            mass * if side == 0 { value } else { 1. - value }
                        })
                    } else {
                        0.
                    };
                    children.push((
                        Pending {
                            script,
                            mass,
                            known: cached,
                        },
                        score,
                    ));
                }
                if self.order_side.is_some() {
                    children.sort_by(|a, b| b.1.total_cmp(&a.1));
                }
                self.pending
                    .extend(children.into_iter().rev().map(|(task, _)| task));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::{enumerate::enumerate_step, PokemonSet};
    use std::collections::BTreeMap;

    #[test]
    fn stopped_streams_partition_native_kernels_without_exceeding_caps() {
        let dex = conformance::load_dex();
        let mon = |species: &str, item: &str, moves: Vec<&str>| {
            serde_json::from_value::<PokemonSet>(
            serde_json::json!({"name":species,"species":species,"item":item,"level":50,"moves":moves})).unwrap()
        };
        let own = vec![
            mon("Marowak", "Quick Claw", vec!["Rock Slide", "Double-Edge"]),
            mon("Snorlax", "", vec!["Tackle"]),
            mon("Tauros", "", vec!["Tackle"]),
        ];
        let foe = vec![
            mon(
                "Zapdos",
                "Focus Band",
                vec!["Thunderbolt", "Hidden Power Ice"],
            ),
            mon("Miltank", "", vec!["Tackle"]),
            mon("Starmie", "", vec!["Tackle"]),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &own, &foe).unwrap();
        let preview = [0, 1].map(|s| Some(b.legal_choices(&dex, s)[0]));
        b.apply_choices(&dex, preview).unwrap();
        let actions = [0, 1].map(|s| {
            b.legal_choices(&dex, s)
                .into_iter()
                .filter(|c| matches!(c, SearchChoice::Move(_)))
                .collect::<Vec<_>>()
        });
        assert_eq!([actions[0].len(), actions[1].len()], [2, 2]);
        for &a in &actions[0] {
            for &z in &actions[1] {
                let choices = [Some(a), Some(z)];
                let full = enumerate_step(&dex, &b, choices, 200000).unwrap();
                let mut expected = BTreeMap::<u128, f64>::new();
                for leaf in full.leaves {
                    *expected.entry(leaf.battle.state_key128()).or_default() += leaf.prob;
                }
                for side in [None, Some(0), Some(1)] {
                    for cap in [0, 1, 2, 4, 8, 16, 64, 256, 200000] {
                        for stop in [1, 3, usize::MAX] {
                            let mut stream = Stream::new(&dex, &b, choices, cap, side);
                            let mut found = BTreeMap::<u128, f64>::new();
                            let mut leaves = 0;
                            while leaves < stop {
                                let Some(leaf) = stream.next() else {
                                    break;
                                };
                                *found.entry(leaf.battle.state_key128()).or_default() += leaf.prob;
                                leaves += 1;
                            }
                            assert!(
                                stream.runs <= cap
                                    && stream.emitted >= 0.
                                    && stream.emitted <= 1. + 1e-10
                            );
                            assert!((found.values().sum::<f64>() - stream.emitted).abs() < 1e-10);
                            for (key, mass) in &found {
                                assert!(*mass <= expected.get(key).copied().unwrap_or(0.) + 1e-10);
                            }
                            if cap == 200000 && stop == usize::MAX {
                                assert!((stream.emitted - 1.).abs() < 1e-10);
                                assert_eq!(found.len(), expected.len());
                                for (key, mass) in &found {
                                    assert!((*mass - expected[key]).abs() < 1e-10);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
