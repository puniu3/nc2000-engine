use std::collections::{HashMap, HashSet};

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::toid;
use serde::Deserialize;

use crate::rng::SplitMix64;

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
pub struct PreviewMon {
    pub species: String,
    pub level: u8,
    pub gender: String,
    pub item: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    schema: String,
    smoothing: f64,
    #[serde(default)]
    backoff_strength: f64,
    source: serde_json::Value,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    team: String,
    enemy_preview: Vec<PreviewMon>,
    side: usize,
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Choice {
    species: [String; 3],
    count: u32,
}

pub struct PickPrior {
    smoothing: f64,
    backoff_strength: f64,
    rows: HashMap<(String, Vec<PreviewMon>, usize), Vec<Choice>>,
}

pub(crate) struct PickDistribution {
    cumulative: Vec<(u8, f64)>,
    total: f64,
    preview: Vec<(SearchChoice, f64)>,
    smoothing: f64,
}

impl PickPrior {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let artifact: Artifact = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if artifact.schema != "nc2000-pick-prior-v1"
            || !artifact.smoothing.is_finite()
            || artifact.smoothing <= 0.0
            || artifact.smoothing > 1.0
            || !artifact.backoff_strength.is_finite()
            || artifact.backoff_strength < 0.0
            || !artifact.source.is_object()
        {
            return Err("invalid pick prior schema, smoothing, or provenance".into());
        }
        let mut rows = HashMap::new();
        for mut row in artifact.rows {
            let mut distinct = HashSet::new();
            if row.side > 1
                || row.team.is_empty()
                || ![0, 6].contains(&row.enemy_preview.len())
                || row.choices.is_empty()
                || row.enemy_preview.iter().any(|mon| {
                    mon.species.is_empty()
                        || toid(&mon.species) != mon.species
                        || !distinct.insert(mon.species.clone())
                        || !(1..=100).contains(&mon.level)
                        || !["", "M", "F"].contains(&mon.gender.as_str())
                })
            {
                return Err("invalid pick prior row".into());
            }
            let mut choices = HashSet::new();
            for choice in &row.choices {
                if choice.count == 0
                    || choice.species.iter().collect::<HashSet<_>>().len() != 3
                    || choice.species.iter().any(|s| s.is_empty() || toid(s) != *s)
                    || !choices.insert(choice.species.clone())
                {
                    return Err("invalid or duplicate pick prior choice".into());
                }
            }
            row.enemy_preview.sort();
            if rows
                .insert((row.team, row.enemy_preview, row.side), row.choices)
                .is_some()
            {
                return Err("duplicate pick prior row".into());
            }
        }
        Ok(Self {
            smoothing: artifact.smoothing,
            backoff_strength: artifact.backoff_strength,
            rows,
        })
    }

    pub(crate) fn condition(
        &self,
        team: &str,
        enemy_preview: &[PreviewMon],
        side: usize,
        roster: &[String],
        appeared: u8,
    ) -> Option<PickDistribution> {
        if roster.len() != 6 || appeared.count_ones() > 3 {
            return None;
        }
        let mut preview = enemy_preview.to_vec();
        preview.sort();
        let specific = self.rows.get(&(team.to_string(), preview, side));
        let general = self.rows.get(&(team.to_string(), Vec::new(), side));
        let sources = match (specific, general) {
            (Some(specific), Some(general)) => {
                let count: f64 = specific.iter().map(|c| c.count as f64).sum();
                let mix = count / (count + self.backoff_strength);
                vec![(specific, mix), (general, 1.0 - mix)]
            }
            (Some(choices), None) | (None, Some(choices)) => vec![(choices, 1.0)],
            (None, None) => return None,
        };
        let mut weights = [0.0; 64];
        let mut preview = Vec::new();
        for (choices, mix) in sources {
            let total: f64 = choices.iter().map(|c| c.count as f64).sum();
            for choice in choices {
                let mut mask = 0u8;
                let mut slots = [0; 3];
                for (i, species) in choice.species.iter().enumerate() {
                    let slot = roster.iter().position(|s| s == species)?;
                    mask |= 1 << slot;
                    slots[i] = slot as u8 + 1;
                }
                if mask.count_ones() != 3 {
                    return None;
                }
                let probability = mix * choice.count as f64 / total;
                weights[mask as usize] += (1.0 - self.smoothing) * probability;
                preview.push((SearchChoice::Team(slots), probability));
            }
        }
        let mut cumulative = Vec::new();
        let mut sum = 0.0;
        for mask in 0u8..64 {
            if mask.count_ones() == 3 && mask & appeared == appeared {
                sum += weights[mask as usize] + self.smoothing / 20.0;
                cumulative.push((mask, sum));
            }
        }
        (!cumulative.is_empty()).then_some(PickDistribution {
            cumulative,
            total: sum,
            preview,
            smoothing: self.smoothing,
        })
    }
}

impl PickDistribution {
    pub(crate) fn sample_allowed(&self, allowed: &[u8], rng: &mut SplitMix64) -> Option<u8> {
        let mut previous = 0.0;
        let weights: Vec<_> = self.cumulative.iter().filter_map(|&(mask, cumulative)| {
            let weight = cumulative - previous;
            previous = cumulative;
            (allowed.contains(&mask) && weight > 0.0).then_some((mask, weight))
        }).collect();
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        if total <= 0.0 { return None; }
        let mut target = rng.next_f64() * total;
        for &(mask, weight) in &weights {
            if target < weight { return Some(mask); }
            target -= weight;
        }
        Some(weights.last().unwrap().0)
    }

    pub(crate) fn sample(&self, rng: &mut SplitMix64) -> u8 {
        let draw = rng.next_f64() * self.total;
        self.cumulative
            .iter()
            .find(|(_, p)| draw < *p)
            .unwrap_or_else(|| self.cumulative.last().unwrap())
            .0
    }

    pub(crate) fn sample_preview(
        &self,
        actions: &[SearchChoice],
        rng: &mut SplitMix64,
    ) -> Option<usize> {
        if actions.is_empty()
            || self
                .preview
                .iter()
                .any(|(choice, _)| !actions.contains(choice))
        {
            return None;
        }
        let draw = rng.next_f64();
        if draw < self.smoothing {
            return Some(rng.below(actions.len()));
        }
        let target = (draw - self.smoothing) / (1.0 - self.smoothing);
        let mut sum = 0.0;
        for (choice, probability) in &self.preview {
            sum += probability;
            if target < sum {
                return actions.iter().position(|action| action == choice);
            }
        }
        actions
            .iter()
            .position(|action| action == &self.preview.last().unwrap().0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_masks_condition_weights_without_retrying_forbidden_mass() {
        let distribution = PickDistribution {
            cumulative: vec![(7, 0.1), (11, 0.4), (19, 1.0)], total: 1.0,
            preview: vec![], smoothing: 0.2,
        };
        let mut rng = SplitMix64::new(41);
        let mut first = 0;
        for _ in 0..10000 {
            let mask = distribution.sample_allowed(&[7, 11], &mut rng).unwrap();
            assert!([7, 11].contains(&mask));
            first += usize::from(mask == 7);
        }
        assert!((2300..2700).contains(&first), "{first}");
        for _ in 0..10 { assert_eq!(distribution.sample_allowed(&[7], &mut rng), Some(7)); }
        let mut untouched = rng.clone();
        assert_eq!(distribution.sample_allowed(&[63], &mut rng), None);
        assert_eq!(rng.next(), untouched.next());
    }

    fn fixture() -> (PickPrior, Vec<PreviewMon>, Vec<String>) {
        let roster: Vec<String> = (b'a'..=b'f').map(|c| (c as char).to_string()).collect();
        let preview: Vec<PreviewMon> = roster
            .iter()
            .map(|s| PreviewMon {
                species: s.clone(),
                level: 50,
                gender: "M".into(),
                item: true,
            })
            .collect();
        let choices = vec![Choice {
            species: ["a".into(), "b".into(), "c".into()],
            count: 4,
        }];
        let rows = HashMap::from([(("team".into(), preview.clone(), 1), choices)]);
        (
            PickPrior {
                smoothing: 0.2,
                backoff_strength: 8.0,
                rows,
            },
            preview,
            roster,
        )
    }

    #[test]
    fn posterior_conditions_the_smoothed_joint_distribution_on_public_reveals() {
        let (prior, preview, roster) = fixture();
        let distribution = prior.condition("team", &preview, 1, &roster, 1).unwrap();
        assert!((distribution.total - 0.9).abs() < 1e-12);
        let mut rng = SplitMix64::new(11);
        let mut preferred = 0;
        let mut support = HashSet::new();
        for _ in 0..10_000 {
            let mask = distribution.sample(&mut rng);
            assert_eq!(mask.count_ones(), 3);
            assert_eq!(mask & 1, 1);
            preferred += usize::from(mask == 7);
            support.insert(mask);
        }
        assert!((8800..9200).contains(&preferred));
        assert_eq!(support.len(), 10);
        let unexpected = prior.condition("team", &preview, 1, &roster, 32).unwrap();
        assert!((unexpected.total - 0.1).abs() < 1e-12);
        assert_eq!(unexpected.cumulative.len(), 10);
    }

    #[test]
    fn public_signature_is_order_invariant_and_unknown_matchups_fall_back() {
        let (prior, mut preview, roster) = fixture();
        preview.reverse();
        assert!(prior.condition("team", &preview, 1, &roster, 0).is_some());
        assert!(prior.condition("other", &preview, 1, &roster, 0).is_none());
        assert!(prior.condition("team", &preview, 0, &roster, 0).is_none());
        preview[0].item = false;
        assert!(prior.condition("team", &preview, 1, &roster, 0).is_none());
    }

    #[test]
    fn preview_sampling_retains_the_lead_and_never_forces_an_illegal_choice() {
        let (prior, preview, roster) = fixture();
        let distribution = prior.condition("team", &preview, 1, &roster, 0).unwrap();
        let actions = [SearchChoice::Team([1, 2, 3]), SearchChoice::Team([2, 1, 3])];
        let mut rng = SplitMix64::new(17);
        let preferred = (0..10_000)
            .filter(|_| distribution.sample_preview(&actions, &mut rng) == Some(0))
            .count();
        assert!((8800..9200).contains(&preferred));
        assert_eq!(distribution.sample_preview(&actions[1..], &mut rng), None);
        assert_eq!(distribution.sample_preview(&[], &mut rng), None);
    }

    #[test]
    fn sparse_matchups_mix_with_team_tendencies_and_unknown_matchups_use_them() {
        let (mut prior, mut preview, roster) = fixture();
        prior.rows.insert(
            ("team".into(), Vec::new(), 1),
            vec![Choice {
                species: ["a".into(), "b".into(), "d".into()],
                count: 20,
            }],
        );
        let specific = prior.condition("team", &preview, 1, &roster, 1).unwrap();
        let mut rng = SplitMix64::new(19);
        let count = (0..10_000)
            .filter(|_| specific.sample(&mut rng) == 7)
            .count();
        assert!((2900..3250).contains(&count));
        preview[0].item = false;
        let general = prior.condition("team", &preview, 1, &roster, 1).unwrap();
        let count = (0..10_000)
            .filter(|_| general.sample(&mut rng) == 11)
            .count();
        assert!((8800..9200).contains(&count));
    }
}
