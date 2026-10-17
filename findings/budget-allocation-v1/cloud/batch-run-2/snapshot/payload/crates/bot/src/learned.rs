use nc2000_engine::dex::Dex;
use serde::{Deserialize, Serialize};

use crate::learning::{
    LearningObservation, Vocabulary, ACTION_FEATURES, GLOBAL_FEATURES, MON_FEATURES,
    OBSERVATION_SCHEMA,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matrix {
    pub shape: [usize; 2],
    pub data: Vec<f32>,
}

impl Matrix {
    fn validate(&self, shape: [usize; 2]) -> Result<(), String> {
        if self.shape != shape
            || self.data.len() != shape[0] * shape[1]
            || self.data.iter().any(|x| !x.is_finite())
        {
            return Err(format!(
                "invalid matrix: expected {shape:?}, got {:?}",
                self.shape
            ));
        }
        Ok(())
    }

    fn row(&self, i: usize) -> Result<&[f32], String> {
        if i >= self.shape[0] {
            return Err(format!("embedding index {i} out of range"));
        }
        Ok(&self.data[i * self.shape[1]..(i + 1) * self.shape[1]])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Linear {
    pub weight: Matrix,
    pub bias: Vec<f32>,
}

impl Linear {
    fn validate(&self, output: usize, input: usize) -> Result<(), String> {
        self.weight.validate([output, input])?;
        if self.bias.len() != output || self.bias.iter().any(|x| !x.is_finite()) {
            return Err("invalid linear bias".into());
        }
        Ok(())
    }

    fn apply(&self, x: &[f32], relu: bool) -> Vec<f32> {
        self.weight
            .data
            .chunks_exact(x.len())
            .zip(&self.bias)
            .map(|(row, &bias)| {
                let y = row.iter().zip(x).fold(bias, |sum, (&w, &x)| sum + w * x);
                if relu {
                    y.max(0.0)
                } else {
                    y
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyValue {
    pub schema: String,
    pub observation_schema: String,
    pub vocabulary: Vocabulary,
    pub species: Matrix,
    pub items: Matrix,
    pub moves: Matrix,
    pub mon: Linear,
    pub context: Linear,
    pub action: Linear,
    pub policy: Linear,
    pub value: Linear,
    #[serde(default)]
    pub value_baseline: Option<String>,
    pub training: serde_json::Value,
}

impl PolicyValue {
    pub fn from_json(text: &str, dex: &Dex) -> Result<Self, String> {
        let model: Self = serde_json::from_str(text).map_err(|e| format!("model JSON: {e}"))?;
        model.validate(dex)?;
        Ok(model)
    }

    pub fn validate(&self, dex: &Dex) -> Result<(), String> {
        let schema_matches = match self.value_baseline.as_deref() {
            None => self.schema == "nc2000-policy-value-v1",
            Some("material") => self.schema == "nc2000-policy-value-v2",
            _ => false,
        };
        if !schema_matches || self.observation_schema != OBSERVATION_SCHEMA {
            return Err("unsupported model/observation schema".into());
        }
        if self.vocabulary != Vocabulary::from_dex(dex) {
            return Err("model dex vocabulary differs".into());
        }
        self.species.validate([self.vocabulary.species.len(), 16])?;
        self.items.validate([self.vocabulary.items.len(), 8])?;
        self.moves.validate([self.vocabulary.moves.len(), 8])?;
        self.mon.validate(32, MON_FEATURES + 32)?;
        self.context.validate(128, 12 * 32 + GLOBAL_FEATURES)?;
        self.action.validate(64, 128 + ACTION_FEATURES + 8)?;
        self.policy.validate(1, 64)?;
        self.value.validate(1, 128)?;
        Ok(())
    }

    fn encode(&self, observation: &LearningObservation) -> Result<Vec<f32>, String> {
        if observation.schema != OBSERVATION_SCHEMA
            || observation.mons.len() != 12
            || observation.global.len() != GLOBAL_FEATURES
        {
            return Err("invalid observation shape".into());
        }
        let mut state = Vec::with_capacity(12 * 32 + GLOBAL_FEATURES);
        for mon in &observation.mons {
            if mon.features.len() != MON_FEATURES {
                return Err("invalid mon feature count".into());
            }
            let mut x = mon.features.clone();
            x.extend(self.species.row(mon.species)?);
            x.extend(self.items.row(mon.item)?);
            let mut moves = [0.0; 8];
            for &id in &mon.moves {
                for (x, &y) in moves.iter_mut().zip(self.moves.row(id)?) {
                    *x += y * 0.25;
                }
            }
            x.extend(moves);
            state.extend(self.mon.apply(&x, true));
        }
        state.extend(&observation.global);
        if state.iter().any(|x| !x.is_finite()) {
            return Err("non-finite observation".into());
        }
        Ok(self.context.apply(&state, true))
    }

    pub fn predict_value(&self, observation: &LearningObservation) -> Result<f32, String> {
        let context = self.encode(observation)?;
        let value = self.value_probability(observation, &context);
        if !value.is_finite() {
            return Err("non-finite model output".into());
        }
        Ok(value)
    }

    fn value_probability(&self, observation: &LearningObservation, context: &[f32]) -> f32 {
        let mut logit = self.value.apply(context, false)[0];
        if self.value_baseline.is_some() {
            let mut health = [observation.global[4] * 6.0, observation.global[5] * 6.0];
            for (side, health) in health.iter_mut().enumerate() {
                for mon in &observation.mons[side * 6..side * 6 + 6] {
                    let f = &mon.features;
                    *health -= (1.0 - f[1]) * f[4] * (1.0 - f[7]);
                }
            }
            logit = 2.0 * (health[0] - health[1]) + 0.5 * logit.tanh();
        }
        1.0 / (1.0 + (-logit).exp())
    }

    pub fn predict(&self, observation: &LearningObservation) -> Result<(Vec<f32>, f32), String> {
        if observation.actions.is_empty() {
            return Err("policy observation has no actions".into());
        }
        let context = self.encode(observation)?;
        let logits = observation
            .actions
            .iter()
            .map(|action| {
                if action.features.len() != ACTION_FEATURES
                    || action.features.iter().any(|x| !x.is_finite())
                {
                    return Err("invalid action features".into());
                }
                let mut x = context.clone();
                x.extend(&action.features);
                x.extend(self.moves.row(action.move_id)?);
                Ok(self.policy.apply(&self.action.apply(&x, true), false)[0])
            })
            .collect::<Result<Vec<_>, String>>()?;
        let value = self.value_probability(observation, &context);
        if !value.is_finite() || logits.iter().any(|x| !x.is_finite()) {
            return Err("non-finite model output".into());
        }
        Ok((logits, value))
    }
}
