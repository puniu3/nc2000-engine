//! Storage-independent identity for completed, blind team-preview decisions.
use nc2000_bot::{Observer, RmConfig};
use nc2000_engine::state::Battle;
use serde_json::{json, Value};

pub(crate) struct Identity {
    driver: &'static str,
    config: String,
    pool: Value,
    prior: Value,
}

impl Identity {
    pub fn new(driver: &'static str, cfg: &RmConfig, pool_json: &str) -> Self {
        Self {
            driver,
            config: format!("{cfg:?}"),
            // The constructor has already parsed and validated this JSON.
            pool: serde_json::from_str(pool_json).expect("validated pool JSON"),
            prior: Value::Null,
        }
    }

    pub fn set_prior(&mut self, json: &str) {
        self.prior = serde_json::from_str(json).expect("applied prior JSON");
    }

    pub fn info(&self, battle: &Battle, observer: &Observer) -> Option<String> {
        let team_key = nc2000_bot::compiled_preview::key(battle, observer.side())?;
        let public: Vec<_> = observer.mons().iter().map(|m| json!([
            m.species.0, m.level, format!("{:?}", m.gender), m.preview_has_item,
            m.hp_type_mask, m.revealed_moves.iter().map(|x| x.0).collect::<Vec<_>>(),
            format!("{:?}", m.item), m.appeared,
        ])).collect();
        Some(json!({
            "format": "nc2000-preview-cache-info-v1",
            "build": nc2000_bot::compiled_preview::build_fingerprint(),
            "driver": self.driver,
            "config": self.config,
            "pool": self.pool,
            "prior": self.prior,
            "teamKey": team_key,
            "opponentPublic": public,
        }).to_string())
    }
}
