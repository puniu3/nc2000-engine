//! nc2000-bot (M5): baseline agents + DUCT MCTS on the engine's search API.
//!
//! Agents see the full battle state (both teams) — self-play evaluation mode.
//! Hidden-information play (opponent-set inference feeding determinization)
//! is a later milestone; the engine API already supports it via
//! `clone` + `reseed`.

pub mod agent;
pub mod mcts;
pub mod rng;
pub mod runner;

pub use agent::{Agent, MaxDamageAgent, RandomAgent};
pub use mcts::{MctsAgent, MctsConfig};
pub use rng::SplitMix64;
pub use runner::{play_game, GameResult};
