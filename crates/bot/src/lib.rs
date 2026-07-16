//! nc2000-bot: baseline agents + DUCT MCTS on the engine's search API.
//! M5: Random / MaxDamage / uniform-rollout MCTS. M6: heavy playouts
//! (ε-greedy max-damage rollout policy, truncation, weighted static eval)
//! plus the duel harness + SPSA tuner that set the eval weights.
//!
//! Agents see the full battle state (both teams) — self-play evaluation mode.
//! Hidden-information play (opponent-set inference feeding determinization)
//! is a later milestone; the engine API already supports it via
//! `clone` + `reseed`.

pub mod agent;
pub mod duel;
pub mod eval;
pub mod mcts;
pub mod rng;
pub mod runner;

pub use agent::{Agent, MaxDamageAgent, RandomAgent};
pub use duel::{run_duel, DuelSpec, DuelStats};
pub use eval::EvalWeights;
pub use mcts::{MctsAgent, MctsConfig, Playout};
pub use rng::SplitMix64;
pub use runner::{play_game, GameResult};
