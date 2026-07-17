//! nc2000-bot: baseline agents + DUCT MCTS on the engine's search API.
//! M5: Random / MaxDamage / uniform-rollout MCTS. M6: heavy playouts
//! (ε-greedy max-damage rollout policy, truncation, weighted static eval)
//! plus the duel harness + SPSA tuner that set the eval weights. M7: mixed
//! strategies — state-keyed SM-MCTS with regret matching (`smmcts`) and the
//! best-response exploitability probe (`exploit`). M8: baked team-preview
//! tables (`preview`) — offline-solved mixed equilibria over the meta pool,
//! consumed by `BakedPreviewAgent` and probed by `CounterPickAgent`. M10a:
//! imperfect-information machinery (`observe` + `belief`) — observation
//! tracker, belief over the meta pool, and the hidden-field determinizer.
//! M10b: `BlindAgent` (`blind`) — the skuct search restricted to the
//! observe/belief surface via per-iteration determinization. M10c:
//! `BlindSearch` — its stepped form (the wasm/ponder substrate), driven
//! internally by `BlindAgent`.
//!
//! Agents see the full battle state (both teams) — self-play evaluation
//! mode — except `BlindAgent`, which restricts itself to the
//! `observe`/`belief` surface plus determinized clones.

pub mod agent;
pub mod belief;
pub mod blind;
pub mod duel;
pub mod eval;
pub mod exploit;
pub mod mcts;
pub mod observe;
pub mod preview;
pub mod rng;
pub mod runner;
pub mod smmcts;

pub use agent::{Agent, MaxDamageAgent, RandomAgent};
pub use belief::Belief;
pub use blind::{baked_preview_pick, open_preview_pick, BlindAgent, BlindSearch};
pub use observe::{ItemObs, MonObs, Observer};
pub use duel::{run_duel, DuelSpec, DuelStats};
pub use eval::EvalWeights;
pub use exploit::BrAgent;
pub use mcts::{MctsAgent, MctsConfig, Playout};
pub use preview::{BakedPreviewAgent, CounterPickAgent, PreviewMode, TableSet};
pub use rng::SplitMix64;
pub use runner::{play_game, GameResult};
pub use smmcts::{RmAgent, RmConfig, SkuctSearch};
