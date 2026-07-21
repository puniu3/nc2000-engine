//! Exhaustive chance-node enumeration of one decision step (M17e).
//!
//! Replaces the seeded LCG with the Oracle mode of `BattleRng` and walks
//! every outcome-class combination of one `apply_choices` step by
//! lexicographic DFS over the recorded draw trace: each run executes the
//! step to completion with a script prefix (unscripted draws take the first
//! non-empty class), then the deepest trace position with an unexplored
//! non-empty sibling class is advanced. Number of engine executions =
//! number of chance leaves.
//!
//! Leaf probabilities are products of per-draw exact rationals
//! (class-count / 2^32) accumulated in f64 — deterministic, no seed noise.

use crate::dex::Dex;
use crate::prng::{BattleRng, Draw};
use crate::state::Battle;

use super::search::SearchChoice;

/// One chance outcome of a decision step. `battle` has advanced to the next
/// request point (or ended); its RNG is a spent Oracle — `reseed` it before
/// any further seeded play.
pub struct ChanceLeaf {
    pub battle: Battle,
    pub prob: f64,
    /// Chance draws consumed along this path.
    pub draws: usize,
}

/// Enumerate every chance outcome of applying `choices` (legal, from
/// `legal_choices`) to `base`, which must sit at a decision point. Returns
/// `None` if more than `cap` leaves would be produced (e.g. Metronome).
///
/// Panics if the choices are rejected — feeding legal choices is the
/// caller's contract.
pub fn enumerate_step(
    dex: &Dex,
    base: &Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
) -> Option<Vec<ChanceLeaf>> {
    let mut leaves: Vec<ChanceLeaf> = Vec::new();
    let mut script: Vec<usize> = Vec::new();
    loop {
        if leaves.len() >= cap {
            return None;
        }
        let mut b = base.clone();
        b.set_log_enabled(false);
        b.prng = BattleRng::enumerating(std::mem::take(&mut script));
        b.apply_choices(dex, choices).expect("enumerate_step: choices must be legal");
        let oracle = b.prng.oracle.take().expect("oracle survives the step");
        let prob = oracle.trace.iter().map(Draw::prob).product();
        leaves.push(ChanceLeaf { battle: b, prob, draws: oracle.trace.len() });

        // Lexicographic successor: deepest position with an unexplored
        // non-empty class; zero-count classes are skipped (they'd enumerate
        // impossible worlds).
        let t = &oracle.trace;
        let next = t.iter().enumerate().rev().find_map(|(i, d)| {
            d.counts[d.chosen + 1..]
                .iter()
                .position(|&c| c > 0)
                .map(|off| (i, d.chosen + 1 + off))
        });
        match next {
            None => return Some(leaves),
            Some((i, class)) => {
                script.extend(t[..i].iter().map(|d| d.chosen));
                script.push(class);
            }
        }
    }
}
