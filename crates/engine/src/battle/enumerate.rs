//! Exhaustive chance-node enumeration of one decision step (M17e).
//!
//! Replaces the seeded LCG with the Oracle mode of `BattleRng` and walks
//! every outcome-class combination of one `apply_choices` step by
//! recursive DFS over the recorded draw trace: each engine run executes
//! the step to completion under a script prefix (unscripted draws take the
//! first non-empty class); the first draw past the prefix is the branch
//! point, and each of its classes is explored in turn.
//!
//! Damage rolls ("droll" draws, 39 classes) are range-merged: if the
//! complete subtrees under the endpoint rolls of a range coincide
//! (leaf-by-leaf: same state_key, same downstream draw trace), damage's
//! monotonicity in the roll pins every interior roll to the same subtree,
//! so the whole range collapses into the endpoint's leaves at the range's
//! total mass — 2 engine explorations instead of 39 wherever overkill or
//! damage-irrelevance makes rolls indistinguishable (owner-approved
//! merge criterion, 2026-07-21). Non-coinciding ranges bisect.
//!
//! Leaf probabilities are products of per-draw exact rationals
//! (class-count / 2^32) accumulated in f64 — deterministic, no seed noise.

use std::collections::HashMap;
use std::rc::Rc;

use crate::dex::Dex;
use crate::prng::{BattleRng, Draw, Oracle};
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

struct LeafRec {
    battle: Battle,
    key: u64,
    prob: f64,
    trace: Rc<Vec<Draw>>,
}

struct Ctx<'a> {
    dex: &'a Dex,
    base: &'a Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
    runs: usize,
}

impl Ctx<'_> {
    fn run(&mut self, script: &[usize]) -> Option<(Battle, Vec<Draw>)> {
        if self.runs >= self.cap {
            return None;
        }
        self.runs += 1;
        let mut b = self.base.clone();
        b.set_log_enabled(false);
        b.prng = BattleRng::enumerating(script.to_vec());
        b.apply_choices(self.dex, self.choices).expect("enumerate_step: choices must be legal");
        let oracle = b.prng.oracle.take().expect("oracle survives the step");
        let Oracle { trace, .. } = *oracle;
        Some((b, trace))
    }
}

fn leaf(b: Battle, trace: Vec<Draw>) -> LeafRec {
    let key = b.state_key();
    let prob = trace.iter().map(Draw::prob).product();
    LeafRec { battle: b, key, prob, trace: Rc::new(trace) }
}

/// All leaves of the subtree whose draw choices are fixed by `script`.
fn subtree(ctx: &mut Ctx, script: &mut Vec<usize>) -> Option<Vec<LeafRec>> {
    let (b, trace) = ctx.run(script)?;
    if trace.len() <= script.len() {
        return Some(vec![leaf(b, trace)]);
    }
    let p = script.len();
    let d = &trace[p];
    let mut out = Vec::new();
    if d.label == "droll" {
        let mut cache: HashMap<usize, Rc<Vec<LeafRec>>> = HashMap::new();
        droll_ranges(ctx, script, p, 0, d.counts.len() - 1, &mut cache, &mut out)?;
    } else {
        let counts = d.counts.clone();
        for (c, &cnt) in counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            script.push(c);
            out.append(&mut subtree(ctx, script)?);
            script.pop();
        }
    }
    Some(out)
}

/// Two subtrees coincide iff they have the same leaves in DFS order: equal
/// state and equal downstream chance structure past the droll at `p`.
fn coincide(a: &[LeafRec], b: &[LeafRec], p: usize) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.key == y.key
                && x.trace.len() == y.trace.len()
                && x.trace[p + 1..].iter().zip(&y.trace[p + 1..]).all(|(dx, dy)| {
                    dx.label == dy.label && dx.chosen == dy.chosen && dx.counts == dy.counts
                })
        })
}

#[allow(clippy::too_many_arguments)]
fn droll_ranges(
    ctx: &mut Ctx,
    script: &mut Vec<usize>,
    p: usize,
    a: usize,
    b: usize,
    cache: &mut HashMap<usize, Rc<Vec<LeafRec>>>,
    out: &mut Vec<LeafRec>,
) -> Option<()> {
    fn endpoint(
        ctx: &mut Ctx,
        script: &mut Vec<usize>,
        cache: &mut HashMap<usize, Rc<Vec<LeafRec>>>,
        c: usize,
    ) -> Option<Rc<Vec<LeafRec>>> {
        if let Some(s) = cache.get(&c) {
            return Some(s.clone());
        }
        script.push(c);
        let s = Rc::new(subtree(ctx, script)?);
        script.pop();
        cache.insert(c, s.clone());
        Some(s)
    }

    let sa = endpoint(ctx, script, cache, a)?;
    if a == b {
        out.extend(sa.iter().map(|l| LeafRec {
            battle: l.battle.clone(),
            key: l.key,
            prob: l.prob,
            trace: l.trace.clone(),
        }));
        return Some(());
    }
    let sb = endpoint(ctx, script, cache, b)?;
    if coincide(&sa, &sb, p) {
        // Merge [a, b]: reweight position p from class a's mass to the
        // range's total mass (per-class counts differ by ±1, so divide the
        // old factor out exactly).
        let counts = &sa.first().map(|l| l.trace[p].counts.clone()).unwrap_or_default();
        let range_mass: u64 = counts[a..=b].iter().sum();
        let scale = range_mass as f64 / counts[a] as f64;
        out.extend(sa.iter().map(|l| LeafRec {
            battle: l.battle.clone(),
            key: l.key,
            prob: l.prob * scale,
            trace: l.trace.clone(),
        }));
        return Some(());
    }
    let m = (a + b) / 2;
    droll_ranges(ctx, script, p, a, m, cache, out)?;
    droll_ranges(ctx, script, p, m + 1, b, cache, out)
}

/// Enumerate every chance outcome of applying `choices` (legal, from
/// `legal_choices`) to `base`, which must sit at a decision point. Returns
/// `None` if more than `cap` engine runs would be needed (e.g. Metronome).
///
/// Panics if the choices are rejected — feeding legal choices is the
/// caller's contract.
pub fn enumerate_step(
    dex: &Dex,
    base: &Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
) -> Option<Vec<ChanceLeaf>> {
    let mut ctx = Ctx { dex, base, choices, cap, runs: 0 };
    let mut script = Vec::new();
    let recs = subtree(&mut ctx, &mut script)?;
    Some(
        recs.into_iter()
            .map(|l| ChanceLeaf { battle: l.battle, prob: l.prob, draws: l.trace.len() })
            .collect(),
    )
}
