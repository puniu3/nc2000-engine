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

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use crate::dex::Dex;
use crate::prng::{BattleRng, DamageRollMode, Draw, Oracle};
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

/// Enumeration result with honest work accounting: `runs` counts actual
/// engine executions (probes + endpoint evaluations included), which is
/// what budgets and benchmarks must use — resolved leaf count is not work.
pub struct StepEnum {
    pub leaves: Vec<ChanceLeaf>,
    pub runs: usize,
    /// Diagnostic observations across scripted transitions, including cache
    /// hits. These are not unique events: probes revisit the same prefix.
    pub damage: DamageEnumStats,
}

pub struct ChancePrefix {
    pub script: Vec<usize>,
    pub mass: f64,
}

pub struct PartialStepEnum {
    pub leaves: Vec<ChanceLeaf>,
    pub pending: Vec<ChancePrefix>,
    pub runs: usize,
    pub exhausted: bool,
}

#[derive(Default)]
struct TracePartition {
    counts: Option<Vec<u64>>,
    children: BTreeMap<usize, TracePartition>,
    terminal: bool,
}

impl TracePartition {
    fn insert(&mut self, trace: &[Draw]) {
        let mut node = self;
        for draw in trace {
            assert!(!node.terminal);
            if let Some(counts) = &node.counts { assert_eq!(counts, &draw.counts); }
            else { node.counts = Some(draw.counts.clone()); }
            node = node.children.entry(draw.chosen).or_default();
        }
        assert!(node.counts.is_none());
        node.terminal = true;
    }

    fn pending(&self, prefix: &mut Vec<usize>, mass: f64, out: &mut Vec<ChancePrefix>) {
        if self.terminal { return; }
        let Some(counts) = &self.counts else {
            out.push(ChancePrefix { script: prefix.clone(), mass });
            return;
        };
        for (class, &count) in counts.iter().enumerate().filter(|(_, n)| **n > 0) {
            prefix.push(class);
            let mass = mass * count as f64 / Draw::TOTAL as f64;
            if let Some(child) = self.children.get(&class) { child.pending(prefix, mass, out); }
            else { out.push(ChancePrefix { script: prefix.clone(), mass }); }
            prefix.pop();
        }
    }
}

/// Leaves and pending prefixes partition the complete chance mass without overlap.
pub fn enumerate_step_partial(
    dex: &Dex, base: &Battle, choices: [Option<SearchChoice>; 2], cap: usize,
) -> PartialStepEnum {
    let mut observed = HashMap::new();
    let mut runs = 0;
    let complete = {
        let mut runner = |script: &[usize]| {
            let (battle, trace) = run_scripted(dex, base, choices, script);
            runs += 1;
            let path: Vec<_> = trace.iter().map(|d| d.chosen).collect();
            observed.entry(path).or_insert_with(|| (battle.clone(), trace.clone()));
            (battle, trace, true)
        };
        enumerate_step_with_runner(dex, base, choices, cap, &mut runner)
    };
    if let Some(step) = complete {
        assert_eq!(step.runs, runs);
        return PartialStepEnum { leaves: step.leaves, pending: Vec::new(), runs, exhausted: false };
    }
    let mut observed: Vec<_> = observed.into_iter().collect();
    observed.sort_by(|a,b| a.0.cmp(&b.0));
    let mut partition = TracePartition::default();
    let leaves = observed.into_iter().map(|(_, (battle, trace))| {
        partition.insert(&trace);
        ChanceLeaf { battle, prob: trace.iter().map(Draw::prob).product(), draws: trace.len() }
    }).collect::<Vec<_>>();
    let mut pending = Vec::new();
    partition.pending(&mut Vec::new(), 1., &mut pending);
    assert!((leaves.iter().map(|l|l.prob).sum::<f64>() + pending.iter().map(|p|p.mass).sum::<f64>() - 1.).abs()<1e-9);
    PartialStepEnum { leaves, pending, runs, exhausted: true }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DamageEnumStats {
    pub exact_draws: usize,
    pub abstract_draws: usize,
    pub offered_classes: usize,
    pub drain_recoil_draws: usize,
    pub multihit_draws: usize,
    pub substitute_draws: usize,
    pub counter_bide_draws: usize,
    pub heal_draws: usize,
}

struct LeafRec {
    battle: Battle,
    key: u128,
    prob: f64,
    trace: Rc<Vec<Draw>>,
}

/// One scripted engine execution of a decision step: applies `choices` to a
/// clone of `base` under the Oracle following `script` (draws past the
/// script take the first non-empty class) and returns the advanced battle
/// plus the full draw trace. The lazy bound solver's expansion atom (e-5);
/// the eager enumerator below remains as its test oracle.
pub fn run_scripted(
    dex: &Dex,
    base: &Battle,
    choices: [Option<SearchChoice>; 2],
    script: &[usize],
) -> (Battle, Vec<Draw>) {
    run_scripted_with_damage_mode(dex, base, choices, script, DamageRollMode::Exact)
}

/// Experimental counterpart of `run_scripted`. The production exact path
/// above always fixes `Exact`; callers must opt into damage abstraction.
pub fn run_scripted_with_damage_mode(
    dex: &Dex,
    base: &Battle,
    choices: [Option<SearchChoice>; 2],
    script: &[usize],
    damage_mode: DamageRollMode,
) -> (Battle, Vec<Draw>) {
    let mut b = base.clone();
    b.set_log_enabled(false);
    b.prng = BattleRng::enumerating_with_damage_mode(script.to_vec(), damage_mode);
    b.apply_choices(dex, choices).expect("run_scripted: choices must be legal");
    let oracle = b.prng.oracle.take().expect("oracle survives the step");
    let Oracle { trace, .. } = *oracle;
    (b, trace)
}

type ScriptRunner<'a> = dyn FnMut(&[usize]) -> (Battle, Vec<Draw>, bool) + 'a;

struct Ctx<'a> {
    dex: &'a Dex,
    base: &'a Battle,
    choices: [Option<SearchChoice>; 2],
    damage_mode: DamageRollMode,
    runner: Option<&'a mut ScriptRunner<'a>>,
    cap: usize,
    runs: usize,
    damage: DamageEnumStats,
}

impl Ctx<'_> {
    fn run(&mut self, script: &[usize]) -> Option<(Battle, Vec<Draw>)> {
        if self.runs >= self.cap {
            return None;
        }
        let result = if let Some(runner) = self.runner.as_mut() {
            let (battle, trace, executed) = runner(script);
            self.runs += usize::from(executed);
            (battle, trace)
        } else {
            self.runs += 1;
            run_scripted_with_damage_mode(self.dex, self.base, self.choices, script, self.damage_mode)
        };
        for draw in &result.1 {
            match draw.label {
                "dmgvar" => self.damage.exact_draws += 1,
                "dmgvar-drain-recoil" => {
                    self.damage.exact_draws += 1;
                    self.damage.drain_recoil_draws += 1;
                }
                "dmgvar-multihit" => {
                    self.damage.exact_draws += 1;
                    self.damage.multihit_draws += 1;
                }
                "dmgvar-substitute" => {
                    self.damage.exact_draws += 1;
                    self.damage.substitute_draws += 1;
                }
                "dmgvar-counter-bide" => {
                    self.damage.exact_draws += 1;
                    self.damage.counter_bide_draws += 1;
                }
                "dmgvar-heal" => {
                    self.damage.exact_draws += 1;
                    self.damage.heal_draws += 1;
                }
                "dmgabs" => self.damage.abstract_draws += 1,
                _ => continue,
            }
            self.damage.offered_classes += draw.counts.len();
        }
        Some(result)
    }
}

fn leaf(b: Battle, trace: Vec<Draw>) -> LeafRec {
    let key = b.state_key128();
    let prob = trace.iter().map(Draw::prob).product();
    LeafRec { battle: b, key, prob, trace: Rc::new(trace) }
}

/// All leaves of the subtree whose draw choices are fixed by `script`.
fn subtree(ctx: &mut Ctx, script: &mut Vec<usize>) -> Option<Vec<LeafRec>> {
    let (b, trace) = ctx.run(script)?;
    subtree_from_run(ctx, script, b, trace)
}

fn subtree_from_run(
    ctx: &mut Ctx,
    script: &mut Vec<usize>,
    b: Battle,
    trace: Vec<Draw>,
) -> Option<Vec<LeafRec>> {
    if trace.len() <= script.len() {
        return Some(vec![leaf(b, trace)]);
    }
    let p = script.len();
    let d = &trace[p];
    let mut out = Vec::new();
    if d.label.starts_with("dmgvar") {
        let chosen = d.chosen;
        let last = d.counts.len() - 1;
        let mut cache: HashMap<usize, Rc<Vec<LeafRec>>> = HashMap::new();
        script.push(chosen);
        let known = subtree_from_run(ctx, script, b, trace)?;
        script.pop();
        cache.insert(chosen, Rc::new(known));
        droll_ranges(ctx, script, p, 0, last, &mut cache, &mut out)?;
    } else {
        let counts = d.counts.clone();
        let chosen = d.chosen;
        let mut known = Some((b, trace));
        for (c, &cnt) in counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            script.push(c);
            let mut leaves = if c == chosen {
                let (b, trace) = known.take().unwrap();
                subtree_from_run(ctx, script, b, trace)?
            } else {
                subtree(ctx, script)?
            };
            out.append(&mut leaves);
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
) -> Option<StepEnum> {
    enumerate_step_with_damage_mode(dex, base, choices, cap, DamageRollMode::Exact)
}

/// Enumerate one decision step while quotienting damage rolls according to
/// `damage_mode`. This is approximate unless `damage_mode == Exact`.
pub fn enumerate_step_with_damage_mode(
    dex: &Dex,
    base: &Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
    damage_mode: DamageRollMode,
) -> Option<StepEnum> {
    enumerate_with_runner(dex, base, choices, cap, damage_mode, None)
}

/// The runner must return a semantically equivalent scripted successor
/// and the identical chance trace, with
/// `executed` true iff it performed an engine execution. Cached calls may
/// cost zero; a depleted cap still stops before invoking the runner.
pub fn enumerate_step_with_runner(
    dex: &Dex,
    base: &Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
    runner: &mut ScriptRunner<'_>,
) -> Option<StepEnum> {
    enumerate_with_runner(dex, base, choices, cap, DamageRollMode::Exact, Some(runner))
}

fn enumerate_with_runner<'a>(
    dex: &'a Dex,
    base: &'a Battle,
    choices: [Option<SearchChoice>; 2],
    cap: usize,
    damage_mode: DamageRollMode,
    runner: Option<&'a mut ScriptRunner<'a>>,
) -> Option<StepEnum> {
    let mut ctx = Ctx {
        dex,
        base,
        choices,
        damage_mode,
        runner,
        cap,
        runs: 0,
        damage: DamageEnumStats::default(),
    };
    let mut script = Vec::new();
    let recs = subtree(&mut ctx, &mut script)?;
    Some(StepEnum {
        leaves: recs
            .into_iter()
            .map(|l| ChanceLeaf { battle: l.battle, prob: l.prob, draws: l.trace.len() })
            .collect(),
        runs: ctx.runs,
        damage: ctx.damage,
    })
}
