# Certified Endgame Solver: Search-Space Reduction Proposal

Status: design proposal; no implementation is included in this document.

Audience: AI agents working on M17e or later exact/certified endgame analysis.

## Executive decision

Do not try to make exhaustive enumeration fast enough by optimizing the current
DFS alone. Use two complementary changes, in this order:

1. Quotient the 39 raw Gen 2 damage rolls by the integer damage value actually
   observed by the rest of the engine.
2. Replace all-or-nothing backward induction with a persistent, lazy,
   lower/upper-bound search that expands only the frontier relevant to the root
   certificate.

Then add simultaneous-move matrix pruning. Keep sampling-based methods as an
explicitly approximate fallback, not as the source of a deterministic
certificate.

The exact HP values created by healthy-position damage rolls are often genuine,
decision-relevant states. No exact memo key can merge all of them. The main
algorithmic win must therefore come from avoiding irrelevant expansion, not
from pretending that all roll-separated states are equivalent.

## Current bottleneck

The current pipeline is:

1. `BattleRng` exposes each damage roll as one of 39 Oracle classes.
2. `enumerate_step` eagerly returns every chance leaf for a joint action.
3. `ExactSolver::ival` does this for every matrix cell.
4. It recursively resolves every aggregated successor before it can back up
   the cell.
5. A budget failure returns `None`; the unfinished node does not retain a
   useful partial interval.
6. The interval memo is cleared between iterative-deepening horizons.

Relevant code:

- `crates/engine/src/prng.rs`: `BattleRng::damage_roll`
- `crates/engine/src/battle/moveexec.rs`: the random-factor calculation in
  `get_damage`
- `crates/engine/src/battle/enumerate.rs`: eager chance enumeration and the
  current endpoint range merge
- `crates/bot/src/exact.rs`: full matrix-cell enumeration, recursive interval
  backup, and horizon-local memoization

The observed healthy-1v1 wall is expected: two attacks may produce a raw
`39 * 39 = 1,521` roll fan before accuracy, critical hits, secondary effects,
Quick Claw, or later turns are considered. Exact HP pairs often remain distinct
until lethal or saturation ranges are reached.

## Phase A: exact damage-outcome quotienting

### Key observation

The engine does not observe the raw roll after this calculation:

```text
final_damage = floor(pre_roll_damage * roll / 255)
roll in 217..=255
```

Therefore, two raw rolls yielding the same `final_damage` are exactly
observationally equivalent at the RNG boundary. They may be merged before any
downstream callback or state mutation runs.

For integer pre-roll damage `D`, the number of distinct outcomes is at most:

```text
D - floor(217 * D / 255) + 1
```

Examples:

| Pre-roll damage | Raw classes | Distinct integer damages | Two-attack fan |
|---:|---:|---:|---:|
| 40 | 39 | 7 | 49 instead of 1,521 |
| 100 | 39 | 16 | 256 instead of 1,521 |
| >= 255 | 39 | at most 39 | no guaranteed reduction |

### Proposed API

Replace the split responsibility of `damage_roll()` plus the following
multiplication/floor with one operation whose output is the value visible to
the engine:

```rust
pub fn apply_damage_variance(&mut self, pre_roll_damage: i32) -> i32;
```

Seeded mode:

1. Consume exactly one raw `random(39)` draw, preserving PS bit parity.
2. Compute `floor(D * (217 + raw) / 255)` exactly as today.

Oracle mode:

1. Compute the final integer damage for every raw class `0..39`.
2. Group contiguous raw classes with equal final damage.
3. Sum their existing `uniform_count(39, raw)` masses. Do not substitute an
   assumed `1/39`; raw u32 counts differ by at most one and the existing exact
   partition should remain the source of truth.
4. Branch over distinct final damages and return that damage directly.

This quotient is sound by construction: all later engine code receives the
same integer in every raw world inside one class.

### Relationship to the current droll range merge

The committed endpoint range merge recursively evaluates complete subtrees for
two roll endpoints, compares their leaves, and bisects on mismatch. It is useful
in lethal/saturation ranges, but it is not the preferred first reduction:

- On healthy positions, unequal endpoints force recursive bisection and add
  internal exploration overhead.
- Equal integer-damage classes can be discovered arithmetically without
  executing any downstream subtree.
- Monotonicity of the scalar damage roll alone does not establish monotonicity
  of the complete engine transition in the presence of threshold callbacks,
  item state, recoil, drain, Substitute, Counter/Mirror Coat bookkeeping, and
  other side effects. A certificate-quality endpoint merge needs a documented
  domain proof or a restriction to mechanically proven saturation cases.

Keep the endpoint merge only as an optional second layer after integer-damage
quotienting, and benchmark it separately. It should be disabled automatically
when its probes cost more than the leaves it removes.

## Phase B: lazy certified interval search

### Required semantic change

An unexpanded chance branch is not an error. It is probability mass with the
sound value interval `[0, 1]`.

For a joint-action cell with expanded successors `k` and unresolved total mass
`p_pending`:

```text
Q_lo = sum_k p_k * L(child_k)
Q_hi = sum_k p_k * U(child_k) + p_pending
```

For lower and upper payoff matrices `Q_lo` and `Q_hi`:

```text
L(state) = value(Q_lo)
U(state) = value(Q_hi)
```

Matrix-game value is monotone in every payoff entry, so this is the same sound
bracketing principle already used by `ExactSolver`, applied incrementally.

The solver must always preserve and return the best current root interval. A
work limit should stop expansion, not discard a partially solved node.

### Lazy Oracle frontier

Do not make `enumerate_step` return a complete `Vec<ChanceLeaf>` before the
solver can inspect bounds. Represent unexplored Oracle scripts explicitly:

```rust
struct PendingChance {
    script: Vec<usize>,
    mass: ExactMass,
}

struct ChanceCell {
    resolved: Vec<(ExactMass, StateId)>,
    pending: Vec<PendingChance>,
}
```

One expansion of a pending script should:

1. Run the synchronous engine once, letting unscripted draws choose the first
   non-empty class as today.
2. Return the resulting decision-state leaf for that complete representative
   trace.
3. Add pending sibling scripts for every non-chosen class encountered after
   the supplied prefix.
4. Assign each pending sibling the exact prefix probability obtained from the
   Oracle trace.

The representative leaf plus all pending sibling subtrees exactly partition
the original pending mass. If fully exhausted, this produces the same leaf
distribution as eager lexicographic enumeration. If stopped early, unresolved
mass remains a valid `[0, 1]` contribution.

This formulation preserves approximately one full engine run per resolved
leaf when exhausted; it does not require a separate run for every internal
chance-tree node.

### Persistent search graph

Replace recursive, horizon-local `ival_memo` with graph nodes that survive work
limits and later calls:

```rust
struct BoundNode {
    lo: f64,
    hi: f64,
    cells: Vec<ChanceCell>,
    status: NodeStatus,
}
```

Properties:

- Terminal outcomes are `[v, v]`.
- Unexpanded nonterminal states start at `[0, 1]`.
- Every expansion performs Bellman/LP backups to the root.
- Bounds are monotone: `lo` never decreases and `hi` never increases, modulo
  tightly controlled floating-point tolerance.
- Fully resolved nodes enter the persistent exact memo.
- Partially resolved nodes remain reusable across budgets, roots, and corpus
  positions.
- Iterative horizons should no longer clear useful work. A depth cap may stop
  scheduling deeper nodes while leaving them as persistent frontier nodes.

The actual game is finite because the state includes the turn and the engine
ties after turn 1000. The implementation may retain a depth policy for
practical scheduling, but it need not throw away the graph at each horizon.

### Frontier selection

Correctness does not depend on the heuristic as long as expansion remains fair
when a tighter interval is requested. Performance does.

Use estimated root impact:

```text
priority = root_reach * pending_mass * (child_hi - child_lo)
```

At simultaneous nodes, obtain strategies from both the lower-matrix and
upper-matrix LP solutions. Prefer cells in:

- either solution's support;
- current best-response rows or columns;
- cells responsible for the largest lower/upper exploitability gap.

Extend `solve_matrix` to return both players' strategies, not only value and
numeric certification gap. Alternate work aimed at raising the lower bound and
lowering the upper bound so one side does not starve.

This is an adaptation of bounded value iteration / trial-based heuristic
search to the repository's concurrent, simultaneous-move state graph.

### Threshold certificate mode

The evaluation-audit harness usually does not need a tight numeric equity. It
needs to prove that the evaluation is materially wrong.

Given `tau_hi = eval + margin` and `tau_lo = eval - margin`, stop as soon as:

```text
L(root) > tau_hi    // proven underestimate
U(root) < tau_lo    // proven overestimate
```

Schedule frontier nodes specifically to answer that threshold query. This is
the chance/simultaneous analogue of null-window alpha-beta search. It should be
the default mode for the M17c regression gate and corpus violation mining.

Use full `U(root) - L(root) <= epsilon` solving only when a numeric equity is
actually required.

## Phase C: simultaneous-move matrix pruning

After chance expansion becomes lazy, avoid resolving every joint-action cell.

Applicable methods:

1. Maintain payoff intervals per cell and remove provably dominated rows or
   columns using LP feasibility.
2. Start with a restricted stage game and add best-response actions (double
   oracle) until neither side can improve.
3. Use the two serialized move orders as lower and upper bounds: giving one
   player knowledge of the other's current action bounds the simultaneous
   game.

The 1v1 action matrix is often only about `4 x 4`, so this is lower priority
than chance quotienting and lazy expansion. It becomes more important in 2v2,
switch-heavy, or larger-support states.

Primary references:

- Bruce W. Ballard, [The *-minimax search procedure for trees containing chance nodes](https://doi.org/10.1016/S0004-3702(83)80015-0)
- Saffidine, Finnsson, and Buro, [Alpha-Beta Pruning for Games with Simultaneous Moves](https://doi.org/10.1609/aaai.v26i1.8148)
- Bosansky et al., [Using Double-Oracle Method and Serialized Alpha-Beta Search for Pruning in Simultaneous Move Games](https://www.ijcai.org/Proceedings/13/Papers/018.pdf)
- Eisentraut, Kretinsky, and Rotar, [Stopping Criteria for Value and Strategy Iteration on Concurrent Stochastic Reachability Games](https://arxiv.org/abs/1909.08348)
- Keller and Helmert, [Trial-Based Heuristic Tree Search for Finite Horizon MDPs](https://doi.org/10.1609/icaps.v23i1.13557)

## Optional exact state reductions

These are secondary optimizations. Apply them only with an explicit future-use
audit.

### Remove semantically dead fields from the solver key

The exact `state_key` hashes several damage-bookkeeping fields. Some fields are
currently written/reset but never read by future engine behavior; others are
observable only by specific moves such as Counter or Mirror Coat.

A solver-specific semantic key may omit a field only when one of these holds:

- repository-wide code analysis proves the field is never read; or
- a state-local liveness guard proves no reachable callback or move can observe
  it before overwrite.

Do not use `state_key_bucketed` for certificates. HP bucketing is an intentional
search abstraction and can change the game value.

### Normalize relative time for cross-root reuse

Absolute turn values prevent otherwise identical corpus positions from sharing
memo entries. A future canonical key may normalize the current turn and every
turn-stamped field to relative offsets, while retaining distance to the
turn-1000 tie. This requires a complete audit of `turn`, `dragged_in`, Future
Sight state, and all other absolute-turn consumers.

## Correctness requirements

### State identity

`state_key()` is a 64-bit hash designed for search statistics, where a collision
is tolerable. It is not, by itself, a mathematical equality proof.

For a certificate-producing solver, use one of:

- a canonical structural state with equality;
- a hash map whose collision bucket performs structural equality; or
- a wider fingerprint plus a debug/full-equality certification pass.

The same requirement applies to endpoint-subtree coincidence checks.

### Probability representation

Each individual Oracle draw has an exact denominator of `2^32`, but products
are currently accumulated in `f64`. Retain the existing behavior initially for
performance, while reporting numeric error bounds separately from game-search
bounds. If the artifact is advertised as formally exact, introduce an exact or
certified probability accumulator.

### Work accounting

After the committed range-merge refactor, `enumerate_step` caps actual engine
runs through `Ctx::runs`, while `ExactSolver` increments `chance_runs` by the
number of returned leaves. These quantities differ when probing or merging
ranges.

Expose and record at least:

- actual full engine executions;
- resolved chance leaves;
- pending chance prefixes;
- expanded decision states;
- LP solves;
- wall time;
- root interval width.

Budget enforcement and benchmark reports must use actual engine executions or
wall time, not returned leaf count mislabeled as runs.

## Approximate fallback

If healthy-position numeric equity remains too expensive after Phases A-C,
separate the product into two modes:

- `certified`: deterministic lower/upper interval, possibly wide;
- `estimated`: stratified chance sampling with an explicit statistical
  confidence interval.

For estimated mode, stratify the discrete non-damage events and the quotient
damage classes instead of sampling raw seeds blindly. Common random numbers
may improve action comparisons, but confidence calculations must account for
the coupling. Chance-sampled MCCFR is a literature-backed option for approximate
equilibria, but the repository's earlier online regret-matching results make it
a fallback rather than the first implementation target.

Reference: Lanctot et al., [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html).

Never mix statistical confidence with deterministic certification in one
unqualified `exact` field.

## Implementation sequence

1. Fix run/leaf accounting so every later benchmark is interpretable.
2. Implement integer-damage quotienting in the RNG/damage boundary.
3. Differentially certify seeded parity and full-distribution parity.
4. Introduce a lazy Oracle pending-prefix API while retaining the eager API as
   a test oracle.
5. Convert `ExactSolver` to persistent partial intervals; never return `None`
   merely because work stopped.
6. Add threshold certificate mode to the corpus harness.
7. Add frontier scheduling by probability-weighted uncertainty.
8. Return LP strategies and add simultaneous-move cell pruning.
9. Reassess the endpoint range merge after isolated A/B measurement.
10. Only then consider solver-specific semantic state keys or approximate mode.

## Acceptance gates

### Correctness

- Seeded battle and PRNG fixture parity remains bit-exact.
- Every Oracle partition sums to `2^32`.
- On a fixed micro-corpus, old eager enumeration and fully exhausted lazy
  enumeration produce the same successor-state probability map.
- Integer-damage quotienting produces that same map with fewer engine runs.
- On small fully solvable games, every partial root interval contains the
  exhaustive value and shrinks monotonically.
- Threshold proofs agree with exhaustive values on all microgames.
- No certificate uses HP-bucketed identity.

### Performance

Measure separately by state class:

- lethal 1v1;
- healthy 1v1;
- 2v1 and 2v2;
- Counter/Mirror Coat/Bide/Substitute/item-threshold cases;
- multi-hit and high-randomness moves.

For every benchmark report:

```text
wall time
actual engine executions
resolved leaves
expanded decision states
root [lo, hi]
proof target, if any
```

The primary gate is not a universal speedup. It is materially more certified
information per engine execution on the real human corpus, especially the
number and margin of proven evaluation violations.

## Expected outcome

Integer-damage quotienting should remove much of the artificial 39-way fan for
ordinary attacks. Lazy bounded search should then prevent the remaining genuine
HP states from being expanded unless they can affect the root interval or the
current threshold proof. Together they directly address both sources of the
explosion: redundant chance outcomes and irrelevant yet genuine states.
