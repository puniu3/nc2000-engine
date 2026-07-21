# Threshold-preserving damage-roll experiment

Status: ACTIVE

## Contract

Measure, before any product integration, how much a selective damage-roll abstraction changes:

1. value/policy accuracy against exact damage enumeration;
2. engine executions, expanded states, wall time, and memory;
3. seed-paired duel score at fixed work and fixed wall-clock.

The approximation starts with one probability-weighted representative damage. It splits only when rolls differ on a battle-relevant threshold (KO/survival, immediate semantic HP predicates, two-hit class, or residual death clock). Damage-sensitive mechanics fall back to exact damage classes.

## Controls

- Branch/worktree: `exp/threshold-damage` at `/home/puniu/nc2000-threshold-exp`
- Base: `cc660bc`
- Current exact implementation remains the control.
- Existing certified anchors remain ground truth; healthy positions use matched finite-horizon exact enumeration.
- Never label representative-mode output `exact` or `certified`.

## Planned variants

- `exact`: current distinct-integer-damage Oracle classes.
- `mean`: one conditional representative, except forced exact fallbacks.
- `threshold1`: split on immediate semantic thresholds.
- `threshold2`: add next-hit and residual-clock classes.

## Required metrics

- value error against the exact bracket/matrix;
- root policy exploitability/action regret;
- exact/abstract chance classes and engine executions;
- expanded decision states, LP solves, wall time, peak RSS;
- seed-paired duel score with 95% CI and think ms/move.

## Progress log

- 2026-07-22: isolated worktree created from committed e-5 baseline.
- 2026-07-22: implemented Oracle-only `mean`/`threshold1`/`threshold2` damage partitions. Seeded play and the existing exact enumeration entry points remain on `exact`.
- 2026-07-22: damage-sensitive mechanics (Counter/Mirror Coat/Bide, drain/recoil, multi-hit, Substitute) fall back to exact damage classes.
- 2026-07-22: added matched finite-horizon value/policy benchmark. A 12-position local horizon-0 smoke completed 10 exact controls; threshold2 used about 9% of exact engine runs on completed roots, with value MAE 0.00011 and zero measured root-policy regret. Two exact roots exhausted the 2M-run cap; treat this only as a harness smoke, not the result.
- 2026-07-22: added a seed-paired duel wrapper: identical SM-MCTS fallback outside 1v1 endgames, exact/abstract finite-horizon search inside, fallback again on work exhaustion. Four-game wiring smoke was 2-2; threshold2 used 10.5 ms/move versus exact 71.2 ms/move at 20k work. Sample is intentionally too small for a strength claim.
- 2026-07-22: full engine/conformance suite passed, including bit-exact replay corpora; the production seeded RNG path is unchanged.
- cx task `20260722-013928` is the current horizon-0 certified-anchor run (12 CPU declaration after larger Spot stockouts). Its outputs return under `/home/puniu/cx/results/20260722-013928/out/`.
- Next writer: do not change the abstraction contract while cx benchmark jobs are active. Inspect this log and `cx status` first.
