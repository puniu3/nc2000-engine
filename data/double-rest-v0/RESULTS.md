# Double-Rest search results

Research snapshot: 2026-07-22. The oracle corpus and split contract are in
`README.md`. All comparisons use one reconstructed final-1v1 position per
source battle (11 positions total), with both Rest moves proven from the full
log and future-revealed moves completed only in the offline harness.

## H0: one decision step

| arm | complete | engine runs vs exact | wall vs exact | value MAE / worst | worst policy regret |
|---|---:|---:|---:|---:|---:|
| exact | 11/11 | 100% | 100% | 0 / 0 | 0 |
| lean-minimal | 11/11 | 5.06% | 3.39% | 0.0000775 / 0.0005493 | 0 |
| one-shot probe/refine | 11/11 | 7.44% | 5.17% | 0.0000276 / 0.0002597 | 0 |
| two-round best-response audit | 11/11 | 11.64% | 8.16% | 0.0000276 / 0.0002597 | 0 |

The audit found no additional policy improvement over one-shot refinement on
this slice. Its extra action-cross probes are therefore optional evidence,
not the live default.

## H1: two decision steps

The exact arm had a 2,000,000-run budget per position. It completed only 8 of
11 positions and spent 283.5 seconds in total. Lean-minimal completed 11/11 in
4.50 seconds. On the eight positions where exact completed, lean-minimal used
2.85% of the runs and 2.44% of the wall time; value MAE was 0.0000186, worst
error 0.0001102, and policy regret was zero.

The two-round audit completed 10/11 in the pre-anytime implementation. On the
eight exact-matched positions it used 6.01% of exact runs and 5.15% of exact
wall time, with no policy improvement over lean-minimal. One unstable exact
cell made the b170 audit exceed its refinement budget. Refinement is now
anytime: an incomplete probe or exact cell keeps the last complete approximate
equilibrium and accounts for all consumed work instead of discarding it and
falling through to MCTS.

## Full-width PP census

The b170 Snorlax-Rest vs Zapdos-Rest root starts with 149 total PP. Exact
full-width expansion produced 4,965 successor states and 61,454 engine runs at
depth 1. A 12,000,000-run cx census reached only a capped 60,000-state depth-2
frontier after 300 seconds; the structural quotient had no early duplicate
hits and 60.1% probability mass was truncated. PP fell only from 149 to
147-148.

Therefore PP/resource layering is useful only after strategic branch pruning;
it is not a replacement for support selection. The live order is:

1. Solve the lean-minimal simultaneous-action matrix.
2. Probe only equilibrium support and pure best-response cells.
3. Exact-refine cells whose low/high representatives can change the policy.
4. Optionally audit excluded actions; stop anytime on budget exhaustion.
5. Reuse bounded LRU one-step transitions across reroots.
6. Apply PP/resource/SCC solving only to the small residual stall subgame.

## Double-Rest duel gate

The duel harness has `--pool rest` (57 frequency-weighted Rest sets) and
`--pool rest-talk` (21 Rest+Sleep Talk sets). It reports mean, p95, p99, and
turn-cap count.

A 100-game seed-paired Rest+Sleep Talk gate at H0 finished 48-48-4: score
0.5000 with 95% CI half-width 0.0965. One-shot probe/refine averaged 420.6 ms
per move versus 2,328.2 ms for exact (5.5x faster). Its p95 was 1,869 ms
versus 10,782 ms and p99 was 3,313 ms versus 17,882 ms. The four ties came
from the run immediately before separate turn-cap accounting was added; future
gates report that count explicitly.
