# Unresolved endgame cases and tactical coverage

The audit found a measurement gap, not a demonstrated strength improvement.
All 59 previously unresolved archived decisions remain unresolved after a
targeted six-state refinement. No product policy changed.

## Census and case review

The frozen `policy-gap-v1` source has 24 self-play games, 72 sampled states,
and 115 actionable endpoints. Its 59 unresolved endpoints occupy 36 states
from 15 games. We reconstructed and inspected all 36 states and selected
20 endpoints from those 15 games for a search-budget probe. Selection was
manual and exploratory, frozen before the new searches; it is not an
independent holdout or a prevalence estimate. `case-notes.json` records each
observation and the question it raises.

| Feature | Unresolved endpoints | Resolved nonforced endpoints |
| --- | --- | --- |
| 2 vs 2 alive | 33/59 | 0/34 |
| 1 vs 1 alive | 2/59 | 16/34 |
| Replacement decision | 18/59 | 3/34 |
| Living Pokemon with healing PP | 51/59 | 28/34 |
| Living Pokemon with setup PP | 59/59 | 34/34 |

Healing and setup presence do not distinguish failure from success of the
reference solver here. These tags describe available kits, not actions
taken or causes of error. Original reference stops for the unresolved
endpoints: 46 node limit, 11 time limit, 2 work limit. Endpoints from one
state or game are correlated; no confidence intervals use them as
independent samples.

## Additional computation

Each selected endpoint received four independent search seeds at nested
27,000 / 108,000 / 432,000 iterations, full-information SkuctSearch c=0.4.
Ten of 80 seed-endpoint pairs changed their action between the first and
last budget. One high-budget empirical mixture is certified within 2
percentage points, and 19 remain unresolved. Search means and action
changes are not ground truth. The four-seed mixture is measured exactly
against the bounded matrix, but its difference from the infinite-seed bot
policy is not bounded. Concurrent times are not 10/60-second device tests.

Reference refinement used six partially resolved states, with 180 seconds,
600,000 graph/cache entries, and 8,000,000 engine executions per state.
Five stopped at time and one at the node limit. Old and new state-value
and payoff-cell intervals were intersected after checking state identity
and action order. Selected archived-loss upper bounds tightened from
7.40 to 4.16 points (G11 D15), 7.04 to 6.45 (G17 D21), and 23.39 to
18.92 (G23 D17), without proving a positive loss or clearing the 2-point
near-optimal threshold. The full archived counts stay 56 near-optimal,
59 unresolved, zero proven losses above 2 points.

The intervals concern one decision with optimal full-information
continuation. They are neither whole-game exploitability nor blind
policy guarantees. G2 D11's archived Rock Slide remains unresolved;
fresh searches instead select Earthquake or Hidden Power, both already
certified near-optimal. This is not evidence of a persistent current bug.

## What self-play never exposed

The full 24 game histories were replayed, including games with no sampled
endgame. Every recorded state key and legal choice was checked.

| Move family | Registered in a game | Actually picked | Chosen at least once | Live sampled states |
| --- | --- | --- | --- | --- |
| Perish Song | 7 | 0 | 0 | 0 |
| Mean Look / Spider Web | 7 | 0 | 0 | 0 |
| Encore | 3 | 0 | 0 | 0 |
| Counter / Mirror Coat | 1 | 1 | 1 | 0 |
| Toxic | 15 | 0 | 0 | 0 |
| Baton Pass | 1 | 0 | 0 | 0 |
| Spikes | 17 | 8 | 2 | 8 |

The first three columns count games out of 24; the last counts states out
of 72. Chosen moves include failed attempts. The absence of several
families starts at team selection, while the Counter game is omitted by
endgame sampling. This audit therefore cannot evaluate those tactics.
Their absence does not prove the bot is weak against them.

Existing Perish investigations and deployed EarlyPerish evidence already
cover that family separately. Existing poison/recovery/bond waiting
certificates demonstrate particular defects or fixes, but synthetic
fixtures do not establish natural prevalence. The parked lock audit's
Magikarp construction is not promoted into an operational strength claim.

The next distinct measurement is a small coverage-forcing corpus using
unaltered catalog teams: include the Encore user from `sample-11`, a
Toxic user from `sample-06`, and the Baton Pass user from
`mjj-2010-garap-a` in legal selections, then inspect actual multi-turn
play. Forced exposure must stay separate from prevalence and overall
win-rate estimates. A candidate change needs a reproducible failure and
independent continuation evidence before a strength claim.

## Validation and reproduction

`tools/headroom-cases` rebuilds the two audit examples, reconstructs the
source corpus, runs the frozen probe and reference protocols, and writes
a fresh report under `tmp/headroom-cases-v1/run-*`. Existing evidence is
preserved. Time-limited reference intervals depend on machine load.

The report independently checks every original and refined policy
assessment. Known matrix games calibrate optimal, uniform, and ablated
policies; matching pennies checks that a reply is chosen against the
mixture rather than against each sampled action. Unknown branches stay
unknown, while forced decisions have zero choice loss. Probe action
ordering, visit totals, seed coverage, and nonempty interval intersections
are asserted. No engine, search, or product code changed.

`records.json.gz` contains all 72 reconstructed snapshots, the full-game
coverage traces, the 20 search probes, six new reference matrices, and
the per-endpoint assessments. `summary.json` includes input hashes,
run manifests, source/binary hashes, and checks. The probe ran before the
same example gained its separate `--coverage` export; `verification.json`
records an exact search-output comparison against the final exporter.
The human-readable report is `index.html`.
