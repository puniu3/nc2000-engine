# Thick Club HP regions: equal-time follow-up

At the only root exposed in the six-root screen (game 10/frame 32), both candidate
passes returned slightly smaller conditional loss upper bounds than controls.
The gain was only 0.00526–0.007855 percentage points. There is no new certificate
of loss at most 0.02, no positive loss lower bound, and no bot-strength evidence.
Keep the extension experimental; do not integrate it by default.

| Serial pass | Elapsed ms | Engine executions | Loss interval | Cache hits |
| --- | ---: | ---: | --- | ---: |
| control-1 | 6008.811 | 248526 | [0, 0.06827917436893671] | 0 |
| regions-1 | 6033.719 | 232300 | [0, 0.06820062381209557] | 197 |
| regions-2 | 6000.108 | 213819 | [0, 0.06822656631932855] | 194 |
| control-2 | 6054.658 | 248526 | [0, 0.06827917436893671] | 0 |

All passes stopped at the registered 6-second research budget, within the
120-ms overshoot gate. Work and node limits did not bind. Independent rational
matrix checks and three corrupted-record rejection checks passed. Two repeats
at one selected root are insufficient to claim general efficiency improvement.
Mac research timing is neither an iPad measurement nor a product response limit.

For direct verification, each timed candidate was replayed deterministically to
its saved trial count with wall-clock stopping disabled. Budgets and traversal
were otherwise preserved. Both replays matched every original payoff, bound,
search counter and cache counter. Stop reason, elapsed time and the explicit
verification configuration were normalized by the checker.

All 197 and 194 reused distributions matched full direct engine enumeration under
the existing guarded semantic identity, with maximum probability difference zero.
The extra 50,746 and 50,596 verification executions are excluded from performance.
Each replay's checker rejected changed hit counts, probability errors, payoff
cells and trial counts. See `tools/native-club-time-verification-audit`.

Protocols, binary hashes and replay plans were committed before their respective
runs. Raw records are deterministic gzip archives; summary hashes refer to
decompressed bytes. The analysis-only adapter patches are retained as artifacts;
the live adapter's patched solver and kernel sources were restored afterward.
The main bound solver, ordinary bot and public deployment remain unchanged.
