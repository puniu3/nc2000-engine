# ISMCTS with eight-turn rollouts

## Scope

Resume the experiment recorded in `4dae00a2` and `tmp/batch-ab/`, isolating
visible-state ISMCTS from the analytical leaf solver. The previous bundled
configuration scored 200–184 at 27,000 iterations and 183–201 at 81,000
iterations against its legacy control (`tmp/batch-ab/report.json`). Those
results do not establish an advantage for the bundle or for ISMCTS alone.

`crates/bot/examples/ismcts_rollout_duel.rs` adapts the preceding session's
scratch duel runner into a tracked, standalone experiment. Its only arms are:

| Setting | `legacy` | `ismcts` |
| --- | --- | --- |
| Base configuration | `RmConfig::legacy_heavy_rollout()` | Same |
| Visible-state interior tree | Off | On |
| Rollout | Eight turns, epsilon 0.2 | Same |
| Leaf evaluation | `BlindSpotLeaf` | Same |
| Analytical solver | Off | Off |
| Semantic HP thresholds | Off | Off |
| Three-rank pruning | Off | Off |
| Sleep-action collapse / damage-history omission | Off / off | Same |
| Root selection | UCB, c=0.4 | Same |
| HP buckets / horizon | 16 / 100 | Same |

No experiment arm derives from `RmConfig::default()`. In particular, merely
selecting `Playout::heavy()` is insufficient: `SolverQuiescence` bypasses the
rollout even when its nominal turn count is eight. The runner explicitly
inherits the legacy settlement mode. Existing ISMCTS HP bookkeeping remains
in the library, but semantic HP merging and solver quadrature are disabled.

This is a configuration-level isolation using the current search library.
The production default and locked preview artifacts are unchanged. A positive
result here would justify further validation, not an automatic production
promotion or a claim that the ISMCTS implementation is formally sound.

## Comparison protocol

- Draw teams using the catalog's existing `drawWeight`, with the existing
  belief pool unchanged. `offpool` and `mixed-offpool` use the previous runner's
  fallback-filtered rental teams.
- Each pair uses the same teams, battle seed, agent seed and preview seeds in
  both games; exchange the algorithms between seats. The schedule depends on
  seed and pair index, so sharding and worker count do not change it.
- By default both arms use the same compiled preview table and choices. A
  missing entry falls back to live preview. Since visible-state ISMCTS is
  bypassed during preview, both arms use the same preview search configuration.
  `--preview live` tests this path explicitly. Reusing a common table controls
  preview differences; it does not certify that table for a new production bot.
- Use equal iteration budgets. Latency is elapsed wall time per non-forced
  battle decision under concurrent local execution, not an equal-time match
  or a controlled isolated microbenchmark.
- Score engine wins as 1, terminal draws as 0.5 and losses as 0. The default
  operational cap is 1001 turns, beyond the engine's turn-limit draw. Any earlier
  nonterminal cap is reported separately from draws, with lower/upper score
  bounds. `score` uses the midpoint of those bounds.
- Compute the approximate 95% normal interval over pair scores, not over
  independent games. Small samples and zero-variance null controls do not
  provide general performance guarantees.
- Print one JSON summary after completion. Optional per-pair records, timing
  arrays and seeds go to `--out`. Summary metadata includes the library build
  fingerprint and SHA-256 hashes of the harness and data inputs.

## Run locally

Run from the repository root:

```sh
cargo build --release -p nc2000-bot --example ismcts_rollout_duel
mkdir -p tmp/ismcts-only
target/release/examples/ismcts_rollout_duel --self-test
target/release/examples/ismcts_rollout_duel \
  --pairs 64 --iters 27000 --threads 8 \
  --out tmp/ismcts-only/catalog-27k.jsonl > tmp/ismcts-only/catalog-27k.json
target/release/examples/ismcts_rollout_duel \
  --pairs 64 --iters 81000 --threads 12 \
  --out tmp/ismcts-only/catalog-81k.jsonl > tmp/ismcts-only/catalog-81k.json
target/release/examples/ismcts_rollout_duel \
  --pairs 32 --iters 27000 --threads 4 --pool offpool \
  --out tmp/ismcts-only/offpool-27k.jsonl > tmp/ismcts-only/offpool-27k.json
```

The default schedule seed is `20261113`. These are exploratory comparisons;
use a fresh seed for confirmation. `--arm-a legacy --arm-b legacy` and
`--arm-a ismcts --arm-b ismcts` are null controls. `--stride N --offset K`
selects pair indices congruent to K modulo N. `--help` lists all options.

The self-test checks that the configurations differ only in the ISMCTS flag,
that a real rollout advances eight turns, that changing actual hidden opponent
happiness preserves search visits/values/actions under the same belief, and
that both arms produce exactly 0.5 in paired identical-policy games. It also
exercises search from both seats and records raw depth counters. The hidden
information check is a targeted regression check, not a complete boundary audit.

## Results: 2026-11-13

All 320 games reached engine terminal outcomes, with no draws or operational
caps. Wins and losses below are from the ISMCTS arm's perspective.

| Pool | Iterations | Pairs | W–L | Score | Approximate paired 95% interval | Mean ms/decision, ISMCTS / legacy |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Catalog | 27,000 | 64 | 57–71 | 44.5% | 36.3–52.7% | 309.78 / 310.36 |
| Catalog | 81,000 | 64 | 63–65 | 49.2% | 41.2–57.2% | 1162.83 / 1187.81 |
| Off-pool rentals | 27,000 | 32 | 33–31 | 51.6% | 41.3–61.9% | 401.01 / 407.88 |

**This experiment did not establish a strength improvement from enabling
visible-state ISMCTS alone on the legacy rollout.** Every interval contains
50%. The estimates do not justify promoting this configuration, and do not
prove that all ISMCTS variants are weaker. The abandoned solver bundle was not
run in these comparisons.

The two catalog experiments used the same 64 schedules, and every catalog
game hit the shared preview table for both arms. All 32 off-pool pairs exercised
fallback belief on both arms, with live preview and no compiled hits. The
81k and off-pool runs overlapped each other and the end of the 27k run, so
interpret latency as descriptive concurrent execution timing. No equal-time
strength claim is made.

Validation passed: release build, self-test (eight paired null controls,
eight-turn rollout advancement, targeted hidden-information invariance),
thread-count and shard invariance, live-preview null control, explicit
nonterminal-cap accounting, invalid-option rejection, and independent
recomputation of all three runs' outcome scores and paired confidence intervals.

Full summary metadata and pair-score histograms are preserved in
[`data/ismcts-rollout-v1/results.json`](../data/ismcts-rollout-v1/results.json).
Raw per-pair records and timing arrays remain under `tmp/ismcts-only/`.
The 27k catalog run's harness hash predates only the addition of preview timing
arrays to disk traces and a terminal-outcome assertion in the null self-test;
its simulation and configuration code are identical to the final harness.
