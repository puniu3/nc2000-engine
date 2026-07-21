# Threshold-preserving damage-roll experiment: first results

Date: 2026-07-22  
Branch/worktree: `exp/threshold-damage` / `/home/puniu/nc2000-threshold-exp`  
Implementation commits: `17d6912`, `b7d4d2a`, `b4c99c7`

## Result

The semantic-threshold scheme is worth continuing. On the matched search benchmark it removes about 90% of horizon-1 engine executions while keeping average value and root-policy errors small. A 400-game fixed-work duel detected no strength loss. It is not worst-case safe yet: one healing/stall position had root-policy regret 0.056, so product integration should add an adaptive refinement/fallback for policy-sensitive heal states.

## Method

- 57 reconstructed real-game endgame positions from the existing certified-anchor corpus.
- Same state, legal actions, horizon, `eval01` leaf evaluator, probability mass, and matrix solver in every arm. Only the damage-roll partition changes.
- `exact`: existing distinct integer damage classes and proven endpoint merge.
- `mean`: one probability-weighted attainable representative.
- `threshold1`: split on KO/survival and HP 1/4, 1/3, 1/2 predicates.
- `threshold2`: `threshold1` plus same-hit two-shot class and status-residual death clock.
- Counter/Mirror Coat/Bide, drain/recoil, multi-hit, and Substitute retain exact damage classes.
- Values below are finite-horizon estimates, not certified bounds. The older certified intervals are used as an independent accuracy check.

## Matched value/policy benchmark

`wall` is the sum of each root's elapsed time, measured with three independent single-thread workers on the local 6-core/12-thread, 8 GiB machine. `regret` evaluates the approximate root policies in the exact root matrix.

### Horizon 0

| mode | complete | engine runs | exact ratio | wall | value MAE | mean policy regret | worst regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact | 57/57 | 81,164 | 1.000 | 7.914 s | 0 | 0 | 0 |
| mean | 57/57 | 27,679 | 0.341 | 2.398 s | 0.001156 | 0.002225 | 0.093748 |
| threshold1 | 57/57 | 29,026 | 0.358 | 2.526 s | 0.000048 | 0.000017 | 0.000502 |
| threshold2 | 57/57 | 29,592 | 0.365 | 2.571 s | 0.000047 | 0.000012 | 0.000502 |

### Horizon 1

| mode | complete | engine runs | exact ratio | states | wall | value MAE | mean policy regret | worst regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 57/57 | 9,871,404 | 1.000 | 20,306 | 956.855 s | 0 | 0 | 0 |
| mean | 57/57 | 840,273 | 0.085 | 3,802 | 75.029 s | 0.003590 | 0.007323 | 0.129655 |
| threshold1 | 57/57 | 944,520 | 0.096 | 4,320 | 84.032 s | 0.000896 | 0.000996 | 0.056044 |
| threshold2 | 57/57 | 987,388 | 0.100 | 4,453 | 89.066 s | 0.000835 | 0.000996 | 0.056044 |

Threshold2 therefore saved 90.0% of engine runs and 90.7% of elapsed work at horizon 1. Its value-error p95 was 0.00794 and worst value error was 0.02520. Three of 57 roots had nonzero policy regret.

For the 38 tight certified anchors (`width <= 0.05`):

| horizon-1 estimate | MAE to certified midpoint | proven outside-interval violations > 0.02 |
|---|---:|---:|
| exact damage | 0.003806 | 2 |
| mean damage | 0.004814 | 3 |
| threshold2 | 0.003953 | 2 |

Against this independent reference, threshold2 worsened MAE by 0.000147 and added no violations. Mean-only added one violation.

### Worst threshold miss

Battle 455, turn 39: Snorlax 53/292 vs Skarmory 97/171, with Rest/healing and stall choices. Threshold2 value error was 0.02520 and exact-matrix policy regret was 0.05604. The important split is not a fixed mechanical HP predicate: the optimal heal/attack mixture changes with post-roll HP. This is evidence for adaptive value/policy refinement (or an exact fallback in a narrowly detected heal/stall state), not for adding many uniform HP buckets.

## Fixed-work duel

Configuration:

- 32 meta-pool teams, seed-paired side swap;
- threshold2 versus exact damage;
- horizon 1, 20,000 work units per eligible 1v1 decision;
- identical 300-iteration SkUct fallback outside 1v1 or on search exhaustion;
- 100-turn cap, 12 threads, seed 1.

Result over 400 games:

- threshold2: 195 wins, 202 losses, 3 ties;
- score `0.4913 +/- 0.0489` (95% CI): no detected strength loss;
- think time: threshold2 `341.28 ms/move`, exact `434.08 ms/move` (21.4% lower);
- average 21.2 turns; batch wall 772.1 s; peak observed RSS about 1.56 GiB.

A 200,000-work/500-turn attempt reached 90/100 games in 950 s, then made no ten-game progress for over six minutes in Rest stalls; it was stopped after 21 minutes. At that setting exact search at every 1v1 decision has unacceptable tail cost. No win-rate result is claimed from the interrupted run.

## Resource projection

- The four-arm, 57-position horizon-1 benchmark consumed about 1,205 single-core seconds (0.335 core-hours). It completed locally with three workers in minutes, not hours.
- Linear projection to 1,200 similar endgames: about 7.0 core-hours for all four arms, before heavy-tail allowance. On a healthy 32–56 vCPU worker this is roughly 15–30 minutes including startup/synchronization; on the local 6-core machine, roughly 1–2 hours.
- The 400-game low-budget duel consumed about 2.6 core-hours (`12 × 772 s`). A 1,200-game confidence run is about 8 core-hours at this setting.
- A 400-game product-fallback (3,000 iterations) gate is expected to need roughly 5–8 core-hours / 25–40 minutes on 12 cores; 1,200 games roughly triples that. Measure again because Rest-stall tails are non-linear.

## cx record

cx was authorized and attempted. Spot stockouts affected 56-, 32-, 16-, and 8-vCPU machine sizes. Fresh smaller images also lacked `cargo`; a portable prebuilt binary was then prepared. The final portable job remained in Spot backoff and was cancelled after the local benchmark finished, to avoid later duplicate spend.

- cancelled stockout jobs: `20260722-013224`, `20260722-013553`, `20260722-013928`, `20260722-015108`;
- failed environment/path probes: `20260722-014144` (no cargo), `20260722-014352` (prebuilt binary had compile-time local repo path; fixed by `b4c99c7`).

## Reproduction

```bash
cargo test -p nc2000-engine
cargo test -p conformance
cargo run --release -p nc2000-bot --example damage_abstraction -- \
  --corpus /home/puniu/nc2000-engine/tmp/corpus-spectator \
  --anchors /home/puniu/nc2000-engine/tmp/eec-all.csv --anchor-only \
  --battles 0-569 --positions 999 --per-battle 99 --alive-max 3 \
  --hp-cap 2000 --horizon 1 --work 2000000 --leaf-cap 100000 \
  --out /tmp/damage-abstraction-h1.csv

cargo run --release -p nc2000-bot --example damage_abstraction_duel -- \
  --a-mode threshold2 --b-mode exact --a-horizon 1 --b-horizon 1 \
  --a-work 20000 --b-work 20000 --fallback-iters 300 \
  --games 400 --threads 12 --pool meta --max-turns 100
```

The single benchmark command is sequential and therefore slower than the six range shards used for the measurements. Split `--battles` into disjoint ranges for parallel runs.
