# Threshold-preserving damage-roll experiment: results and waste audit

Date: 2026-07-22
Branch/worktree: `exp/threshold-damage` / `/home/puniu/nc2000-threshold-exp`
Implementation commits: `17d6912`, `b7d4d2a`, `b4c99c7`, `7cfd38d`, `538e4fe`

> **Provenance note (2026-07-23):** This experiment used the pre-v3 57-row anchor artifact. Its anchor counts, 38-row tight subset, interval comparisons, and derived MAE/violation figures are historical controls, not current ground truth; reconstruction drift was discovered afterward. The fail-closed M17e v3 artifact covers all 570 corpus coordinates with 72/72 rows reconstructed, 29 tight rows, 45 violations at ε=0.02, total margin 6.088, and tight MAE 0.0952. The matched exact-vs-approximate experiment numbers below are preserved as measured.

## Result

The semantic-threshold scheme is worth continuing. The waste audit found that unconditional HP fractions, the residual clock, and conservative exact escapes consume work without measurable policy benefit in this sample. The leanest fixed abstraction uses 5.38% of exact runs on the 57 anchors and 2.41% on an 8-position holdout.

Adaptive root-cell refinement fixes the healing/stall miss without another global HP partition. The final `probe-refine` arm uses 7.90% of exact work on the 57 anchors and 6.83% on holdout; worst exact-matrix policy regret is `0.0000959` and `0` respectively. A 400-game fixed-work duel against exact detected no strength loss and used 10.9% less think time. It remains experimental: the `0.01` probe threshold has only eight untuned exact-complete holdout positions, the result is not certified, an action omitted by the approximate support can still be missed, and Rest stalls retain a heavy runtime tail. The product agent remains untouched.

## Method

- 57 reconstructed real-game endgame positions from the then-current pre-v3 anchor corpus.
- Same state, legal actions, horizon, `eval01` leaf evaluator, probability mass, and matrix solver in every arm. Only the damage-roll partition changes.
- `exact`: existing distinct integer damage classes and proven endpoint merge.
- `mean`: one probability-weighted attainable representative.
- `threshold1`: split on KO/survival and HP 1/4, 1/3, 1/2 predicates.
- `threshold2`: `threshold1` plus same-hit two-shot class and status-residual death clock.
- Counter/Mirror Coat/Bide, drain/recoil, multi-hit, and Substitute retain exact damage classes.
- Values below are finite-horizon estimates, not certified bounds. The then-current pre-v3 intervals were used as an independent historical accuracy check.

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

For the 38 tight rows in the pre-v3 artifact (`width <= 0.05`; historical, not the current formal gate):

| horizon-1 estimate | MAE to certified midpoint | proven outside-interval violations > 0.02 |
|---|---:|---:|
| exact damage | 0.003806 | 2 |
| mean damage | 0.004814 | 3 |
| threshold2 | 0.003953 | 2 |

Against this independent reference, threshold2 worsened MAE by 0.000147 and added no violations. Mean-only added one violation.

### Worst threshold miss

Battle 455, turn 39: Snorlax 53/292 vs Skarmory 97/171, with Rest/healing and stall choices. Threshold2 value error was 0.02520 and exact-matrix policy regret was 0.05604. The important split is not a fixed mechanical HP predicate: the optimal heal/attack mixture changes with post-roll HP. This is evidence for adaptive value/policy refinement (or an exact fallback in a narrowly detected heal/stall state), not for adding many uniform HP buckets.

## Waste audit

The second pass independently ablated fixed thresholds, clocks, heal handling, and each conservative exact escape. `runs` remains scripted engine executions. Damage-draw reason counts are diagnostic probe observations, not unique battle events.

### Meaning-backed thresholds and clocks

`lean` removes the unconditional 1/4, 1/3, and 1/2 partitions. It retains KO plus only predicates that are actually observable in the state: usable Substitute (1/4), usable Belly Drum or a held Berry/Gold Berry/Berry Juice (1/2), and usable Flail/Reversal bands. PP-zero moves no longer create thresholds or Counter/Bide escapes.

| horizon-1 arm, 57 anchors | runs | exact ratio | value MAE | mean policy regret | worst regret |
|---|---:|---:|---:|---:|---:|
| threshold2 | 987,388 | 0.10003 | 0.000834556 | 0.000995576 | 0.056043558 |
| lean | 889,749 | 0.09013 | 0.000903880 | 0.000993065 | 0.056043558 |
| lean + next-hit clock | 930,887 | 0.09430 | 0.000842048 | 0.000993065 | 0.056043558 |
| lean + residual clock | 890,237 | 0.09018 | 0.000903880 | 0.000993065 | 0.056043558 |
| lean + both clocks | 931,441 | 0.09436 | 0.000842048 | 0.000993065 | 0.056043558 |

Findings:

- Removing the unconditional fractions saves 5.7% of threshold2 work; the value-MAE change is `+0.0000693`, with no policy-regret loss.
- The residual clock costs 488 runs and changes no measured value or policy metric. Remove it.
- The next-hit clock costs 41,138 runs (4.6% over lean). It improves anchor value MAE by `0.0000618`, but not policy regret; on the holdout it improves neither. Keep it only as an optional value-fidelity arm, not the default.

### Heal/stall strategies

| arm, 57 anchors | runs | exact ratio | value MAE | worst value error | worst regret |
|---|---:|---:|---:|---:|---:|
| lean | 889,749 | 0.09013 | 0.000903880 | 0.025197219 | 0.056043558 |
| exact rolls vs low-HP healer | 1,911,238 | 0.19361 | 0.000750565 | 0.025197219 | 0.056043558 |

Broad heal exactness more than doubles lean work. It fixes one non-policy value error (`0.005213`) but does not change mean/worst policy regret or the worst value error. Its largest false positive is battle 446: `+294,666` runs with zero value or policy change. On the horizon-1 holdout it consumes 15.9% of exact runs versus lean's 3.10%, with identical accuracy.

A two-representative low/high split when damage can enter a usable-heal region also fails. In the battle-455 pathological root it leaves the regret-causing side unchanged; on the 8-position holdout it costs 304,808 runs versus lean's 185,553 and slightly worsens regret. The bad root-matrix entry is specifically Curse vs Drill Peck: exact continuation value `0.375957`, abstract `0.193145`. A global HP rule cannot identify that strategic cell.

### Conservative exact escapes

At horizon 1, lean's forced-exact probe observations are:

| reason | observations | share of forced exact |
|---|---:|---:|
| drain/recoil | 227,501 | 62.8% |
| multi-hit | 77,273 | 21.3% |
| Substitute | 52,463 | 14.5% |
| Counter/Bide | 4,855 | 1.3% |

Independent removal results:

| arm, 57 anchors | runs | exact ratio | value MAE | mean policy regret | worst regret |
|---|---:|---:|---:|---:|---:|
| lean | 889,749 | 0.09013 | 0.000903880 | 0.000993065 | 0.056043558 |
| no Counter/Bide escape | 885,829 | 0.08974 | 0.000903880 | 0.000993065 | 0.056043558 |
| no drain/recoil escape | 599,599 | 0.06074 | 0.000903946 | 0.000992588 | 0.056016411 |
| no multi-hit escape | 853,455 | 0.08646 | 0.000903880 | 0.000993065 | 0.056043558 |
| no Substitute escape | 861,002 | 0.08722 | 0.000903880 | 0.000993065 | 0.056043558 |
| no exact escapes (`lean-minimal`) | 530,638 | 0.05376 | 0.000903946 | 0.000992588 | 0.056016411 |

All escapes are waste on this sample. Drain/recoil is the dominant sink: removing it alone saves 32.6% of lean work. Removing all four saves 40.4%, while changing MAE by `+0.000000066` and worst error by `+0.000003769`; policy regret is marginally lower. This is an empirical result, not a semantic proof, so adversarial mechanic fixtures remain necessary before product use.

### Untuned holdout

The holdout excludes all 57 tuning anchors and samples late 1v1 states across six disjoint corpus ranges. Exact completed 8 of the 12 horizon-1 positions within the 2,000,000-run cap. The final comparison reran those 8 completed positions:

| arm | runs | exact ratio | value MAE | mean policy regret | worst regret |
|---|---:|---:|---:|---:|---:|
| threshold2 | 225,620 | 0.03774 | 0.000030006 | 0.000086139 | 0.000689114 |
| lean | 185,553 | 0.03104 | 0.000029999 | 0.000086106 | 0.000688850 |
| lean-minimal | 144,315 | 0.02414 | 0.000030008 | 0.000086106 | 0.000688850 |
| heal two-way split | 304,808 | 0.05098 | 0.000029996 | 0.000086157 | 0.000689256 |

The holdout confirms the ranking: `lean-minimal` saves another 22.2% over lean with no material accuracy loss; the heal split is dominated.

### Decision

- Base abstraction: `lean-minimal` (KO plus only live mechanical thresholds, no clocks or exact escapes).
- Default research candidate: low/high-screened equilibrium-support refinement described below.
- Optional accuracy ablation: add only the next-hit clock.
- Reject: residual clock, broad heal exactness, heal two-way split, unconditional 1/3, and unconditional 1/4/1/2.
- Before product integration: broaden the exact-complete holdout, add adversarial Counter/Bide/drain/recoil/multi-hit/Substitute fixtures, test support-discovery failures, and rerun the product-fallback duel gate.

## Adaptive equilibrium-cell refinement

The final search is one-shot and preserves the same finite-horizon recursion, evaluator, and matrix solver:

1. Solve the full root matrix with `lean-minimal` conditional-mean representatives.
2. Select the cross product of equilibrium-support actions and approximate pure best responses.
3. Re-evaluate only those cells twice with the **same buckets and probability masses**, substituting each bucket's lowest and highest attainable damage. This adds no HP partition.
4. If a cell's low/high continuation values differ by more than `0.01`, re-evaluate that cell with exact damage rolls. Low, high, and exact passes each share their recursive memo across selected root cells.
5. Replace only refined entries in the mean matrix and solve it again.

### Why full support exactness is wasteful

Exact-recomputing every selected support/best-response cell is an effective control: it matched the full exact value and policy on all 57 anchors and all 8 holdout positions. It is still overkill.

| arm | sample | runs | exact ratio | value MAE | worst regret | selected/exact cells |
|---|---|---:|---:|---:|---:|---:|
| all selected cells exact | 57 anchors | 2,096,579 | 0.21239 | 0 | 0 | 234 |
| all selected cells exact | 8 holdout | 575,439 | 0.09625 | 0 | 0 | 25 |

On the anchors, `lean-minimal` already matched exact at 41/57 roots. Full support refinement nevertheless spent 599,199 exact runs on those roots and exact-recomputed 203/234 selected cells. Only 4 roots had value error or regret above `0.001`; an impossible hindsight oracle would need 238,612 total runs, 2.42% of full exact. This motivated the cheap pre-exact instability probe.

### Low/high-screened result

| arm | sample | runs | exact ratio | wall ratio | value MAE | worst value error | mean regret | worst regret | exact cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lean-minimal | 57 anchors | 530,638 | 0.05376 | 0.04381 | 0.000903946 | 0.025200988 | 0.000992588 | 0.056016411 | 0 |
| probe-refine | 57 anchors | 779,791 | 0.07899 | 0.06772 | 0.000004019 | 0.000077036 | 0.000003638 | 0.000095855 | 11 |
| lean-minimal | 8 holdout | 144,315 | 0.02414 | 0.02070 | 0.000030008 | 0.000239698 | 0.000086106 | 0.000688850 | 0 |
| probe-refine | 8 holdout | 408,146 | 0.06827 | 0.06363 | 0.000000046 | 0.000000293 | 0 | 0 | 2 |

The 57-anchor probe-refine work decomposes into 530,638 mean runs, 118,166 low/high probe runs, and 130,987 exact-refinement runs. The holdout decomposition is 144,315 + 25,572 + 238,259: two high-HP exact cells dominate, so the heavy tail is concentrated rather than removed.

At the battle-455 failure pair, the probe refines 2 cells total and uses 55,192 runs (14.15% of exact), versus 157,294 runs (40.34%) for exacting all five selected cells. Worst policy regret falls from `0.056016` to `0.0000299`.

Limitations:

- `0.01` was selected from the 57-anchor low/high separation and only then checked on the 8-position holdout; it is not a universal constant.
- Low/high values are diagnostics, not mathematical bounds. Damage can be non-monotone through choices and mechanics.
- Candidate discovery starts from the mean matrix. A strategically important action misranked outside its support and approximate best responses is not probed.
- Refinement is one-shot. It does not iteratively add support actions after exact entries move the equilibrium.

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

The final probe-refine arm was then run against exact under the same low-budget gate (same 32 teams, paired schedule, horizon 1, 20,000 work per mean/probe/exact phase, fallback 300, turn cap 100, seed 1):

- probe-refine: 194 wins, 202 losses, 4 ties;
- score `0.4900 +/- 0.0488` (95% CI): no detected strength loss;
- think time: probe-refine `381.86 ms/move`, exact `428.76 ms/move` (10.9% lower);
- average 21.0 turns; batch wall 790.5 s.

The per-phase limit means probe-refine can theoretically spend up to four nominal work budgets (mean, low, high, exact), although its selected-cell passes used much less in the measured corpus. The think-time result, not nominal budget multiplication, is the relevant duel resource measurement.

A 200,000-work/500-turn attempt reached 90/100 games in 950 s, then made no ten-game progress for over six minutes in Rest stalls; it was stopped after 21 minutes. At that setting exact search at every 1v1 decision has unacceptable tail cost. No win-rate result is claimed from the interrupted run.

## Resource projection

- The four-arm, 57-position horizon-1 benchmark consumed about 1,205 single-core seconds (0.335 core-hours). It completed locally with three workers in minutes, not hours.
- Linear projection to 1,200 similar endgames: about 7.0 core-hours for all four arms, before heavy-tail allowance. On a healthy 32–56 vCPU worker this is roughly 15–30 minutes including startup/synchronization; on the local 6-core machine, roughly 1–2 hours.
- The 400-game low-budget duel consumed about 2.6 core-hours (`12 × 772 s`). A 1,200-game confidence run is about 8 core-hours at this setting.
- The probe-refine 57-anchor pass consumed 65.2 summed single-core seconds versus 963.2 for exact; its 8-position holdout consumed 35.6 versus 559.4 seconds. These are measured sums, not parallel batch wall.
- The 400-game probe-refine duel also consumed about 2.6 core-hours (`12 × 790.5 s`).
- A 400-game product-fallback (3,000 iterations) gate is expected to need roughly 5–8 core-hours / 25–40 minutes on 12 cores; 1,200 games roughly triples that. Measure again because Rest-stall tails are non-linear.

## cx record

cx was authorized and attempted. Spot stockouts affected 56-, 32-, 16-, and 8-vCPU machine sizes. Fresh smaller images also lacked `cargo`; a portable prebuilt binary was then prepared. The final portable job remained in Spot backoff and was cancelled after the local benchmark finished, to avoid later duplicate spend.

- cancelled stockout jobs: `20260722-013224`, `20260722-013553`, `20260722-013928`, `20260722-015108`;
- failed environment/path probes: `20260722-014144` (no cargo), `20260722-014352` (prebuilt binary had compile-time local repo path; fixed by `b4c99c7`).

## Reproduction

```bash
cargo test -p nc2000-engine
cargo test -p nc2000-bot
cargo test -p conformance
cargo run --release -p nc2000-bot --example chance_conformance
cargo run --release -p nc2000-bot --example damage_abstraction -- \
  --corpus /home/puniu/nc2000-engine/tmp/corpus-spectator \
  --anchors /home/puniu/nc2000-engine/tmp/eec-all.csv --anchor-only \
  --battles 0-569 --positions 999 --per-battle 99 --alive-max 3 \
  --hp-cap 2000 --horizon 1 --work 2000000 --probe-work 2000000 \
  --refine-work 2000000 --leaf-cap 100000 --modes exact,lean-minimal \
  --probe-refine --probe-threshold 0.01 \
  --out /tmp/damage-abstraction-h1.csv

cargo run --release -p nc2000-bot --example damage_abstraction_duel -- \
  --a-probe --a-mode lean-minimal --b-mode exact \
  --a-horizon 1 --b-horizon 1 --a-work 20000 --b-work 20000 \
  --probe-work 20000 --probe-threshold 0.01 --fallback-iters 300 \
  --games 400 --threads 12 --pool meta --max-turns 100
```

The single benchmark command is sequential and therefore slower than the six range shards used for the measurements. Split `--battles` into disjoint ranges for parallel runs.
