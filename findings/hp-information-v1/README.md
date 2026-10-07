# Hidden Power evidence × opponent move information

Research diagnosis only. The ordinary bot and the default-off HP evidence feature are unchanged.

The protocol was fixed before the 27,000-iteration outcomes. It reuses three constructed stress fixtures and one on-pool control, with fresh seeds: 24 stress blocks and two control blocks, four games per block. Arms 0/1 use normal hidden-move inference with public HP evidence absent/present; arms 2/3 supply the exact four base move identities of all six opponent roster slots with evidence absent/present. HP remains generic. The opponent uses ordinary move inference and public HP evidence in every arm.

The move oracle feeds a persistent research observer a separate battle clone with base-slot usage flags set. Actual battles and search roots stay untouched. A test covers both sides of every forced prefix, stable revisions on repeated observation, unchanged battle state and move usage, and preservation of every non-move observation field. Full move information can indirectly condition prior genotypes and preserve live PP; the intervention does not hold those downstream effects fixed. It cannot be shipped as a hidden-information policy.

All 104 games run regardless of outcomes or missed exposures. Capped results remain unknown. Type effects are paired within each move-information condition; their difference is exploratory. These fixtures, opponent and small sample do not estimate general strength, human superiority, or distance to a game-theoretic ceiling. There is no adoption gate based on these outcomes.

A 64-iteration smoke on a separate seed completed 32 games and 1,001 replayed decisions. The audit samples 64 determinizations at every non-forced, multi-action decision. All oracle loadouts matched all six true canonical move sets; every evidence-enabled active model satisfied the public type mask. Full-strength results are recorded separately after completion.

Run `tools/hp-information-audit PROTOCOL RESULTS OUTPUT` to replay all actions, state keys, seeds, observation masks, oracle views and outcomes; audit modeled loadouts; and archive records and paired summaries. The checker requires the recorded source and executable hashes. It does not rerun every search.


## Full-strength result

All 104 games terminated without caps or ties. Stress wins (24 games per arm):

| Opponent move information | No HP evidence | Public HP evidence |
| --- | ---: | ---: |
| Usual inference | 16 | 12 |
| Exact base move identities | 8 | 15 |

The paired HP effect was 4 improvements, 8 regressions and 12 ties under usual inference; with exact moves it was 9 improvements, 2 regressions and 13 ties. The two control seed blocks had identical action traces in all four arms (all losses). Each constructed case remains visible in `summary.json`: Articuno 5/4/1/6, Dragonair 7/5/6/8, Exeggutor 4/3/1/1, in arm order.

The observed interaction is +45.8 percentage points. A descriptive 20,000-draw bootstrap resampling four-arm seed blocks within each case gives an interaction interval of +8.3 to +79.2 points. The HP-effect intervals are -41.7 to +12.5 points with usual inference and +8.3 to +50.0 points with exact moves. These small, fixed strata do not support a general strength or adoption claim; the intervals are neither simultaneous nor a preregistered decision gate. Both information interventions together score 15/24, below the baseline's 16/24. Correct moves alone score 8/24. The data do not support simply enabling type evidence or assuming that correcting move inference alone solves the problem.

Replay verification covered 2,541 decisions. At 2,129 non-forced multi-action decisions, 64 modeled battles each gave 136,256 samples. All 64,256 oracle samples had the exact six base loadouts. Enabled-evidence samples had zero public type contradictions; disabled-evidence arms had 12,416 and 11,264 contradictory active samples. These denominators follow different game trajectories and are not prevalence or direct causal estimates. The first HP-driven action divergence occurred in 23 usual-inference blocks (21 with missing actual active moves) and all 24 oracle stress blocks (none with missing moves).

An independent fixture-DV check covered 2,397 observations and 2,398 constrained mon observations. No true HP type was excluded; all 96 stress games exposed type evidence. Two invalid-mask controls were rejected. The main replay checker also rejected score, seed, oracle-input and missing-arm mutations on the smoke records. The smoke's 32 games reproduce every decision and observation except timing; the full run is replay-verified, not wholly search-repeated.

The actionable result is a dependency between loadout modeling and type evidence, not a strength improvement. Future diagnosis should examine decision values at common public states and separate remaining model error from search/value error. Do not tune these 24 outcomes or treat this oracle as a deployable policy. No production source, default configuration, or deployed bot changed.

`records.json.gz` contains the manifest, full games and model diagnostics. `uncertainty.json` and `truth-check.json` record separate analyses with source/data hashes. `tools/hp-information-uncertainty` and `tools/hp-information-truth-check` reproduce them from the frozen archive.
