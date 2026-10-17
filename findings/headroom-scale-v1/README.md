# Headroom scale v1

Status: **complete — extra-compute playing-strength headroom demonstrated**.
The registered confirmation comparison finished at block 6 on 2026-10-16.

**756 confirmation games: 427 wins / 329 losses / 0 ties**, score **0.564815**.
The alpha=0.025 anytime-valid betting interval is **[0.504114, 0.625394]**.
This is the first eligible complete-block prefix whose lower bound exceeds 0.5.
All earlier eligible checkpoints failed that rule. The driver and controller
stopped cleanly; no block 7 was started and no further computation is pending.

There were **896 executed comparison games**: 128 discovery games and 12 games
from the six excluded first pairs are not used in the confirmation result.
Two identical-agent null controls added 16 games. All comparisons and controls
had zero harness caps and zero invalid games. The discovery block's 70/58 result
remains archived and was not included in the positive decision.

Conclusion: the current local blind algorithm can still improve through additional
search under the registered catalog distribution. This establishes an implementable
same-information route; it does not establish an improvement at equal response time,
measure human custom-team performance, or quantify distance to game-theoretic optimality.

Across all 896 comparison games, including excluded games, mean choose-call time
was **3.459 s vs 1.083 s (3.19x)**; p95 was 5.992 s vs 1.859 s. These are measurements
under ten-thread load on the native Mac runtime, not iPad/Web response times.

`final-audit.json`, reproducible with `python3 findings/headroom-scale-v1/final_audit.py`,
checks all 389 frozen inputs, binary consistency, raw/compact timing agreement,
counts and exclusions, clean termination and both null controls. An independent
80-digit Decimal calculation reproduces all six confirmation intervals and the
first positive checkpoint. Its e-value at 0.5 is 179.098, above the registered
two-sided threshold 80. The frozen implementation and its prior test records remain
unchanged. Raw and aggregate results are in `results/` and `summary.json`.

The direct test was `blind:81000:0.4` against `blind:27000:0.4`: same local
algorithm, same player-visible information, same frozen belief prior, three times
the search work at preview and in battle. No pondering, oracle, learned model,
community-prior change or production default change. Both teams are drawn uniformly
from the 24-team catalog. This differs from the product's drawWeight distribution
and measures the current local runtime, not a fresh audit of the public deployment.

`protocol.json` was written before games started. It fixes sixteen
128-game blocks (2,048-game cap) and an anytime-valid fixed-fraction betting
interval with alpha 0.025 on side-swapped pair means. The primary positive stop
requires its lower bound above 0.5 at a complete block. An interval including 0.5
is not evidence of saturation. The first execution segment contains four blocks
(512 games); later segments use the remaining registered seeds if needed.
The original runner's checkpoints remain archived but do not control the amended
confirmation decision described next. The remaining 0.025 error budget is reserved for other, separately registered
improvement routes. No interim per-game score peeking changes the stopping rule.

Seed audit and confirmation amendment: `duel.rs` derives the two agent seeds as
`base_seed XOR (game_index * multiplier)`, with different multipliers. At index
zero both are therefore exactly `base_seed`, even with `crn_agent_seeds=false`.
Enumeration of all registered indices found only that collision. Before any block
1 outcomes were available, `confirmation-amendment.json` froze this correction:
block 0 is discovery-only; blocks 1–15 supply confirmation after excluding each
first side-swap pair (games 0 and 1). This leaves at most **1,890 confirmation games
/ 945 pairs**, within the unchanged 2,048 executed-game cap. Neither exclusions
nor future block allocation depend on the excluded outcomes.
`schedule-seed-audit.json` also checks all 945 retained pair schedules: each
has a distinct battle seed, shared only by its intended two side-swapped games.
`retained-seed-audit.json` checks uniqueness of the 3,780 retained agent seeds.

`summarize.py` validates every raw game, including excluded games, then applies
the same alpha=0.025 betting interval only to the retained confirmation pairs.
It records the first qualifying complete-block prefix; if an execution segment
finishes extra blocks before the controller notices, those data remain archived
and are not substituted for that first qualifying prefix. The original runner
can stop segments early under its old raw-data rule, but the controller continues
with unused registered blocks unless the amended analysis passes or the cap is
reached. No game is rerun. Original protocol, discovery raw data and pre-amendment
summary/auditor are preserved. `confirmation-implementation.json` binds the new
analysis and five passing correction tests, before block 1 outcomes were produced.
`summary.json` distinguishes total executed `games` from eligible primary
`analysis_games`; only the latter accompany its confirmatory score and interval.

All games must terminate without invalid moves or a harness turn cap. The cap is
1,100, above the engine's 1,001-turn tie; engine ties score 0.5. A null comparison
of identical blind agents at 100 iterations must return exactly 0.5 and no split
pairs under common agent seeds before the main independent-seed comparison starts.

AWS submission failed before a job was created because the login session expired
(`tmp/headroom-scale-cloud-00/receipt.json`, cleanup `not_started`). The experiment
therefore runs locally with ten threads rather than the planned 32. Only throughput
changes: arena schedules and agent seeds do not depend on thread count. The override
is recorded in `execution.json`; actual thread count is in every outcome artifact.

The completed initial segment was run from the project root (do not rerun into these artifacts):

```sh
python3 findings/headroom-scale-v1/run.py --start 0 --count 4 --threads 10
```

The controller then ran `--start 4 --count 4 --threads 10`, stopping cleanly
after block 6 passed the criterion. Existing complete blocks were used for the cumulative interval. Do not
start a second copy while the current process is live. Completed blocks are never
overwritten; if a process fails inside a block, inspect and preserve the partial
artifact and restart only that uncompleted block. No complete-block checkpoint
may be counted twice.

`advance.py` now waits on the identified initial driver, audits every completed
segment independently, and runs only the remaining registered blocks in groups
of four. It stops at the first verified positive checkpoint or the existing
2,048-game cap. A failed or partial segment stops the controller for inspection;
it never discards or silently replaces a result. `controller.json` and
`controller.log` record its state. Verify the actual process as well as these
files before restarting anything. This controller changes orchestration only;
the frozen experiment runner, inputs, seeds and statistical rule are unchanged.

The driver validates all frozen inputs before computation and prints each compact
complete result to stdout for independent recovery. `results/block-*-raw.jsonl`
retains the full arena artifact, including per-choice timings. The manifest records
the binary, source, platform and compiler; aggregate JSON retains all pair and game
scores. This is a measurement of extra-compute headroom, not a same-latency release
gate, and does not determine the game-theoretic optimum.

Conclusion requirements: a feasible same-information policy must beat the baseline
under the registered statistical rule before claiming demonstrated headroom from
this route. Failed earlier feature changes and loose exact-solver bounds cannot
prove that the game is solved or that no improvement exists. A positive result
must be reported together with its compute cost, tested distribution and uncertainty.

Inference validation: all four existing `BettingIntervalTests` pass. The selected
alpha=0.025 was also checked by exact enumeration of repeated checks under three
Bernoulli null means, plus interval nesting for fractional pair scores; results
and the evaluator digest are in `inference-validation.json`.
