# Certified one-decision policy loss

The frozen holdout contains 72 states from 18 of 24 source games, with 115 actionable endpoints. This audit fixes the first decision and assumes optimal full-information continuation. Rewards are win=1, draw=0.5, loss=0. It does not measure the deployed blind policy's full-game exploitability.

## Results

- Archived 10-second decisions: 56 endpoints certified within 0.02 of optimal; 59 unresolved; no loss above 0.02 proved. Of the 56, 22 are forced decisions. Across all nonforced endpoints, 34 are certified near-optimal and 59 remain unresolved.
- 35 endpoints have every legal action certified within 0.02; this includes the 22 forced endpoints.
- 12 endpoints across 8 source games have both a certified near-optimal pure action and a certified action losing more than 0.02. On this discriminating subset, the uniform policy loses 0.1408–0.3067 on average; max damage without search loses 0.0419–0.0514. Means weight source games equally.
- Current full-information SkuctSearch, c=0.4, empirical mixtures of four search seeds on that subset:

| Iterations | Mean loss bounds | Endpoints certified within 0.02 |
|---|---|---|
| 27,000 | 0–0.03458 | 9/12 |
| 108,000 | 0–0.00605 | 10/12 |
| 432,000 | 0–0.00500 | 12/12 |

At 432,000 iterations the largest individual upper bound is below 0.01. These are certificates for the measured empirical mixtures; they omit uncertainty about the infinite-seed bot distribution. Bounds overlap, so reduced loss from a larger search budget is not proved. The entire archived corpus still has mean loss bounds of 0–0.35025: unresolved states prevent an overall near-optimality claim. No proven large loss is not a proof that no large loss exists.

## Method

`BoundSolver::solve_policy_losses` continues refining first-decision payoff cells even after the root value closes. It targets all pure actions plus uniform mixtures; both sides play optimally after the fixed first decision. Unexpanded probability and unresolved successors retain bounds. The opponent best-responds to the entire mixture, without seeing the sampled action. Forced decisions have zero choice loss without requiring the state value to be solved.

The reference budget per state is 60 seconds, 120,000 graph/cache entries, and 2,000,000 engine executions. Root action pruning is disabled for this audit; continuation bounds retain the existing sound solver machinery. Stopping reasons were 22 policy targets met, 35 node limits, 10 time limits, and 5 work limits. Reference work totaled 52,787,388 engine executions and 1,802 aggregate seconds.

Search budgets are nested within each seed. The budget ladder runs only where oracle bounds establish a good and a bad pure action, independently of the new search outputs. That selection is explicitly conditional, not representative of all endgames. Archived realized actions and new empirical mixtures are separate measurements. The interrupted development run and short smoke pilot are excluded.

## Reproduction

From the repository root:

```sh
tools/policy-gap
```

This builds the harness, extracts the content-bound corpus, and creates a fresh run and report under `tmp/policy-gap-v1`. Existing results are preserved. Time-limited references may differ with machine load, so compare certified intervals rather than expecting byte-identical outputs.

`summary.json` includes provenance and aggregate bounds. `records.json.gz` contains all matrices, policies, source identifiers, and search outputs. `corpus.jsonl.gz` and `archived-baseline.json` freeze the inputs. `verification.json` records 156 passing bot/endgame tests and independent Python interval checks. Tests include exhaustive stochastic microgames, mixtures requiring hidden randomization, continued refinement after value closure, and forced decisions with unknown value. The human-readable result is `index.html`.

No deployed policy or configuration was changed.
