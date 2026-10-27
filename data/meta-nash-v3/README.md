# meta-nash-v3 — the `?nash` mixture under the shipped blind profile

`pool-artifact.json` is generated (`node tools/build-team-products-v2.js`)
from `solution.json`; it is what `?nash` draws the bot's party from, once per
battle, the play door's draw when the setup panel's "solved mixture" choice
is on, and the evaluator's default opponents. Every support party is
`selected` in `data/team-selection-v2/selection.json`; weights live here,
never as labels.

## Conditions

Blind vs blind, c = 0.4 on both sides, the frozen prior
`data/belief-pool-v3` (`fnv1a64:ffa73c4245fe3774`), live preview search,
500-turn cap scored 0.5, no pondering; harness
`crates/bot/examples/team_eval.rs`, solver `tools/nash-solve.py` (RM+, seed
bootstrap), procedure `data/team-selection-v2/PROTOCOL.md` §Nash.

## Files

| File | What |
|---|---|
| `solution-blind3000.json` | the 24 selected parties, every pair 16 seed blocks at blind:3000:0.4 |
| `support-relevant-blind3000.txt` | its 19 support-relevant parties (weight ≥ 0.01 or support in ≥ 10% of bootstraps) |
| `solution-blind27000-r0.json` | those 19, every pair 16 blocks at blind:27000:0.4 (seed base 20261161) |
| `rounds.json` | best-response rounds: S members outside the game against the 27k mixture (16 mixture-sampled blocks); sample-16-orig and sample-02, then rental-cban-14 joined |
| `solution-blind27000.json` | the final 22-party game at 27k (pooled with stage B/C and +16 refinement blocks per pair on the top 10 support candidates at seed base 20262161) |
| `solution.json` | the shipped weights (support ≥ 0.01, renormalized) and conditions |
| `challenge-blind3000.json`, `challenge-blind27000.json`, `challenge-blind27000-refine.json` | non-selected candidates, held-out page teams and off-prior chimeras against the mixture |

## Result

sample-07 0.261, nb-sample-13-0 0.234, sample-08 0.187, rental-cban-3 0.159,
sample-10 0.159 (initial 16-block solve: sample-08 0.412, sample-07 0.314,
魔人島 ソード＆シールド 0.127, sample-13 0.127, sample-10 0.020; held-out 64-block
CRN duel of the refined mixture against the initial mixture at blind:27000:0.4
on seed base 20263161: 0.523 ± 0.074 overall, 0.531 ± 0.097 on the 49 non-mirror
blocks). The three parties that joined in the best-response rounds take no
weight in the re-solved game. Across seed bootstraps the five support parties
lead consistently (rental-cban-3 in 88% of resamples, sample-08 84%, sample-10
76%, nb-sample-13-0 75%, sample-07 74%).

Challenges at 27k: no challenger's 95% interval clears 0.5. The two whose
point estimate did (off-prior chimeras offprior-03 0.547 and offprior-16
0.531 on 32 blocks) came back at 0.500 ± 0.085 and 0.516 ± 0.090 on 64 fresh
blocks — near parity, neither shown to beat the mixture nor ruled out. At
3000 the chimera offprior-12 scored 0.781 but 0.453 at 27k: exploit evidence
does not transfer across budgets.

Scope: the equilibrium of a finite game over the selected catalog at the
bot's product budget without pondering — not an equilibrium of the whole
format.
