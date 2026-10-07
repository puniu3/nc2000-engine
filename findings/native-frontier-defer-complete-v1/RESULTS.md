# Complete-frontier deferral under bounded work and state count

The opt-in candidate activated: 1,127,524 nonterminal leaves were deferred.
Both repeated searches produced identical non-time outputs. Default controls
reproduced the frozen parent exactly. The registered work ceiling was 2,000,000
engine executions and state-count ceiling 200,000 for each search.

| Mode | Fixed-policy guarantee | Engine executions | Retained states | Stop | Elapsed seconds |
| --- | --- | ---: | ---: | --- | --- |
| Default | [0.29620143242332375, 1] | 921,554 | 200,068 | NodeBudget | 39.79 / 39.79 |
| Defer complete leaves | [0.30619502368918616, 1] | 2,000,000 | 50,677 | WorkExhausted | 63.99 / 65.27 |

The same ceilings allowed a roughly 75% reduction in retained state count and
about 1.00 percentage point improvement in the lower bound. Actual execution
counts and elapsed times increased; this is not a matched-time improvement.
Retained state count is not measured resident memory usage. Batch insertion
can overshoot the state cap. These are fixed Rest-first/Sleep-Talk policy bounds
at game87/frame49, not optimal first-action values or ordinary-bot scores.

The implementation retains full representative scripts after complete eager
enumeration, only when deferral is requested. A script representing merged
chance leaves must carry their combined probability when materialized; taking
the raw probability of the representative trace would lose mass. Existing
partial enumeration and default solver settings remain unchanged.

All 63 bound-solver release tests passed, including exhaustive two-turn best
responses for both sides and complete/partial successor reconstruction. A
separate native verification re-enumerated all 256 saved sampled cells with zero
probability difference and replayed all 31,560 representative scripts to the
same exact state keys and full trace lengths. Of these, 13,144 carried merged
probability. Verification used 39,270 enumeration executions, 31,560 script
replays and 64 policy checks (70,894 total), excluded from performance samples.
Independent rational matrix checks and five record corruption controls passed.

The state-count bottleneck is reduced, but the Rest-versus-attack ordering remains
unresolved. No ordinary-bot strength gain, public deployment or matched-time
advantage is claimed. A separately registered matched-time comparison is the
next decision point before wider use; default deferral remains disabled.
