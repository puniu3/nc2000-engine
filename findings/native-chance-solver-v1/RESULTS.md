# Probability-first integration does not improve this root certificate

The new enumerator is available through the default-off probability_first_eager
analysis setting. It retains exact leaf and pending-prefix masses, uses no damage
range merging, and preserves the existing default path. All63 bound-solver tests
passed, including partial/complete successor reconstruction with deferral both
on and off and exhaustive two-turn fixed-policy bounds for both sides.

The library reproduced all16 frozen standalone cells:47,872 leaf scripts,
probabilities and outcomes plus66,680 pending scripts/probabilities matched
exactly. Fresh replay also matched every leaf's exact state key. Verification
used95,744 extra executions and is excluded from the solver comparison.

Three modes ran serially in ABCCBA order at250,000 execution and50,000 state
ceilings on game62/frame92. Every run exhausted250,000 executions; all four
first-action value and loss intervals remain[0,1]. All non-time repeats matched,
and the default and retained-prefix modes reproduced their frozen parent outputs
apart from the new false configuration field.

| Mode | Retained states | Elapsed seconds, two runs | Selected loss |
| --- | ---: | --- | --- |
| Default | 682 | 4.459 / 4.208 | [0,1] |
| Retain old eager prefixes | 6,677 | 6.656 / 6.367 | [0,1] |
| Probability-first | 3,449 | 3.652 / 3.403 | [0,1] |

Times are descriptive for this small fixed-work diagnostic. The candidate is
faster here but does not improve the decision-relevant certificate. Probability
coverage is not a substitute for reaching informative terminal continuations.
Independent rational matrix checks and12 deliberate corruption rejections passed.
Keep the candidate opt-in; no ordinary-bot behavior or public deployment changes,
and no strength or near-optimality claim follows from these results.
