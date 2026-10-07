# Waking-priority witness strengthens the Rest-first lower bound slightly

At the registered 2-million-execution/200,000-state ceilings, the old policy
reproduced the frozen interval [0.29620143242332375, 1], 921,554 executions and
200,068 retained states. Waking-priority yielded [0.3012754726891027, 1], 946,576
executions and 200,113 retained states. Both stopped at the state limit, with
batch overshoot. The lower-bound improvement is about 0.5074 percentage points.
Both repeated runs were identical apart from time and trial identifier.

The retained expanded waking-state action census confirms activation: the old
policy chose Sleep Talk at 308 states and the new policy chose Double Edge at
314. Counts are graph-specific and not probability-weighted. State expansion
and all opponent choices remain unconstrained apart from the registered budgets.
Independent rational matrix checks and five corruption controls passed. The
new policy API was separately checked by the 63-test bound suite and native
64-state/256-cell replay in native-frontier-wake-v1.

The intervals overlap; their lower endpoints do not rank true policy values.
Neither the actual Rest first move's error nor its superiority over Double Edge
is proved. This improves a research witness only. Ordinary-bot behavior and
sleep-counter rules are unchanged. Deferral remains off for both arms following
its negative fixed-time comparison. No public deployment or strength claim.
