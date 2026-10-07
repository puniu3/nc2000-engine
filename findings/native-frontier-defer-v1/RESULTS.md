# Deferred frontier mode did not activate

All four registered searches reproduced the same Rest-first policy interval
[0.29620143242332375, 1], 921,554 engine executions and 200,068 retained nodes,
stopping at the node cap. Both non-time repeats matched exactly and both controls
reproduced the frozen parent. Independent rational matrix checks and five record
corruption controls passed. The two targeted existing release tests for deferred
successor reconstruction and partial policy bounds passed.

The candidate deferred zero leaves and retained zero exhausted-prefix runs.
This is an activation failure, not evidence that effective deferral cannot help.
Source inspection identifies the cause: enumerate_step_partial returns leaf
scripts only after cap exhaustion. On complete enumeration it returns None;
the solver then eagerly retains every nonterminal child even when deferral is
requested. This search never exhausted the per-cell cap, so both modes did the
same search. Candidate wall time therefore measures bookkeeping overhead, not
an effective memory-saving method. No solver benefit or bot strength is shown.

A follow-up implementation must preserve replayable representative scripts on
complete enumeration as well, validate them against exact successors including
merged damage ranges, and bind a new experiment before comparing outcomes.
Ordinary bot behavior and default analysis configuration are unchanged.
