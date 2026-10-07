# Fixed-state frontier refinement

Raising the enumeration cap from 256 to 4096 completed all 256 joint-action
cells at the same 64 states, using 39,270 engine executions plus 64 policy checks.
The 207 previously complete cells are unchanged. All 1,024 outcome-probability
interval comparisons are consistent. Independent replay reproduced all cells
with maximum probability error zero; four deliberately corrupted records fail.

The number of lower bounds above 0.1 rose from seven to 49. For 41 sleeping
states, the one-step guarantee is now approximately 0.269–0.333; the other 11
sleeping states still have zero immediate-win lower bounds. Eight awake states
have guarantees approximately 0.729–0.902, and four frozen states remain zero.
All upper bounds remain one. Thus truncation, rather than only a short horizon,
explains much of the weak sleeping-state lower bounds in the initial probe.
A higher cap is unnecessary for this sample because every cell completed.

These are one-step diagnostic bounds, not optimal values. Bounds have not been
injected into the solver. The root search fields in the output are copied from
the frozen parent, not a newly timed search. The sample is not probability
weighted. No solver speedup or ordinary-bot strength gain has been demonstrated.
The existing solver can defer eager nonterminal leaves as replayable chance
scripts; a separately registered integrated comparison can test this mechanism
without adding a new bound-injection implementation.
