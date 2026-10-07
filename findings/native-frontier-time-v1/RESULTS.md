# Frontier deferral did not improve the fixed-time bound

All four registered ABBA runs stopped on the 40-second time budget. Elapsed
40,004.6–40,037.5 ms passed the 800 ms overshoot gate; neither the 10-million
execution ceiling nor the 200,000-state ceiling bound these searches.

Default lower guarantees were 0.29112396195450885 and 0.29167290387717026;
deferral yielded 0.2837048150389542 and 0.28978841156229573. Every upper endpoint
remained one. Both candidate lower bounds are below both control lower bounds.
The candidate retained 25,507–25,936 states versus 183,372–184,584, but required
1,025,122–1,057,918 executions versus 839,190–845,058. Smaller retained state
count does not establish reduced resident memory or a better bound per second.

Independent rational matrix checks, source/hash and timing gates, and five
record-corruption rejection controls passed. Timed runs naturally have different
trial counts; no deterministic replay of these timed outputs is claimed. The
previous fixed-work runs and native representative-script verification remain
separate evidence for implementation correctness.

The registered candidate fails its equal-ceiling improvement criterion. Keep
deferral disabled by default; do not extend this sample or retune the time budget
to reverse the result. This is a single fixed-policy witness on this Mac, not a
product latency or ordinary-bot strength test. The next mechanistic lead is the
fixed policy choosing Sleep Talk on states whose existing sleep counter makes
it wake before acting; this is not evidence of a normal-bot error.
