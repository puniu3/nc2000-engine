# One-step bounds on retained frontier states

The frozen Rest-first search reproduced every non-time output from the parent
Snorlax witness study. It retained 197,737 unexpanded eligible states. The
registered sample selects 64 evenly spaced entries after sorting exact state
keys; it is not weighted by reach probability.

Keeping the same own fixed policy and all four opponent replies, 30,048 extra
engine executions bounded one step at these states. Of 256 cells, 207 completed
and 49 retained unknown probability mass. Lower bounds were positive at 49 of
64 states, but only seven exceeded 0.1. Those seven were awake attack states
with lower bounds approximately 0.808 or 0.902. The remaining 42 positive bounds
were approximately 0.002–0.011. All upper bounds remained one. Among 52 sleeping
states, 41 had small positive bounds and 11 had zero lower bounds; all four
frozen states had zero lower bounds.

Fresh reconstruction and enumeration verified all 256 cells with maximum
probability difference zero, using 30,048 replay executions and 64 additional
policy-validation executions. These are verification costs, not performance
samples. The auditor checks sampling, state hashes, mass conservation, bound
directions, parent-search identity and replay accounting. Five deliberate
record corruptions were rejected. Summary bounds include outward padding of
1e-10; raw floating-point helper outputs are not unpadded certificates for
future solver injection.

No bounds were injected into the graph, and neither the solver nor the ordinary
bot changed. The diagnostic does not demonstrate speedup, useful coverage over
all reachable states, or strength improvement. The small sleeping-state bounds
leave open whether a higher per-cell enumeration cap helps; any follow-up must
distinguish enumeration truncation from an intrinsically weak one-step horizon.
