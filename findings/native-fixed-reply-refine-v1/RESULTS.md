# Fixed first reply with a different continuation

Game 104/frame 52 remains unresolved. Against first Rock Slide followed by
Cross Chop priority, Rest's unrestricted security upper bound is
0.09572707038387768. Repeated Rock Slide gives 0.10546875. Neither beats the
previous unrestricted upper bound of 0.08124596542620566; the alternative Drill
Peck lower bound remains 0.045156087695040474. No positive loss was certified.

Both jobs used the registered 2,000,000-execution budget and 200,000-node stopping
threshold. They stopped at the node limit after 1,459,618 and 547,698 executions.
Reported live nodes exclude the closed memo, which also counts toward the limit.
Repeats matched every output field except time. The initial opponent action was
held at Rock Slide in both jobs; only its later priority differed. Every future
choice of the target player remained available.

The follow-up auditor reuses the parent identity, score orientation and reference
checks. It first explicitly checks the new fixed first reply, then normalizes
only that metadata for the inherited priority-first matrix check. It rejects
missing jobs, a changed first reply, a changed continuation, and a false upper
bound. Prior payoff bounds are independently recomputed by rational matrix LP.
The eight fixed-policy/reply release tests from the parent study cover the
existing solver API, including first-action overrides and exhaustive continuations.

Do not export a fixed-opponent lower bound as a guarantee against unrestricted
opponents. Only the target upper bound transfers in this direction. A positive
loss certificate would need an alternative unrestricted guarantee exceeding
that upper bound. This condition was not met. The result neither refutes the
actual choice nor proves its optimality, and does not improve the ordinary bot.

The strongest existing bounds remain authoritative. Further generic budget
increases for these two fixed replies are not justified by this trial. All
protocols and binaries were bound before execution; no public deployment or
main solver change occurred.
