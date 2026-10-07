# Retained HP resource families

The first two still-unresolved roots in each direct-healing stratum were selected from the frozen 70-root loss screen. Read-only diagnostics reproduce every frozen bound, selected action, stop, execution count, node count, configuration and pre-existing statistic. Only elapsed time and diagnostic fields differ.

| Source game/frame | Retained states | Groups ignoring active HP | Reduction |
|---|---:|---:|---:|
| 0/18 | 14,435 | 287 | 98.01% |
| 3/97 | 36,127 | 242 | 99.33% |
| 4/60 | 33,114 | 426 | 98.71% |
| 10/32 | 35,099 | 472 | 98.66% |
| 13/41 | 21,118 | 303 | 98.57% |
| 16/18 | 14,583 | 1,241 | 91.49% |

Ignoring active PP alone, turn alone, or both together reduces no retained-state count. Adding these omissions to HP does not merge any more groups in this sample. Between 97.33% and 99.85% of live nodes have not initialized their action matrix. No resource-closure scheduling was activated in these runs. Game 16/frame 18 hits the trial depth cap 284 times; the other five roots have no depth-cap hit.

The count comparison identifies HP variation as the immediate retained-frontier multiplicity in these six graphs. It does not establish a safe state quotient, equal continuation values, HP monotonicity, or the total reachable game size. Normalized states were hashed only and never simulated or inserted into the solver. The existing guards for dead damage bookkeeping and unobservable Quick Claw rolls were preserved.

Independent checks reconcile all projection histograms and subset counts, live node counts, trial depth totals and each root-cell width against pending chance mass plus propagated child uncertainty. The independent rational payoff checker passed, and four deliberate data corruptions were rejected.

Next implementation work should target joint computation across HP families while retaining distinct HP outcomes and their probability mass. The existing per-script HP cache is not sufficient evidence for speed: prior policy-hp-eager-v1 diagnostics measured greater elapsed time despite cache hits. A new method needs complete successor-distribution comparisons across KO, healing and item boundaries before value bounds may share any work. No ordinary bot change or strength improvement was established.
