# Optional region cache in the full analysis solver

Decision: do not integrate this cache into the main bound solver. The complete control/regions/regions/control comparison finished for all six preregistered roots at 250,000 actual engine executions and a 50,000-node stopping threshold. Each pass is serial, with no overlapping benchmarks.

The controls reproduce frozen loss bounds, every payoff cell, actions, stops, execution counts, node counts and pre-existing statistics exactly after removing the added disabled config field. Both repeats agree for every non-time output in each arm. The independent rational matrix audit passes; four deliberate record corruptions are rejected.

Across six roots, there are 7,403 eager enumeration calls per candidate pass. Only 12 calls pass the existing attack-translation admission guard, all in game 4/frame 60. None reuses a stored kernel. That root stores 10,822 leaves without a hit. All root/cell bounds, losses and engine execution counts remain unchanged. Both arms spend exactly 1.5 million engine executions per pass.

Total measured times are 34.609 and 33.613 seconds for controls, versus 33.781 and 33.845 seconds for candidates. The 0.9% mean difference, with zero reuse and no work or bound improvement, is not evidence of useful solver acceleration. The preceding 11.9% conditional microbenchmark does not transfer to the solver's actual traversal.

The retained graph contains many HP-only variants, but most are unexpanded; this multiplicity is not the number of repeated admitted computations. Further reuse work must first examine actual expanded action pairs and guard exclusions. Do not widen conditions merely to increase hit counts: successor distributions and KO/item/healing boundaries must remain exact.

Integration changes exist only as `bounds.patch` and the isolated compiled experiment. Main BoundConfig, ordinary bot behavior, game rules and deployment are unchanged. The guarded kernel remains an opt-in research API. No strength improvement or game-wide optimality result is established.
