# Complete bounded screen

All 360 roots from 82 games were retained. The fixed-world minimax value interval was at most 0.02 wide at 36 roots; the selected pure first action's minimax loss was at most 0.02 at the same 36 roots. No selected action had a loss lower bound above 0.02, and no selected action was proved strictly dominated. The remaining 324 selected-action loss intervals are unresolved, not evidence of optimality. No randomized blind-policy error or strength gain is established.

Stopping reasons were 20 StrategyMet, 177 NodeBudget and 163 WorkExhausted. The 5,000-node setting is a stopping threshold checked between expansions; a batched expansion can cross it. The largest nodes-plus-closed count was 8,161. Work did not exceed the 25,000-engine-execution limit per root.

The checker verifies every root against the frozen census, every legal action and selected action, interval orientation, pure-action guarantees/losses and strict domination gaps. It independently brackets root values by enumerating the payoff matrix LP vertices using rational arithmetic. Three known matrix games cover mixing, a dominated row and side reversal. Four corrupted-record controls are rejected. The four preliminary smoke roots reproduce in the complete scan exactly except wall time; helper and executable hashes match the protocol.

The next deeper screen selects the earliest unresolved root in each of the 70 games with unresolved roots, before seeing deeper results. This reduces concentration in long games. It remains a fixed-world diagnostic, and all unresolved cases must remain visible.
