# Counterfactual response loss results

All 151 preregistered novel eligible successors from 57 source games completed. These were selected from the no-search census, before bot responses and loss outcomes.

- 29 selected actions have fixed-world pure-action loss at most 0.02.
- 122 remain unresolved; stops: {'WorkExhausted': 81, 'AtMostThreshold': 29, 'NodeBudget': 41}.
- No selected action has positive loss lower bound above 1e-8 or certified strict domination.
- 28,115,725 engine executions; every root respected the 250,000 execution budget. Node count is a batched-expansion stopping threshold, not a hard allocation cap.

All matrices were independently recomputed with rational LP vertex enumeration. Five corruptions were rejected: missing fork, wrong source frame, wrong selected action, invalid cell interval and false threshold-stop claim. Source snapshots are bound to the preceding full fork replay and 45 exact native-search controls.

The analysis fixes actual hidden state and assumes full-information optimal continuation with independent chance classes. Pure-action loss is not blind randomized-policy exploitability. The 151 constructed continuations are not independent games and must not be combined with the original 360 census as a random sample. No ordinary bot or deployment change; no evidence of improved overall strength or a game-wide ceiling.

Expanding the same bounded analysis to additional reachable continuations has not yielded a new actionable mistake. Unresolved bounds remain the limiting evidence, so further work should address solving structure rather than treating more corpus entries as strengthening progress.
