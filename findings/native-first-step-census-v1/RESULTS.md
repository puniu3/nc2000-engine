# Native one-step candidate census, cap 256

All 360 roots from 82 games completed: 5,352 legal joint actions, 655,147 engine
executions, 3,570 complete cells and 1,782 partially enumerated cells. All missing
mass remains unknown. Forty-one previously small-loss roots are excluded from
candidate ranking, leaving 319 unresolved roots.

Six roots have an alternative with a strictly greater lower immediate-win
probability against every first reply than the selected action's immediate-win
upper probability. Game104/frame52 (Rest versus Drill Peck) has the largest
minimum difference, about 4.107 percentage points. The remaining five minima
are below one percentage point. The first candidate was already examined by
the fixed-reply studies without certifying an error.

This ranking does not measure long-term regret: recovery, setup, paralysis and
flinching can change later outcomes. It selects diagnostics only. No selected
move was proved wrong by this census and no normal bot behavior changed.

The checker reconstructs the prior exclusion rule, validates coverage and mass
conservation, reproduces all eight complete Marowak reference cells, passes four
synthetic controls, and rejects four record corruptions. The unknown-mass control
ensures missing outcomes cannot be silently treated as zero immediate-win chance.
A registered higher-cap follow-up tests whether unresolved mass hides more
candidates; the initial records and protocol remain frozen.
