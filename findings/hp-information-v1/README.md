# Hidden Power evidence × opponent move information

Research diagnosis only. The ordinary bot and the default-off HP evidence feature are unchanged.

The protocol was fixed before the 27,000-iteration outcomes. It reuses three constructed stress fixtures and one on-pool control, with fresh seeds: 24 stress blocks and two control blocks, four games per block. Arms 0/1 use normal hidden-move inference with public HP evidence absent/present; arms 2/3 supply the exact four base move identities of all six opponent roster slots with evidence absent/present. HP remains generic. The opponent uses ordinary move inference and public HP evidence in every arm.

The move oracle feeds a persistent research observer a separate battle clone with base-slot usage flags set. Actual battles and search roots stay untouched. A test covers both sides of every forced prefix, stable revisions on repeated observation, unchanged battle state and move usage, and preservation of every non-move observation field. Full move information can indirectly condition prior genotypes and preserve live PP; the intervention does not hold those downstream effects fixed. It cannot be shipped as a hidden-information policy.

All 104 games run regardless of outcomes or missed exposures. Capped results remain unknown. Type effects are paired within each move-information condition; their difference is exploratory. These fixtures, opponent and small sample do not estimate general strength, human superiority, or distance to a game-theoretic ceiling. There is no adoption gate based on these outcomes.

A 64-iteration smoke on a separate seed completed 32 games and 1,001 replayed decisions. The audit samples 64 determinizations at every non-forced, multi-action decision. All oracle loadouts matched all six true canonical move sets; every evidence-enabled active model satisfied the public type mask. Full-strength results are recorded separately after completion.

Run `tools/hp-information-audit PROTOCOL RESULTS OUTPUT` to replay all actions, state keys, seeds, observation masks, oracle views and outcomes; audit modeled loadouts; and archive records and paired summaries. The checker requires the recorded source and executable hashes. It does not rerun every search.
