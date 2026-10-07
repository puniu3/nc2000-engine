# Completed native current-default versus deployed comparison

All 192 games (96 swapped pairs) completed. Current default won 99 and lost 93, with no ties: score 0.515625. The prespecified pair-based betting95 interval is [0.4083498179, 0.6222688367]. Superiority over deployed c7b11e8 is not established. This is not an equivalence result or evidence of proximity to the optimal ceiling.

The frozen native replay reconstructed all 4,498 decision states and 8,534 player decisions, checking serialized-state fingerprints, legal choices, terminal outcomes, turns, keys and final PRNG seeds. Hidden Power executed 231 times across 76 games. All binary/data/configuration hashes still match registration. No games were discarded, replaced or added.

This measures the bot revisions on one corrected referee and a uniform draw over the current 24-team catalog, using native observations (including the existing exact-HP contract), the same 103-team belief prior and 27,000 iterations. Each version retains its own search and internal simulation engine. It does not reproduce the old public application's incorrect rules, browser timing or human-dependent pondering. Type inference is disabled in both arms.

No runtime setting or deployment is changed from this result. Correctness repairs remain, but overall strength improvement has not been demonstrated. The separate protocol-path HP combination study was also inconclusive (94 wins, 96 losses, 2 ties) and does not isolate the effect of type inference. Further progress should identify remaining decision losses through game-state analysis, rather than treating either near-50% matchup as an optimality certificate.
