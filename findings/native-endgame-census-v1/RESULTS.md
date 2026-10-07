# Native endgame census

Replaying all 192 completed native comparison games recovered 4,498 decision states and 8,534 player decisions. The current bot made 4,266 decisions; 3,754 were idle move requests. Both players had one selected Pokemon remaining at 422 current decisions, of which 365 were idle move requests. Removing five forced-action roots leaves 360 decisions from 82 games.

All four opponent move IDs were publicly revealed at 106 roots from only six games. Game 62 contributes 87 of these roots. Revealed moves plus known current item cover 95 roots from five games. These are correlated positions, not 106 independent observations. Revealed Hidden Power identifies the generic move, not its type or power; stats, unobserved PP and other hidden state can also remain uncertain. No root is certified fully observed by this census.

All 360 roots retained at least one belief candidate (242 have one, 95 have two, 23 have three); none used fallback or had transformed active Pokemon. One catalog candidate does not certify true hidden-state knowledge or a complete legal-world model. Both active Pokemon retain Rest PP at 185 roots from 14 games, so many roots require resource-aware analysis rather than only a short damage race.

The helper preserves the current worker's observer/belief lifecycle from team preview, synchronizing at each actual current-side decision, and verifies all saved search seeds. It does not rerun or alter bot decisions. Actual stats, items and move loadouts are diagnostic truth in a separate field; they are never supplied to search. Full snapshots retain history and the original state fingerprint. Every snapshot round-trips through the isolated native serialization adapter with identical debug clone and state key.

A complete repeat reproduced both census and snapshots byte for byte. The Python audit independently checks 360 snapshot hashes against saved decision frames, selection conditions, saved responses and move-reveal classification. It rejects snapshot-HP corruption, response corruption and a missing snapshot.

This census proves neither a policy error nor a strength improvement. The next analysis must bound action losses and separate hidden-information limitations from search/evaluation mistakes. It should retain unresolved cases and report results by game as well as by root, because one long game dominates the fully revealed stratum. No runtime behavior, public deployment, sleep-counter semantics or Counter rules changed.
