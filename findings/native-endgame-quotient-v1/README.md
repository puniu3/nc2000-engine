# Reproduce the quotient confirmation

Use the frozen native_endgame_scheduling helper and executable from native-endgame-scheduling-v1 with this protocol.json, the original census roots and frozen duel dex. Invoke PROTOCOL ROOTS DEX OUTPUT; OUTPUT must not exist. Only dead_quick_claw_quotient is enabled; eager-prefix retention and selected-only targeting stay disabled.

The complete subset was fixed at 2b2d631: all 64 original deeper-screen indices outside the six diagnostic indices. Each root retains 250,000 engine executions, a 50,000-node stopping threshold, cell cap 4,096 and all legal pure actions as refinement targets.

Repeat (game,frame) pairs (1,37), (7,21), (52,54) with identical settings, saving repeat.jsonl. These are the newly narrow diagnostic root, first improved confirmation root and first worsened confirmation root. Run `python3 tools/native-endgame-quotient-audit tmp/native-endgame-quotient-v1 tmp/native-endgame-census-v1/roots.jsonl` to recreate summary.json. Deterministic gzip archives use mtime zero and retain all unresolved records.
