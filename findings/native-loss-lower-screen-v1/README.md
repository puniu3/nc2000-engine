# Reproduce the complete loss-lower screen

Use the frozen helper/binary from native-loss-threshold-v1, implementation commit 2a63a63, this protocol.json, original census roots and frozen duel dex. Invoke PROTOCOL ROOTS DEX OUTPUT. The 70 indices were fixed at 3e554f2; all targets are actual archived actions, with no overrides or outcome-based exclusions.

Run `python3 tools/native-loss-lower-screen-audit tmp/native-loss-lower-screen-v1 tmp/native-endgame-census-v1/roots.jsonl` to recreate summary.json while the source files and binary match their recorded hashes. The verifier checks complete index coverage, source provenance, all matrices, threshold decisions, overlap with prior bounds and the six repeated pilot roots. records.jsonl.gz uses mtime zero.
