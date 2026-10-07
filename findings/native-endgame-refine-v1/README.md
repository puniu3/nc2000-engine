# Reproduce the deeper native endgame screen

Use the frozen helper and executable identified in native-endgame-bounds-v1, with this directory's protocol.json and the original native-endgame-census-v1 roots. The protocol fixes all 70 indices before execution (commit 7f4ae3f). The common helper invocation is PROTOCOL ROOTS DEX OUTPUT; the output file must not exist.

The repeat uses the same settings for game/frame pairs (7,21), (181,97), (0,18), (1,37), covering both newly narrow roots and two unresolved controls. Save it as repeat.jsonl. Run `python3 tools/native-endgame-refine-audit tmp/native-endgame-refine-v1 tmp/native-endgame-census-v1/roots.jsonl tmp/native-endgame-bounds-v1/main.jsonl`. The verifier checks the original corpus hashes, deterministic earliest-unresolved-per-game selection, all interval nesting and repeat equality, and invokes the frozen rational matrix checker.

records.jsonl.gz and repeat-records.jsonl.gz are deterministic gzip archives (mtime zero). The raw hashes are in summary.json. No previous evidence archive or runtime source was modified.
