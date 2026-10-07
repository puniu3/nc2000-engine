# Reproduce the counterfactual census

Build native_counterfactual_forks.rs as a bot example in a serialization-adapter checkout with the current bot runtime (the existing tmp/native-loss-threshold checkout is suitable). Copy its release binary to tmp/native-counterfactual-forks-v1/generator before building another worktree. Source and binary hashes are in protocol.json.

Invoke the helper with PROTOCOL GAMES OUTPUT, redirecting stdout to census.json. The source is the frozen native-deployed-duel-v1 games JSONL; all dex, catalog and belief hashes are checked. Output is created exclusively. With measure_search false, no bot search is run. Repeat into repeat-branches.jsonl and repeat-census.json and compare bytes.

Run `python3 tools/native-counterfactual-forks-audit findings/native-counterfactual-forks-v1/protocol.json tmp/native-deployed-v1/main/games.jsonl tmp/native-counterfactual-forks-v1/branches.jsonl tmp/native-counterfactual-forks-v1/census.json`. branches.jsonl.gz preserves every native child Battle; census.json.gz preserves sorted metadata. Both use gzip mtime zero. Fork IDs encode source game, source current-decision frame and opponent-action ordinal; they are not new archived frame indices.
