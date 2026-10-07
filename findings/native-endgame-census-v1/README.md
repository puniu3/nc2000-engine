# Reproduce the endgame census

The input is the frozen `native-deployed-duel-v1/records.jsonl.gz`, also retained at `tmp/native-deployed-v1/main/games.jsonl`. Its manifest references the frozen dex, catalog and belief pool. `summary.json` records its SHA-256 and the helper/binary/output hashes.

Build `native_endgame_census.rs` as a bot example in the existing `tmp/hp-native-current` serialization-adapter checkout (runtime base `6c2bd9e`), following `native-wire-v1/README.md`. The ordinary main engine deliberately has no Battle serialization adapter. Use no extra bot features. Copy the resulting release executable to `tmp/native-endgame-census-v1/census`.

Run `census GAMES.jsonl ROOTS.jsonl`, redirecting stdout to `census.json`. Repeat with `repeat-roots.jsonl` and `repeat-census.json`. Then run `python3 tools/native-endgame-census-audit tmp/native-endgame-census-v1 tmp/native-deployed-v1/main/games.jsonl`. Output is the committed summary. Both gzip archives use mtime zero.

The helper derives from the frozen native replay without modifying that archive. `census.json.gz` contains root metadata sorted by game/frame; `roots.jsonl.gz` contains the corresponding full native Battle snapshots in recorded game completion order. The filter is current agent 0, both selected teams down to one Pokemon, idle joint move request, and more than one current legal action. No winner, action quality or solver output enters the filter. This is an exploratory census, not a preregistered strength test.
