# Native counterfactual loss v1

Plan, protocol and helper registered at `9a7c19c`. The helper is built in the existing native serialization adapter tree with the additive loss-threshold analysis API. The ordinary bot is unchanged.

`records.jsonl.gz` preserves the full raw output with zero gzip timestamp. Source identifiers are `source_game`, `source_frame`, `branch` and `fork_id`; the branch ordinal is never an archived decision frame.

Recheck after decompression:

```sh
python3 tools/native-counterfactual-loss-audit findings/native-counterfactual-loss-v1/protocol.json tmp/native-counterfactual-search-v1/branches.jsonl tmp/native-counterfactual-loss-v1/main.jsonl
```
