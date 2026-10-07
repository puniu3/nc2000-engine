# Native counterfactual search v1

Protocol committed at `614635d`; uses the frozen generator in `native-counterfactual-forks-v1`. Raw branch snapshots and metadata are gzip-compressed with zero timestamp. `summary.json` records the audit and uncompressed hashes.

Recheck after decompression:

```sh
python3 tools/native-counterfactual-forks-audit findings/native-counterfactual-search-v1/protocol.json tmp/native-deployed-v1/main/games.jsonl tmp/native-counterfactual-search-v1/branches.jsonl tmp/native-counterfactual-search-v1/census.json
```
