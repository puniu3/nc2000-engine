# Native attack regions v1

Implementation `69f8da1`; benchmark binding `f3c277f`. Inputs reuse the complete archived retained-family census in `native-attack-kernels-v1`. The main ordinary bot and main bound solver do not call this prototype.

```sh
python3 tools/native-attack-kernels-audit findings/native-attack-regions-v1/protocol.json tmp/native-attack-kernels-v1/census.jsonl tmp/native-attack-regions-v1/benchmark.json tmp/native-attack-regions-v1/bench
```
