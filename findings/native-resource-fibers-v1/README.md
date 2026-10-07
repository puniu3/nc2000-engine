# Native resource fibers v1

Protocol and read-only helper registered at `212cbde`. Projection masks remove active HP (1), active move-slot PP (2), and absolute turn (4) from hash comparisons only. Zero mask is the current guarded solver key. Gzip output preserves raw records; SHA256 hashes refer to uncompressed bytes.

```sh
python3 tools/native-resource-fibers-audit findings/native-resource-fibers-v1/protocol.json tmp/native-resource-fibers-v1/main.jsonl tmp/native-endgame-census-v1/roots.jsonl tmp/native-loss-lower-screen-v1/main.jsonl
```
