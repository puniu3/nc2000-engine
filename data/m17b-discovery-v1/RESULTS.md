# M17b blind budget discovery — 20k vs 10k

Status: **INCONCLUSIVE** (2026-07-24). The deployed/native blind budget
therefore remains unchanged at 10,000 iterations; this artifact does not
authorize either promotion or a lower-budget recommendation.

## Result

- Population: meta pool, no baked preview tables, max 500 turns.
- Design: four preregistered seeds, 200 side-swapped games per seed.
- Total: 800 games / 400 side-swap pairs; zero caps and zero invalid games.
- `blind:20000:1:16`: 419 wins, 381 losses, score **0.52375**.
- Normal 95% interval over side-swap pairs: **[0.49176, 0.55574]**.
- Gate thresholds: promote only if mean ≥ 0.53 and lower bound > 0.50;
  stop only if upper bound < 0.55. Neither fired.

The machine-readable decision is [`gate-result.json`](gate-result.json).
Reproduce it (expected exit status: 1 because the gate is inconclusive):

```bash
python3 tools/aggregate-arena.py \
  data/m17b-discovery-v1/m17b-*.jsonl \
  --gate-manifest data/m17b-tier-gate-v1.json \
  --gate-out /tmp/m17b-gate.json
```

## Provenance

- CX task: `20260724-112350`,
  `m17b-20k10k-discovery-18c9e1b-16c`
- Machine: `c2d-highcpu-16`; elapsed 4,205 seconds.
- Staged arena binary SHA-256:
  `ec524227d4229b899779267714a7e6992d3e0989644819097a6fecc8718e805c`
- Arena build fingerprint:
  `fnv1a64:8b33cc92799829ea:arena-build-v1:1parts`
- Gate manifest hash:
  `sha256:873c391ceb71e0dfae19b2301eb198db23f0ddc8398282c52f694637775b1455:arena-tier-gate-manifest-v1`
- Evaluator hash:
  `sha256:b0978d6f12e660899cffbaee42357e1a6076be4d6b15b69f89673fba566826b9:arena-tier-gate-evaluator-v1`

Raw shard SHA-256:

```text
f4ec51de27de8c8fd64c2c20057e79f95cfb269b0178080d78498dc326dbc552  m17b-20000v10000-discovery-1700000001.jsonl
afd6fc9de26eaec0a2a3fcc5784a64279f1acd2d2830fc5a39eed334955913e5  m17b-20000v10000-discovery-1700000002.jsonl
c926cdf5559024f990c40923d8dd9f31c015c9928ac2ba2274fe428235be17eb  m17b-20000v10000-discovery-1700000003.jsonl
2017ca67df4dacfa8254b6a2be1f1d064619d221e917212339110f1b8dfbaf71  m17b-20000v10000-discovery-1700000004.jsonl
```

## Closure (2026-07-25, owner)

Closed with the conservative option below: **no resolution run, no confirmation
seeds, nothing above reinterpreted.** The gate's evidence stands exactly as
recorded and nominates no knee.

Separately — and *not* on the strength of this artifact — the `--iters` default
of `tools/ps-client.js` moved 10,000 → 30,000 to match the shipped Web budget
(open sheet, 30k + ponder), so that ladder and postmortem evidence describes
the configuration that actually ships. That 10,000 was a harness flag default
whose origin was a seed-stability floor (battle-3623 T6), never a tuned
operating point and never a shipped product parameter. **This is configuration
alignment, not promotion**: nothing here measured 30k, and no reader should
cite this artifact as evidence for it.

## Next-session rule

Do not spend the confirmation seeds: discovery nominated no knee candidate.
Before collecting more games, commit a separate, fresh-seed inconclusive-
resolution manifest and its combination rule. Alternatively, explicitly close
M17b with the conservative operational decision to retain the current 10k
budget. Do not reinterpret or extend this manifest post hoc.
