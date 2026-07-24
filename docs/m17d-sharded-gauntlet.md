# M17d spot-safe full gauntlet

The full off-pool fallback A/B workload is deterministic but long-running:
18 custom teams × 4 pilots × 64 games = 4,608 paired jobs. Run it as
independent, resume-safe shards:

```bash
tools/run-m17d-shards.sh tmp/m17d-full 64 full --threads 16
```

This produces 72 half-open job ranges, `shard-START-END.jsonl`, and then
`merged.jsonl`. Re-running the same command validates and skips completed
shards. A changed executable, input dataset, selected team, semantic config,
seed, shard plan, or workload refuses resume instead of mixing lineages.
The wrapper fixes the full profile's turn cap at 1,000; the earlier 500-turn
diagnostic capped one layered arm and was uncertified.

For a staged/spot worker, copy the release example binary, `tools/`, and the
data/fixture inputs, then set:

```bash
NC2000_REPO_ROOT="$PWD" \
NC2000_M17D_BIN=./offpool_fallback_gauntlet \
tools/run-m17d-shards.sh "$CX_OUT" 64 full --threads 16
```

`manifest.json` binds the executable/data/team/config identity and every
planned battle/agent seed. The merger requires exactly one row for every job,
checks the seed and team lineage, recomputes every shard summary/fingerprint,
and rejects any missing/duplicate/reordered row, invalid arm, turn cap, score
inconsistency, or mismatched shard. Each adjacent, same-battle-seed
side-swapped orientation pair is one inference block. The preregistered effect
estimate is the mean of those block-level layered-minus-legacy deltas; its 95%
interval is `mean ± 1.96 × sample standard error` over blocks. A full merge
requires all 2,304 blocks and reports the block mean, half-width, and bounds.
Shard writes use same-directory temporary files followed by no-clobber atomic
publication; an existing merged artifact is accepted only when byte-identical.

Standalone operations:

```bash
# Plan only (no games).
target/release/examples/offpool_fallback_gauntlet \
  --profile full --max-turns 1000 --shard-size 64 \
  --manifest-out tmp/m17d-full/manifest.json

# Validate one shard.
python3 tools/merge-m17d-shards.py \
  --manifest tmp/m17d-full/manifest.json \
  --check-shard tmp/m17d-full/shard-000000-000064.jsonl \
  --start 0

# Fail-closed merge after all shards exist.
python3 tools/merge-m17d-shards.py \
  --manifest tmp/m17d-full/manifest.json \
  --out tmp/m17d-full/merged.jsonl
```
