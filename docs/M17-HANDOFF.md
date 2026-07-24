# M17 handoff

Updated: 2026-07-25.

## Current state

No local or CX task is running; the CX spot VM is terminated. The worktree is
clean apart from pre-existing user-owned untracked inputs.

- M17e formal exact gate: complete and shipped.
- M17c calibrated cutoff value: complete and shipped.
- M17a formal blind regret gate: complete; no root-ranking patch justified.
- Open-sheet replay provenance: v3 private sheets and fail-closed validation
  complete.
- M17b blind 20k-vs-10k discovery: complete but inconclusive. The full raw
  artifact and decision are in `data/m17b-discovery-v1/RESULTS.md`; 10k remains
  the operational native/PS budget.
- Web open-sheet budget gate: implementation and preregistered 15k/30k/60k
  centered manifest complete; no CPU run has started.
- M17d: the gauntlet is now deterministic, shardable, resume-safe, no-clobber,
  and fail-closed. Its inference unit and paired 95% interval are bound into
  the run fingerprint. No full 4,608-row run has started.

Key checkpoints:

- `89ead3b` — record the M17b raw discovery and inconclusive result.
- `86ebac2` — certify M17d side-swapped inference blocks and 1,000-turn full
  profile.
- `5991d24` / `2ba76cf` — no-clobber, resumable M17d shards and merger.
- `a5f8599` / `3b9ad43` — Web open-sheet centered gate and open stage runner.
- `6f3464e` — fail-closed open-sheet regret provenance.

## Next decisions and runs

1. **M17b blind:** do not use confirmation seeds; discovery nominated no
   candidate. Either retain 10k and close the milestone conservatively, or
   first commit a separate fresh-seed inconclusive-resolution manifest and
   combination rule. Do not add games to or reinterpret the existing
   manifest.
2. **M17d full profile:** launch the resumable 4,608-job run when a 16-CPU CX
   worker is available:

   ```bash
   tools/run-m17d-shards.sh "$CX_OUT" 64 full --threads 16
   ```

   The wrapper fixes the full-profile turn cap at 1,000. Expected duration is
   roughly 12 hours on 16 CPUs. Merge is automatic and requires 2,304 complete
   side-swapped inference blocks with no cap or invalid arm.
3. **Web open-sheet:** run 30k-vs-15k discovery using the staged arena binary
   and `tools/run-m17b-stage.sh ... discovery open`, then apply
   `tools/evaluate-open-centered-gate.py` with
   `data/m17b-open-centered-tier-gate-v1.json`. Ponder is intentionally outside
   this fixed-floor gate.
4. After the CPU results, update the M17 entries in `README.md`, decide which
   remaining research tails are explicitly parked, run the product regression
   suite, and close M17.

## Last regression

At this checkpoint:

- `cargo test --workspace`: green.
- blind arena gate evaluator self-tests: 19/19.
- Web centered gate self-tests: 11/11.
- M17d Rust tests: 5/5; merger self-test and end-to-end shard/resume smoke:
  green.
- All four tracked M17b raw shards pass the arena JSONL validator and reproduce
  the recorded inconclusive gate result.
