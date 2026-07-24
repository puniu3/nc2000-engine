# M17 handoff

Updated: 2026-07-25 (second revision — M17b closed, Web budget gate parked).

## Current state

No local or CX task is running; the CX spot VM is terminated. The worktree is
clean apart from pre-existing user-owned untracked inputs.

- M17e formal exact gate: complete and shipped.
- M17c calibrated cutoff value: complete and shipped.
- M17a formal blind regret gate: complete; no root-ranking patch justified.
- Open-sheet replay provenance: v3 private sheets and fail-closed validation
  complete.
- M17b blind 20k-vs-10k discovery: complete but inconclusive, and **closed
  conservatively (owner)** — no resolution run, confirmation seeds unspent, the
  manifest not reinterpreted. Full raw artifact in
  `data/m17b-discovery-v1/RESULTS.md`.
- PS client budget: `tools/ps-client.js --iters` default moved 10,000 → 30,000
  to match the shipped Web budget, so ladder/postmortem evidence describes the
  shipped configuration (closes M17a's blind-10k-vs-open-30k scope caveat).
  Configuration alignment, **not** a gate promotion. `--mode open` stays opt-in:
  it needs the opponent's real sheet via `--opp-team-file`.
- Web open-sheet budget gate: implementation and preregistered 15k/30k/60k
  centered manifest complete, **PARKED unrun (owner, UX grounds)** — 30k +
  ponder is the product sweet spot; see the Parked section of `README.md` for
  the full rationale and the reopen conditions.
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

1. **M17d full profile:** launch the resumable 4,608-job run when a 16-CPU CX
   worker is available:

   ```bash
   tools/run-m17d-shards.sh "$CX_OUT" 64 full --threads 16
   ```

   The wrapper fixes the full-profile turn cap at 1,000. Expected duration is
   roughly 12 hours on 16 CPUs. Merge is automatic and requires 2,304 complete
   side-swapped inference blocks with no cap or invalid arm.
2. **Ladder re-exposure at the aligned budget:** the next PS batch runs on the
   new 30k default, so its postmortems are finally about the shipped bot. Add
   `--mode open --opp-team-file F` wherever the opponent's sheet is genuinely
   available. Expect roughly 11–20 s/move (single-threaded wasm-in-node) — well
   inside the 150 s per-turn budget, but budget the wall-clock per game.
3. **Strengthening tails — ALL ACCEPTED (owner), none parked.** Full status and
   the re-derived cluster ranking are in the M17 README entry. Work order:
   0. **Corpus position source for `eval_calibration`** — now a prerequisite,
      not an optional extra. The harness generates positions by skuct
      self-play only, so conditions the bot under-uses are almost absent from
      it (Spikes: 2 of 36 in a smoke run vs 21.4% turn-weighted in the
      corpus). Any condition-feature weight calibrated on that distribution is
      calibrated on a distribution the blind spot itself produced. Build the
      corpus arm the M16a plan specified, reusing the importer path that
      `human_agreement.rs` already drives.
   1. **Spikes eval feature** — term IMPLEMENTED and shipped inert
      (`EvalWeights::spikes`, default 0.0, gated behind `!= 0.0`).
      **Calibrated: the weight is 1.5.** Corpus run, 570 positions x 32
      playouts, GT skuct:300, spikes slice 115 positions (105 one-sided):
      the oriented bias crosses zero at 1.5 (+0.061 off, −0.002 at 1.5) and
      r / Brier / MSE all optimise there too (0.580→0.588, 0.2123→0.2107,
      0.0860→0.0844). Three independent criteria agree. **Gate: parity.**
      Seed-paired, 400 games each: 0.530±0.046 at 300 iters, 0.498±0.047 at
      1000 — the 300-iter edge did not replicate, so read the pair as no
      measurable strength change, at identical think time (280 vs 280 ms).
      **DONE — default flipped to 1.5** on the Rev-1 bar (better calibration
      + parity + no cost) plus the owner's product call: at equal strength,
      behaving the way a human expects is worth more here than a strength
      delta too small to measure.

      Not yet measured, and the natural next check: whether the term moves
      M16b top-1 agreement, especially the switching cluster. Spikes is a
      switching tax, so it should show up exactly there. `human_agreement`
      picks up `EvalWeights::default()` through `corpus::cfg()`, so a
      before/after needs a weight override plumbed into `reconstruct_*` —
      do it as part of the M16b cluster item rather than as a one-off.
   2. **Voluntary switching** — M16b's worst stratum by a wide margin
      (`kind=switch` top-1 24.9% vs move 42.8%). Touches L2 rollout as much as
      L1 eval, so expect it to be the biggest of the five.
   3. **Confusion eval feature** (8.6%), the second missing term.
   4. **Status-move valuation**, then the Curse/Body Slam, Rest, and Perish
      Song clusters — likely overlapping causes, so re-measure M16b between
      them rather than fixing all four blind.
   5. **Weight re-tuning** last: it re-runs M6's SPSA plateau conclusion against
      whatever eval the items above leave behind, so doing it earlier wastes it.

   Every item takes the standing gate: seed-paired arena (`--ab` +
   `eval_ab_duel`) vs the M16-exit bot, then ladder. Re-run
   `tools/aggregate-human-agreement.py` after each to check the cluster moved.
4. After the M17d result and the tails, update the M17 entries in `README.md`,
   run the product regression suite, and close M17.

Do **not** run the Web open-sheet budget gate: it is parked, and its manifest
exists so that reopening is cheap, not as a queued task. Reopen conditions are
in the Parked section of `README.md`.

## Last regression

At this checkpoint:

- `cargo test --workspace`: green.
- blind arena gate evaluator self-tests: 19/19.
- Web centered gate self-tests: 11/11.
- M17d Rust tests: 5/5; merger self-test and end-to-end shard/resume smoke:
  green.
- All four tracked M17b raw shards pass the arena JSONL validator and reproduce
  the recorded inconclusive gate result.
