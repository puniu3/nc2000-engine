# M17 handoff

Updated: 2026-07-27 (third revision — the exchange term's verdict closed the
additive-term route; Phase B and the confusion term are in, inert, and waiting
on the screen).

## 2026-07-27 in one page

- **The additive-term route is measurement-bound, and that is now a rule.** The
  exchange bonus posted the largest corpus calibration gain on record
  (r 0.585→0.644) and did not ship: four duel arms parity, every
  perceptibility instrument flat, because it changes the bot's top-1 on 13.8%
  of decisions while the *seed alone* changes it on 15.5%.
  `data/exchange-term-verdict.txt` is the record. **New standing step:
  `tools/eval-candidate-screen.py` before any duel** — it reads two
  `human_agreement` artifacts, reports the candidate's top-1 change rate
  against the seed-flip floor, and refuses a verdict without a floor. Its
  exit code is the gate (0 measurable / 1 inconclusive / 2 unmeasurable / 3 no
  floor). First applications: the exchange term 0.89x the floor, the M16c
  rollout arm 1.10x — both unresolvable by any duel.
- **Confusion term + Phase B exchange scheme implemented, shipped inert**
  (`4443548`): `EvalWeights::confusion`, `EvalWeights::exchange_v2`,
  `EvalWeights::exchange_scheme(w, damp)`, `eval::exchange_matrix` as the
  diagnostic surface, harness arms on all three instruments, 10 new tests in
  `crates/bot/tests/exchange.rs`. Details and the smoke numbers are in the M17
  README entry; the gate is open, not passed.
- **The belief prior reaches blind play** (`2606e8d`, `4c6dc59`): a wasm
  `setBeliefPrior(json)` on `ProtocolSearcher` that refuses (out loud, without
  throwing) on an empty/unparseable table, after the first request, or in
  pinned mode, plus `tools/ps-client.js --belief-prior FILE`. `BlindSearcher`
  deliberately did **not** get the setter: the plan named it, but its
  `BlindSearch` has no such method (the setter belongs to `BlindAgent`, which
  the bridge does not hold), it has no blind consumer, and it is the class the
  shipped open-sheet Web app drives — a contamination surface with nothing on
  the far end. What is still unverified is the owner's step: a local-clone game
  in `--mode blind` with and without a table.

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

1. **M17d full profile: DONE (2026-07-25).** 4,608/4,608 pairs, 2,304/2,304
   blocks, zero caps/invalids/exclusions, certified. Paired delta **+0.0087,
   95% [−0.0063, +0.0237]** — no regression, no established gain. Took 2.6 h on
   16 CPUs, not the 12 h this doc estimated. Artifact:
   `~/cx/results/20260725-084811/m17d-full-v2/merged.jsonl`. The first attempt
   died on the turn-cap boundary bug; see the M17d README entry.
2. **Ladder re-exposure: budget SPEC-FIXED at 30k (owner, 2026-07-25), not a
   pending validation.** The web product runs 30k, so 30k is the spec; the PS
   client default already matches. Running games is human-in-the-loop — it needs
   an opponent on a reachable server, and main-ladder botting needs PS staff
   permission — so this is not a dev task that blocks M17. Do it when the
   opportunity arises, with `--mode open --opp-team-file F` wherever the
   opponent's sheet is genuinely available. Expect 11–20 s/move.
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
   2. **Voluntary switching — read `docs/SWITCHING-QUESTION-HANDOFF.md` FIRST.**
      It supersedes the notes below: both obvious fixes are now done and null,
      the two cheap instruments cannot answer the question, and **a CX job is
      in flight (`20260725-211409`) whose result is the next step.** Do not
      start switching work without reading it.

      **Do NOT re-attack this at the rollout layer.**
      It is M16b's worst stratum (`kind=switch` top-1 24.9% vs move 42.8%),
      and the obvious L2 fix is already built, already measured, and already
      parked: `RmConfig::rollout_m16c` (default **false**) carries
      bad-matchup voluntary switching plus `status_pseudo_score`, and the
      2026-07-21 measurement was null — corpus agreement 39.3%→38.7% overall
      and **switches 25.0%→23.8%**, i.e. slightly worse, at self-play parity.
      The recorded diagnosis is that at product budgets the tree, not the
      rollout tail, owns the root values. An earlier revision of this work
      order listed "voluntary switching" as a fresh item; that was written
      without checking, the same staleness that hid two already-shipped eval
      features.

      The one lever that postdates that null result is the **Spikes term
      shipped 2026-07-25**, which is a switching tax by construction and
      should move switch valuation if anything does. So the first step here is
      a measurement, not a fix: re-run M16b with the term on and off and see
      whether the switch stratum moves. That also tests the product rationale
      the term shipped on. Needs the weight override plumbed into
      `corpus::reconstruct_*` (today `corpus::cfg()` silently takes
      `EvalWeights::default()`), which is why the plumbing is worth building
      properly rather than hacking.
   3. **Confusion eval feature** (8.6%), the second missing term — **built,
      inert, gate open.** `EvalWeights::confusion` default 0.0, priced per
      expected lost turn off the engine's confusion clock. Smoke (30 battles /
      90 positions / 8 playouts / GT 150) improves monotonically to 1.5
      (r 0.596→0.607, Brier 0.2060→0.2042); that scale's last prediction was
      noise, so it decides nothing.

      **The Phase B exchange scheme is the same gate and should be run in the
      same batch** (`EvalWeights::exchange_scheme(w, damp)`, `exchange_v2`).
      Smoke peaks at exchange **0.5 with damp 1.0** (r 0.596→0.640, Brier
      0.2060→0.1984) — i.e. the matrix pays as an *addition*, and moving the
      status/Substitute/Spikes weights out of the additive sum makes
      calibration worse at every weight. Order, and do not skip a step:

      ```
      # 1. full corpus calibration (the CX-scale run; ~80 min at this shape)
      eval_calibration --corpus tmp/corpus-spectator --battles 0-569 \
          --playouts 32 --gt-iters 300 --ab --confusion-sweep --scheme-sweep
      # 2. footprint screen at PRODUCT budget, against the seed floor
      human_agreement --corpus tmp/corpus-spectator --battles 0-59 \
          --iters 30000 --seed 1 [--scheme W --scheme-damp D | --confusion W]
      tools/eval-candidate-screen.py BASE CAND BASE-OTHER-SEED
      # 3. seed-paired duel ONLY if the screen clears the floor
      eval_ab_duel --games 800 --iters 3000  --scheme W --scheme-damp D
      ```

      A baseline must be re-measured through today's code rather than reused:
      `tmp/ha-30k-base2.jsonl` predates the 2026-07-27 reveal-channel fixes
      (`db7d7cb`, `a5495f9`, `3d34eaf`), which move belief imputation and hence
      the rows. The seed floor from `ha-30k-s1`/`s2` (0.1547) is a property of
      search noise and is expected to carry, but it is cheap to re-derive in
      the same batch.
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
