# Solving the battle game

Status: **CUT by the owner on 2026-10-02** after S1 reached but did not clear
the current search (see [Verdict](#verdict-cut)). Kept as the record of what
was built and measured; the code stays in-tree, inert.

## Owner decisions (2026-10-02)

1. Goal: a bot that clearly outplays strong human players. The route is to
   approximately solve the game, not to tune the heuristic bot further or to
   gate work on scarce human matches.
2. Two README non-goals are lifted: whole-game equilibrium solving, and neural
   networks. Inference stays CPU/wasm-sized (no GPU at play time).
3. Compute is local. The cx VM contract is cancelled; if local resources run
   out, capacity is acquired to fit the measured problem size.

## Formulation

With both teams' sets known, a battle is a finite two-player zero-sum
stochastic game with simultaneous moves (a Shapley game; the engine's turn
limit makes it terminate). Every state s has a value V*(s), and

    V*(s) = val( M_s ),   M_s[i][j] = E_chance[ V*(s') | s, i, j ]

where `val` is the matrix-game value (row = side 0, maximizing) and the
expectation is over the engine's chance outcomes of joint action (i, j).
Optimal play at s is the equilibrium mixture of M_s.

Exact solution is out of reach: the HP combinations alone exceed 10^13 states.
The program approximates V* with a small network V_θ trained by fitted value
iteration on sampled Shapley backups, and plays by shallow search (exact or
sampled chance, matrix games at every node) with V_θ at the leaves. The
backgammon programs (TD-Gammon, GNU BG, XG) reached superhuman play the same
way: a learned evaluation plus a 2–3 ply chance-exact lookahead.

The current bot is not this. Its rollouts plus handcrafted eval estimate the
value of its own policy rather than V*, and argmax-visits play is a pure
strategy at simultaneous nodes, which a reading opponent can exploit.

Hidden information (the opponent's two benched picks, at most 10
combinations, and unrevealed sets under a known prior) is small. S2 handles it
after the perfect-information core works.

## Stages and gates

- **S0 — feasibility (done).** Cost of one backup, and the current eval's
  one-step residual `val(M[eval01]) − eval01(s)` on corpus positions.
- **S1 — perfect-information core.** Fitted value iteration. States come from
  self-play in which both sides play the 1-ply equilibrium of the current V_θ
  (with exploration), plus corpus positions; the matrix computed to choose
  each move supplies the training target. Network: per-mon encoder with
  species/move/item embeddings, side-antisymmetric head (V(swap s) =
  1 − V(s) by construction), wasm-sized. Gates: held-out residual against
  eval01's (table below); the certified M17e anchors; a perfect-information
  duel against the current search (sovereign).
- **S2 — hidden information.** First V_θ inside the existing blind
  determinization, then a best-response exploitability probe; move to
  public-belief-state solving (ReBeL-type) over the small private spaces if
  the probe shows leakage or strategy fusion costs.
- **S3 — team layer.** With V_θ, a team pair's preview matrix (picks × leads)
  is computed directly instead of from thousands of games; double oracle over
  team space, confirming finalists by play. This replaces the evaluation cost
  and noise that bounded M11 and the 2026-09 pool rebuild.
- **S4 — product.** Shallow search + equilibrium mixtures + V_θ in wasm.
  Strength is measured by exploitability and duels; human games confirm only.

## Progress

### S0 — feasibility (done, GO)

`crates/bot/examples/s0_backup_probe.rs`, 570 corpus battles × 4 positions =
2,280 reconstructed states (imputed hidden sets), leaf = `eval01`, this Mac
(Apple M6, 12 cores, 24 GB), 216 s wall on 10 threads. Rows:
`data/game-solving-s0/s0-backup-full.csv`.

Cost per backup (whole joint-action matrix, one core):

| alive (both sides) | n | cells p50 | sample K=32 ms p50 / p90 | mean-damage ms p50 / p90 | exact ms p50 / p90 | exact capped at 50k runs |
|---|---|---|---|---|---|---|
| ≤ 2 | 126 | 16 | 2.0 / 3.2 | 7.4 / 42.7 | 126 / 802 | 5% |
| 3–4 | 735 | 24 | 2.6 / 4.2 | 9.1 / 174 | 115 / 902 | 8% |
| 5–6 | 1,419 | 36 | 3.7 / 5.9 | 11.9 / 115 | 175 / 1,013 | 8% |

Sampled backups are unbiased and cheap: |sample − exact| p50 0.0025, p90
0.013–0.018 (0.051 in endgames). The mean-damage collapse looks accurate here
(p90 ≤ 0.0034) only because `eval01` is smooth in HP; a true V has jumps at KO
thresholds, so it is not used for training targets. At ~3 ms per backup, 11
cores give ~3,000 backups/s, about 10^7 per hour with the eval as leaf.

`eval01` is far from a fixed point:

| alive | residual bias | RMS | abs p50 | abs p90 |
|---|---|---|---|---|
| ≤ 2 | +0.037 | 0.240 | 0.036 | 0.419 |
| 3–4 | +0.004 | 0.087 | 0.034 | 0.138 |
| 5–6 | +0.002 | 0.084 | 0.041 | 0.128 |

(exact-expectation column; the sampled and mean columns agree within 0.004
RMS.) A one-step inconsistency of 0.08 RMS is larger than the typical value
gap between root candidates (~0.05), and endgames are worst, matching the
M17e anchor violations. Zero residual would not prove correctness, but this
much residual proves room.

Verdict: GO; local compute suffices for S1.

### S1 — perfect-information core (in progress)

Tooling (commit 890eab9): `bot::value` (codec, side-antisymmetric network,
sampled Shapley backup, `ValueAgent` = 1-ply equilibrium player,
`NetSearchAgent` = state-keyed UCB with the network at expansions),
`examples/fvi_gen.rs` (self-play records; the move-choice backup is the TD
target), `tools/fvi-train.py` (PyTorch 2.8 in `tmp/venv`; λ-returns over
backup values and outcomes; exports a model whose Rust forward pass must match
within 1e-3 — measured 1e-6), `examples/fvi_eval.rs` (held-out corpus residual
+ certified anchors), `examples/fvi_duel.rs` (seed-paired duels over the
training pools with one shared team preview, so only in-battle play differs).
Tests (`tests/value.rs`): volatile/side-condition coverage over 2,000 random
games, the team-swap mirror of the record, the network's antisymmetry.

Anchors: the July v3 artifact no longer reconstructs (0/72 state keys match
current reconstruction), so the 72 anchors were re-certified on the current
engine (`tmp/m17e-merged-v4.json`, three formal shards).

Training pools: `team-pool-v2` (24) + `belief-pool-v3` (103), random picks.
Iteration t generates 20,000 games (K = 16, ε = 0.1) with model t's leaf
(eval01 for t = 0) and trains model t+1 warm-started on iterations t−1 and t
(λ = 0.5). Loop driver: `tmp/fvi/loop.sh`.

| leaf | corpus residual RMS (570 pos.) | anchor violations / margin | tight MAE (n 30) | 1-ply duel vs eval1ply (400) |
|---|---|---|---|---|
| eval01 | 0.107 | 44 / 5.83 | 0.110 | — |
| model-1 | 0.081 | 26 / 4.70 | 0.130 | 0.560 ± 0.038 |
| model-2 | 0.077 | 25 / 3.89 | 0.104 | 0.600 ± 0.040 |
| model-3 | 0.071 | 26 / 4.16 | 0.116 | 0.603 ± 0.042 |
| model-4 | 0.073 | 23 / 4.01 | 0.114 | 0.595 ± 0.038 |
| model-5 | 0.071 | 25 / 3.49 | 0.099 | 0.647 ± 0.040 |
| model-6 | 0.070 | 22 / 3.33 | 0.086 | 0.583 ± 0.042 |
| model-7 | 0.073 | 21 / 3.52 | 0.088 | — |

Successive models at 1-ply (400 games each): 2>1 0.510 ± 0.039, 3>2 0.552 ±
0.043, 4>3 0.515 ± 0.039, 5>4 0.545 ± 0.039, 6>5 0.540 ± 0.038, 7>6 0.487 ±
0.041. Against the fixed eval1ply reference the 1-ply strength levels off at
about 0.6 from model-2 on; train and validation loss stay within 0.01, so the
network is not overfitting.

Against the current search (the clock column is per move, one core):

| A | B | games | score A | ms/move A / B |
|---|---|---|---|---|
| eval1ply | skuct:27000 (perfect info) | 100 | 0.340 ± 0.071 | 2 / 723 |
| netsearch model-1 (UCB, 3000 it., c 1.0) | skuct:27000 | 60 | 0.367 ± 0.114 | 154 / 963 |
| model-2 depth 2 (8×4) | skuct:27000 | 100 | 0.430 ± 0.084 | 263 / 877 |
| blindnet model-3 (1-ply, 8 determinizations) | blind:27000:0.4 (shipped) | 200 | 0.400 ± 0.054 | 72 / 1,058 |

UCB with a value leaf did not beat the 1-ply equilibrium player: argmax
visits is a pure strategy at simultaneous nodes. The matrix-game search is the
line kept.

### Throughput: where the time goes and what depends on what

Measured by `examples/fvi_timing.rs` (instrumented copy of `deep_backup`,
overhead 1.0–1.7%, values equal to the library's). Self-play at ks 6x3 with
the sparse forward: 124.8 ms per decision under a concurrent 10-thread loop —
network forward 56%, state hash 17%, engine step 12%, encode 8%, clone 5%,
equilibrium solve 0.03%; cache hit rate 46% at the root and 38% one ply down;
3,685 network evaluations per decision, ~48 per depth-1 node; 97% of samples
are spent in the depth-1 child backups. A free forward pass would make
generation at most 2.3x faster. Cells whose first sample consumes no
randomness hold 0.5% of samples, so skipping them is not worth doing.

Order and dependencies:

1. Mon-encoding cache — no dependencies, bit-identical values; DONE
   (7aa9f32, forward 19.9 → 15.0 µs per evaluation), live from deep iteration d2.
2. Smaller network — needs a same-data duel; the half-width head played even
   (0.505 ± 0.040) at 24% less forward time, the all-half network 0.477 ±
   0.039 at 49% less (equivalence not shown). An architecture change starts a
   new lineage, so it is adopted only at an experiment boundary.
3. Sample allocation in child backups — depends on whether depth-2 generation
   is kept (the running d-series decides; 1-ply generation is 17x cheaper and
   its profile differs: hash 24%, forward 46%).
4. Batched leaf evaluation — designed with item 1's cache; prerequisite for
   any GPU path, which is re-decided only after items 1–3 lower the forward
   share.
5. Engine-level state hash / step / clone — needs an audit of every
   `state_key` consumer (duel preview seeds, search trees, wasm determinism
   tests) and the conformance suite; last.

Duels against the 27k searches spend 75–90% of their time in the reference
arm, which none of the above speeds up; only fewer games (sequential
stopping) shortens them.

## Verdict: CUT

Owner decision 2026-10-02, after the go/no-go gates.

- Depth-2 targets did not break the 1-ply plateau: d1/d2/d3 vs model-9 at
  1-ply = 0.522 ± 0.042 / 0.472 ± 0.041 / 0.472 ± 0.039; residual 0.075 →
  0.073; anchors 21–22 violations.
- Against the current perfect-information search, model-d3 at depth 2 scored
  0.465 ± 0.041 at 8×4 samples (153 vs 949 ms/move) and **0.495 ± 0.042 at
  matched time** (32×10, 781 vs 1,114 ms/move), 400 games each, seed 31.
  Fivefold sampling bought ~+0.03; the value function is the binding limit.
- This is exactly the bar the 2026-08-14 RL-headroom study set for a learned
  value (`docs/EXP-rl-headroom.md` on the unmerged branch
  `origin/claude/bot-reinforcement-learning-potential-bcf77n`: replace the
  rollout and beat 8-turn rollout + eval at equal wall clock). It was reached,
  not cleared. That study was not read before S1 started; the 2026-09
  imitation/PPO pipeline also failed to establish a strength improvement
  (owner confirmation); no numerical result is available.

What stays: `bot::value` (codec, network, Shapley backup, depth-limited matrix
search, Bayesian root, the value agents), `SkuctSearch::step_with_leaf`, the
`fvi_*` examples, `tools/fvi-train.py`, `tests/value.rs`, and the re-certified
v4 anchors (`tmp/m17e-merged-v4.json`, gitignored). None is on a product path.
Models (`tmp/fvi/model-*.json`), logs and gate outputs remain in `tmp/fvi/`;
the training records were deleted (reproducible from `fvi_gen` and the seeds in
the loop scripts).
