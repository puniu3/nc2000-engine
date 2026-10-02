# Solving the battle game

Status: **S0 done (GO), S1 next.** Owner direction 2026-10-02. This is the
authoritative plan and handoff for the game-solving program; update it in place.

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
