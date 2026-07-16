# preview-tables-v0 — baked team-preview equilibria (M8)

One JSON per meta-pool matchup (`pair-<i>-<j>.json`, `i ≤ j` = pool indices in
`data/meta-pool-v0/meta-pool.json` rank order; the `(j,i)` orientation is the
transpose complement and is not stored). Produced by
`crates/bot/examples/bake_preview.rs`; regenerate or extend with:

```bash
cargo run --release -p nc2000-bot --example bake_preview -- --teams 0-33   # resumable: existing pairs are skipped
cargo run --release -p nc2000-bot --example bake_preview -- --summarize    # gate A report over everything baked
```

## What one file contains

- `actions` — the canonical 60-action preview space: 20 three-subsets × 3
  leads, `[lead, bench_lo, bench_hi]`, 1-based display positions. Bench order
  is quotiented out on purpose: it only affects which display slot a random
  drag-in (Roar/Whirlwind) resolves to, uniform either way, so win
  probability is invariant. 120 ordered picks → 60 cells per side, 4x fewer
  matrix cells.
- `screen` — full-width 60×60 payoff estimate (side-a mean score per joint
  preview) from cheap ε-greedy max-damage self-play games. Ranking signal
  only: the policy never switches voluntarily, so its values are biased for
  trap/stall lines. Mirror pairs bake the upper triangle and reflect
  (`M[b][a] = 1 − M[a][b]`, diagonal 0.5 by symmetry).
- `support` — the refined action subset per side: top screen-equilibrium
  best responses ∪ picks from a real `skuct` search at the preview root (the
  advisor — insurance against systematic screen bias).
- `refine` — the matrix of record: support×support re-estimated with
  `skuct` self-play games, P1/P2 alternated per game to cancel any engine
  side bias, fresh battle seed per game, 300-turn cap scored 0.5.
- `sol` — the shipped policies + gate numbers:
  - `p_a`/`p_b`: RM+-solved mixed equilibrium (full-width, linear averaging;
    `smmcts::solve_rm_plus`), dust below 5% of the modal probability shed.
  - `argmax_a`/`argmax_b`: best pure reply to the opponent's equilibrium —
    what a non-mixing bot would ship.
  - `value`: side-a equilibrium value.
  - `guarantee_{mixed,argmax}_{a,b}`: exact counter-picking guarantees on
    the refined matrix — the payoff each policy keeps against a
    best-responding opponent (restricted to the refined support). The M8
    gate quantity: mixed ≥ argmax, margin = what counter-picking costs the
    pure policy.
- `cfg`, `secs` — the budgets that produced the file, and wall-clock.

## How it is consumed

`nc2000_bot::preview::TableSet` loads the directory once and resolves live
battles to pool teams by roster signature (sorted (species, level, item,
moves) — full sets, so the one species-set collision in the pool
disambiguates). `BakedPreviewAgent` plays preview from the table (mixed
sample or argmax) and delegates the battle to any inner agent; unknown
matchups fall back to the inner agent's own preview search.
`CounterPickAgent` is the adversary: it best-responds at preview to a known
table policy. Arena specs: `baked:`/`bakedarg:`/`counter:`/`counterarg:`
+ `--pool meta[:LO-HI]` + `--tables`.

Note for M9 (wasm) / M10 (hidden info): team preview reveals species+levels
only. Two pool teams share a species set (`Cloyster/Exeggutor/Machamp/
Miltank/Snorlax/Zapdos`), and out-of-pool opponents never match a signature —
runtime identification against humans is a belief problem deferred to M10;
the wasm client can meanwhile fall back to online preview search whenever the
signature lookup misses.

## Determinism

Bakes are deterministic for a given `--seed` at any `--threads` (verified:
identical JSON minus the `secs` field at 3 vs 11 threads). Battle seeds and
agent seeds derive from (pair, cell, game) indices only.
