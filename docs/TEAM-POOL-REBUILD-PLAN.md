# Blind Product and Team-Pool Architecture (`team-pool-rebuild`)

This document defines the three-role team-data architecture and offline build
pipeline for blind play (`search-profiles.json`).

## Three distinct team-data roles

Never substitute one artifact for another ([`web/src/data.ts`](../web/src/data.ts),
[`web/src/app.tsx`](../web/src/app.tsx)):

1. **Catalog (`data/team-pool-v3/team-pool.json`; historical `data/team-pool-v2/team-pool.json`)**: Playable built-in party list
   for humans and ordinary bot draws (`drawWeight`). Built-in player options and
   the ordinary bot draw share exact canonical sets.
2. **Opponent prior (`data/belief-pool-v3/belief-pool.json`)**: Broad known-party
   candidate pool used by `Belief` across all doors and tools. Own-pool
   pruning never strips legal/human-facing archetypes out of the opponent prior.
3. **Primary mixture (`data/team-pool-v3/flagship-mixture.json` primary;
   `data/kodoku-r3/conclusion-equilibrium-deme.json`,
   `data/kodoku-r3/conclusion-equilibrium-r1.json`, and
   `data/meta-nash-v3/pool-artifact.json` historical)**: Weighted party draw
   tables when `?nash` / mixture selection is active. The
   [adaptive bot audit](../data/nash-refresh-v1/README.md) records the measured
   responses to the historical draw and the five-strategy candidate.

Information mode (`blind` vs historical `open` for fork/kifu replay), search
coefficients (`data/search-profiles.json`), and production iteration budgets
(`data/live-search.json`) are governed by config files rather than hardcoded here.

## Pipeline steps & authoritative artifacts

The offline pipeline is split into explicit versioned directories and tools:

1. **Source normalization & import** (`tools/import-majinjima.js`,
   `tools/normalize-training.js`, `tools/build-training-products.js`): Inspect
   [`data/training-v3/README.md`](../data/training-v3/README.md) for the common
   EV/IV/happiness spread policy and exhaustive adversarial verifier
   (`crates/bot/examples/verify_training.rs`).
2. **Candidate inventory & opponent prior** (`tools/build-team-inventory.js`,
   `tools/belief-pool-v3.py`): Inventory provenance lives in
   [`data/team-inventory-v1/PROTOCOL.md`](../data/team-inventory-v1/PROTOCOL.md)
   and [`data/team-inventory-v1/README.md`](../data/team-inventory-v1/README.md);
   prior assembly and verification live in
   [`data/belief-pool-v3/README.md`](../data/belief-pool-v3/README.md).
3. **Own-pool evaluation & selection protocol (steps 3–5)** (`tools/team-eval-inputs.js`,
   `crates/bot/examples/team_eval.rs`, `tools/nash-solve.py`,
   `tools/nash-challenge.py`): Measure against a fixed panel under explicit
   agent/prior fingerprints (`PROTOCOL.md` rules freeze before scoring). See
   [`data/team-selection-v2/PROTOCOL.md`](../data/team-selection-v2/PROTOCOL.md),
   [`data/team-selection-v2/README.md`](../data/team-selection-v2/README.md), and
   [`data/meta-nash-v3/README.md`](../data/meta-nash-v3/README.md).
4. **Product generation & verification (step 6)** (`tools/build-team-products-v2.js`, `tools/build-team-products-v3.js`):
   Assembles the shipped catalog and checks memberships, canonical sets, draw
   weights, and prior/training alignment. Browser Team Preview now runs live
   search directly on `CompactBattle` (the former pre-baked `compiled-preview-v1.json`
   table was retired once live preview search became fast enough).

## Progress

The blind rebuild (`c = 0.4`, unified catalog + broad opponent prior) and
subsequent `training-v3` normalization and prior expansion are complete and live.
For current artifact counts, fingerprints, and verification outputs, consult the
authoritative data directories directly rather than duplicating snapshots here:

- Catalog: [`data/team-pool-v2/README.md`](../data/team-pool-v2/README.md)
- Opponent prior: [`data/belief-pool-v3/README.md`](../data/belief-pool-v3/README.md)
- Training & spread checks: [`data/training-v3/README.md`](../data/training-v3/README.md)
