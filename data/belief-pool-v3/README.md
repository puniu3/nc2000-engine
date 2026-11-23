# belief-pool-v3 — the known-party opponent prior

`belief-pool.json` (`522` entries) is the deployed known-party opponent prior:
real-world parties including variants, plus generated sets only when present in
the current playable catalog. Together with the bundled zero-prior Gen 2 species
supplement (`crates/bot/src/belief_supplement_v1.json`) and off-pool EV/IV
spread synthesis (`crates/bot/src/belief.rs`), it forms `Belief`'s prior
baseline across all blind doors and tools. The catalog and Nash draw mixtures
remain separate (`data/team-pool-v2/`, `data/kodoku-r3/`). `bake-prior.json`
(`221` entries) preserves the earlier snapshot for provenance and historical
comparisons.

`python3 tools/belief-pool-v3.py build` invokes `tools/build-known-prior.mjs`
(and `node tools/build-training-products.js` verifies deterministic alignment).
It retains admitted baseline weights, adds admitted real-world types with
weight 1, collapses duplicate source aliases without adding mass, and applies
shared [`training-v3`](../training-v3/README.md) normalization across all sets
(including exact full-set/stat support for all 33 current catalog parties).
Authoritative counts, sources, exclusions for invalid-format parties, and file
hashes live in `known-expansion.json` (excluded records are never silently
converted into different legal parties). For native verification, run `cargo test`.

## Allocation & Historical Provenance

- **Weight rule**: Baseline cluster entries carry `weight = 1 / cluster_size`
  (`×0.25` for machine-generated teams), while expanded real-world entries carry
  weight `1` (`1 / cluster_size` for multi-variant page series; see
  `known-expansion.json`). Weights affect collision shells and
  weighted fallback sampling (`crates/bot/src/belief.rs`).
- **Historical `bake-prior.json` (`221` entries)**: Built from `data/team-inventory-v1`
  (`103`), `data/team-selection-v2/evolved-teams.json` (`8`),
  `data/partybox-v0/teams.json` (`101`), and `9` `<id>-lowatk` Female-optimized
  twins (`fnv1a64:7d23346d1218e317` after `training-v3`). Historical selection
  measurements under `measurements/prior-c04-b{3000,27000}.json` record the
  earlier pre-`training-v3` spreads (`fnv1a64:957b746a94b97708`).
