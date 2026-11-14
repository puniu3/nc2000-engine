# nc2000-engine

Rust battle engine, imperfect-information search bot, and WebAssembly/Preact client for Pokemon Showdown's `[Gen 2] Nintendo Cup 2000 No OHKO Stadium2 Strict` (`gen2stadium2nc2000`).

## Invariants

### 1. Format & Engine (`crates/engine`, `crates/conformance`)
- **Format regulation**: `gen2nintendocup2000noohkostadium2strict` (`data/gen2stadium2.json`, learnsets in `data/learnsets-gen2.json`). Picked team size = 3 of 6; level range 50–55; picked 3-mon level sum $\le 155$ (`battle::MAX_TOTAL_LEVEL`).
- **Gen 2 canonicalization (`crates/engine/src/validate.rs`)**: `ability: "No Ability"`, `nature: "Serious"`, SpA DV = SpD DV, SpA EV = SpD EV, HP DV derived from Atk/Def/Spe/Spc low bits, gender/shiny/Hidden Power type derived from DVs. After team preview, active side party size truncates from 6 to the 3 picked Pokémon.
- **Operational Sleep Clause (`crates/engine/tests/sleep_clause.rs`)**: Normal engine, bot, solver, and WASM construction enforce operational sleep-forfeit: if a side already has a living sleeping Pokémon (including from Rest), an opponent action that inflicts sleep on that side causes immediate loss by the inflictor (curing berries, Encore, Metronome, or switches do not exempt it; Rest itself never forfeits). Freeze prevents a second freeze on the same side; one sleeping and one frozen Pokémon may coexist.
- **PS reference parity (`crates/conformance`)**: Frozen PS fixtures (`fixtures/corpus-v1/`) use the older prevention-based Sleep Clause via `conformance::ps_reference_battle`. Every snapshot immediately following a log-growing choice must match PS bit-for-bit on state, `prngSeed` (Gen 5 64-bit LCG four-limb seed), and protocol log.
- **Total-destructure drift guards**: `Battle::state_key*` (`crates/engine/src/battle/search.rs`) and `Belief::determinize` (`crates/bot/src/belief.rs`) destructure state structs exhaustively. Any new state field must fail compilation until classified for state hashing and public/hidden determinization.
- **Compact, replay & source-hash fingerprints (`crates/engine/build.rs`, `crates/bot/build.rs`, `web/vite.config.ts`)**:
  - `crates/engine/build.rs` hashes all `crates/engine/src/**/*.rs` bytes (including comments) into `ENGINE_FINGERPRINT`, checked alongside `CompactMoveSpec` layout by `crates/conformance/tests/kifu.rs` and `crates/conformance/tests/replay.rs`. Non-semantic edits to `crates/engine/src/` require updating the digest allowlist in `crates/engine/build.rs`.
  - `crates/bot/build.rs` hashes `crates/{engine,conformance,bot}/src/**/*.rs` into `M17E_SOLVER_BUILD_FINGERPRINT`.
  - `web/vite.config.ts` (`verifyCompiledPreview`) checks source/data digests and semantic shell-prior compatibility for `data/compiled-preview-v1.json` artifacts. Training-v3 preview tables have been recomputed against the corrected sets and prior; `npm run build:research` disables lookup.

### 2. Imperfect-Information Search & Data Roles (`crates/bot`)
- **Blind information policy**: Live Web search receives only the submitted own team and filtered player frames. Evaluation, PS play, native `BlindAgent` and the legacy WASM facade reconstruct their search roots through the same protocol importer. `ObservedPosition` restricts the product search API to reconstructed input; full-world research APIs remain explicit. Every candidate preserves the announced HP bucket, and product heuristic root exclusions are disabled so uncertain guesses cannot permanently remove an action. See the [repair and limited adversarial recheck](data/blind-boundary-repair-v1/README.md) and the [historical vulnerability audit](data/blind-boundary-audit-v1/README.md).
- **Retired `open` mode**: Pinned-sheet (`open`) mode exists only to replay historical fork/kifu records (`data/search-profiles.json`).
- **Three distinct team-data roles (`web/src/data.ts`, `docs/TEAM-POOL-REBUILD-PLAN.md`)** — never substitute one for another:
  1. **Catalog (`data/team-pool-v2/team-pool.json`)**: Human built-in party list and ordinary bot draw (weighted by `drawWeight`).
  2. **Opponent prior (`data/belief-pool-v3/belief-pool.json`)**: Weighted known-party candidate set used by `Belief` across all doors and tools: real-world variants plus only catalog-generated sets. User pool overrides never replace this prior. The optional marginal species/move prior (`crates/bot/src/prior.rs`) is opt-in and inert by default (never place a file at `data/belief-prior-v0.json`).
  3. **Historical solved mixtures (`data/kodoku-r3/conclusion-equilibrium-deme.json` primary; `data/kodoku-r3/conclusion-equilibrium-r1.json` and `data/meta-nash-v3/pool-artifact.json` preserved under `<details>`)**: Bot party draw when mixture draw (`?nash` or setup selector) is active.
- **Budgets & compiled preview**: Search coefficients and iteration budgets are read from `data/search-profiles.json` and `data/live-search.json` (never hardcoded in docs). The [training-v3 preview bake](data/preview-training-v3/README.md) retains 810 compatible keys after the known-party prior expansion. The 180 affected keys are excluded from production bundles and use live preview search, as do unrecorded matchups. Source/data guards remain active; conditional distributions within both public shells determine prior compatibility. See [prior compatibility](docs/PREVIEW-PRIOR-COMPATIBILITY.md).

### 3. WASM & Web App (`crates/wasm`, `web`)
- **WASM bridge (`crates/wasm/src/lib.rs`)**: Exposes `Dex`, `Battle`, `Searcher`, `BlindSearcher`, `ProtocolSearcher`, `Validator`, and `ForkArena`. Verified against native behavior via `crates/wasm/tests-node/`.
- **Web app (`web/src/app.tsx`, `web/src/info-mode.ts`)**: Doors are `/` (`?blind` alias; `?nash` / `?nash=v3` pre-selects bot mixture draw), `?evaluate` (`web/src/evaluate.tsx`), `?fork` (`web/src/fork.tsx`), and `?solver` (`web/src/solver.tsx`). Runtime data is fetched from `<base>data/*` and copied during deploy (`.github/workflows/pages.yml`), never bundled into JS chunks.

## Pointers

### Codebase Map
- Engine core & search API: [`crates/engine/src/lib.rs`](crates/engine/src/lib.rs), [`crates/engine/src/battle/search.rs`](crates/engine/src/battle/search.rs)
- Team validator & canonicalizer: [`crates/engine/src/validate.rs`](crates/engine/src/validate.rs)
- Compact battle state & `NC2-…` replay codec: [`crates/engine/src/compact.rs`](crates/engine/src/compact.rs), [`crates/engine/src/replay.rs`](crates/engine/src/replay.rs)
- Conformance harness & snapshot comparator: [`crates/conformance/src/lib.rs`](crates/conformance/src/lib.rs), [`fixtures/corpus-v1/`](fixtures/corpus-v1/)
- Bot modules (search, belief, eval, accounting, endgame, fork, solver): [`crates/bot/src/lib.rs`](crates/bot/src/lib.rs)
- Metagame evolution & Kodoku equilibrium pipeline: [`crates/bot/examples/meta_eco.rs`](crates/bot/examples/meta_eco.rs)
- WASM bindings & build script: [`crates/wasm/src/lib.rs`](crates/wasm/src/lib.rs), [`crates/wasm/build.sh`](crates/wasm/build.sh), [`crates/wasm/tests-node/`](crates/wasm/tests-node/)
- Web application & E2E suites: [`web/src/app.tsx`](web/src/app.tsx), [`web/src/data.ts`](web/src/data.ts), [`web/package.json`](web/package.json), [`web/tests/`](web/tests/)
- PS local/websocket interop & preview cache helper: [`tools/ps-arena.js`](tools/ps-arena.js), [`tools/ps-client.js`](tools/ps-client.js), [`tools/preview-cache.js`](tools/preview-cache.js)
- Core Engine SDK builder & deterministic position diagnosis CLI: [`tools/bundle-core-sdk.js`](tools/bundle-core-sdk.js), [`tools/diagnose-position.js`](tools/diagnose-position.js) (downstream environment adapters like Gym Leader's Castle live in the operational fork [`hm0615g/nc2000-engine`](https://github.com/hm0615g/nc2000-engine))
- CI / GitHub Pages deployment contract: [`.github/workflows/pages.yml`](.github/workflows/pages.yml)

### Specifications & Reference Docs
- Format mechanics & PS porting notes: [`PORT-NOTES.md`](PORT-NOTES.md), [`PORTING.md`](PORTING.md)
- Server-side preview cache specification: [`docs/PREVIEW-CACHE.md`](docs/PREVIEW-CACHE.md) (rendered by [`tools/render-preview-cache-doc.py`](tools/render-preview-cache-doc.py))
- Native fork integration handoff: [`docs/NATIVE-FORK-HANDOFF.md`](docs/NATIVE-FORK-HANDOFF.md) (rendered by [`tools/render-native-handoff.py`](tools/render-native-handoff.py))
- Search speed & off-pool belief diagnostics handoff: [`docs/SEARCH-AND-BELIEF-HANDOFF.md`](docs/SEARCH-AND-BELIEF-HANDOFF.md)
- Engine anomaly report & triage protocol: [`.github/ISSUE_TEMPLATE/engine_anomaly_report.yml`](.github/ISSUE_TEMPLATE/engine_anomaly_report.yml)
- Compact replay (`NC2-…`) specification: [`docs/replay.md`](docs/replay.md)
- Endgame exact/bound solver specification: [`ENDGAME-SOLVER-ALGORITHM.md`](ENDGAME-SOLVER-ALGORITHM.md)
- Team catalog, belief prior & Nash rebuild protocol: [`docs/TEAM-POOL-REBUILD-PLAN.md`](docs/TEAM-POOL-REBUILD-PLAN.md)
- License & third-party notices: [`LICENSE`](LICENSE), [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)

## Canonical Verification Commands

```bash
# Rust workspace tests (engine, conformance, bot)
cargo test

# WASM Node build + parity/contract suites (mirrors .github/workflows/pages.yml)
crates/wasm/build.sh nodejs
for suite in smoke determinism parity kifu fork solver destiny-bond priority-heal residual-wait own-recovery-wait preview-cache; do
  node "crates/wasm/tests-node/$suite.js"
done

# Web build + worker & Playwright E2E gates
crates/wasm/build.sh
cd web && npm run build && npm run test:worker && npm run test:evaluate-worker && npm run test:policy && npm run test:e2e
```
