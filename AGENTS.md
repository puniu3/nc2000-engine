# Agent & Contributor Guide (`nc2000-engine`)

## 1. Fast Build & Test Workflows

- **Rust workspace (fast dev/test profile in `.cargo/config.toml`)**:
  - `cargo t` (alias for `cargo test --workspace --lib --tests`, skipping 0-test doc-test discovery)
  - `cargo test -p nc2000-engine` / `cargo test -p conformance` / `cargo test -p nc2000-bot`
  - CI contract binaries preserved: `crates/engine/tests/sleep_clause.rs`, `crates/conformance/tests/kifu.rs`, `crates/conformance/tests/replay.rs`, `crates/conformance/tests/conformance_suite.rs`, and `crates/bot/tests/bot_suite.rs`.
- **WASM & Web**:
  - `crates/wasm/build.sh nodejs` + `for s in smoke determinism parity kifu fork solver endgame destiny-bond priority-heal residual-wait own-recovery-wait preview-cache; do node "crates/wasm/tests-node/$s.js"; done`
  - `crates/wasm/build.sh && cd web && npm run build && npm run test:worker && npm run test:evaluate-worker && npm run test:policy && npm run test:e2e`

## 2. Epoch & Cache Versioning

- **2-axis epoch tracking**:
  - `ENGINE_EPOCH = 1` in `crates/engine/src/replay.rs` governs compact replay (`NC2-…`) and kifu structural compatibility; bump only on breaking mechanics or replay format changes.
  - `POLICY_EPOCH = "policy-v4"` (`CASTLE_POLICY_EPOCH = "castle-policy-v4"`) in `crates/bot/src/compiled_preview.rs` governs server-side preview cache identity (`crates/wasm/src/preview_cache.rs`, `tools/preview-cache.js`) and evaluator build versioning (`web/vite.config.ts`). Browser Team Preview runs live search directly (no pre-baked preview table).
- `tools/bundle-core-sdk.js` writes `dist/nc2000-core-sdk/engine-manifest.json` (untracked).

## 3. Research & Strength Roadmap

- Playing-strength architecture, measured-null dead ends, and open `81k` frontiers: see [`docs/BOT-STRENGTH-HANDOFF.md`](docs/BOT-STRENGTH-HANDOFF.md).
- Do NOT consult gitignored local scratch/archive directories (`findings/`, `tmp/`, `.git/sota/`) for current architecture or open bottlenecks; verify claims directly against [`docs/BOT-STRENGTH-HANDOFF.md`](docs/BOT-STRENGTH-HANDOFF.md) and live code under `crates/`.
