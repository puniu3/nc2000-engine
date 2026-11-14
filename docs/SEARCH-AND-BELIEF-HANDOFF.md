# Search Speed & Belief Inference Handoff

> **Status (`2026-11-15`)**: **Track 3** zero-drift speedups (`69c49c3b`), **Track 2** Phases 1–5 Imperfect-Information Architecture Upgrade (`a2e19a26`–`4f516159`), and the non-optional player-boundary unification (`ce6485d9`, routing `OpenAgent`, `omniscient_audit` T3, and WASM preview helpers through `LocalPlayer` / `ObservedPosition`) are landed, re-baked (`990/990` preview policies), and verified across Rust, Node WASM, Web Worker/Policy, and Playwright E2E suites.
>
> This document replaces the parked Residual NN / TPU track (`docs/RESIDUAL-NN-HANDOFF.md`), whose runtime code and JAX/Colab pipeline were removed to eliminate dual-engine technical debt while retaining generated datasets under [`data/`](../data/).

---

## 0. Context: Why Residual NN & `tpu/` Code Were Removed

Earlier ceiling-raising experiments explored adding a 2-layer antisymmetric evaluation residual MLP + rank-4 rollout action residual (`EvalResidualWeights`, `RolloutResidualWeights`) on top of [`EvalWeights`](../crates/bot/src/eval.rs) and maintaining a 32-bit branchless JAX/TPU mirror (`tpu/`). Both subsystems were deleted from active code for three reasons:

1. **Structural Ceiling Mismatch**: As shown in [`findings/headroom-probe-v1/README.md`](../findings/headroom-probe-v1/README.md), imperfect-information belief error on off-prior teams (`+112` to `+181` Elo gap between `open` and `blind` at `9k`/`27k` iterations) dominates catalog leaf-eval nuances (`+26` Elo, not statistically significant in `open:9000` vs `blind:9000`).
2. **Latency Budget Sensitivity**: MCTS playing strength scales tightly with tree iterations per second in 32-bit WebAssembly; extra per-leaf/per-rollout neural feature extraction and matrix multiplies tax the exact hot loop where search depth matters most.
3. **Dual-Engine Maintenance Tax**: Keeping a separate branchless JAX simulator in bit-exact parity with Rust (`crates/engine/src/`, `crates/bot/src/compact.rs`, `crates/bot/src/eval.rs`) created high synchronization friction compared to improving the single authoritative Rust engine directly.

**What was kept**: All historical datasets, equilibrium matrices, compiled preview artifacts, and audit directories under `data/` (`data/compiled-preview-v1.json`, `data/tpu-nash-bake-v1/`, `data/kodoku-r3/`, `data/preview-training-v3/`, `data/belief-pool-v3/`, `data/team-pool-v2/`) plus historical reports in `findings/` and `docs/` remain untouched.

---

## 1. Track 2 — Off-Pool Belief Inference & Subsampled World Scaling

### 1.1 Empirical Headroom (Why Belief Is the Highest-ROI Bottleneck)

Measurements recorded in [`findings/headroom-probe-v1/README.md`](../findings/headroom-probe-v1/README.md) establish where Elo is currently left on the table:

- **In-catalog matchups** (`team-pool-v2`, 24 of 33 teams present in prior):
  - `open:9000` vs `blind:9000`: `0.537 ± 0.065` (`+26` Elo, not statistically significant).
  - `skuct:9000` vs `open:9000`: `0.525 ± 0.074` (`+17` Elo, not statistically significant).
- **Off-prior matchups** (`120` held-out non-prior teams):
  - `open:9000` vs `blind:9000`: **`0.656 ± 0.054` (`+112` Elo)**.
  - `open:27000` vs `blind:27000`: **`0.738 ± 0.078` (`+181` Elo)**.
- **Real human ladder corpus** (`logs_20260804.zip`, 190 bot-vs-human games):
  - **`188 / 190` games triggered off-pool fallback** (`0 / 25` exact full-team matches, despite strong macro species correlation `r = 0.928`).

Improving how [`Belief`](../crates/bot/src/belief.rs) and [`BlindSearch`](../crates/bot/src/blind.rs) handle out-of-catalog teams and hidden opponent state is therefore the primary addressable ceiling.

---

### 1.2 Concrete Gaps in Current Off-Pool Belief (`crates/bot/src/{belief,observe,prior,import}.rs`)

Code inspection of `HEAD` (`25ed6f33`) identified seven structural information losses:

1. **Zero-prior species gaps & EV/IV narrowness**:
   - The union of `data/belief-pool-v3/belief-pool.json` (468 teams / 2,808 sets / 229 species; only 1 of 2,808 sets has `spe ev < 252` or `iv < 30`), `data/training-v3/supplemental.json` (202 species, all already present in `belief-pool-v3`), and `rentals.json` (adds `Ditto`) covers `230 / 246` species in [`data/learnsets-gen2.json`](../data/learnsets-gen2.json), leaving **16 Gen 2 species with zero set priors**.
2. **Hard-filtering cliff ([`belief.rs`](../crates/bot/src/belief.rs#L1047-L1068))**:
   - `Belief::refilter` uses strict equality checks in [`build_refs`](../crates/bot/src/belief.rs#L2518-L2540) and [`consistent`](../crates/bot/src/belief.rs#L2542-L2552). A single off-pool tech move, unexpected item, or level mismatch empties `self.alive` across the entire candidate pool, jumping directly to fallback synthesis without attempting near-pool set repair on unchanged slots.
3. **Unappeared bench Pokémon are frozen ([`belief.rs`](../crates/bot/src/belief.rs#L1813-L1826))**:
   - [`bayesian_loadout_draw`](../crates/bot/src/belief.rs#L1813-L1822) returns `None` immediately when `!mo.appeared`. Unseen bench mons are pinned to a single MAP `fallback` loadout across every determinization instead of sampling plausible unrevealed threats (e.g., Surprise Explosion / Belly Drum / Perish Trap / coverage moves).
4. **Appeared but switched-out bench Pokémon are frozen ([`belief.rs`](../crates/bot/src/belief.rs#L1238-L1315))**:
   - Under `FallbackPolicy::BayesianPointMass | FallbackPolicy::Bayesian`, both [`predict_fallback_draws`](../crates/bot/src/belief.rs#L1248-L1254) and [`sample_fallback_refs`](../crates/bot/src/belief.rs#L1294-L1300) gate slot activation on `!r.fainted && (r.is_active || battle.active_id(opp).is_none())`. Living bench mons with `previously_switched_in > 0` (already seen in battle with unrevealed move slots or unknown item) are starved of `loadout_draws`.
5. **Tier A item collapse ([`belief.rs`](../crates/bot/src/belief.rs#L1851-L1859))**:
   - `move_sig` keys candidates by `(mo.item.original.is_some().then(|| p.item.map(|i| i.0)), mvs)`. While the opponent's item is unrevealed (`mo.item.original.is_none()`), candidates sharing the same moveset but different plausible items (e.g. `Leftovers` vs `Miracle Berry` vs `Mint Berry` vs `Quick Claw`) collapse into one representative.
6. **Tier B synthesis gate ([`belief.rs`](../crates/bot/src/belief.rs#L2141-L2261)) & `det_cache` bypass ([`belief.rs`](../crates/bot/src/belief.rs#L1271-L1273))**:
   - Novel combo synthesis in [`bayesian_loadout_draw`](../crates/bot/src/belief.rs#L2149) is gated on `has_novel_baton_pass` and capped at `MAX_NOVEL_LOADOUTS = 2` (`L2227`), missing non-Baton-Pass surprises (Curse/Restalk variants, Explosion techs, Perish Trap shells).
   - `predict_fallback_draws` returns `None` whenever `self.draws` is `Some` for an active slot (`L1271-L1273`), bypassing `BlindSearch::det_cache`.
   - `SpeciesPrior` (`crates/bot/src/prior.rs:68-79`) parses `items` and `lead`, both currently unused by sampling.
7. **Unexploited public item & Hidden Power evidence ([`observe.rs`](../crates/bot/src/observe.rs#L127-L145), [`observe.rs`](../crates/bot/src/observe.rs#L675-L683), [`import.rs`](../crates/bot/src/import.rs))**:
   - `MonObs.hp_type_mask` (`observe.rs:141`) defaults to `u16::MAX` and `hp_effectiveness_inference` is disabled by default (`observe.rs:675-683`).
   - Safe passive **negative item evidence** from protocol frames is largely unexploited:
     - Staying at `< 100%` HP at end-of-turn without a Leftovers heal eliminates `Leftovers` (unless suppressed/full HP).
     - Remaining statused (`slp`, `par`, `psn`, `tox`, `brn`, `frz`) at end-of-turn eliminates auto-cure berries matching that status (`Miracle Berry`, `Mint Berry` for sleep, etc.).
     - Switching in without triggering `Berserk Gene` eliminates `Berserk Gene` (for species/sets where it matters).
   - **High-Risk Non-Goal (avoid unless rigorously proven)**: Converting raw observed damage rolls or turn-order ties directly into hard exclusions is fragile in Gen 2 due to `[217..255]/255` damage variance, `/100` PS HP percentage rounding, critical hits, and `Quick Claw` (`60/256` trigger). Prefer soft likelihood reweighting over hard filtering if continuous damage/speed likelihoods are ever explored.

---

### 1.3 `MAX_SUBSAMPLED_WORLDS = 24` & Lessons from Reverted `ecc6954c` / `d1cebfed`

Current [`BlindSearch`](../crates/bot/src/blind.rs#L416-L418) configuration:
- `MAX_SUBSAMPLED_WORLDS = 24`
- `WORLD_WARMUP_SAMPLES = 96` (`CELL_WARMUP_SAMPLES = 6`)
- Team preview (`is_preview()`) bypasses `world_pool` (`blind.rs:920`).
- In battle, `24` worlds receive `1,125` iterations/world at `27k` budget and `3,375` iterations/world at `81k` budget (`L939-L965`).

**Why `ecc6954c` / `d1cebfed` had to be reverted**:
That experiment bundled six unrelated changes together:
1. Unconditional `24 -> 192` widening (`8x` worlds),
2. `obs_nodes` cross-world continuation tree sharing,
3. `public_dominated_reason` heuristic root exclusions (which falsely pruned winning last-mon self-KO / Sleep Clause / Quick Claw lines),
4. `best()` argmax tie-breaking changes,
5. `MechTracker`, and
6. `ShellDraw` / Tier B changes.

**Guardrails for future `BlindSearch` world scaling**:
1. **Never re-enable heuristic root exclusions**: [`use_public_exclusions`](../crates/bot/src/blind.rs#L550-L556) explicitly documents why guessed worlds cannot prove action dominance.
2. **Isolate variables**: Never bundle `MAX_SUBSAMPLED_WORLDS` changes with tree sharing (`obs_nodes`), root masks, or root selection rule changes.
3. **Avoid unconditional `8x` widening (`192`)**:
   - Multiplies `world_pool` heap footprint by `~8x` inside 32-bit WASM linear memory.
   - Cuts `27k` per-world tree depth from `1,125` down to `~141` iterations/world.
   - `192 * 96 = 18,432` warmup samples exhaust `late_cells` budget before `WORLD_WARMUP_SAMPLES` finishes at `27k`.
4. **Promising direction to benchmark in isolation**: Adaptive widening (`24 -> 48..64` worlds *only* when `belief.is_fallback()` or opponent entropy is high) paired with proportionally smaller `WORLD_WARMUP_SAMPLES` and cached `det_cache` keys (`predict_fallback_draws` fixed for active M18 draws), evaluated at **equal wall-clock time** (`27k` / `81k`).

---

### 1.4 Validation & Benchmarking Pitfalls (Do Not Repeat)

- **Avoid self-play-only validation**: [`crates/bot/examples/offpool_fallback_gauntlet.rs`](../crates/bot/examples/offpool_fallback_gauntlet.rs) runs `BlindAgent` vs `BlindAgent` self-play, which hides shared evaluation blind spots. Pair off-pool benchmarks against non-self-play controls (e.g., `OpenAgent` oracle or external human/held-out rosters) and check exact belief likelihood / action agreement too.
- **Avoid skewed corpus priors**: Spectator logs average `~2.35 / 4.0` observed moves per mon, systematically under-counting passive/reactive moves (`Whirlwind`, `Protect`, `Rest`, `Toxic`, `Sleep Talk`). Follow the unforced-distribution lessons documented alongside `early-rest-natural-v1` when fitting marginal priors.
- **Pre-existing test check**: [`data/training-v3/README.md`](../data/training-v3/README.md) previously noted 8 unit failures in `cargo test -p nc2000-bot --lib belief::`. All workspace `belief::` and `nc2000-bot` unit/integration tests pass after Phases 1–4 (`226` bot lib tests, `0` failed).

---

### 1.5 Landed Imperfect-Information Upgrade (Phases 1–5, `a2e19a26`–`baa4f977`)

Design boundary (**Choice A — Strict Pool Trust Until Contradiction & Turn 0 Invariance**): while `!self.alive.is_empty()`, candidate pool filtering and Turn 0 Team Preview sampling behave identically to baseline. Near-pool soft shell repair, quantile-coupled fallback synthesis, and unappeared-bench `LoadoutDraw` diversification activate only once `self.alive.is_empty()` and in-battle (`obs.mons().iter().any(|m| m.appeared)`).

1. **Phase 1 (`a2e19a26`) — Symmetric Hidden Runtime Timer Resampling & `belief_probe` Harness**:
   - [`crates/bot/src/player_state.rs`](../crates/bot/src/player_state.rs), [`crates/bot/src/import.rs`](../crates/bot/src/import.rs), [`crates/bot/src/blind.rs`](../crates/bot/src/blind.rs): track `HiddenTimerSpec` (`Sleep`, `Confusion`, `Encore`, `Disable`, `PartiallyTrapped`, `Bide`, `LockedMove`) symmetrically across both sides and resample conditioned uniform remaining durations in `BlindSearch` on a dedicated `timer_rng` (including between reused `world_pool` evaluations).
   - [`crates/bot/examples/belief_probe.rs`](../crates/bot/examples/belief_probe.rs): non-self-play evaluation suite with `corpus_audit` (simulated + spectator corpus false-exclusion/drop audit), `calibration` (posterior recall, unrevealed item likelihood, Hidden Power accuracy, effective loadout entropy), and `gauntlet` (`BlindAgent` vs `OpenAgent` oracle across `offpool` and `catalog`).
2. **Phase 2 (`6407a92b`) — Live Hidden Power Narrowing, Deductive Item Exclusions & `|split|` Parity**:
   - [`crates/bot/src/hp_evidence.rs`](../crates/bot/src/hp_evidence.rs), [`crates/bot/src/observe.rs`](../crates/bot/src/observe.rs), [`crates/bot/src/position.rs`](../crates/bot/src/position.rs), [`web/src/position.ts`](../web/src/position.ts): live `Hidden Power` effectiveness narrowing (`hp_type_mask`), safe passive item exclusions (`Leftovers` missing when end-of-turn HP `< 100%`, auto-cure berries inconsistent with remaining status, `Berserk Gene` absent on switch-in, `Thief` state), `|split|` parity, `Heal Bell`/`Safeguard` handling, `excluded_items` serialization round-trip, unprojected `fallback` vs projected `fallback_visible`, and `resolve_non_excluded_item`.
3. **Phase 3 (`8682fe58`) — Near-Pool Repair, Joint Fallback Synthesis & Loadout Diversity**:
   - [`crates/bot/src/belief.rs`](../crates/bot/src/belief.rs): per-slot `bayesian_prior_carriers` item + explicit HP mask scoring; `bayesian_loadout_draw` supporting Tier A (`locked_staples`, secondary item variants, unappeared-bench `LoadoutDraw` once in battle), Tier B novel move combinations, and unprojected `"hiddenpower"` preservation; and `build_fallback_cands` (`FallbackCandPool`) combining Source 1 Near-Pool Soft Shell Repair (retaining exact untouched pool slots) with Source 2 joint synthesis enforcing global Item Clause and per-slot exclusions.
4. **Phase 4 (`baa4f977`) — Low-Discrepancy Systematic World Resampling & Quantile-Coupled Fallback Pool**:
   - [`crates/bot/src/belief.rs`](../crates/bot/src/belief.rs), [`crates/bot/src/blind.rs`](../crates/bot/src/blind.rs): replaced Cartesian `BEAM_WIDTH = 32` dilution (`24` flat-weight rosters, which fragmented `3k`-budget MCTS subtrees and dropped `offpool 128 @ 3k` to `0.3984..0.4414`) with **8-stratum Latin-Hypercube Quantile Coupling** capped at `8` joint fallback rosters (`source1.truncate(4)`, `merged.truncate(8)`), `TeamPlanPick` (`Pool`, `FallbackCand`, `FallbackDefault`), `Belief::root_sampling_plan` with `systematic_resample_indices` + Fisher-Yates shuffle on a dedicated `SplitMix64` stream (preserving `belief.sample` at `is_preview()`), `predict_plan_draws` / `determinize_in_place_planned` `det_cache` parity, `BlindSearch::with_max_worlds`, and adaptive `world_warmup = (192 / self.root_types.len().max(2)).clamp(48, 96)` in `BlindSearch::record_joint`.

#### Measured Phase 0 vs Final Verification (`belief_probe`)

- **`corpus_audit`** (18 simulated games / 361 decisions + 570 human spectator battles / 3,414 decisions): `0` across all 5 safety counters (`in_pool_drops`, `hp_false_exclusions`, `item_false_exclusions`, `corpus_hp_false_exclusions`, `corpus_item_false_exclusions`).
- **`calibration`** (48 off-pool games, 111 positions):

| Metric | Phase 0 (`69c49c3b`) | Final (Phase 4, `baa4f977`) |
| --- | ---: | ---: |
| `active_unseen_move_recall` | `0.8082` | **`0.8301`** |
| `bench_unseen_move_recall` | `0.8571` | **`0.8966`** |
| `item_log_likelihood` | `-2.7839` | **`-2.6411`** |
| `hp_type_acc` (`4` HP slots) | `0.5000` | **`0.5625`** |
| `eff_distinct_loadouts` | `1.1475` | **`2.8525`** |

- **`gauntlet`** (`BlindAgent` vs `OpenAgent` oracle; `0` capped, `0` invalids across all 1,408 scored games):

| Matchup & Budget | Pairs (Games) | Phase 0 | Final |
| --- | ---: | ---: | ---: |
| `offpool` @ `3000` | `128` (`256`) | `0.4688` (`64`p, `-21.7` Elo) | **`0.4883`** (`-8.1` Elo) |
| `offpool` @ `3000` | `256` (`512`) | — | **`0.5000`** (`±0.0` Elo) |
| `offpool` @ `9000` | `128` (`256`) | — | **`0.4805`** (`-13.6` Elo) |
| `offpool` @ `27000` | `64` (`128`) | — | **`0.5156`** (`+10.9` Elo) |
| `catalog` @ `3000` | `128` (`256`) | `0.4375` (`48`p, `-43.7` Elo) | **`0.5117`** (`+8.1` Elo) |

- **Phase 5 Verification & Bake (`4f516159`, `ce6485d9`)**: Re-baked `data/compiled-preview-v1.json` (`990/990` policies); all workspace Rust tests (`cargo t` + `sleep_clause`, `kifu`, `replay`, `conformance_suite`, `bot_suite`), all 12 WASM/Node suites (`smoke`..`endgame`..`preview-cache`), Core SDK bundle, Vite build, `test:worker`, `test:evaluate-worker`, `test:policy`, `compiled-preview-coverage.mjs` (`1089` matchups / `990` valid keys / `69,696` slot checks), `compiled-preview-production.mjs` (`1089` catalog pairs / `2178` cases, `hiddenLoadoutInvariant: true`), and Playwright `test:e2e` (`70/70`) passed.

---

## 2. Track 3 — Zero-Drift Search Speedups & Native Server Execution

### 2.1 Critical Baseline Correction (`43a5c58a` Profile Is Obsolete)

[`findings/performance-scout-v1/README.md`](../findings/performance-scout-v1/README.md) measured commit `43a5c58a` (`2026-10-24`: `40.7%` state hashing, `21.7%` event dispatch, `3.8%` visible alloc, `1.2%` `clone_from`). **Do not use those numbers to size work on current `HEAD` (`25ed6f33`).** Between `2026-10-24` and `2026-11-14`, most of the October bottlenecks were already shipped:

- **`3ca86602` (`portable-search-v1`)**: `-7.7%` to `-11.1%` Chromium WASM, `-12.1%` Node WASM, `-16.2%..-16.3%` native.
- **`ced93e74` / `9c5bceb3`**: `WordHasher::batch` (`PolyBatch`), `EvCbs::mask` + `CbMask` fast skip in `Battle::run_event`, `RV::Str(&'static str)`.
- **`4eb424a7` / `207ff6e1`**: Moved 8-turn rollouts & leaf eval off `Battle::apply_choices` onto 192-byte `Copy` [`CompactBattle`](../crates/engine/src/compact.rs) (`mcts.rs`); eliminated UCB/path heap allocations.
- **`9194b776` / `d8096e0d` / `849a3545`**: Added `BlindSearch::det_cache`, `predict_fallback_draws`, and `world_pool` (`24` worlds with `reset_search_sim`).
- **`5d1dcec9` / `2fb86ac8`**: Added `4096 × 4-way` SplitMix64 Bellman cache + factored 2D endgame solver in [`crates/bot/src/linear.rs`](../crates/bot/src/linear.rs).
- **`5a0abbb7`**: Added `PolyStep`/`PolyRecorder`, [`BattleSearchCache`](../crates/engine/src/battle/search.rs) (`bench_steps`), and `BlindSearch::root_node_type`.
- **Compiler/Allocator defaults already active**: Root [`Cargo.toml`](../Cargo.toml) already enables `lto = "fat"` and `codegen-units = 1`; [`crates/bot/Cargo.toml`](../crates/bot/Cargo.toml) already enables `mimalloc` on non-`wasm32`; `crates/wasm/Cargo.toml` already runs `wasm-opt = ["-O3"]`.

**Mandatory Step 0 before any optimization work**: Re-profile current `HEAD` first using:
- Native micro-benchmarks: `cargo run --release -p nc2000-bot --example perf_probe`
- Native flamegraph on recorded games: `cargo run --release -p nc2000-bot --example profile_blind -- <RECORDED_GAMES> tmp/blind.svg`
- Portable WASM/browser regression harness: [`web/tests/portable-perf.mjs`](../web/tests/portable-perf.mjs)

Realistic remaining **portable (WASM + native) zero-drift speedup** on `HEAD` is estimated (inferred from inspection, pending fresh profiling) at **~5%–15%**, not another `2x`.

---

### 2.2 Already Rejected Optimization Approaches (Do Not Repeat)

- Direct tuple-hash expansion (`WordHasher::batch` / `PolyBatch` already outperforms naive tuple unrolling).
- Relying on `-C target-cpu=native` alone in search (`-3.0%` at `3k`, noisy `-6.6%` at `27k` on Apple Silicon; non-portable across hosts/WASM).
- GPU/Metal offload at 1-dispatch-per-MCTS-iteration (host/device sync latency dominates tree iteration cost).

---

### 2.3 Remaining Zero-Drift Code Hotspots (`HEAD`)

Every change inside `crates/engine/src/` modifies `ENGINE_FINGERPRINT` ([`crates/engine/build.rs`](../crates/engine/build.rs)) and requires updating its allowlist plus verifying bit-exact parity (`conformance` `kifu`, `replay`, `conformance_suite`, `sleep_clause`) and `data/compiled-preview-v1.json` lock rebuilds.

#### A. Hashing & Team Preview Setup
1. **Batch post-move `Pokemon::hash_state_impl` tail ([`crates/engine/src/state.rs`](../crates/engine/src/state.rs#L1338-L1436))**:
   - `fast_flags` requires `last_move.is_none()` (`L1347`). As soon as [`Pokemon::move_used`](../crates/engine/src/battle/pokemon.rs#L698) sets `last_move = Some(...)`, virtually every depth `>= 1` active mon falls out of `hash_batch` into serial field hashing (`L1399-L1435`). Wrapping the fallback dynamic tail in `h.hash_batch` (or widening `fast_flags` to separate only `last_move*` / `move_this_turn`) is strictly bit-identical / zero-drift.
2. **PP deduction step-cache invalidation ([`pokemon.rs`](../crates/engine/src/battle/pokemon.rs#L675-L694))**:
   - `Battle::deduct_pp` mutates `move_slots` on Turn 1, breaking `full_step` (`base_move_slots == init_base_moves`) equivalence checks in `search.rs`. Separately caching static vs PP/volatile dynamic hashes preserves Turn 1+ prefix reuse without altering hash bits.
3. **Team Preview bench hashing & `det_cache` misses ([`search.rs`](../crates/engine/src/battle/search.rs#L109-L166), [`blind.rs`](../crates/bot/src/blind.rs#L433))**:
   - During Team Preview (`selected == false`), `bench_steps` in `BattleSearchCache` is empty, so all 12 roster mons hash uncollapsed on preview tree evaluations (up to `729k` budget in live config), and `root_search_key_cached` runs uncached on `det_cache` misses. Indexing Team Preview roster prefixes cleanly accelerates Preview without changing outputs.
4. **Non-goal — pervasive dirty tracking**: Do **not** add per-field dirty bitmasks across `Battle`/`Pokemon` mutation sites (`9,500+` lines); untouched bench mons already reuse `CachedBenchMon` in two integer operations, making dirty-flag desync bugs far costlier than the residual hash savings.

#### B. Event Dispatch (`Battle::apply_choices` inside MCTS Tree) & `CompactBattle`
1. **Replace string-slice callback ladders (`conditions.rs`, `items.rs`, `moveexec.rs`)**:
   - Callback dispatch currently routes through string-matched `(key: &str, cb: &str)` functions even though runtime handlers already hold typed integer IDs (`CondId`, `ItemId`, `MoveId`, `Cb`). Compiling a flat integer-indexed jump table at `Dex::from_json` eliminates repeated `&str` hashing/matching on hot tree transitions.
2. **Lazy priority resolution in [`find_*_event_handlers`](../crates/engine/src/battle/events.rs#L464-L745)**:
   - Handler discovery calls `self.resolve_priority(dex, listener, cb)` (dex metadata lookups, `f64` priority fields, speed-order lookups) eagerly for every matching listener even when `handlers.len() <= 1`. Deferring priority resolution until `handlers.len() > 1` avoids sorting metadata work in single-listener events.
3. **`field_event` fast-skip & permanent format rules ([`events.rs`](../crates/engine/src/battle/events.rs#L691-L745), [`L1177-L1266`](../crates/engine/src/battle/events.rs#L1177-L1266))**:
   - `field_event` lacks a `CbMask` field-wide fast skip analogous to `run_event`, and `find_field_event_handlers` iterates permanent format-rule pseudo-weathers (`maxtotallevel`, `stadiumsleepclause`, `freezeclausemod`) on `Residual` checks (`L702`: `pseudo_weather.len() > 3` still scans once present).
4. **Pre-cache invariant `MatchTable` entries per `(world_id, party)` ([`compact.rs`](../crates/engine/src/compact.rs#L1469))**:
   - Once Team Preview is complete (`selected == true`), matchup specs inside `CompactBattle::from_battle_with_roster_cache` are static whenever `!transformed`. Caching the immutable `MatchTable` slices per `SampledWorld` removes leaf-extraction work.

---

### 2.4 Deployment & Runtime Lever (Highest Server-Side Speedup)

- **Move Node WASM servers (`crates/wasm/pkg-node`) to Native Rust ([`docs/NATIVE-FORK-HANDOFF.md`](./NATIVE-FORK-HANDOFF.md))**: Running `ProtocolAgent` as a native binary with release LTO + host `-C target-cpu` + Profile-Guided Optimization (PGO) typically delivers a much larger wall-clock gain than micro-optimizing inside V8/WASM, and eliminates the 32-bit WASM linear memory ceiling if wider `world_pool` budgets are adopted server-side.
- **Browser Web Workers remain WASM-only**: If browser throughput needs additional headroom after algorithmic profiling, evaluate targeted `wasm32` `simd128` vectorization specifically for `linear.rs` / `WordHasher::batch` (subject to Safari/Chromium baseline support).

---

## 3. Recommended Sequence for Future Work

1. **Profile `HEAD` before touching hot paths**: Capture fresh native (`perf_probe`, `profile_blind`) and WASM (`web/tests/portable-perf.mjs`) baselines before choosing optimization targets.
2. **Remaining Track 2 Belief Gaps**: Fill priors/synthetic set generation for the 16 zero-prior Gen 2 species and uncommon EV spreads, and benchmark adaptive `with_max_worlds` widening above `24` exclusively at high search budgets (`>= 27k`) / high belief entropy under equal wall-clock time.
3. **Remaining Zero-Drift Track 3 & Deployment Wins**: Explore remaining items in Section 2.3 (post-move `hash_state_impl` batching, PP-deduction step-cache split, Team Preview bench hashing, static `MatchTable` caching) and migrate Node server runners to native `ProtocolAgent` ([`docs/NATIVE-FORK-HANDOFF.md`](./NATIVE-FORK-HANDOFF.md)).
