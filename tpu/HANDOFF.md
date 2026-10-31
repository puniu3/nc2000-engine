# Handoff: Gen 2 NC2000 TPU Engine, Bot Residual Distillation & Kodoku ("蠱毒")

## 1. Current Repository State
- **Unified Location**: `/Users/puniu/nc2000-engine/tpu` (integrated directly into `/Users/puniu/nc2000-engine`).
- **Shared Dex Data**: Reads [`../data/gen2stadium2.json`](file:///Users/puniu/nc2000-engine/data/gen2stadium2.json) directly locally, and bundles it automatically via `uv run gen2-tpu bundle-colab --out <bundle.tar.gz>` for remote Colab TPU execution.
- **Test Status**: All 14 tests in [`tests/test_tpu_gen2.py`](file:///Users/puniu/nc2000-engine/tpu/tests/test_tpu_gen2.py) pass (`cd tpu && uv run pytest -v`) both on local 8-core CPU TPU emulation and on real Google Colab TPU **`v6e1` (Trillium)** hardware, including step-by-step bit-exact parity across 20 golden multi-turn games (488 steps) against Rust [`nc2000_engine::compact::CompactBattle`](file:///Users/puniu/nc2000-engine/crates/engine/src/compact.rs).
- **Empirical Google Colab TPU `v6e1` (Trillium, 32 GiB HBM, 32 MiB VMEM) Benchmarks**:
  - **Zero HLO `gather` / `scatter` Operations**: Replaced all dynamic `.at[...].set(...)` and array indexing in [`kernel.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kernel.py) and [`search.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/search.py) with `_set1`, `_get1`, and static `jnp.stack`, achieving **`gather_ops = 0` and `scatter_ops = 0`** across `apply_actions`, `sample_heavy_playout_action`, and `eval_compact_leaf`.
  - **Actor-Loop HLO Compaction**: Using `jax.lax.fori_loop(0, 2, ...)` in `_run_two_actor_steps` ([`kernel.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kernel.py#L2304)) reduced `apply_actions` lowered HLO from `96,599` to `56,073` lines (`42%` smaller), cutting XLA TPU compile time by `2.2x` (`76.38 s -> 34.38 s`) and improving per-step VPU execution speed by `1.34x` (`0.923 ms -> 0.690 ms`).
  - **Fused Pair-Grid + Side-Swap CRN Subgame Kernel**: [`evaluate_3v3_picks_batch_tpu`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kodoku.py#L943) fuses `(n_pairs, games_per_pair)` including both normal `(A, B)` and side-swapped `(B, A)` Common Random Numbers orientations into a single XLA call with bucketed static shapes (`64..8192`, including `4608` for full `48 x 48 x 2` Stage-1 screening), reaching **4.57 million turn-steps/sec** (`40.37 ms` for 4,608 40-turn games, **`7.70x` faster** than the `310.88 ms` baseline).
  - **In-Kernel 5-Turn Block Early Termination**: [`_get_jit_multi_table_rollout_fn`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kodoku.py#L740) wraps 5-turn `jax.lax.scan` blocks inside a `jax.lax.while_loop` conditioned on `jnp.any(st.outcome == 0)`, accelerating 60-turn Stage-2 rollouts by **`2.08x`** (`4.36 ms -> 2.09 ms`) with `max_diff = 0.0`.
  - **Vectorized 6-Slot Leave-One-Out (`ΔV_{-s}`) RM+**: [`_solve_batched_row_masked_rm_plus_values`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kodoku.py#L1281) solves all 6 Leave-One-Out row-masked subgames in parallel via `jax.vmap` (`69.67 ms -> 0.44 ms`, **`158x` faster**).
  - **Multi-Matchup Fused Stage-2 Gauntlet**: [`evaluate_kodoku_gauntlet_tpu`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kodoku.py#L1812) fuses Stage-2 refinement across all `(P_i, Q_j)` team pairs via [`_get_jit_multi_roster_subgame_fn`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kodoku.py#L1782), reaching **`58.91 ms` per full 6v6 matchup** (`8.74x` faster than the `515.09 ms` baseline).

| Workload on Colab TPU `v6e1` | Baseline | Optimized (`v6e1`) | Speedup / Throughput |
| :--- | ---: | ---: | :--- |
| `sharded_heavy_rollout` (`B=1024`, `8t`) | `19.54 ms` | **`14.21 ms`** | **`1.38x`** (`0.58M` turn-steps/s) |
| `sharded_heavy_rollout` (`B=16384`, `8t`) | `75.87 ms` | **`33.57 ms`** | **`2.26x`** (**`3.90M` turn-steps/s**) |
| `evaluate_root_payoff_matrix` (`49x32`, `8t`) | `31.92 ms` | **`25.12 ms`** | **`1.27x`** |
| `evaluate_depth2_counterfactual_oracle` (`top_k=3`) | `39.22 ms` | **`25.49 ms`** | **`1.54x`** |
| `core_nash_6v6_pair_tpu` (`48x48x2@40t` + `3x3x6@60t` + 12x LOO) | `515.09 ms` (compile `247.7s`) | **`68.79 ms`** (compile `148.0s`) | **`7.49x` latency** (`4.57M` Stage-1 steps/s) |
| `core_nash_6v6_pair_tpu` (`inherited_subgame` `@mut_slot`) | — | **`20.50 ms`** | **~48.8 6v6 matchups/sec** |
| `evaluate_kodoku_gauntlet_tpu` (`4x4 = 16` 6v6 matchups) | `8,241 ms` (`515 ms/pair`) | **`942.49 ms`** (`58.91 ms/pair`) | **`8.74x` latency** |

---

## 2. Module Map (`/Users/puniu/nc2000-engine/tpu/src/gen2_tpu/`)

| File | Responsibility | Key Entry Points |
| :--- | :--- | :--- |
| [`tpu_emu.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/tpu_emu.py) | 8-core CPU TPU emulation (`jax_enable_x64=False`), HLO static analysis, SPMD mesh (`TPU_V5E`, `TPU_V5P`, `TPU_V6E`) | `init_tpu_emulator`, `verify_tpu_compatibility`, `make_tpu_mesh` |
| [`prng.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/prng.py) | 32-bit limb port of Showdown 64-bit Gen5 LCG (`uint32[2]`) | `prng_random`, `prng_random_chance`, `seed_from_str` |
| [`state.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/state.py) | SoA `MatchTable`, `CompactBattle` (`516 B`), bit-packed `PackedBattle` (`uint32[30] = 120 B`, 4.3x compression) | `MatchTable`, `CompactBattle`, `PackedBattle`, `pack_battle`, `unpack_battle` |
| [`dex_loader.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/dex_loader.py) | Loader for [`../data/gen2stadium2.json`](file:///Users/puniu/nc2000-engine/data/gen2stadium2.json) and 3v3 state builder | `load_dex`, `build_compact_battle` |
| [`kernel.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kernel.py) | Branchless 32-bit Gen 2 turn & forced-switch transition kernel (zero TPU `gather` / `scatter` ops) | `apply_actions`, `legal_actions_mask`, `expected_hit_fraction`, `best_hit_fraction` |
| [`residual.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/residual.py) | Drop-in port of `nc2000_bot::eval::{EvalResidualWeights, RolloutResidualWeights, ResidualBundle}` + JAX trainer | `default_residual_bundle`, `compact_eval_residual_sym`, `compute_rollout_action_residuals`, `train_residual_bundle_jax`, `export_residual_bundle_json` |
| [`search.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/search.py) | `EvalWeights`, `Playout::Heavy`, root RM+ solver, Depth-2 counterfactual simultaneous-move oracle, batch position labeler | `eval_compact_leaf`, `sharded_heavy_rollout`, `evaluate_root_payoff_matrix`, `evaluate_depth2_counterfactual_oracle`, `label_positions_batch_tpu` |
| [`kodoku.py`](file:///Users/puniu/nc2000-engine/tpu/src/gen2_tpu/kodoku.py) | 6v6 `RosterTable6`, pure-JAX 3v3 assembly, fused multi-table & multi-roster batched rollouts, Two-Stage 6v6 Core-Nash & vectorized Leave-One-Out (`ΔV_{-s}`), Population WAR OLS, `PairTable` & `dominance-audit` JSON export | `compile_roster6`, `assemble_3v3_from_rosters_jax`, `simulate_multi_table_heavy_games`, `core_nash_6v6_pair_tpu`, `evaluate_kodoku_gauntlet_tpu`, `export_pair_table_json` |

---

## 3. CLI Subcommands & Drop-In Integration Boundaries

Run from the repository root (`/Users/puniu/nc2000-engine`):

```bash
# 1. Run tests on 8-core local TPU emulator
uv run --project tpu pytest tpu/tests -v

# 2. Run TPU compatibility audit & rollout/oracle benchmark
uv run --project tpu gen2-tpu benchmark

# 3. Export a 6v6 PairTable (SPACE_VERSION = 1) JSON for crates/bot/src/preview.rs
uv run --project tpu gen2-tpu pair-table --team-a <teamA.json> --team-b <teamB.json> --out <pair.json>

# 4. Evaluate a 6v6 pool gauntlet and export cells/unit_cells JSON for meta_eco
uv run --project tpu gen2-tpu kodoku-cells --pool-p <poolP.json> [--pool-q <poolQ.json>] --out <cache-cells.json>

# 5. Build a self-contained tarball (including gen2stadium2.json) for Colab TPU v6e1
uv run --project tpu gen2-tpu bundle-colab --out /tmp/gen2_tpu_bundle.tar.gz
```
