# gen2-tpu

Local Cloud TPU emulation environment and 32-bit branchless JAX kernel for `[Gen 2] NC 2000` (`gen2stadium2`), mirroring [`../crates/engine/src/compact.rs`](file:///Users/puniu/nc2000-engine/crates/engine/src/compact.rs) and empirically optimized for **Google Colab TPU `v6e1` (Trillium)**.

## Architecture

1. **`gen2_tpu.tpu_emu`**: Emulates a multi-core Cloud TPU host on CPU (`jax_num_cpu_devices=8`, `jax_enable_x64=False`):
   - Enforces strict 32-bit execution (`int32`, `uint32`, `float32`) with zero 64-bit integer/float operations.
   - `verify_tpu_compatibility(fn, *args)`: Audits lowered XLA HLO for forbidden 64-bit types, Python host callbacks, dynamic shapes, and VMEM working-set capacity (`TPU_V5E`, `TPU_V5P`, `TPU_V6E`).
   - `make_tpu_mesh` & `shard_batch`: Multi-core SPMD execution via `jax.shard_map` across the 8 virtual TPU cores.
2. **`gen2_tpu.prng`**: Bit-exact 32-bit implementation of Pokemon Showdown's 64-bit Gen5 LCG (`seed * 0x5D588B656C078965 + 0x269EC3`) using 16-bit limb arithmetic in `uint32[2]`.
3. **`gen2_tpu.state` & `gen2_tpu.dex_loader`**: Loads [`../data/gen2stadium2.json`](file:///Users/puniu/nc2000-engine/data/gen2stadium2.json) and constructs Structure-of-Arrays (SoA) `MatchTable`, `CompactBattle` (`516 B`), and bit-packed `PackedBattle` (`uint32[30] = 120 B`, 4.3x compression via `pack_battle` / `unpack_battle`).
4. **`gen2_tpu.kernel`**: Branchless 32-bit turn and switch step functions (`step_turn`, `step_switch`, `apply_actions`, `legal_actions_mask`, `eval_leaf`, `evaluate_hit_simd16`) with zero XLA HLO `gather` or `scatter` instructions on TPU.
5. **`gen2_tpu.rollout`**: Fixed-horizon `jax.lax.scan` rollouts (`vmap_rollout`), 8-core SPMD `sharded_tpu_rollout`, and Pallas VMEM tiled kernels (`pallas_evaluate_hits(..., interpret=True)`).
6. **`gen2_tpu.search`**: Offline TPU evaluation & solver primitives mirroring `nc2000-bot`:
   - `EvalWeights` & `eval_compact_leaf` (`nc2000_bot::eval::compact_eval_leaf` + `compact_race_margin`).
   - `sample_heavy_playout_action` & `sharded_heavy_rollout` (`Playout::Heavy`: $\varepsilon$-greedy max-damage 8-turn rollout over 120-byte `PackedBattle` batches sharing one `MatchTable` per matchup; **3.90M turn-steps/sec** at `B=16384` on `v6e1`).
   - `evaluate_root_payoff_matrix` & `solve_rm_plus_jax`: Counterfactual joint-action (`7 x 7`) rollout matrix probe and simultaneous-move Regret Matching+ Nash equilibrium solver (`25.12 ms` on `v6e1`).
   - `evaluate_depth2_counterfactual_oracle` & `label_positions_batch_tpu`: Batched Depth-2 simultaneous-move counterfactual oracle on TPU (`25.49 ms` on `v6e1`) for distilling tactical search into `ResidualBundle`.
7. **`gen2_tpu.residual`**: Drop-in port of `nc2000_bot::eval::{EvalResidualWeights, RolloutResidualWeights, ResidualBundle}` (`32 -> 16 -> 1` antisymmetric eval residual + `12 x 6` rank-4 bilinear rollout action residual) with constrained JAX Adam trainer (`train_residual_bundle_jax`) and `NC2000_RESIDUAL_JSON` import/export.
8. **`gen2_tpu.kodoku`**: TPU-accelerated 6v6 Two-Stage Core-Nash & Kodoku ("蠱毒") evaluator mirroring `nc2000_bot::preview` and `meta_eco.rs` (`RosterTable6`, pure-JAX 3v3 subgame assembly `assemble_3v3_from_rosters_jax`, fused CRN pair-grid rollouts `evaluate_3v3_picks_batch_tpu` reaching **4.57M turn-steps/sec** on `v6e1`, `select_top_k_orthogonal` with Leave-One-Out Anti-Monopoly guarantee, `core_nash_6v6_pair_tpu` in **`68.79 ms`** full / **`20.50 ms`** incremental, fused multi-matchup `evaluate_kodoku_gauntlet_tpu` in **`58.91 ms/matchup`**, `recompute_population_wars_np`, `export_pair_table_json`).

See [`HANDOFF.md`](file:///Users/puniu/nc2000-engine/tpu/HANDOFF.md) for Colab TPU `v6e1` benchmark tables, architecture details, and CLI integration with `nc2000-engine`.

## Usage

```bash
uv run --project tpu pytest tpu/tests -v
uv run --project tpu gen2-tpu --help
```
