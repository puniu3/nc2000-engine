"""Batched TPU rollout, multi-core `shard_map`, and Pallas `interpret=True` kernels.

Provides:
1. `rollout_single` & `vmap_rollout`: Multi-turn fixed-horizon trajectory rollouts
   compiled into a single XLA `while`/`scan` loop with zero host round-trips.
2. `sharded_tpu_rollout`: Distributes a batch of `N` rollouts across all virtual TPU
   cores (`jax.devices()`) using `shard_map` and `NamedSharding`.
3. `pallas_evaluate_hits`: Explicit Pallas (`jax.experimental.pallas`) kernel running
   in `interpret=True` mode to emulate TPU VMEM block tiling (`BlockSpec`) and grid
   execution locally on CPU.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .kernel import (
    SIMD16_ROLLS,
    apply_actions,
    eval_leaf,
    sample_uniform_legal_action,
)
from .state import CompactBattle, MatchTable
from .tpu_emu import make_tpu_mesh


class RolloutResult(NamedTuple):
    """Summary of a batched rollout on TPU."""

    final_state: CompactBattle
    leaf_values: jax.Array  # float32[batch]
    outcomes: jax.Array  # int32[batch]
    turns_played: jax.Array  # int32[batch]


def rollout_single(
    init_state: CompactBattle,
    table: MatchTable,
    policy_seed: jax.Array,
    horizon: int = 32,
) -> RolloutResult:
    """Run a single `CompactBattle` for `horizon` steps using `jax.lax.scan`."""

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, p_seed = carry
        p_seed, a0 = sample_uniform_legal_action(st, table, 0, p_seed)
        p_seed, a1 = sample_uniform_legal_action(st, table, 1, p_seed)
        acts = jnp.stack([a0, a1], axis=0)
        next_st = apply_actions(st, table, acts)
        return (next_st, p_seed), None

    (final_st, _), _ = jax.lax.scan(
        _step, (init_state, policy_seed), None, length=horizon
    )
    val = eval_leaf(final_st, table, 0)
    return RolloutResult(
        final_state=final_st,
        leaf_values=val,
        outcomes=final_st.outcome,
        turns_played=final_st.turn,
    )


def clone_battle_batch(
    base_state: CompactBattle,
    batch_size: int,
    base_seed_int: int = 0x123456789ABCDEF0,
) -> tuple[CompactBattle, jax.Array]:
    """Replicate a single `CompactBattle` into a Structure-of-Arrays batch `[batch_size, ...]`
    with distinct battle PRNG seeds and policy PRNG seeds.
    """
    batched_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), base_state
    )
    idx = jnp.arange(batch_size, dtype=jnp.uint32)
    hi_base = jnp.uint32((base_seed_int >> 32) & 0xFFFFFFFF)
    lo_base = jnp.uint32(base_seed_int & 0xFFFFFFFF)

    # Give each trajectory a distinct battle RNG seed and policy RNG seed
    battle_seeds = jnp.stack(
        [
            hi_base ^ (idx * jnp.uint32(0x9E3779B9)),
            lo_base + (idx * jnp.uint32(0x85EBCA6B) + jnp.uint32(1)),
        ],
        axis=-1,
    )
    policy_seeds = jnp.stack(
        [
            (hi_base + jnp.uint32(0xC2B2AE35)) ^ (idx * jnp.uint32(0x27D4EB2F)),
            lo_base ^ (idx * jnp.uint32(0x165667B1) + jnp.uint32(7)),
        ],
        axis=-1,
    )
    batched_state = batched_state._replace(prng_seed=battle_seeds)
    return batched_state, policy_seeds


def _vmap_rollout_impl(
    batched_state: CompactBattle,
    table: MatchTable,
    policy_seeds: jax.Array,
    horizon: int = 32,
) -> RolloutResult:
    """On Cloud TPU, uses scan-outside-vmap so XLA TPU steps the entire `[B, ...]` batch
    per turn without `vmap`-wrapped `while_loop` temporary buffer explosion.
    On CPU, uses vmap-outside-scan for L1 cache locality per trajectory and smaller HLO.
    """
    if jax.default_backend() != "tpu":
        return jax.vmap(
            lambda st, ps: rollout_single(st, table, ps, horizon=horizon),
            in_axes=(0, 0),
        )(batched_state, policy_seeds)

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, ps = carry
        ps, a0 = jax.vmap(
            sample_uniform_legal_action, in_axes=(0, None, None, 0)
        )(st, table, 0, ps)
        ps, a1 = jax.vmap(
            sample_uniform_legal_action, in_axes=(0, None, None, 0)
        )(st, table, 1, ps)
        acts = jnp.stack([a0, a1], axis=-1)
        next_st = jax.vmap(apply_actions, in_axes=(0, None, 0))(st, table, acts)
        return (next_st, ps), None

    (final_st, _), _ = jax.lax.scan(
        _step, (batched_state, policy_seeds), None, length=horizon
    )
    val = jax.vmap(eval_leaf, in_axes=(0, None, None))(final_st, table, 0)
    return RolloutResult(
        final_state=final_st,
        leaf_values=val,
        outcomes=final_st.outcome,
        turns_played=final_st.turn,
    )


@partial(jax.jit, static_argnames=("horizon",))
def vmap_rollout(
    batched_state: CompactBattle,
    table: MatchTable,
    policy_seeds: jax.Array,
    horizon: int = 32,
) -> RolloutResult:
    """Vectorized rollout over a batch of battles on a single TPU core."""
    return _vmap_rollout_impl(batched_state, table, policy_seeds, horizon=horizon)


@lru_cache(maxsize=16)
def _build_sharded_rollout_fn(mesh: Mesh, horizon: int):
    state_spec = CompactBattle(*([P("core")] * len(CompactBattle._fields)))
    table_spec = MatchTable(*([P()] * len(MatchTable._fields)))
    seed_spec = P("core", None)
    out_spec = RolloutResult(
        final_state=state_spec,
        leaf_values=P("core"),
        outcomes=P("core"),
        turns_played=P("core"),
    )

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(state_spec, table_spec, seed_spec),
        out_specs=out_spec,
        check_vma=False,
    )
    def _per_core_fn(
        local_states: CompactBattle, shared_table: MatchTable, local_seeds: jax.Array
    ):
        return _vmap_rollout_impl(
            local_states, shared_table, local_seeds, horizon=horizon
        )

    return jax.jit(_per_core_fn)


def sharded_tpu_rollout(
    batched_state: CompactBattle,
    table: MatchTable,
    policy_seeds: jax.Array,
    horizon: int = 32,
    mesh: Mesh | None = None,
) -> RolloutResult:
    """Distribute `vmap_rollout` across all virtual TPU cores via `shard_map`."""
    if mesh is None:
        mesh = make_tpu_mesh(axis_names=("core",))
    fn = _build_sharded_rollout_fn(mesh, horizon)
    return fn(batched_state, table, policy_seeds)


# =========================================================================
# Pallas TPU VMEM Emulation Kernel (`interpret=True`)
# =========================================================================


def _pallas_hit_16_kernel(
    base_dmg_ref,
    hp_ref,
    sub_hp_ref,
    bypasses_ref,
    rolls_ref,
    dealt_out_ref,
    post_hp_out_ref,
    post_sub_out_ref,
):
    """Pallas kernel computing 16 damage rolls and branchless `[hp, sub_hp]` updates
    per battle in a VMEM tile of size `block_b`.
    """
    bd = base_dmg_ref[:]  # int32[block_b]
    hp = hp_ref[:]  # int32[block_b]
    sub = sub_hp_ref[:]  # int32[block_b]
    byp = bypasses_ref[:]  # int32[block_b] (0 or 1)
    rolls = rolls_ref[:]  # int32[16]

    # Broadcast [block_b, 1] x [1, 16] -> [block_b, 16] inside VMEM
    bd_2d = bd[:, None]
    rolls_2d = rolls[None, :]
    varied = jnp.maximum(jnp.int32(1), (bd_2d * rolls_2d) // jnp.int32(255))
    d_lanes = jnp.where(
        bd_2d <= 0,
        jnp.int32(0),
        jnp.where(bd_2d == 1, jnp.int32(1), varied),
    )

    hits_sub = (sub > 0) & (byp == 0)
    hits_sub_2d = hits_sub[:, None]
    hp_2d = jnp.broadcast_to(hp[:, None], d_lanes.shape)
    sub_2d = jnp.broadcast_to(sub[:, None], d_lanes.shape)

    before = jnp.where(hits_sub_2d, sub_2d, hp_2d)
    dealt = jnp.minimum(d_lanes, before)
    after = before - dealt

    dealt_out_ref[:, :] = dealt
    post_hp_out_ref[:, :] = jnp.where(hits_sub_2d, hp_2d, after)
    post_sub_out_ref[:, :] = jnp.where(hits_sub_2d, after, sub_2d)


@partial(jax.jit, static_argnames=("block_size",))
def pallas_evaluate_hits(
    base_dmg: jax.Array,
    target_hp: jax.Array,
    target_sub_hp: jax.Array,
    bypasses_sub: jax.Array,
    block_size: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute the 16-roll damage + Substitute/HP pool kernel via Pallas `interpret=True`.

    Tiles the batch of size `B` (multiple of `block_size`, default 128 matching TPU VPU
    lane width) into VMEM blocks using `pl.BlockSpec`.
    """
    batch_size = base_dmg.shape[0]
    if batch_size % block_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be a multiple of VPU block_size ({block_size})"
        )
    grid_len = batch_size // block_size
    out_shape = jax.ShapeDtypeStruct((batch_size, 16), jnp.int32)

    return pl.pallas_call(
        _pallas_hit_16_kernel,
        out_shape=(out_shape, out_shape, out_shape),
        grid=(grid_len,),
        in_specs=[
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((16,), lambda _: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((block_size, 16), lambda i: (i, 0)),
            pl.BlockSpec((block_size, 16), lambda i: (i, 0)),
            pl.BlockSpec((block_size, 16), lambda i: (i, 0)),
        ],
        interpret=True,
    )(
        base_dmg.astype(jnp.int32),
        target_hp.astype(jnp.int32),
        target_sub_hp.astype(jnp.int32),
        bypasses_sub.astype(jnp.int32),
        SIMD16_ROLLS,
    )
