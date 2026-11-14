"""Batched TPU rollout and multi-core `shard_map`.

Provides:
1. `rollout_single` & `vmap_rollout`: Multi-turn turn-capped trajectory rollouts
   compiled into a single XLA `while`/`scan` loop with zero host round-trips.
2. `sharded_tpu_rollout`: Distributes a batch of `N` rollouts across all virtual TPU
   cores (`jax.devices()`) using `shard_map` and `NamedSharding`.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from .kernel import (
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
    """Run a single `CompactBattle` for up to `horizon` turns using `jax.lax.scan`."""
    if horizon <= 0:
        val = eval_leaf(init_state, table, 0)
        return RolloutResult(
            final_state=init_state,
            leaf_values=val,
            outcomes=init_state.outcome,
            turns_played=init_state.turn,
        )

    cutoff_turn = init_state.turn + jnp.int32(horizon - 1)

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, p_seed = carry
        active = (st.outcome == 0) & (st.turn <= cutoff_turn)
        next_seed, a0 = sample_uniform_legal_action(st, table, 0, p_seed)
        next_seed, a1 = sample_uniform_legal_action(st, table, 1, next_seed)
        acts = jnp.stack([a0, a1], axis=0)
        cand_st = apply_actions(st, table, acts)
        next_st = jax.tree_util.tree_map(
            lambda new, old: jnp.where(active, new, old), cand_st, st
        )
        out_seed = jnp.where(active, next_seed, p_seed)
        return (next_st, out_seed), None

    (final_st, _), _ = jax.lax.scan(
        _step, (init_state, policy_seed), None, length=max(1, horizon * 2)
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

    if horizon <= 0:
        val = jax.vmap(eval_leaf, in_axes=(0, None, None))(batched_state, table, 0)
        return RolloutResult(
            final_state=batched_state,
            leaf_values=val,
            outcomes=batched_state.outcome,
            turns_played=batched_state.turn,
        )

    cutoff_turn = batched_state.turn + jnp.int32(horizon - 1)

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, ps = carry
        active = (st.outcome == 0) & (st.turn <= cutoff_turn)
        next_ps, a0 = jax.vmap(
            sample_uniform_legal_action, in_axes=(0, None, None, 0)
        )(st, table, 0, ps)
        next_ps, a1 = jax.vmap(
            sample_uniform_legal_action, in_axes=(0, None, None, 0)
        )(st, table, 1, next_ps)
        acts = jnp.stack([a0, a1], axis=-1)
        cand_st = jax.vmap(apply_actions, in_axes=(0, None, 0))(st, table, acts)

        def _select(new: jax.Array, old: jax.Array) -> jax.Array:
            mask = active.reshape((active.shape[0],) + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        next_st = jax.tree_util.tree_map(_select, cand_st, st)
        out_ps = jnp.where(active[:, None], next_ps, ps)
        return (next_st, out_ps), None

    (final_st, _), _ = jax.lax.scan(
        _step, (batched_state, policy_seeds), None, length=max(1, horizon * 2)
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
