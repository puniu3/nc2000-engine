"""Comprehensive test suite for the Gen 2 (`gen2stadium2`) TPU Emulation & Kernel.

Verifies:
1. TPU emulation environment constraints (8 virtual cores, 32-bit `jax_enable_x64=False`,
   static XLA HLO compilation with zero host callbacks or dynamic shapes, VMEM fit).
2. Bit-exact parity of the 32-bit `uint32[2]` Gen5 PRNG against the 64-bit reference LCG.
3. Exhaustive parity of branchless primitives (`ActionSortKey`, `get_stat_branchless`,
   `lookup_accuracy_threshold`, `apply_roll_branchless`).
4. Trajectory & state parity of `CompactBattle` (`step_turn`, `step_switch`, `apply_actions`,
   `legal_actions_mask`, `eval_leaf`, `compact_race_margin`) against
   `nc2000-engine`'s `compact.rs` tests.
5. Multi-core `shard_map` rollout equivalence and Kodoku / Bayesian equilibrium solvers.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gen2_tpu import (
    PACKED_BATTLE_BYTES,
    PACKED_BATTLE_WORDS,
    TPU_V5E,
    ActionPhase,
    CompactBattle,
    CompactWeather,
    ItemStatKind,
    Status,
    VolBit,
    action_sort_key,
    action_sort_key_new,
    apply_actions,
    apply_roll_branchless,
    build_compact_battle,
    clone_battle_batch,
    clone_packed_battle_batch,
    compact_base_logit_and_eff,
    compact_eval_residual_sym,
    compact_race_margin,
    default_eval_weights,
    default_residual_bundle,
    eval_compact_01,
    eval_compact_leaf,
    eval_leaf,
    eval_residual_forward,
    evaluate_depth2_counterfactual_oracle,
    evaluate_root_payoff_matrix,
    export_residual_bundle_json,
    extract_eval_features,
    extract_rollout_state_and_actions,
    get_stat_branchless,
    init_tpu_emulator,
    label_positions_batch_tpu,
    legal_actions_mask,
    load_residual_bundle_json,
    lookup_accuracy_threshold,
    make_tpu_mesh,
    pack_battle,
    prng_next_u32,
    prng_random,
    prng_random_chance,
    resolve_turn_order,
    sample_heavy_playout_action,
    seed_from_str,
    seed_from_u64_int,
    seed_to_str,
    seed_to_u64_int,
    select_shell_belief_minimax_action_jax,
    sharded_heavy_rollout,
    sharded_tpu_rollout,
    solve_rm_plus_jax,
    step_turn,
    train_residual_bundle_jax,
    unpack_battle,
    update_shell_belief_mask_jax,
    verify_tpu_compatibility,
    vmap_rollout,
)


_JIT_APPLY = jax.jit(apply_actions)


def _mk_set(species: str, level: int, item: str, moves: list[str]) -> dict:
    return {
        "name": species,
        "species": species,
        "item": item,
        "ability": "No Ability",
        "moves": moves,
        "level": level,
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
        "happiness": 255,
    }


def test_tpu_emulator_topology_and_32bit_mode():
    devices = init_tpu_emulator()
    if jax.default_backend() == "cpu":
        assert len(devices) == 8
    else:
        assert len(devices) >= 1
    assert not jax.config.read("jax_enable_x64")

    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    table, state = build_compact_battle("1,2,3,4", p1, p2)
    acts = jnp.array([0, 0], dtype=jnp.int32)

    report = verify_tpu_compatibility(apply_actions, state, table, acts, spec=TPU_V5E)
    assert report.passed
    assert report.x64_disabled
    assert report.forbidden_dtypes == []
    assert not report.has_host_callbacks
    assert not report.has_dynamic_shapes
    assert report.fits_in_vmem


def test_prng32_matches_64bit_gen5_lcg_bit_for_bit():
    mult = 0x5D588B656C078965
    add = 0x269EC3
    mask64 = 0xFFFFFFFFFFFFFFFF

    seed_str = "1,2,3,4"
    seed32 = seed_from_str(seed_str)
    assert seed_to_str(seed32) == seed_str
    seed64 = seed_to_u64_int(seed32)

    jitted_next = jax.jit(prng_next_u32)
    jitted_rand = jax.jit(prng_random)
    jitted_chance = jax.jit(prng_random_chance)

    for step in range(500):
        # Reference 64-bit step
        seed64 = (seed64 * mult + add) & mask64
        expected_u32 = (seed64 >> 32) & 0xFFFFFFFF

        seed32, draw_u32 = jitted_next(seed32)
        assert int(draw_u32) == expected_u32
        assert seed_to_u64_int(seed32) == seed64

        # Test random(n)
        n = (step % 39) + 2
        seed64 = (seed64 * mult + add) & mask64
        expected_r = (((seed64 >> 32) & 0xFFFFFFFF) * n) >> 32
        seed32, r_val = jitted_rand(seed32, n)
        assert int(r_val) == expected_r
        assert seed_to_u64_int(seed32) == seed64

        # Test random_chance(num, den)
        seed64 = (seed64 * mult + add) & mask64
        expected_c = ((((seed64 >> 32) & 0xFFFFFFFF) * 256) >> 32) < 60
        seed32, c_val = jitted_chance(seed32, 60, 256)
        assert bool(c_val) == expected_c
        assert seed_to_u64_int(seed32) == seed64


def test_action_sort_key_orders_switches_priority_quick_claw_and_speed_ties():
    # Switch beats +2 Protect
    sw = action_sort_key_new(ActionPhase.SWITCH, 0, False, 100, 0)
    prot = action_sort_key_new(ActionPhase.MOVE, 2, True, 400, 1)
    first, second, is_tie = resolve_turn_order(sw, prot)
    assert (int(first), int(second), bool(is_tie)) == (0, 1, False)

    # Forced Hyper Beam recharge (slot_refs & 0x80) uses MOVE phase even if switch action >= 4 is passed
    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Hyper Beam", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    tbl, st = build_compact_battle("1,2,3,4", p1, p2)
    st_rech = st._replace(slot_refs=st.slot_refs.at[0].set(0x80))
    key_rech = action_sort_key(st_rech, tbl, 0, jnp.int32(5), jnp.int32(0))
    assert int(key_rech >> 15) == int(ActionPhase.MOVE)

    # Quick Claw on slower Pokemon beats faster Pokemon at equal priority
    slow_qc = action_sort_key_new(ActionPhase.MOVE, 0, True, 90, 0)
    fast_no_qc = action_sort_key_new(ActionPhase.MOVE, 0, False, 350, 1)
    first, second, is_tie = resolve_turn_order(slow_qc, fast_no_qc)
    assert (int(first), int(second), bool(is_tie)) == (0, 1, False)

    # Quick Attack (+1) beats Quick Claw (0 priority)
    quick_atk = action_sort_key_new(ActionPhase.MOVE, 1, False, 90, 1)
    first, second, is_tie = resolve_turn_order(slow_qc, quick_atk)
    assert (int(first), int(second), bool(is_tie)) == (1, 0, False)

    # Double Quick Claw ties on speed regardless of underlying base speeds
    qc_p0 = action_sort_key_new(ActionPhase.MOVE, 0, True, 85, 0)
    qc_p1 = action_sort_key_new(ActionPhase.MOVE, 0, True, 300, 1)
    _, _, is_tie = resolve_turn_order(qc_p0, qc_p1)
    assert bool(is_tie)
    assert int((qc_p0 >> 1) & 1023) == 1023
    assert int((qc_p1 >> 1) & 1023) == 1023


def test_stat_branchless_matches_reference_formula():
    stat_num = [25, 28, 33, 40, 50, 66, 100, 150, 200, 250, 300, 350, 400]

    stored_vals = [1, 15, 95, 130, 178, 255, 358, 500, 999]
    items = [
        ItemStatKind.NONE,
        ItemStatKind.THICK_CLUB,
        ItemStatKind.LIGHT_BALL,
        ItemStatKind.METAL_POWDER,
    ]
    rows = []
    expected_rows = []
    for stored in stored_vals:
        for stat_idx in range(5):
            for unboosted in (False, True):
                for unmodified in (False, True):
                    for fast_return in (False, True):
                        for par in (False, True):
                            for brn in (False, True):
                                for screen in (False, True):
                                    for item in items:
                                        rows.append(
                                            (
                                                stored,
                                                stat_idx,
                                                int(unboosted),
                                                int(unmodified),
                                                int(fast_return),
                                                int(par),
                                                int(brn),
                                                int(screen),
                                                int(item),
                                            )
                                        )
                                        exp_b = []
                                        for boost in range(-6, 7):
                                            eff_boost = 0 if unboosted else boost
                                            val = (stored * stat_num[eff_boost + 6]) // 100
                                            if par and stat_idx == 4:
                                                val >>= 2
                                            if brn and (not unmodified) and stat_idx == 0:
                                                val >>= 1
                                            val = max(1, min(999, val))
                                            if fast_return:
                                                exp_b.append(int(val))
                                            else:
                                                post_val = val
                                                if (not unboosted) and screen and stat_idx in (1, 3):
                                                    post_val <<= 1
                                                if (item == ItemStatKind.THICK_CLUB and stat_idx == 0) or (
                                                    item == ItemStatKind.LIGHT_BALL and stat_idx == 2
                                                ):
                                                    post_val <<= 1
                                                if item == ItemStatKind.METAL_POWDER and stat_idx in (1, 3):
                                                    post_val += post_val >> 1
                                                exp_b.append(int(post_val))
                                        expected_rows.append(exp_b)

    cfg = jnp.asarray(rows, dtype=jnp.int32)
    boosts = jnp.arange(-6, 7, dtype=jnp.int32)
    actual_grid = np.asarray(
        jax.jit(
            jax.vmap(
                lambda row: jax.vmap(
                    lambda b: get_stat_branchless(
                        row[0],
                        b,
                        row[1],
                        row[2] != 0,
                        row[3] != 0,
                        row[4] != 0,
                        row[5] != 0,
                        row[6] != 0,
                        row[7] != 0,
                        row[7] != 0,
                        row[8],
                    )
                )(boosts)
            )
        )(cfg)
    )
    np.testing.assert_array_equal(actual_grid, np.asarray(expected_rows, dtype=np.int32))


def test_accuracy_threshold_and_damage_rolls_branchless():
    # Verify lookup_accuracy_threshold across all pct (0..100), acc (-6..6), eva (-6..6), BrightPowder, and ignore flags
    acc_mults = [33, 36, 43, 50, 60, 75, 100, 133, 166, 200, 233, 266, 300]
    pcts = jnp.arange(101, dtype=jnp.int32)[:, None, None]
    accs = jnp.arange(-6, 7, dtype=jnp.int32)[None, :, None]
    evas = jnp.arange(-6, 7, dtype=jnp.int32)[None, None, :]

    pcts_np = np.arange(101, dtype=np.int32)[:, None, None]
    accs_np = np.arange(-6, 7, dtype=np.int32)[None, :, None]
    evas_np = np.arange(-6, 7, dtype=np.int32)[None, None, :]
    acc_mults_arr = np.asarray(acc_mults, dtype=np.int32)
    num_acc_np = np.take(acc_mults_arr, accs_np + 6)
    num_eva_np = np.take(acc_mults_arr, 6 - evas_np)
    base_pct = (pcts_np * 255) // 100
    expected_base = np.clip((base_pct * num_acc_np * num_eva_np) // 10000, 1, 255)
    expected_bp = np.maximum(0, expected_base - 20)

    actual_base = np.asarray(
        jax.jit(lambda p, a, e: lookup_accuracy_threshold(p, a, e, False, False, False, False))(
            pcts, accs, evas
        )
    )
    np.testing.assert_array_equal(actual_base, expected_base)

    actual_bp = np.asarray(
        jax.jit(lambda p, a, e: lookup_accuracy_threshold(p, a, e, False, False, False, True))(
            pcts, accs, evas
        )
    )
    np.testing.assert_array_equal(actual_bp, expected_bp)
    assert int(lookup_accuracy_threshold(30, -6, 6, True, True, False, False)) == int(
        lookup_accuracy_threshold(30, 0, 0, False, False, False, False)
    )
    assert int(lookup_accuracy_threshold(50, 0, 6, False, False, True, False)) == int(
        lookup_accuracy_threshold(50, 0, 0, False, False, False, False)
    )
    assert int(lookup_accuracy_threshold(50, 0, -6, False, False, True, False)) == int(
        lookup_accuracy_threshold(50, 0, -6, False, False, False, False)
    )

    # Verify 32-bit apply_roll_branchless matches floor(dmg * roll / 255) for all dmg 1..4000, rolls 217..255
    dmgs = jnp.arange(1, 4001, dtype=jnp.int32)[:, None]
    rolls = jnp.arange(217, 256, dtype=jnp.int32)[None, :]
    fast_grid = np.asarray(jax.jit(lambda d, r: apply_roll_branchless(d, r, False))(dmgs, rolls))

    dmgs_np = np.arange(1, 4001, dtype=np.float64)[:, None]
    rolls_np = np.arange(217, 256, dtype=np.float64)[None, :]
    exact_grid = np.maximum(1, np.floor(dmgs_np * rolls_np / 255.0).astype(np.int32))
    np.testing.assert_array_equal(fast_grid, exact_grid)


def test_compact_battle_turn_and_forced_switch_parity():
    # Mirrors compact_battle_from_battle_step_and_hash & compact_battle_forced_switch_after_ko_steps_safely
    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    table, cb = build_compact_battle("1,2,3,4", p1, p2)
    assert int(cb.turn) == 1
    assert list(np.asarray(cb.active_idx)) == [0, 0]
    assert list(np.asarray(cb.alive_mask)) == [7, 7]

    # Step turn: Snorlax Double-Edge (0) vs Marowak Earthquake (0)
    cb_step1 = _JIT_APPLY(cb, table, jnp.array([0, 0], dtype=jnp.int32))
    assert int(cb_step1.turn) == 2
    assert list(np.asarray(cb_step1.hp_pool[:, 0])) == [121, 196, 156, 97, 171, 196]
    assert seed_to_u64_int(cb_step1.prng_seed) == 0xD402AA9393EB613E

    # Switch Snorlax to party slot 1 (action 5), Marowak Swords Dance (2)
    cb_sw = _JIT_APPLY(cb, table, jnp.array([5, 2], dtype=jnp.int32))
    assert int(cb_sw.active_idx[0]) == 1
    assert int(cb_sw.boosts[3, 0]) == 2  # Swords Dance +2 Attack

    # Multi-turn trajectory with Rest, Sleep Talk, forced switch, and terminal KO
    traj_cb = cb_step1
    for acts in [[2, 2], [3, 0], [5, 7], [1, 1], [3, 0], [6, 0], [1, 0]]:
        traj_cb = _JIT_APPLY(traj_cb, table, jnp.array(acts, dtype=jnp.int32))
    assert int(traj_cb.outcome) == 2  # P2Win
    assert list(np.asarray(traj_cb.alive_mask)) == [0, 7]
    assert seed_to_u64_int(traj_cb.prng_seed) == 0xD3900F72C51548C9
    assert float(eval_leaf(traj_cb, table, 0)) == 0.0
    assert float(eval_leaf(traj_cb, table, 1)) == 1.0

    # Forced switch after Explosion double-KO
    p1_exp = [
        _mk_set("Cloyster", 50, "Leftovers", ["Explosion", "Surf", "Ice Beam", "Spikes"]),
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
    ]
    p2_exp = [
        _mk_set("Marowak", 50, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    table_e, cb_e = build_compact_battle("1,2,3,4", p1_exp, p2_exp)
    cb_e = cb_e._replace(hp_pool=cb_e.hp_pool.at[3, 0].set(10))

    # Cloyster uses Explosion (0), Marowak uses Swords Dance (2)
    cb_after_exp = _JIT_APPLY(cb_e, table_e, jnp.array([0, 2], dtype=jnp.int32))
    assert list(np.asarray(cb_after_exp.request)) == [2, 2]
    assert list(np.asarray(cb_after_exp.alive_mask)) == [6, 6]

    # Legal actions for both sides should be [5, 6] (switch 2, switch 3)
    for s in range(2):
        mask = np.asarray(legal_actions_mask(cb_after_exp, table_e, s))
        legal_indices = list(np.flatnonzero(mask))
        assert legal_indices == [5, 6]

    # Step forced switch [5, 6]
    cb_resumed = _JIT_APPLY(cb_after_exp, table_e, jnp.array([5, 6], dtype=jnp.int32))
    assert list(np.asarray(cb_resumed.active_idx)) == [1, 2]
    assert list(np.asarray(cb_resumed.request)) == [1, 1]


def test_sharded_8core_rollout_matches_vmap_bit_for_bit():
    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    table, base_state = build_compact_battle("1,2,3,4", p1, p2)
    batch_size = 64  # 8 per virtual TPU core
    horizon = 16
    batched_state, policy_seeds = clone_battle_batch(base_state, batch_size)

    res_vmap = vmap_rollout(batched_state, table, policy_seeds, horizon=horizon)
    res_sharded = sharded_tpu_rollout(batched_state, table, policy_seeds, horizon=horizon)

    np.testing.assert_allclose(
        np.asarray(res_vmap.leaf_values), np.asarray(res_sharded.leaf_values), rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(res_vmap.outcomes), np.asarray(res_sharded.outcomes))
    np.testing.assert_array_equal(
        np.asarray(res_vmap.turns_played), np.asarray(res_sharded.turns_played)
    )
    np.testing.assert_array_equal(
        np.asarray(res_vmap.final_state.hp_pool),
        np.asarray(res_sharded.final_state.hp_pool),
    )
    np.testing.assert_array_equal(
        np.asarray(res_vmap.final_state.prng_seed),
        np.asarray(res_sharded.final_state.prng_seed),
    )


def test_packed_battle_120b_lossless_roundtrip():
    assert PACKED_BATTLE_WORDS == 30
    assert PACKED_BATTLE_BYTES == 120

    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    table, cb = build_compact_battle("1,2,3,4", p1, p2)

    # Step through multiple turns (status, boosts, switches, Substitute/Spikes/timers) and
    # verify pack_battle -> unpack_battle is 100% bit-exact on every field
    jit_pack = jax.jit(pack_battle)
    jit_unpack = jax.jit(unpack_battle)

    cur = cb
    for acts in [[0, 2], [2, 0], [3, 0], [5, 7], [2, 1], [6, 0]]:
        packed = jit_pack(cur)
        assert packed.words.shape == (30,)
        assert packed.words.dtype == jnp.uint32
        restored = jit_unpack(packed)
        for field in CompactBattle._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(cur, field)),
                np.asarray(getattr(restored, field)),
                err_msg=f"Mismatch in field {field}",
            )
        cur = _JIT_APPLY(cur, table, jnp.array(acts, dtype=jnp.int32))

    # Also verify batched [B, 30] pack/unpack roundtrip
    batched_cb, _ = clone_battle_batch(cur, 16)
    batched_packed = jit_pack(batched_cb)
    assert batched_packed.words.shape == (16, 30)
    batched_restored = jit_unpack(batched_packed)
    for field in CompactBattle._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(batched_cb, field)),
            np.asarray(getattr(batched_restored, field)),
            err_msg=f"Batched mismatch in field {field}",
        )


def test_heavy_playout_eval_weights_and_root_rm_plus():
    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    table, cb = build_compact_battle("1,2,3,4", p1, p2)
    w = default_eval_weights()

    # Symmetric healthy opening should evaluate near 0.5, and damaging P2 increases P1 eval
    v_init = float(eval_compact_leaf(cb, table, w))
    assert 0.35 < v_init < 0.65
    cb_damaged_p2 = cb._replace(hp_pool=cb.hp_pool.at[3, 0].set(20))
    v_adv = float(eval_compact_leaf(cb_damaged_p2, table, w))
    assert v_adv > v_init + 0.10

    # Verify Playout::Heavy with eps=0.0 picks max-damage move (Double-Edge = 0 for Snorlax vs Marowak)
    _, greedy_a0 = sample_heavy_playout_action(cb, table, 0, seed_from_str("1,2,3,4"), eps=0.0)
    assert int(greedy_a0) == 0  # Double-Edge deals more than Earthquake to Marowak

    # Verify Sharded Heavy Rollout on 120-byte PackedBattle passes TPU compatibility audit
    packed_batch, heavy_seeds = clone_packed_battle_batch(cb, 64)
    heavy_res = sharded_heavy_rollout(packed_batch, table, heavy_seeds, turns=8, eps=0.2)
    assert heavy_res.final_packed.words.shape == (64, 30)
    assert heavy_res.leaf_values.shape == (64,)
    assert np.all((np.asarray(heavy_res.leaf_values) >= 0.0) & (np.asarray(heavy_res.leaf_values) <= 1.0))

    # Verify Counterfactual Root Payoff Matrix & Regret Matching+ Nash solver
    eq = evaluate_root_payoff_matrix(cb, table, w, samples_per_cell=16, rollout_turns=8)
    assert eq.payoff_matrix.shape == (7, 7)
    assert np.isclose(float(jnp.sum(eq.sigma0)), 1.0, atol=1e-5)
    assert np.isclose(float(jnp.sum(eq.sigma1)), 1.0, atol=1e-5)
    assert 0.0 <= float(eq.game_value) <= 1.0
    # Illegal action slots (index 4 = switch to already-active slot 0) must have 0 probability
    assert float(eq.sigma0[4]) == 0.0
    assert float(eq.sigma1[4]) == 0.0


def test_20_diverse_games_486_steps_rust_parity():
    """Step-by-step differential test against 20 multi-turn golden games (486 steps)
    generated directly from `nc2000-engine` (`CompactBattle`) and `nc2000-bot` (`eval.rs`).
    Covers Baton Pass, Perish Song, Mean Look, Spikes, Roar/Whirlwind, Counter/Mirror Coat,
    Destiny Bond, Encore, Disable, Substitute, Belly Drum, Curse, Leech Seed, Nightmare,
    Confusion, Safeguard, Reflect/Light Screen, Weather, Berries, Thick Club/Light Ball/Metal Powder,
    Return/Frustration/Hidden Power/Flail/Reversal/Seismic Toss/Super Fang/Hyper Beam/Explosion.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "golden_20games.json"
    with open(fixture_path) as f:
        games = json.load(f)

    w = default_eval_weights()

    @jax.jit
    def _inspect_game_steps(cbs, tbl):
        def _one(cb):
            restored = unpack_battle(pack_battle(cb))
            l0 = legal_actions_mask(cb, tbl, 0)
            l1 = legal_actions_mask(cb, tbl, 1)
            ev0 = eval_leaf(cb, tbl, 0)
            rm = compact_race_margin(cb, tbl, w)
            return restored, l0, l1, ev0, rm

        return jax.vmap(_one)(cbs)

    state_fields = [
        "hp_pool", "pp", "status_packed", "item", "vol_flags", "timers",
        "boosts", "slot_refs", "active_idx", "party_roster", "alive_mask",
        "reflect_turns", "lightscreen_turns", "safeguard_turns", "side_flags",
        "last_move_selfko", "request", "turn", "weather_kind", "weather_turns",
        "quick_claw_roll", "outcome", "speed_order", "clause_flags",
        "hurt_damage_this_turn", "hurt_cat_this_turn",
    ]

    total_steps = 0
    for g_idx, g in enumerate(games):
        table, cb = build_compact_battle(g["seed"], g["p1"], g["p2"])
        states = []
        for step in g["steps"]:
            if step["actions"] is not None:
                cb = _JIT_APPLY(cb, table, jnp.array(step["actions"], dtype=jnp.int32))
            states.append(cb)
        total_steps += len(states)

        stacked_cb = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)
        restored_batch, l0_batch, l1_batch, ev0_batch, rm_batch = _inspect_game_steps(
            stacked_cb, table
        )
        cb_np = {fld: np.asarray(getattr(stacked_cb, fld)) for fld in CompactBattle._fields}
        restored_np = {
            fld: np.asarray(getattr(restored_batch, fld)) for fld in CompactBattle._fields
        }
        for fld in CompactBattle._fields:
            np.testing.assert_array_equal(
                cb_np[fld],
                restored_np[fld],
                err_msg=f"Pack roundtrip mismatch game={g_idx} field={fld}",
            )

        l0_np = np.asarray(l0_batch)
        l1_np = np.asarray(l1_batch)
        ev0_np = np.asarray(ev0_batch)
        rm_np = np.asarray(rm_batch)

        for s_idx, step in enumerate(g["steps"]):
            ref = step["state"]
            for fld in state_fields:
                jax_val = cb_np[fld][s_idx]
                ref_val = np.asarray(ref[fld], dtype=jax_val.dtype)
                np.testing.assert_array_equal(
                    jax_val,
                    ref_val,
                    err_msg=f"State mismatch game={g_idx} step={s_idx} field={fld}",
                )

            assert int(cb_np["prng_seed"][s_idx, 0]) == ref["prng_seed_hi"], f"PRNG hi game={g_idx} step={s_idx}"
            assert int(cb_np["prng_seed"][s_idx, 1]) == ref["prng_seed_lo"], f"PRNG lo game={g_idx} step={s_idx}"

            if int(cb_np["outcome"][s_idx]) == 0:
                l0 = list(np.flatnonzero(l0_np[s_idx]))
                l1 = list(np.flatnonzero(l1_np[s_idx]))
                assert l0 == sorted(ref["legal0"]), f"Legal0 game={g_idx} step={s_idx}"
                assert l1 == sorted(ref["legal1"]), f"Legal1 game={g_idx} step={s_idx}"

            assert abs(float(ev0_np[s_idx]) - float(ref["eval_leaf0"])) < 1e-5
            assert abs(float(rm_np[s_idx]) - float(ref["compact_race_margin"])) < 1e-5

    assert total_steps == 486


def test_residual_zero_identity_monotonicity_and_baton_pass_antisymmetry():
    """Mirrors `nc2000_bot::eval::residual_tests::{
        test_default_residual_zero_identity_and_monotonicity,
        test_residual_exact_antisymmetry_and_baton_pass_transfer
    }`."""
    bundle = default_residual_bundle()
    m = bundle.eval
    assert bool(m.active)
    assert not bool(bundle.rollout.active)

    z = jnp.zeros((32,), dtype=jnp.float32)
    assert float(eval_residual_forward(z, m)) == 0.0

    # Synergy features x[0..16] must be non-decreasing (>= 0)
    for j in range(16):
        ej = z.at[j].set(1.0)
        val = float(eval_residual_forward(ej, m))
        assert val >= 0.0, f"synergy feature x[{j}] produced negative residual {val}"

    # Vulnerability features x[16..32] must be non-increasing (<= 0)
    for j in range(16, 32):
        ej = z.at[j].set(1.0)
        val = float(eval_residual_forward(ej, m))
        assert val <= 0.0, f"vulnerability feature x[{j}] produced positive residual {val}"

    p1_team = [
        _mk_set("Scizor", 50, "Miracle Berry", ["Swords Dance", "Baton Pass", "Agility", "Hidden Power"]),
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Starmie", 50, "Leftovers", ["Surf", "Psychic", "Recover", "Thunder Wave"]),
    ]
    p2_team = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        _mk_set("Misdreavus", 50, "Leftovers", ["Mean Look", "Perish Song", "Protect", "Thunder"]),
    ]
    tbl_ab, cb_ab = build_compact_battle("1,2,3,4", p1_team, p2_team)
    tbl_ba, cb_ba = build_compact_battle("1,2,3,4", p2_team, p1_team)

    # Turn 1: Scizor Swords Dance (move 0 -> +2 Atk), Snorlax Rest (move 2)
    cb_ab = _JIT_APPLY(cb_ab, tbl_ab, jnp.array([0, 2], dtype=jnp.int32))
    cb_ba = _JIT_APPLY(cb_ba, tbl_ba, jnp.array([2, 0], dtype=jnp.int32))

    w = default_eval_weights()
    _, eff_ab = compact_base_logit_and_eff(cb_ab, tbl_ab, w)
    _, eff_ba = compact_base_logit_and_eff(cb_ba, tbl_ba, w)

    x0_ab = extract_eval_features(cb_ab, tbl_ab, 0)
    x1_ba = extract_eval_features(cb_ba, tbl_ba, 1)
    np.testing.assert_allclose(np.asarray(x0_ab), np.asarray(x1_ba), rtol=0, atol=1e-6)
    assert float(x0_ab[0]) > 0.20  # Baton Pass physical boost x Marowak receiver

    sym_ab = float(compact_eval_residual_sym(cb_ab, tbl_ab, eff_ab, m))
    sym_ba = float(compact_eval_residual_sym(cb_ba, tbl_ba, eff_ba, m))
    assert abs(sym_ab + sym_ba) < 1e-6
    assert sym_ab > 0.05

    # Verify rollout capability extractor identifies Scizor's Baton Pass (`a[0] > 0` on slot 1)
    l0 = legal_actions_mask(cb_ab, tbl_ab, 0)[:7]
    phi_s, act_feats, valid = extract_rollout_state_and_actions(
        cb_ab, tbl_ab, 0, l0, jnp.float32(0.25)
    )
    assert bool(valid)
    assert float(phi_s[10]) == 1.0  # own_has_self_switch
    assert float(act_feats[1, 0]) > 0.30  # move 1 (Baton Pass) transfer feature a[0]


def test_depth2_oracle_and_residual_training_pipeline(tmp_path: Path):
    """End-to-end verification of the TPU Depth-2 simultaneous-move oracle, batch labeling,
    constrained `train_residual_bundle_jax`, and drop-in `NC2000_RESIDUAL_JSON` export.
    """
    p1_team = [
        _mk_set("Scizor", 50, "Miracle Berry", ["Swords Dance", "Baton Pass", "Agility", "Hidden Power"]),
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Starmie", 50, "Leftovers", ["Surf", "Psychic", "Recover", "Thunder Wave"]),
    ]
    p2_team = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        _mk_set("Misdreavus", 50, "Leftovers", ["Mean Look", "Perish Song", "Protect", "Thunder"]),
    ]
    tbl, cb0 = build_compact_battle("1,2,3,4", p1_team, p2_team)
    # Step 1: Scizor Swords Dance (+2 Atk), Snorlax Rest
    cb_sd = _JIT_APPLY(cb0, tbl, jnp.array([0, 2], dtype=jnp.int32))

    d2 = evaluate_depth2_counterfactual_oracle(
        cb_sd,
        tbl,
        top_k=3,
        ply1_samples=16,
        ply2_samples=4,
        rollout_turns=8,
        rm_sweeps=256,
    )
    assert d2.payoff_matrix.shape == (7, 7)
    assert 0.0 <= float(d2.game_value) <= 1.0
    # With +2 Atk on Scizor and Marowak on bench vs sleeping Snorlax, Depth-2 oracle
    # settles Baton Pass -> Marowak switch and evaluates Side 0 as strongly favored (> 0.60)
    assert float(d2.game_value) > 0.60
    assert float(d2.q_values0[1]) > 0.60  # Baton Pass (action 1) has high Depth-2 Q-value

    # Label a small batch of states on TPU and run constrained JAX residual training
    cb_hurt = cb_sd._replace(hp_pool=cb_sd.hp_pool.at[3, 0].set(80))
    packed_batch = pack_battle(
        jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), cb0, cb_sd, cb_hurt)
    )
    labeled = label_positions_batch_tpu(
        packed_batch,
        tbl,
        top_k=3,
        ply1_samples=16,
        ply2_samples=4,
        rollout_turns=8,
        rm_sweeps=256,
    )
    assert labeled.x0.shape == (3, 32)
    assert labeled.v_oracle.shape == (3,)

    bundle, metrics = train_residual_bundle_jax(
        labeled.x0,
        labeled.x1,
        labeled.base_logit,
        labeled.v_oracle,
        labeled.is_1v1_unboosted,
        epochs=60,
        lr=5e-3,
    )
    assert metrics.trained_mse <= metrics.base_mse + 1e-6
    assert metrics.max_antisymmetry_err < 1e-6
    assert metrics.zero_input_out == 0.0

    # Export and reload JSON compatible with `NC2000_RESIDUAL_JSON`
    json_path = tmp_path / "residual_bundle.json"
    export_residual_bundle_json(bundle, json_path)
    reloaded = load_residual_bundle_json(json_path)
    np.testing.assert_allclose(
        np.asarray(bundle.eval.w1), np.asarray(reloaded.eval.w1), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(bundle.eval.w2), np.asarray(reloaded.eval.w2), rtol=0, atol=1e-6
    )


def _sample_6v6_teams() -> tuple[list[dict], list[dict]]:
    t_a = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 55, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
        _mk_set("Exeggutor", 50, "Miracle Berry", ["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
        _mk_set("Gengar", 50, "Leftovers", ["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
        _mk_set("Miltank", 50, "Leftovers", ["Body Slam", "Earthquake", "Milk Drink", "Heal Bell"]),
    ]
    t_b = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Raikou", 55, "Mint Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Starmie", 50, "Leftovers", ["Surf", "Psychic", "Recover", "Thunder Wave"]),
        _mk_set("Tyranitar", 50, "Leftovers", ["Rock Slide", "Crunch", "Earthquake", "Roar"]),
        _mk_set("Blissey", 50, "Leftovers", ["Ice Beam", "Thunderbolt", "Soft-Boiled", "Heal Bell"]),
    ]
    return t_a, t_b


def test_roster6_assembly_matches_build_compact_battle_and_multi_table_tpu_audit():
    """Verify `compile_roster6` + `assemble_3v3_from_rosters_jax` matches `build_compact_battle`
    bit-for-bit across arbitrary 3-mon picks, and `simulate_multi_table_heavy_games` passes
    strict TPU HLO verification.
    """
    from gen2_tpu import MatchTable, PackedBattle
    from gen2_tpu.kodoku import (
        assemble_3v3_from_rosters_jax,
        build_shared_dex_table,
        compile_roster6,
        simulate_multi_table_heavy_games,
    )

    t_a, t_b = _sample_6v6_teams()
    shared_dex = build_shared_dex_table()
    roster_a = compile_roster6(t_a)
    roster_b = compile_roster6(t_b)

    jit_assemble = jax.jit(
        lambda pa, pb, bseed: assemble_3v3_from_rosters_jax(
            roster_a, roster_b, pa, pb, bseed, shared_dex
        )
    )

    test_picks = [
        ((0, 2, 3), (0, 2, 4)),
        ((2, 0, 5), (3, 0, 5)),
        ((1, 3, 4), (1, 2, 3)),
        ((5, 2, 3), (4, 2, 5)),
    ]
    tables_list = []
    packed_list = []

    for idx, (pa, pb) in enumerate(test_picks):
        seed_str = f"100,{200 + idx},300,400"
        ref_tbl, ref_cb = build_compact_battle(seed_str, t_a, t_b, p1_pick=pa, p2_pick=pb)
        asm_tbl, asm_cb = jit_assemble(
            jnp.array(pa, dtype=jnp.int32),
            jnp.array(pb, dtype=jnp.int32),
            seed_from_str(seed_str),
        )
        for fld in MatchTable._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(ref_tbl, fld)),
                np.asarray(getattr(asm_tbl, fld)),
                err_msg=f"MatchTable mismatch on pick={pa}x{pb} field={fld}",
            )
        for fld in CompactBattle._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(ref_cb, fld)),
                np.asarray(getattr(asm_cb, fld)),
                err_msg=f"CompactBattle mismatch on pick={pa}x{pb} field={fld}",
            )
        tables_list.append(asm_tbl)
        packed_list.append(pack_battle(asm_cb))

    # Stack 4 distinct 3v3 matchups into a multi-table batch and verify TPU HLO compatibility
    batched_tables = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *tables_list)
    batched_packed = PackedBattle(words=jnp.stack([p.words for p in packed_list], axis=0))
    p_seeds = jnp.stack([seed_from_str(f"9,8,7,{i}") for i in range(4)], axis=0)

    report = verify_tpu_compatibility(
        lambda pk, tb, ps: simulate_multi_table_heavy_games(pk, tb, ps, turns=12, eps=0.2),
        batched_packed,
        batched_tables,
        p_seeds,
        spec=TPU_V5E,
    )
    assert report.passed, f"Multi-table TPU audit failed: {report}"
    assert len(report.forbidden_dtypes) == 0

    multi_res = simulate_multi_table_heavy_games(
        batched_packed, batched_tables, p_seeds, turns=12, eps=0.2
    )
    assert multi_res.leaf_values.shape == (4,)
    assert np.all((np.asarray(multi_res.leaf_values) >= 0.0) & (np.asarray(multi_res.leaf_values) <= 1.0))


def test_kodoku_two_stage_core_nash_leave_one_out_and_pair_table_export(tmp_path: Path):
    """Verify 60-action level legality, `select_top_k_orthogonal` Leave-One-Out Anti-Monopoly
    guarantee, `core_nash_6v6_pair_tpu`, Monotonic Subgame Inheritance, `recompute_population_wars_np`,
    and `PairTable` (`SPACE_VERSION = 1`) JSON export.
    """
    from gen2_tpu.kodoku import (
        PREVIEW_ACTIONS_1BASED,
        SPACE_VERSION,
        build_shared_dex_table,
        compile_roster6,
        core_nash_6v6_pair_tpu,
        evaluate_kodoku_gauntlet_tpu,
        export_dominance_audit_cells_json,
        export_pair_table_json,
        legal_action_indices,
        recompute_population_wars_np,
        select_top_k_orthogonal,
    )

    t_a, t_b = _sample_6v6_teams()
    # Both teams have two Lv55 (slots 1, 2) and four Lv50 (slots 3, 4, 5, 6).
    # Picking both Lv55 + any Lv50 gives 55+55+50 = 160 > 155 (illegal!).
    # So 4 subsets of size 3 containing {1, 2} (4 * 3 = 12 actions) are illegal -> 48 legal actions.
    legal_a = legal_action_indices([s["level"] for s in t_a])
    assert len(legal_a) == 48
    for idx in legal_a:
        a, b, c = PREVIEW_ACTIONS_1BASED[idx]
        assert not ({a, b, c} >= {1, 2})

    # Verify Leave-One-Out Anti-Monopoly Guarantee in `select_top_k_orthogonal`:
    # Suppose top-ranked actions all include slot 1; Pass 2 must append a backup excluding slot 1.
    eq = np.zeros((len(legal_a),), dtype=np.float32)
    br = np.zeros((len(legal_a),), dtype=np.float32)
    for i, act_idx in enumerate(legal_a):
        if 1 in PREVIEW_ACTIONS_1BASED[act_idx]:
            eq[i] = 0.25
            br[i] = 0.80
        else:
            eq[i] = 0.0
            br[i] = 0.40
    sup = select_top_k_orthogonal(legal_a, eq, br, k=2)
    assert len(sup) >= 3  # 2 cores containing slot 1 + 1 anti-monopoly backup excluding slot 1
    assert any(1 not in PREVIEW_ACTIONS_1BASED[idx] for idx in sup)

    shared_dex = build_shared_dex_table()
    roster_a = compile_roster6(t_a)
    roster_b = compile_roster6(t_b)

    res = core_nash_6v6_pair_tpu(
        roster_a,
        roster_b,
        shared_dex,
        screen_games=2,
        support_k=3,
        refine_games=2,
        screen_turns=4,
        refine_turns=4,
        seed=20260405,
    )
    assert 0.0 <= res.value <= 1.0
    assert len(res.legal_a) == 48
    assert len(res.legal_b) == 48
    assert res.screen_matrix.shape == (48, 48)
    assert len(res.sup_a) >= 3
    assert len(res.sup_b) >= 3
    assert np.isclose(float(np.sum(res.x_star)), 1.0, atol=1e-5)
    assert np.isclose(float(np.sum(res.y_star)), 1.0, atol=1e-5)
    assert np.all(res.slot_marginals_a >= 0.0)
    assert np.all(res.slot_marginals_b >= 0.0)
    assert res.effective_cores_a >= 1.0

    # Test Monotonic Subgame Inheritance for single-slot mutation (`mutated_slot_a = 5`)
    res_mut = core_nash_6v6_pair_tpu(
        roster_a,
        roster_b,
        shared_dex,
        screen_games=2,
        support_k=3,
        refine_games=2,
        screen_turns=4,
        refine_turns=4,
        seed=20260405,
        inherited_subgame=(res.sup_a, res.sup_b),
        mutated_slot_a=5,
    )
    assert 0.0 <= res_mut.value <= 1.0
    assert any(6 in PREVIEW_ACTIONS_1BASED[idx] for idx in res_mut.sup_a)

    # Test PairTable JSON export (`SPACE_VERSION = 1`)
    pt_path = tmp_path / "pair_table.json"
    export_pair_table_json(pt_path, res, team_a="team_a", team_b="team_b")
    with pt_path.open() as f:
        pt_data = json.load(f)
    assert pt_data["space_version"] == SPACE_VERSION
    assert len(pt_data["actions"]) == 60
    assert len(pt_data["sol"]["p_a"]) == 60
    assert len(pt_data["sol"]["p_b"]) == 60

    # Test Population WAR OLS regression (`recompute_population_wars_np`)
    lvls = np.array([[55, 55, 50, 50, 50, 50], [55, 55, 50, 50, 50, 50]], dtype=np.float32)
    margs = np.stack([res.slot_marginals_a, res.slot_marginals_b], axis=0)
    wars, _, _ = recompute_population_wars_np(margs, lvls, games=1.0)
    assert wars.shape == (2, 6)
    assert abs(float(np.mean(wars))) < 1e-5

    # Test Multi-Team Gauntlet & dominance-audit cell JSON export
    gauntlet = evaluate_kodoku_gauntlet_tpu(
        [{"id": "team_a", "sets": t_a}, {"id": "team_b", "sets": t_b}],
        screen_games=2,
        support_k=2,
        refine_games=2,
        screen_turns=4,
        refine_turns=4,
    )
    assert gauntlet.payoff_matrix.shape == (2, 2)
    assert np.isclose(gauntlet.payoff_matrix[0, 1] + gauntlet.payoff_matrix[1, 0], 1.0, atol=1e-5)
    cells_path = tmp_path / "dominance_cells.json"
    export_dominance_audit_cells_json(cells_path, gauntlet)
    with cells_path.open() as f:
        cells_data = json.load(f)
    assert len(cells_data["cells"]) == 4


def test_blindspot_leaf_false_boost_refund_and_1v1_invariance():
    """Verify `compact_false_boost_refund_diff` refunds Swagger false `+Atk` boosts on pure
    special attackers and discounts Encore-locked boosts by 50%, while returning `0.0` in 1v1.
    """
    from gen2_tpu.search import (
        compact_false_boost_refund_diff,
        default_eval_weights,
        eval_compact_01,
    )
    from gen2_tpu.state import VolBit

    def mk(species: str, level: int, item: str, moves: list[str]) -> dict:
        return {
            "name": species,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "level": level,
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255,
        }

    # P1 active is pure special Zapdos (no physical moves, no self-setup, no Baton Pass on team)
    p1 = [
        mk("Zapdos", 55, "Leftovers", ["Thunderbolt", "Hidden Power", "Rest", "Sleep Talk"]),
        mk("Suicune", 50, "Leftovers", ["Surf", "Ice Beam", "Rest", "Roar"]),
        mk("Blissey", 50, "Leftovers", ["Ice Beam", "Thunderbolt", "Soft-Boiled", "Heal Bell"]),
    ]
    # P2 active is physical Marowak (has Earthquake, Swords Dance)
    p2 = [
        mk("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        mk("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        mk("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    table, base_st = build_compact_battle("1,2,3,4", p1, p2)
    w = default_eval_weights()

    # Give P1 Zapdos +2 Atk (e.g. from opponent's Swagger):
    # Without BlindSpotLeaf refund, +2 Atk would inflate Zapdos's score by +0.30.
    # With BlindSpotLeaf refund, the false +2 Atk is refunded (-0.30) so eval is unchanged!
    st_swagged = base_st._replace(boosts=base_st.boosts.at[0, 0].set(2))
    ref_diff = float(compact_false_boost_refund_diff(st_swagged, table, w))
    assert np.isclose(ref_diff, -2.0 * float(w.boost[0]), atol=1e-6)
    assert np.isclose(
        float(eval_compact_01(st_swagged, table, w)),
        float(eval_compact_01(base_st, table, w)),
        atol=1e-6,
    )

    # Give P2 Marowak +2 Atk + Encore lock (and no Baton Pass):
    # 50% of positive boost score is refunded on Side 1 -> positive diff for Side 0.
    st_encored = base_st._replace(
        boosts=base_st.boosts.at[3, 0].set(2),
        vol_flags=base_st.vol_flags.at[3].set(VolBit.ENCORE),
    )
    ref_enc = float(compact_false_boost_refund_diff(st_encored, table, w))
    assert np.isclose(ref_enc, 0.5 * 2.0 * float(w.boost[0]), atol=1e-6)

    # In 1v1 (alive_mask = [1, 1]), refund_diff must be strictly 0.0 (1v1 invariance)
    st_1v1 = st_swagged._replace(alive_mask=jnp.array([1, 1], dtype=jnp.int32))
    assert float(compact_false_boost_refund_diff(st_1v1, table, w)) == 0.0


def test_dominated_mask7_perish_trap_combo_and_tpu_hlo():
    """Verify `compact_dominated_mask7`, Perish-trap combo/escape rollout overrides in
    `sample_heavy_playout_action`, and zero gather/scatter ops in lowered TPU XLA HLO.
    """
    from gen2_tpu.search import compact_dominated_mask7, sample_heavy_playout_action
    from gen2_tpu.state import Status, VolBit

    def mk(species: str, level: int, item: str, moves: list[str]) -> dict:
        return {
            "name": species,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "level": level,
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255,
        }

    # P1 Lead Misdreavus with Mean Look (slot 0), Perish Song (slot 1), Protect (slot 2), Rest (slot 3)
    p1 = [
        mk("Misdreavus", 55, "Leftovers", ["Mean Look", "Perish Song", "Protect", "Rest"]),
        mk("Snorlax", 50, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        mk("Cloyster", 50, "Leftovers", ["Surf", "Spikes", "Explosion", "Rest"]),
    ]
    p2 = [
        mk("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        mk("Blissey", 55, "Leftovers", ["Ice Beam", "Soft-Boiled", "Heal Bell", "Sing"]),
        mk("Raikou", 50, "Leftovers", ["Thunderbolt", "Reflect", "Roar", "Rest"]),
    ]
    table, st0 = build_compact_battle("1,2,3,4", p1, p2)
    seed = seed_from_str("5,6,7,8")

    # 1. At full HP on Turn 1, Misdreavus is faster than Skarmory -> Rest (move 3) is a provable no-op
    dom0 = np.asarray(compact_dominated_mask7(st0, table, 0))
    assert bool(dom0[3]) is True  # Full-HP Rest when faster

    # 2. Perish-trap Step 1: Foe Skarmory is not trapped and not perished -> picks Mean Look (move 0)
    _, act_step1 = sample_heavy_playout_action(st0, table, 0, seed, eps=0.0)
    assert int(act_step1) == 0

    # 3. Perish-trap Step 2: Once Foe Skarmory has VolBit.TRAPPED_OR_LOCKED -> picks Perish Song (move 1)
    st_trapped = st0._replace(vol_flags=st0.vol_flags.at[3].set(VolBit.TRAPPED_OR_LOCKED))
    _, act_step2 = sample_heavy_playout_action(st_trapped, table, 0, seed, eps=0.0)
    assert int(act_step2) == 1

    # 4. Counter-phaze escape: Foe Skarmory is trapped & perished at timer 2 -> picks Whirlwind (move 1)
    st_both_perish = st_trapped._replace(
        vol_flags=st_trapped.vol_flags.at[0]
        .set(VolBit.PERISH_SONG)
        .at[3]
        .set(VolBit.TRAPPED_OR_LOCKED | VolBit.PERISH_SONG),
        timers=st_trapped.timers.at[0, 3].set(2).at[3, 3].set(2),
    )
    _, act_skarm_phaze = sample_heavy_playout_action(st_both_perish, table, 1, seed, eps=0.0)
    assert int(act_skarm_phaze) == 1  # Skarmory's Whirlwind

    # 5. Perish escape at timer == 1: Misdreavus switches out to bench (action 5 or 6)
    st_cnt1 = st_both_perish._replace(timers=st_both_perish.timers.at[0, 3].set(1))
    _, act_escape = sample_heavy_playout_action(st_cnt1, table, 0, seed, eps=0.0)
    assert int(act_escape) in (5, 6)

    # 6. Leftovers `own_recovery_wait` exemption: when Misdreavus is chipped and Skarmory is asleep
    # with sleep counter 3 (and no Sleep Talk), no-ops are exempted so Misdreavus can stall for Leftovers!
    st_lefties_wait = st0._replace(
        hp_pool=st0.hp_pool.at[0, 0].set(100),
        status_packed=st0.status_packed.at[3].set(Status.SLP | (3 << 3)),
    )
    dom_wait = np.asarray(compact_dominated_mask7(st_lefties_wait, table, 0))
    assert not np.any(dom_wait[:4])

    # 7. Verify TPU HLO compatibility (zero gather/scatter, zero 64-bit dtypes)
    report = verify_tpu_compatibility(
        lambda s, tb, sd: sample_heavy_playout_action(s, tb, 0, sd, eps=0.2),
        st0,
        table,
        seed,
        spec=TPU_V5E,
    )
    assert report.passed, f"TPU compatibility failed: {report}"
    assert len(report.forbidden_dtypes) == 0
    assert not report.has_host_callbacks
    assert not report.has_dynamic_shapes


def test_coupled_shell_equilibrium_tpu(tmp_path: Path):
    """Verify `build_coupled_shell_units` and `solve_coupled_shell_equilibrium_tpu` group
    same-shell variants into `DEME2` units and solve the skew-symmetric coupled-shell equilibrium.
    """
    from gen2_tpu.kodoku import (
        build_coupled_shell_units,
        evaluate_kodoku_gauntlet_tpu,
        export_dominance_audit_cells_json,
        solve_coupled_shell_equilibrium_tpu,
    )

    t_a, t_b = _sample_6v6_teams()
    # Create a second variant of t_a with the exact same 6-(species, level) preview shell
    t_a_var2 = [dict(m) for m in t_a]
    t_a_var2[0] = dict(t_a[0], moves=["Body Slam", "Curse", "Rest", "Sleep Talk"])

    teams_p = [
        {"id": "a_v1", "sets": t_a},
        {"id": "a_v2", "sets": t_a_var2},
        {"id": "b_single", "sets": t_b},
    ]
    units = build_coupled_shell_units(teams_p)
    assert len(units) == 2
    assert units[0]["unit_id"] == "DEME2:a_v1+a_v2"
    assert units[1]["unit_id"] == "DEME1:b_single"

    summary = evaluate_kodoku_gauntlet_tpu(
        teams_p,
        screen_games=2,
        support_k=2,
        refine_games=2,
        screen_turns=4,
        refine_turns=4,
    )
    coupled = solve_coupled_shell_equilibrium_tpu(summary, teams_p=teams_p)
    assert coupled.m_solve.shape == (2, 2)
    assert np.isclose(coupled.m_solve[0, 1] + coupled.m_solve[1, 0], 1.0, atol=1e-5)
    assert np.isclose(float(np.sum(coupled.unit_weights)), 1.0, atol=1e-5)

    out_json = tmp_path / "coupled_cells.json"
    export_dominance_audit_cells_json(out_json, summary, coupled_equilibrium=coupled)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "coupled_equilibrium" in payload
    assert len(payload["unit_cells"]) == 2 * 3


def test_bayesian_shell_preview_nash_and_hitchhiker_detection():
    """Verify `solve_bayesian_shell_preview_nash_np` solves asymmetric information
    Shell-vs-Shell Team Preview games and `detect_hitchhiker_slots` pinpoints dead-bench
    Hitchhikers (`pick_s < 0.015`, `LOO_dV_s < 0.0005`).
    """
    from gen2_tpu.kodoku import (
        detect_hitchhiker_slots,
        solve_bayesian_shell_preview_nash_np,
    )

    # 2 variants in Shell U (`ka = 3` legal lineups) vs 1 variant in Shell V (`kb = 3` legal lineups).
    # Variant U0 dominates with lineup 0 against V's lineup 0, while U1 dominates with lineup 1 against V's lineup 1.
    m_00 = np.array(
        [
            [0.90, 0.20, 0.60],
            [0.30, 0.40, 0.60],
            [0.40, 0.40, 0.50],
        ],
        dtype=np.float64,
    )
    m_10 = np.array(
        [
            [0.30, 0.40, 0.60],
            [0.20, 0.90, 0.60],
            [0.40, 0.40, 0.50],
        ],
        dtype=np.float64,
    )
    mats = np.stack([[m_00], [m_10]], axis=0)  # shape (2, 1, 3, 3)
    val, per_u, per_v, p_strats, q_strats = solve_bayesian_shell_preview_nash_np(
        mats,
        alpha=np.array([0.5, 0.5], dtype=np.float64),
        beta=np.array([1.0], dtype=np.float64),
        sweeps=400,
    )
    assert 0.50 < val < 0.65
    assert per_u.shape == (2,)
    assert per_v.shape == (1,)
    assert p_strats.shape == (2, 3)
    assert q_strats.shape == (1, 3)
    assert np.allclose(np.sum(p_strats, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.sum(q_strats, axis=1), 1.0, atol=1e-5)

    # Test pinpoint Hitchhiker detection (`detect_hitchhiker_slots`)
    t_a, _ = _sample_6v6_teams()
    t_seaking = [dict(m) for m in t_a]
    t_seaking[5] = _mk_set("Seaking", 50, "Leftovers", ["Surf", "Horn Drill", "Rest", "Sleep Talk"])

    flagged_hh = detect_hitchhiker_slots(
        t_seaking,
        slot_picks=[0.80, 0.75, 0.70, 0.45, 0.30, 0.00],
        slot_marginals=[0.12, 0.08, 0.05, 0.03, 0.02, 0.00],
    )
    assert len(flagged_hh) == 1
    assert flagged_hh[0]["slot"] == 5
    assert flagged_hh[0]["species"] == "Seaking"
    assert flagged_hh[0]["reason"] == "dead_nash_slot"

    flagged_clean = detect_hitchhiker_slots(
        t_a,
        slot_picks=[0.55, 0.55, 0.50, 0.50, 0.45, 0.45],
        slot_marginals=[0.05, 0.05, 0.04, 0.04, 0.03, 0.03],
    )
    assert len(flagged_clean) == 0


def test_bayesian_election_call_and_shell_belief_minimax_tpu():
    """Verify Sequential Bayesian Election-Call (`選挙当確速報`) and In-Battle Intra-Shell
    Bayesian Belief + 1-Ply Simultaneous-Move RM+ Search on TPU.
    """
    from gen2_tpu.dex_loader import load_dex
    from gen2_tpu.kodoku import (
        assemble_3v3_from_rosters_jax,
        bayesian_election_call_bounds_np,
        build_shared_dex_table,
        compile_roster6,
        evaluate_6v6_bayesian_election_call_tpu,
        simulate_shell_belief_minimax_games_tpu,
        update_shell_belief_mask_jax,
    )

    # 1. Verify credible bounds shrink immediately on 1.0/0.0 blowouts while staying wide on 50/50
    sum_s = np.array([2.0, 1.0], dtype=np.float64)
    sq_s = np.array([2.0, 1.0], dtype=np.float64)
    cnt_s = np.array([2, 2], dtype=np.int32)
    mean_b, lcb_b, ucb_b, rad_b = bayesian_election_call_bounds_np(sum_s, sq_s, cnt_s, z=2.326)
    assert np.isclose(mean_b[0], 1.0)
    assert np.isclose(mean_b[1], 0.5)
    assert rad_b[0] < rad_b[1]
    assert lcb_b[0] > 0.40

    # 2. Verify `evaluate_6v6_bayesian_election_call_tpu` evaluates across waves with early cell calling
    shared_dex = build_shared_dex_table()
    t_a, t_b = _sample_6v6_teams()
    r_a = compile_roster6(t_a)
    r_b = compile_roster6(t_b)

    res_elec = evaluate_6v6_bayesian_election_call_tpu(
        r_a,
        r_b,
        shared_dex,
        wave_games=(2, 2),
        turns=4,
        win_threshold=0.95,
        loss_threshold=0.05,
        cell_radius_tol=0.35,
    )
    assert 0.0 <= res_elec.value <= 1.0
    assert 0.0 <= res_elec.lcb99 <= res_elec.ucb99 <= 1.0
    assert res_elec.election_status in ("OPEN", "WIN_CALLED", "LOSS_CALLED")
    assert res_elec.waves_used >= 1
    assert res_elec.cells_called_early >= 0

    # 3. Verify `update_shell_belief_mask_jax` eliminates incompatible opponent shell variants when PP drops
    t_b_alt = [dict(m) for m in t_b]
    t_b_alt[0] = dict(t_b[0], moves=["Thunderbolt", "Surf", "Ice Beam", "Rest"])
    r_b_alt = compile_roster6(t_b_alt)

    pick0 = jnp.array([0, 1, 2], dtype=jnp.int32)
    b_seed = jnp.array([0x12345678, 0x9ABCDEF0], dtype=jnp.uint32)
    tb_true, st_init = assemble_3v3_from_rosters_jax(r_a, r_b, pick0, pick0, b_seed, shared_dex)
    tb_alt, _ = assemble_3v3_from_rosters_jax(r_a, r_b_alt, pick0, pick0, b_seed, shared_dex)
    cand_tbs = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y], axis=0), tb_true, tb_alt)

    # Simulate P1's lead (slot 3, move 0) losing 1 PP
    st_after = st_init._replace(pp=st_init.pp.at[3, 0].set(st_init.pp[3, 0] - 1))
    bmask_init = jnp.array([True, True], dtype=jnp.bool_)
    bmask_upd = update_shell_belief_mask_jax(st_init, st_after, tb_true, cand_tbs, 0, bmask_init)
    assert bool(bmask_upd[0]) is True
    assert bool(bmask_upd[1]) is False

    # 4. Verify `simulate_shell_belief_minimax_games_tpu` runs end-to-end on batched trajectories
    pk_batch = pack_battle(jax.tree_util.tree_map(lambda x: x[None, ...], st_init))
    true_tb_batch = jax.tree_util.tree_map(lambda x: x[None, ...], tb_true)
    cand_p0_batch = jax.tree_util.tree_map(lambda x: x[None, ...], cand_tbs)
    prior_p0_batch = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
    cand_p1_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], tb_true)
    prior_p1_batch = jnp.array([[1.0]], dtype=jnp.float32)
    p_seeds = jnp.array([[0x11112222, 0x33334444]], dtype=jnp.uint32)

    ro_res, final_m0, final_m1 = simulate_shell_belief_minimax_games_tpu(
        pk_batch,
        true_tb_batch,
        cand_p0_batch,
        prior_p0_batch,
        cand_p1_batch,
        prior_p1_batch,
        p_seeds,
        turns=3,
        rollout_turns=0,
        rm_sweeps=16,
    )
    assert ro_res.leaf_values.shape == (1,)
    assert 0.0 <= float(ro_res.leaf_values[0]) <= 1.0
    assert bool(final_m0[0, 0]) is True
    assert bool(final_m1[0, 0]) is True


def test_solve_bayesian_equilibrium_jax_and_unified_nash_bake_pipeline_tpu(tmp_path: Path):
    """Verify:
    1. `solve_bayesian_equilibrium_jax` prevents wishful-thinking across determinized worlds
       (each world `v` minimizes independently over its own legal action mask `masks_opp_v[v]`)
       and passes strict TPU HLO audit (0 gather/scatter ops, 0 64-bit dtypes).
    2. `run_unified_nash_and_bake_pipeline_tpu` runs all 3 stages in 1 TPU workflow:
       - Gate 1: Non-inferiority election-call (`bayesian_eq=True` vs `bayesian_eq=False`),
       - Gate 2: Sequential election-call Nash verification & rebuild,
       - Gate 3: Team Preview `PairTable` baking (`SPACE_VERSION = 1`).
    """
    from gen2_tpu.kodoku import (
        run_unified_nash_and_bake_pipeline_tpu,
        solve_bayesian_equilibrium_jax,
    )

    # 1. Construct a 2-world Bayesian game where:
    # - Action 0 is punished in World 0 by opp action 0 (payoff 0.10, while opp action 1 gives 0.95)
    #   and punished in World 1 by opp action 1 (payoff 0.10, while opp action 0 gives 0.95).
    #   World-averaging `0.5*(V0 + V1)` falsely makes Action 0 look like 0.525 against everything,
    #   whereas coupled per-world Bayesian Equilibrium sees that in each world the opponent chooses
    #   that world's best response (giving 0.10) and therefore selects safe Action 1 (payoff 0.62).
    v0 = np.full((7, 7), 0.50, dtype=np.float32)
    v1 = np.full((7, 7), 0.50, dtype=np.float32)
    v0[0, 0], v0[0, 1] = 0.10, 0.95
    v1[0, 0], v1[0, 1] = 0.95, 0.10
    v0[1, :2] = 0.62
    v1[1, :2] = 0.62

    mats_v = jnp.stack([jnp.asarray(v0), jnp.asarray(v1)], axis=0)
    mask_own = jnp.array([True, True, False, False, False, False, False], dtype=jnp.bool_)
    masks_opp_v = jnp.stack([mask_own, mask_own], axis=0)
    post_w = jnp.array([0.5, 0.5], dtype=jnp.float32)

    sigma0, act_vals, pure_guar, eq_val = solve_bayesian_equilibrium_jax(
        mats_v, mask_own, masks_opp_v, post_w, sweeps=96
    )
    assert float(sigma0[1]) > 0.90
    assert float(sigma0[0]) < 0.10
    assert np.isclose(float(eq_val), 0.62, atol=0.05)

    report = verify_tpu_compatibility(
        lambda mv, mo, mov, pw: solve_bayesian_equilibrium_jax(mv, mo, mov, pw, sweeps=64),
        mats_v,
        mask_own,
        masks_opp_v,
        post_w,
        spec=TPU_V5E,
    )
    assert report.passed, f"solve_bayesian_equilibrium_jax TPU audit failed: {report}"
    assert len(report.forbidden_dtypes) == 0
    assert not report.has_host_callbacks
    assert not report.has_dynamic_shapes

    # 2. Verify 1-run `run_unified_nash_and_bake_pipeline_tpu`
    t_a, t_b = _sample_6v6_teams()
    teams_pool = [
        {"id": "nash_a", "sets": t_a},
        {"id": "challenger_b", "sets": t_b},
    ]
    out_dir = tmp_path / "unified_bake"
    pipe_res = run_unified_nash_and_bake_pipeline_tpu(
        teams_pool,
        incumbent_nash_weights={"nash_a": 1.0, "challenger_b": 0.0},
        non_inferiority_waves=(2,),
        nash_wave_games=(2,),
        policy_turns=3,
        policy_rollout_turns=0,
        policy_followup_turns=4,
        policy_rm_sweeps=16,
        bake_screen_games=2,
        bake_support_k=2,
        bake_refine_games=2,
        bake_screen_turns=4,
        bake_refine_turns=4,
        bake_out_dir=out_dir,
        seed=20260405,
    )
    assert pipe_res.non_inferiority_status in (
        "SUPERIOR_CONFIRMED",
        "NON_INFERIOR_CONFIRMED",
        "INFERIOR_REJECTED",
    )
    assert pipe_res.nash_status in ("NASH_HELD", "NASH_REBUILT")
    assert np.isclose(float(np.sum(pipe_res.final_nash_weights)), 1.0, atol=1e-5)
    assert len(pipe_res.baked_pair_tables) >= 2
    assert (out_dir / "unified_summary.json").exists()


def test_kodoku_evolution_and_training_normalization_tpu(tmp_path):
    """Verify `normalize_training_set`, `mutate_team_for_kodoku`, `evaluate_population_tpu`, and `run_kodoku_tpu`."""
    from gen2_tpu.kodoku import (
        evaluate_population_tpu,
        mutate_team_for_kodoku,
        normalize_training_set,
        normalize_training_team,
        preview_signature_sets,
        run_kodoku_tpu,
    )

    # 1. Verify training-v3 normalization parity with `tools/normalize-training.js`
    starmie_special = normalize_training_set(
        {
            "species": "Starmie",
            "level": 50,
            "gender": "N",
            "item": "Miracle Berry",
            "moves": ["Surf", "Ice Beam", "Psychic", "Recover"],
            "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        }
    )
    assert starmie_special["ivs"]["atk"] == 2
    assert starmie_special["evs"]["atk"] == 0
    assert starmie_special["ivs"]["hp"] == 31

    zapdos_hp_ice = normalize_training_set(
        {
            "species": "Zapdos",
            "level": 50,
            "gender": "N",
            "item": "Mint Berry",
            "moves": ["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"],
            "ivs": {"hp": 31, "atk": 31, "def": 26, "spa": 31, "spd": 31, "spe": 31},
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        }
    )
    assert zapdos_hp_ice["ivs"]["atk"] == 22
    assert zapdos_hp_ice["ivs"]["def"] == 26
    assert zapdos_hp_ice["evs"]["atk"] == 0

    # 2. Verify shell-preserving Kodoku mutation
    t_a, t_b = _sample_6v6_teams()
    parent = {"id": "parent_a", "sets": normalize_training_team(t_a), "variantCluster": "vc-test-a"}
    rng = np.random.default_rng(42)
    child = mutate_team_for_kodoku(parent, rng, preserve_shell=True, new_team_id="child_a")
    assert preview_signature_sets(child["sets"]) == preview_signature_sets(parent["sets"])
    assert len({s["species"] for s in child["sets"]}) == 6
    non_empty_items = [s["item"] for s in child["sets"] if s.get("item")]
    assert len(non_empty_items) == len(set(non_empty_items))

    # 3. Verify two-stage `evaluate_population_tpu` & `run_kodoku_tpu`
    seed_pool = [
        {"id": "seed_a", "sets": t_a, "variantCluster": "vc-test-a"},
        {"id": "seed_b", "sets": t_b, "variantCluster": "vc-test-b"},
    ]
    pop_eval = evaluate_population_tpu(
        seed_pool,
        screen_games=2,
        support_k=2,
        refine_games=2,
        screen_turns=4,
        refine_turns=4,
        survivor_top_k=2,
        policy_games_per_pair=2,
        policy_turns=2,
        policy_rollout_turns=0,
        policy_followup_turns=0,
        policy_rm_sweeps=16,
        max_shell_variants=1,
        bayesian_eq=True,
    )
    assert pop_eval.blended_payoff_matrix.shape == (2, 2)
    assert len(pop_eval.survivor_indices) >= 1

    evo_out = tmp_path / "kodoku_evo"
    evo_res = run_kodoku_tpu(
        seed_pool,
        generations=2,
        offspring_per_gen=1,
        max_population_size=3,
        screen_games=2,
        support_k=2,
        refine_games=2,
        screen_turns=4,
        refine_turns=4,
        survivor_top_k=2,
        policy_games_per_pair=2,
        policy_turns=2,
        policy_rollout_turns=0,
        policy_followup_turns=0,
        policy_rm_sweeps=16,
        max_shell_variants=1,
        out_dir=evo_out,
    )
    assert len(evo_res.generation_history) == 2
    assert len(evo_res.nash_teams) >= 1
    assert (evo_out / "kodoku_evolution_summary.json").exists()


def test_pack_battle_61pp_and_stall_counter_roundtrip():
    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    _, cb = build_compact_battle("1,2,3,4", p1, p2)
    pp_hi = cb.pp.at[0, 0].set(61).at[3, 2].set(63)
    timers_hi = cb.timers.at[0, 3].set(2 | (3 << 3)).at[4, 3].set(1 | (5 << 3))
    cb_hi = cb._replace(pp=pp_hi, timers=timers_hi)

    restored = jax.jit(lambda s: unpack_battle(pack_battle(s)))(cb_hi)
    np.testing.assert_array_equal(np.asarray(restored.pp), np.asarray(cb_hi.pp))
    np.testing.assert_array_equal(np.asarray(restored.timers), np.asarray(cb_hi.timers))


def test_kernel_gen2_mechanics_regression_guards():
    # 1. Protect/Endure stall counter, moved_second failure, and Endure 1-HP clamp
    p1 = [
        _mk_set("Snorlax", 55, "Leftovers", ["Protect", "Endure", "Body Slam", "Rest"]),
        _mk_set("Zapdos", 50, "Leftovers", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Rapid Spin", "Surf", "Spikes", "Explosion"]),
    ]
    p2 = [
        _mk_set("Golem", 55, "Leftovers", ["Explosion", "Earthquake", "Rock Slide", "Curse"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        _mk_set("Starmie", 50, "Leftovers", ["Surf", "Psychic", "Recover", "Rapid Spin"]),
    ]
    tbl, st = build_compact_battle("1,2,3,4", p1, p2)

    # Slower/forced Endure (move 1) survives lethal Explosion (move 0) at 1 HP cleanly
    st_end = _JIT_APPLY(st, tbl, jnp.array([1, 0], dtype=jnp.int32))
    assert int(st_end.hp_pool[0, 0]) == 1
    assert (int(st_end.timers[0, 3]) >> 3) == 1
    assert int(st_end.alive_mask[1]) == 6

    # Protect moving second after opponent switch fails with stall counter remaining 0
    st_prot_sw = _JIT_APPLY(st, tbl, jnp.array([0, 5], dtype=jnp.int32))
    assert (int(st_prot_sw.timers[0, 3]) >> 3) == 0

    # 2. Hyper Beam sets 0x80 recharge bit restricting legal_actions_mask to [0], then clears
    p1_hb = [
        _mk_set("Snorlax", 55, "Leftovers", ["Hyper Beam", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Leftovers", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    tbl_hb, st_hb = build_compact_battle("9,8,7,6", p1_hb, p2)
    st_hb1 = _JIT_APPLY(st_hb, tbl_hb, jnp.array([0, 3], dtype=jnp.int32))
    assert (int(st_hb1.slot_refs[0]) & 0x80) != 0
    assert list(np.flatnonzero(np.asarray(legal_actions_mask(st_hb1, tbl_hb, 0)))) == [0]
    st_hb2 = _JIT_APPLY(st_hb1, tbl_hb, jnp.array([5, 3], dtype=jnp.int32))
    assert (int(st_hb2.slot_refs[0]) & 0x80) == 0
    assert int(st_hb2.active_idx[0]) == 0

    # 3. Struggle fallback when all PP == 0 does not wrap PP negative and inflicts 1/4 damage dealt recoil
    st_struggle = st_hb._replace(
        pp=st_hb.pp.at[0].set(jnp.zeros((4,), dtype=jnp.int32)),
        item=st_hb.item.at[0].set(jnp.int32(0)),
    )
    assert list(np.flatnonzero(np.asarray(legal_actions_mask(st_struggle, tbl_hb, 0)))) == [0, 5, 6]
    st_str_after = _JIT_APPLY(st_struggle, tbl_hb, jnp.array([0, 3], dtype=jnp.int32))
    assert list(np.asarray(st_str_after.pp[0])) == [0, 0, 0, 0]
    assert int(st_str_after.hp_pool[0, 0]) < int(tbl_hb.maxhp[0])

    # 4. Moonlight weather healing ignores stale weather_kind when weather_turns == 0
    p1_moon = [
        _mk_set("Umbreon", 55, "Leftovers", ["Moonlight", "Curse", "Body Slam", "Rest"]),
        _mk_set("Zapdos", 50, "Leftovers", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Rapid Spin", "Surf", "Spikes", "Explosion"]),
    ]
    tbl_m, st_m = build_compact_battle("1,2,3,4", p1_moon, p2)
    maxhp_u = int(tbl_m.maxhp[0])
    st_m_stale = st_m._replace(
        hp_pool=st_m.hp_pool.at[0, 0].set(maxhp_u // 4),
        weather_kind=jnp.int32(CompactWeather.RAIN),
        weather_turns=jnp.int32(0),
    )
    st_m_after = _JIT_APPLY(st_m_stale, tbl_m, jnp.array([0, 3], dtype=jnp.int32))
    expected_clear_hp = min(maxhp_u, (maxhp_u // 4) + (maxhp_u // 2) + int(tbl_m.quantum_1_16[0]))
    assert int(st_m_after.hp_pool[0, 0]) == expected_clear_hp

    # 5. Rapid Spin clears Spikes, Leech Seed, and TRAPPED_OR_LOCKED
    st_rs = st._replace(
        active_idx=jnp.array([2, 0], dtype=jnp.int32),
        side_flags=jnp.array([3, 0], dtype=jnp.int32),
        vol_flags=st.vol_flags.at[2].set(VolBit.LEECH_SEED | VolBit.TRAPPED_OR_LOCKED),
    )
    st_rs_after = _JIT_APPLY(st_rs, tbl, jnp.array([0, 3], dtype=jnp.int32))
    assert (int(st_rs_after.side_flags[0]) & 0x03) == 0
    assert (int(st_rs_after.vol_flags[2]) & (VolBit.LEECH_SEED | VolBit.TRAPPED_OR_LOCKED)) == 0

    # 6. AncientPower raises all 5 stats (+1) with chance forced by high-probability seed check
    p1_ap = [
        _mk_set("Omastar", 50, "Leftovers", ["Ancient Power", "Surf", "Ice Beam", "Rest"]),
        _mk_set("Zapdos", 50, "Leftovers", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Rapid Spin", "Surf", "Spikes", "Explosion"]),
    ]
    tbl_ap, st_ap = build_compact_battle("1,2,3,4", p1_ap, p2)
    omni_hit = False
    for k in range(40):
        cand = _JIT_APPLY(
            st_ap._replace(prng_seed=jnp.array([k * 0x1234567 + 1, k * 0x89ABCDEF + 3], dtype=jnp.uint32)),
            tbl_ap,
            jnp.array([0, 3], dtype=jnp.int32),
        )
        if np.all(np.asarray(cand.boosts[0, :5]) == 1):
            omni_hit = True
            break
    assert omni_hit


def test_wait_action_handling_and_counter_immunity_and_belief_invariance():
    # 1. Forced-switch Wait state (`request == [2, 0]`): Wait side uses action 7 cleanly
    p1 = [
        _mk_set("Gengar", 55, "Leftovers", ["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
        _mk_set("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Leftovers", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
    ]
    p2 = [
        _mk_set("Gengar", 50, "Leftovers", ["Shadow Ball", "Thunderbolt", "Counter", "Mirror Coat"]),
        _mk_set("Blissey", 55, "Leftovers", ["Ice Beam", "Soft-Boiled", "Heal Bell", "Seismic Toss"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
    ]
    tbl, st = build_compact_battle("1,2,3,4", p1, p2)

    # Simulate P1 needing forced switch (`request = [2, 0]`)
    st_wait = st._replace(
        hp_pool=st.hp_pool.at[0, 0].set(0),
        alive_mask=jnp.array([6, 7], dtype=jnp.int32),
        request=jnp.array([2, 0], dtype=jnp.int32),
    )
    w = default_eval_weights()
    eq = evaluate_root_payoff_matrix(st_wait, tbl, w, samples_per_cell=4, rollout_turns=0)
    assert np.isfinite(float(eq.game_value))
    d2 = evaluate_depth2_counterfactual_oracle(st_wait, tbl, w, rollout_turns=0)
    assert np.isfinite(float(d2.game_value))

    # `select_shell_belief_minimax_action_jax` returns action 7 for waiting side 1
    cand_tbs = jax.tree_util.tree_map(lambda x: x[None, ...], tbl)
    _, a1_wait = select_shell_belief_minimax_action_jax(
        st_wait,
        tbl,
        cand_tbs,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([True], dtype=jnp.bool_),
        1,
        jnp.array([0x1111, 0x2222], dtype=jnp.uint32),
        w,
        rollout_turns=0,
        rm_sweeps=8,
    )
    assert int(a1_wait) == 7

    # 2. Counter immunity guard in `sample_heavy_playout_action(..., rollout_residual=None)`:
    # P2 Gengar (Ghost/Poison) hit by Normal damage cannot hit Ghost with Counter -> must NOT choose Counter
    p1_norm = [
        _mk_set("Snorlax", 55, "Leftovers", ["Body Slam", "Earthquake", "Rest", "Sleep Talk"]),
        _mk_set("Zapdos", 50, "Leftovers", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        _mk_set("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2_cnt = [
        _mk_set("Wobbuffet", 50, "Leftovers", ["Counter", "Mirror Coat", "Safeguard", "Destiny Bond"]),
        _mk_set("Blissey", 55, "Leftovers", ["Ice Beam", "Soft-Boiled", "Heal Bell", "Seismic Toss"]),
        _mk_set("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
    ]
    tbl_cnt, st_cnt = build_compact_battle("1,2,3,4", p1_norm, p2_cnt)
    # Force opponent (P1) active = Gengar (Ghost type -> Counter Normal move is immune!)
    tbl_ghost, st_ghost = build_compact_battle("1,2,3,4", p1, p2_cnt)
    st_g_hurt = st_ghost._replace(
        hurt_damage_this_turn=jnp.array([0, 120], dtype=jnp.int32),
        hurt_cat_this_turn=jnp.array([0, 1], dtype=jnp.int32),
    )
    _, a1_cnt_imm = sample_heavy_playout_action(
        st_g_hurt, tbl_ghost, 1, jnp.array([1, 2], dtype=jnp.uint32), eps=0.0, rollout_residual=None
    )
    assert int(a1_cnt_imm) != 0

    # Against Snorlax (Normal type, not immune), Counter SHOULD be preferred when hurt_cat == 1
    st_s_hurt = st_cnt._replace(
        hurt_damage_this_turn=jnp.array([0, 120], dtype=jnp.int32),
        hurt_cat_this_turn=jnp.array([0, 1], dtype=jnp.int32),
    )
    _, a1_cnt = sample_heavy_playout_action(
        st_s_hurt, tbl_cnt, 1, jnp.array([1, 2], dtype=jnp.uint32), eps=0.0, rollout_residual=None
    )
    assert int(a1_cnt) == 0

    # 3. `update_shell_belief_mask_jax` move-slot-order invariance and Thief item-gain guard
    p2_perm = [
        _mk_set("Wobbuffet", 50, "Leftovers", ["Mirror Coat", "Counter", "Safeguard", "Destiny Bond"]),
        p2_cnt[1],
        p2_cnt[2],
    ]
    tbl_perm, _ = build_compact_battle("1,2,3,4", p1_norm, p2_perm)
    cands_p = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y], axis=0), tbl_cnt, tbl_perm)
    # P2 uses Counter (slot 0 in tbl_cnt, slot 1 in tbl_perm) AND gains an item from 0 -> 1 via Thief
    st_prev = st_cnt._replace(item=st_cnt.item.at[3].set(0))
    st_curr = st_prev._replace(
        pp=st_prev.pp.at[3, 0].set(st_prev.pp[3, 0] - 1),
        item=st_prev.item.at[3].set(23),
    )
    upd = update_shell_belief_mask_jax(
        st_prev, st_curr, tbl_cnt, cands_p, 0, jnp.array([True, True], dtype=jnp.bool_)
    )
    assert bool(upd[0]) is True
    assert bool(upd[1]) is True

