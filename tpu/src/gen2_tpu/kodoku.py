"""TPU-accelerated 6v6 Two-Stage Core-Nash & Kodoku ("蠱毒") Evaluator for Gen 2 NC2000.

Mirrors and drops into `nc2000-engine`:
1. `crates/bot/src/preview.rs` (`SPACE_VERSION = 1`, `preview_actions`, `legal_action_indices`,
   `PairTable`, `SubMatrix`, `PairSolution`, `solve_pair`).
2. `crates/bot/examples/meta_eco.rs` (`core_nash_eval_weighted`, `select_top_k_orthogonal`,
   Leave-One-Out slot marginal value `ΔV_{-s}`, Monotonic Subgame Inheritance for single-slot
   mutations, `recompute_population_wars`, `solve_rectangular_minimax`, and `dominance-audit` /
   `converge-conclusion` cell caches).

Key TPU Architecture:
- `RosterTable6`: Precompiles a 6-Pokemon roster into static 32-bit arrays (`int32[6, ...]`).
- `assemble_3v3_from_rosters_jax`: Pure-JAX branchless function that slices any 3-mon pick
  `(pick_a, pick_b)` from `(roster_a, roster_b)` and advances the battle PRNG through gender
  draws, `speed_sort` tie-breakers, lead switch tie-breaker, and Turn-0 Quick Claw roll,
  matching `build_compact_battle` bit-for-bit.
- `simulate_multi_table_heavy_games`: Batched multi-matchup `Playout::Heavy` simulator where
  every trajectory in the batch can carry its own `MatchTable` (`in_axes=(0, 0, 0)`), enabling
  simultaneous evaluation of thousands of distinct 3v3 subgames in a single TPU kernel.
"""

from __future__ import annotations

from functools import lru_cache, partial
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .dex_loader import (
    DEFAULT_DEX_PATH,
    _COMPACT_TYPE_BOOST_ITEMS,
    _HP_TYPES,
    _STAT_KEYS,
    _compile_move_spec,
    _tr,
    load_dex,
    toid,
)
from .kernel import apply_actions, legal_actions_mask
from .prng import prng_random, prng_random_chance, seed_from_str
from .residual import ResidualBundle, RolloutResidualWeights
from .search import (
    EvalWeights,
    HeavyRolloutResult,
    compact_dominated_mask7,
    default_eval_weights,
    eval_compact_leaf,
    sample_heavy_playout_action,
    solve_rm_plus_jax,
)
from .state import (
    CompactBattle,
    CompactWeather,
    ItemFlag,
    ItemStatKind,
    MatchTable,
    MoveFlag,
    PackedBattle,
    pack_battle,
    unpack_battle,
)

SPACE_VERSION: int = 1
MAX_TOTAL_LEVEL_NC2000: int = 155


# =========================================================================
# 1. Canonical 60 Team-Preview Action Space (`nc2000_bot::preview`)
# =========================================================================


def preview_actions_1based() -> list[tuple[int, int, int]]:
    """Generate the 60 canonical 1-based triples `(lead, bench_lo, bench_hi)` in `1..=6`.

    Matches `nc2000_bot::preview::preview_actions()` in exact order:
    20 subsets `(a < b < c)` in `1..=6`, each expanded into `(a, b, c)`, `(b, a, c)`, `(c, a, b)`.
    """
    out: list[tuple[int, int, int]] = []
    for a in range(1, 7):
        for b in range(a + 1, 7):
            for c in range(b + 1, 7):
                out.append((a, b, c))
                out.append((b, a, c))
                out.append((c, a, b))
    return out


PREVIEW_ACTIONS_1BASED: tuple[tuple[int, int, int], ...] = tuple(preview_actions_1based())
PREVIEW_ACTIONS_0BASED_NP: np.ndarray = np.array(
    [[a - 1, b - 1, c - 1] for (a, b, c) in PREVIEW_ACTIONS_1BASED], dtype=np.int32
)
PREVIEW_CORE_MASKS_60_NP: np.ndarray = np.array(
    [
        (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
        for (a, b, c) in PREVIEW_ACTIONS_1BASED
    ],
    dtype=np.int32,
)


def legal_action_indices(
    levels: list[int] | np.ndarray,
    max_total_level: int = MAX_TOTAL_LEVEL_NC2000,
) -> list[int]:
    """Return indices into `PREVIEW_ACTIONS_1BASED` (`0..60`) satisfying `sum(level) <= 155`.

    Matches `nc2000_bot::preview::legal_action_indices`.
    """
    lv = [int(x) for x in levels]
    n_slots = len(lv)
    legal: list[int] = []
    for idx, (a, b, c) in enumerate(PREVIEW_ACTIONS_1BASED):
        if a <= n_slots and b <= n_slots and c <= n_slots:
            if lv[a - 1] + lv[b - 1] + lv[c - 1] <= max_total_level:
                legal.append(idx)
    return legal


def legal_preview_mask_60(
    levels: list[int] | np.ndarray,
    max_total_level: int = MAX_TOTAL_LEVEL_NC2000,
) -> np.ndarray:
    """Return boolean mask `bool[60]` over the 60 canonical preview actions."""
    mask = np.zeros((60,), dtype=bool)
    for idx in legal_action_indices(levels, max_total_level=max_total_level):
        mask[idx] = True
    return mask


# =========================================================================
# 2. Precompiled 6-Mon `RosterTable6` & Pure-JAX 3v3 Subgame Assembly
# =========================================================================


class SharedDexTable(NamedTuple):
    """Global Gen 2 Stadium 2 Dex constants shared across all rosters and matchups."""

    type_chart: jax.Array  # int32[20, 20]
    normal_type: jax.Array  # int32
    fighting_type: jax.Array  # int32
    flying_type: jax.Array  # int32
    ground_type: jax.Array  # int32
    ghost_type: jax.Array  # int32
    grass_type: jax.Array  # int32
    fire_type: jax.Array  # int32
    electric_type: jax.Array  # int32
    ice_type: jax.Array  # int32
    sandstorm_immune_mask: jax.Array  # uint32
    poison_immune_mask: jax.Array  # uint32


class RosterTable6(NamedTuple):
    """Precompiled 6-Pokemon team table (`[6, ...]`) for zero-overhead JAX subgame slicing."""

    type_mask: jax.Array  # uint32[6]
    maxhp: jax.Array  # int32[6]
    quantum_1_8: jax.Array  # int32[6]
    quantum_1_16: jax.Array  # int32[6]
    stored_stats: jax.Array  # int32[6, 5]
    max_pp: jax.Array  # int32[6, 4]
    types: jax.Array  # int32[6, 2]
    num_types: jax.Array  # int32[6]
    level: jax.Array  # int32[6]
    num_moves: jax.Array  # int32[6]
    item_stat_kind: jax.Array  # int32[6]
    item_boost_type: jax.Array  # int32[6]
    item_flags: jax.Array  # int32[6]
    init_item: jax.Array  # int32[6]

    move_base_power: jax.Array  # int32[6, 4]
    move_fixed_damage: jax.Array  # int32[6, 4]
    move_primary_vol: jax.Array  # int32[6, 4]
    move_sec_vol: jax.Array  # int32[6, 4]
    move_type: jax.Array  # int32[6, 4]
    move_flags: jax.Array  # int32[6, 4]
    move_priority: jax.Array  # int32[6, 4]
    move_base_acc_pct: jax.Array  # int32[6, 4]
    move_crit_ratio: jax.Array  # int32[6, 4]
    move_primary_status: jax.Array  # int32[6, 4]
    move_sec_chance_256: jax.Array  # int32[6, 4]
    move_sec_status: jax.Array  # int32[6, 4]
    move_sec_boost_stat: jax.Array  # int32[6, 4]
    move_sec_boost_delta: jax.Array  # int32[6, 4]
    move_boost_stats: jax.Array  # int32[6, 4, 3]
    move_boost_deltas: jax.Array  # int32[6, 4, 3]
    move_special_kind: jax.Array  # int32[6, 4]
    move_var_bp_kind: jax.Array  # int32[6, 4]
    move_multihit_mean_x2: jax.Array  # int32[6, 4]

    num_gender_draws: jax.Array  # int32 scalar (0..=6)
    legal_mask_60: jax.Array  # bool[60]


def build_shared_dex_table(dex_path: str | Path = DEFAULT_DEX_PATH) -> SharedDexTable:
    """Build the immutable `SharedDexTable` from `gen2stadium2.json`."""
    dex = load_dex(dex_path)
    return SharedDexTable(
        type_chart=jnp.asarray(dex.type_chart, dtype=jnp.int32),
        normal_type=jnp.int32(dex.normal_type),
        fighting_type=jnp.int32(dex.fighting_type),
        flying_type=jnp.int32(dex.flying_type),
        ground_type=jnp.int32(dex.ground_type),
        ghost_type=jnp.int32(dex.ghost_type),
        grass_type=jnp.int32(dex.grass_type),
        fire_type=jnp.int32(dex.fire_type),
        electric_type=jnp.int32(dex.electric_type),
        ice_type=jnp.int32(dex.ice_type),
        sandstorm_immune_mask=jnp.uint32(dex.sandstorm_immune_mask),
        poison_immune_mask=jnp.uint32(dex.poison_immune_mask),
    )


def compile_roster6(
    team_sets: list[dict[str, Any]],
    dex_path: str | Path = DEFAULT_DEX_PATH,
) -> RosterTable6:
    """Compile a 6-Pokemon roster (or 3..6 Pokemon roster) into a `RosterTable6`.

    Matches the stat, Hidden Power DV, item, move, and gender-draw logic of
    `dex_loader.build_compact_battle`.
    """
    if len(team_sets) < 3 or len(team_sets) > 6:
        raise ValueError(f"Expected 3..6 Pokemon sets in roster, got {len(team_sets)}")

    dex = load_dex(dex_path)
    n_actual = len(team_sets)

    # Count gender draws across the actual roster (matching `Battle::from_fixture`)
    gender_draws = 0
    for s in team_sets:
        sp_key = toid(s["species"])
        sp_data = dex.raw["species"][sp_key]
        g = s.get("gender") or sp_data.get("gender")
        if g not in ("M", "F", "N"):
            gender_draws += 1

    type_mask_arr = np.zeros((6,), dtype=np.uint32)
    maxhp_arr = np.ones((6,), dtype=np.int32)
    q1_8_arr = np.ones((6,), dtype=np.int32)
    q1_16_arr = np.ones((6,), dtype=np.int32)
    stored_stats_arr = np.zeros((6, 5), dtype=np.int32)
    max_pp_arr = np.zeros((6, 4), dtype=np.int32)
    types_arr = np.zeros((6, 2), dtype=np.int32)
    num_types_arr = np.ones((6,), dtype=np.int32)
    level_arr = np.full((6,), 200, dtype=np.int32)  # Padded slots get level 200 -> illegal in 60-mask
    num_moves_arr = np.zeros((6,), dtype=np.int32)
    item_stat_kind_arr = np.zeros((6,), dtype=np.int32)
    item_boost_type_arr = np.full((6,), 0xFF, dtype=np.int32)
    item_flags_arr = np.zeros((6,), dtype=np.int32)
    init_item_arr = np.zeros((6,), dtype=np.int32)

    move_base_power = np.zeros((6, 4), dtype=np.int32)
    move_fixed_damage = np.zeros((6, 4), dtype=np.int32)
    move_primary_vol = np.zeros((6, 4), dtype=np.int32)
    move_sec_vol = np.zeros((6, 4), dtype=np.int32)
    move_type = np.zeros((6, 4), dtype=np.int32)
    move_flags = np.zeros((6, 4), dtype=np.int32)
    move_priority = np.zeros((6, 4), dtype=np.int32)
    move_base_acc_pct = np.zeros((6, 4), dtype=np.int32)
    move_crit_ratio = np.zeros((6, 4), dtype=np.int32)
    move_primary_status = np.zeros((6, 4), dtype=np.int32)
    move_sec_chance_256 = np.zeros((6, 4), dtype=np.int32)
    move_sec_status = np.zeros((6, 4), dtype=np.int32)
    move_sec_boost_stat = np.full((6, 4), 0xFF, dtype=np.int32)
    move_sec_boost_delta = np.zeros((6, 4), dtype=np.int32)
    move_boost_stats = np.full((6, 4, 3), 0xFF, dtype=np.int32)
    move_boost_deltas = np.zeros((6, 4, 3), dtype=np.int32)
    move_special_kind = np.zeros((6, 4), dtype=np.int32)
    move_var_bp_kind = np.zeros((6, 4), dtype=np.int32)
    move_multihit_mean_x2 = np.full((6, 4), 2, dtype=np.int32)

    for c_idx in range(n_actual):
        pset = team_sets[c_idx]
        sp_key = toid(pset["species"])
        sp_data = dex.raw["species"][sp_key]
        level = int(pset.get("level", 50))
        happiness = int(pset.get("happiness", 255))

        ivs = [31] * 6
        evs = [0] * 6
        raw_ivs = pset.get("ivs") or {}
        raw_evs = pset.get("evs") or {}
        for i, sk in enumerate(_STAT_KEYS):
            if sk in raw_ivs:
                ivs[i] = int(raw_ivs[sk])
            if sk in raw_evs:
                evs[i] = int(raw_evs[sk])
            ivs[i] = max(0, min(31, ivs[i])) & 30
            evs[i] = max(0, min(255, evs[i]))

        bs = sp_data["baseStats"]
        bases = [bs["hp"], bs["atk"], bs["def"], bs["spa"], bs["spd"], bs["spe"]]
        stats = [0] * 6
        for i in range(6):
            ev_term = _tr(evs[i] / 4.0)
            if i == 0:
                stats[i] = int(
                    _tr(_tr(2.0 * bases[i] + ivs[i] + ev_term + 100.0) * level / 100.0 + 10.0)
                )
            else:
                stats[i] = int(
                    _tr(_tr(2.0 * bases[i] + ivs[i] + ev_term) * level / 100.0 + 5.0)
                )

        maxhp = max(1, stats[0])
        maxhp_arr[c_idx] = maxhp
        q1_8_arr[c_idx] = max(1, maxhp // 8)
        q1_16_arr[c_idx] = max(1, maxhp // 16)
        stored_stats_arr[c_idx] = stats[1:6]
        level_arr[c_idx] = level

        sp_types = sp_data["types"]
        t0 = dex.type_index[sp_types[0]]
        t1 = dex.type_index[sp_types[1]] if len(sp_types) > 1 else t0
        types_arr[c_idx, 0] = t0
        types_arr[c_idx, 1] = t1
        num_types_arr[c_idx] = len(sp_types)
        tmask = (1 << (t0 & 31)) | (1 << (t1 & 31))
        type_mask_arr[c_idx] = np.uint32(tmask)

        atk_dv = ivs[1] // 2
        def_dv = ivs[2] // 2
        spe_dv = ivs[5] // 2
        spc_dv = ivs[3] // 2
        hp_type_name = _HP_TYPES[4 * (atk_dv % 4) + (def_dv % 4)]
        hp_type_id = dex.type_index[hp_type_name]
        hp_power = (
            5 * ((spc_dv >> 3) + 2 * (spe_dv >> 3) + 4 * (def_dv >> 3) + 8 * (atk_dv >> 3))
            + (spc_dv % 4)
        ) // 2 + 31

        ikey = toid(pset.get("item", ""))
        if ikey and ikey in dex.item_map:
            iid = dex.item_map[ikey]
            init_item_arr[c_idx] = (iid & 0x7F) + 1
            if ikey in _COMPACT_TYPE_BOOST_ITEMS:
                item_boost_type_arr[c_idx] = dex.type_index[_COMPACT_TYPE_BOOST_ITEMS[ikey]]
            if ikey == "thickclub" and sp_key in ("cubone", "marowak"):
                item_stat_kind_arr[c_idx] = ItemStatKind.THICK_CLUB
            elif ikey == "lightball" and sp_key == "pikachu":
                item_stat_kind_arr[c_idx] = ItemStatKind.LIGHT_BALL
            elif ikey == "metalpowder" and sp_key == "ditto":
                item_stat_kind_arr[c_idx] = ItemStatKind.METAL_POWDER

            iflags = 0
            if ikey == "quickclaw":
                iflags |= ItemFlag.QUICK_CLAW
            if ikey == "leftovers":
                iflags |= ItemFlag.LEFTOVERS
            if ikey == "brightpowder":
                iflags |= ItemFlag.BRIGHTPOWDER
            if ikey == "mintberry":
                iflags |= ItemFlag.MINTBERRY
            if ikey == "miracleberry":
                iflags |= ItemFlag.MIRACLEBERRY
            if ikey == "goldberry":
                iflags |= ItemFlag.GOLDBERRY
            if ikey in ("berry", "berryjuice"):
                iflags |= ItemFlag.BERRY
            if ikey == "scopelens":
                iflags |= ItemFlag.SCOPE_LENS
            item_flags_arr[c_idx] = iflags

        raw_moves = pset.get("moves") or []
        nm = min(4, len(raw_moves))
        num_moves_arr[c_idx] = nm
        return_bp = (happiness * 10) // 25
        frustration_bp = ((255 - happiness) * 10) // 25

        for m_i in range(nm):
            mkey = toid(raw_moves[m_i])
            if mkey.startswith("hiddenpower") and mkey != "hiddenpower":
                mkey = "hiddenpower"
            cm = _compile_move_spec(dex, mkey)
            pp_ups = 0 if cm["no_pp_boosts"] else 3
            pp_val = (cm["pp"] * (5 + pp_ups)) // 5
            if cm["pp"] == 40:
                pp_val -= pp_ups
            pp_val = max(0, min(64, pp_val))
            max_pp_arr[c_idx, m_i] = pp_val

            bp = cm["base_power"]
            mt = cm["move_type"]
            mfl = cm["move_flags"]
            if cm["var_bp_kind"] == 1:
                bp = return_bp
            elif cm["var_bp_kind"] == 2:
                bp = frustration_bp
            elif cm["var_bp_kind"] == 3:
                mt = hp_type_id
                bp = hp_power
                is_phys = (dex.special_type_mask & (1 << (hp_type_id & 31))) == 0
                if is_phys:
                    mfl |= MoveFlag.IS_PHYSICAL
                else:
                    mfl &= ~MoveFlag.IS_PHYSICAL
                if hp_type_id == dex.water_type:
                    mfl |= MoveFlag.IS_WATER
                else:
                    mfl &= ~MoveFlag.IS_WATER
                if hp_type_id == dex.fire_type:
                    mfl |= MoveFlag.IS_FIRE
                else:
                    mfl &= ~MoveFlag.IS_FIRE

            move_base_power[c_idx, m_i] = bp
            move_fixed_damage[c_idx, m_i] = cm["fixed_damage"]
            move_primary_vol[c_idx, m_i] = cm["primary_vol"]
            move_sec_vol[c_idx, m_i] = cm["sec_vol"]
            move_type[c_idx, m_i] = mt
            move_flags[c_idx, m_i] = mfl
            move_priority[c_idx, m_i] = cm["priority"]
            move_base_acc_pct[c_idx, m_i] = cm["base_acc_pct"]
            move_crit_ratio[c_idx, m_i] = cm["crit_ratio"]
            move_primary_status[c_idx, m_i] = cm["primary_status"]
            move_sec_chance_256[c_idx, m_i] = cm["sec_chance_256"]
            move_sec_status[c_idx, m_i] = cm["sec_status"]
            move_sec_boost_stat[c_idx, m_i] = cm["sec_boost_stat"]
            move_sec_boost_delta[c_idx, m_i] = cm["sec_boost_delta"]
            move_boost_stats[c_idx, m_i] = cm["boost_stats"]
            move_boost_deltas[c_idx, m_i] = cm["boost_deltas"]
            move_special_kind[c_idx, m_i] = cm["special_kind"]
            move_var_bp_kind[c_idx, m_i] = cm["var_bp_kind"]
            move_multihit_mean_x2[c_idx, m_i] = cm["multihit_mean_x2"]

    legal_60 = legal_preview_mask_60(level_arr[:n_actual])

    return RosterTable6(
        type_mask=jnp.asarray(type_mask_arr, dtype=jnp.uint32),
        maxhp=jnp.asarray(maxhp_arr, dtype=jnp.int32),
        quantum_1_8=jnp.asarray(q1_8_arr, dtype=jnp.int32),
        quantum_1_16=jnp.asarray(q1_16_arr, dtype=jnp.int32),
        stored_stats=jnp.asarray(stored_stats_arr, dtype=jnp.int32),
        max_pp=jnp.asarray(max_pp_arr, dtype=jnp.int32),
        types=jnp.asarray(types_arr, dtype=jnp.int32),
        num_types=jnp.asarray(num_types_arr, dtype=jnp.int32),
        level=jnp.asarray(level_arr, dtype=jnp.int32),
        num_moves=jnp.asarray(num_moves_arr, dtype=jnp.int32),
        item_stat_kind=jnp.asarray(item_stat_kind_arr, dtype=jnp.int32),
        item_boost_type=jnp.asarray(item_boost_type_arr, dtype=jnp.int32),
        item_flags=jnp.asarray(item_flags_arr, dtype=jnp.int32),
        init_item=jnp.asarray(init_item_arr, dtype=jnp.int32),
        move_base_power=jnp.asarray(move_base_power, dtype=jnp.int32),
        move_fixed_damage=jnp.asarray(move_fixed_damage, dtype=jnp.int32),
        move_primary_vol=jnp.asarray(move_primary_vol, dtype=jnp.int32),
        move_sec_vol=jnp.asarray(move_sec_vol, dtype=jnp.int32),
        move_type=jnp.asarray(move_type, dtype=jnp.int32),
        move_flags=jnp.asarray(move_flags, dtype=jnp.int32),
        move_priority=jnp.asarray(move_priority, dtype=jnp.int32),
        move_base_acc_pct=jnp.asarray(move_base_acc_pct, dtype=jnp.int32),
        move_crit_ratio=jnp.asarray(move_crit_ratio, dtype=jnp.int32),
        move_primary_status=jnp.asarray(move_primary_status, dtype=jnp.int32),
        move_sec_chance_256=jnp.asarray(move_sec_chance_256, dtype=jnp.int32),
        move_sec_status=jnp.asarray(move_sec_status, dtype=jnp.int32),
        move_sec_boost_stat=jnp.asarray(move_sec_boost_stat, dtype=jnp.int32),
        move_sec_boost_delta=jnp.asarray(move_sec_boost_delta, dtype=jnp.int32),
        move_boost_stats=jnp.asarray(move_boost_stats, dtype=jnp.int32),
        move_boost_deltas=jnp.asarray(move_boost_deltas, dtype=jnp.int32),
        move_special_kind=jnp.asarray(move_special_kind, dtype=jnp.int32),
        move_var_bp_kind=jnp.asarray(move_var_bp_kind, dtype=jnp.int32),
        move_multihit_mean_x2=jnp.asarray(move_multihit_mean_x2, dtype=jnp.int32),
        num_gender_draws=jnp.int32(gender_draws),
        legal_mask_60=jnp.asarray(legal_60, dtype=jnp.bool_),
    )


def _sel6_row(arr: jax.Array, idx: jax.Array) -> jax.Array:
    """Select row `idx` (`0..=5`) from `arr` (`[6, ...]`) without dynamic gather on TPU."""
    if jax.default_backend() != "tpu":
        return arr[idx]
    b01 = jnp.where(idx == 0, arr[0], arr[1])
    b23 = jnp.where(idx == 2, arr[2], arr[3])
    b45 = jnp.where(idx == 4, arr[4], arr[5])
    b03 = jnp.where(idx < 2, b01, b23)
    return jnp.where(idx < 4, b03, b45)


def _slice_3_from_roster6(arr: jax.Array, pick3: jax.Array) -> jax.Array:
    """Slice 3 rows `pick3` (`int32[3]`) from `arr` (`[6, ...]`) into `[3, ...]`."""
    if jax.default_backend() != "tpu":
        return arr[pick3]
    r0 = _sel6_row(arr, pick3[0])
    r1 = _sel6_row(arr, pick3[1])
    r2 = _sel6_row(arr, pick3[2])
    return jnp.stack([r0, r1, r2], axis=0)


def _concat_pick_pair(
    arr_a: jax.Array,
    arr_b: jax.Array,
    pick_a: jax.Array,
    pick_b: jax.Array,
) -> jax.Array:
    """Slice `pick_a` (`[3]`) from `arr_a` and `pick_b` (`[3]`) from `arr_b` into `[6, ...]`."""
    sa = _slice_3_from_roster6(arr_a, pick_a)
    sb = _slice_3_from_roster6(arr_b, pick_b)
    return jnp.concatenate([sa, sb], axis=0)


def assemble_3v3_from_rosters_jax(
    roster_a: RosterTable6,
    roster_b: RosterTable6,
    pick_a: jax.Array,
    pick_b: jax.Array,
    battle_seed: jax.Array,
    shared_dex: SharedDexTable,
) -> tuple[MatchTable, CompactBattle]:
    """Assemble `(MatchTable, CompactBattle)` for 0-based 3-mon picks `(pick_a, pick_b)` in JAX.

    Matches `dex_loader.build_compact_battle(seed, p1_sets, p2_sets, pick_a, pick_b)` bit-for-bit,
    including gender PRNG draws, TeamPreview `speed_sort` tie-breakers, lead switch tie-breaker,
    and Turn-0 Quick Claw roll.
    """
    rng_seed = battle_seed.astype(jnp.uint32)
    total_gender_draws = roster_a.num_gender_draws + roster_b.num_gender_draws
    for d_i in range(12):
        next_seed, _ = prng_random(rng_seed, jnp.int32(2))
        rng_seed = jnp.where(jnp.int32(d_i) < total_gender_draws, next_seed, rng_seed)

    type_mask = _concat_pick_pair(roster_a.type_mask, roster_b.type_mask, pick_a, pick_b)
    maxhp = _concat_pick_pair(roster_a.maxhp, roster_b.maxhp, pick_a, pick_b)
    quantum_1_8 = _concat_pick_pair(roster_a.quantum_1_8, roster_b.quantum_1_8, pick_a, pick_b)
    quantum_1_16 = _concat_pick_pair(roster_a.quantum_1_16, roster_b.quantum_1_16, pick_a, pick_b)
    stored_stats = _concat_pick_pair(roster_a.stored_stats, roster_b.stored_stats, pick_a, pick_b)
    max_pp = _concat_pick_pair(roster_a.max_pp, roster_b.max_pp, pick_a, pick_b)
    types = _concat_pick_pair(roster_a.types, roster_b.types, pick_a, pick_b)
    num_types = _concat_pick_pair(roster_a.num_types, roster_b.num_types, pick_a, pick_b)
    level = _concat_pick_pair(roster_a.level, roster_b.level, pick_a, pick_b)
    num_moves = _concat_pick_pair(roster_a.num_moves, roster_b.num_moves, pick_a, pick_b)
    item_stat_kind = _concat_pick_pair(
        roster_a.item_stat_kind, roster_b.item_stat_kind, pick_a, pick_b
    )
    item_boost_type = _concat_pick_pair(
        roster_a.item_boost_type, roster_b.item_boost_type, pick_a, pick_b
    )
    item_flags = _concat_pick_pair(roster_a.item_flags, roster_b.item_flags, pick_a, pick_b)
    init_item = _concat_pick_pair(roster_a.init_item, roster_b.init_item, pick_a, pick_b)

    move_base_power = _concat_pick_pair(
        roster_a.move_base_power, roster_b.move_base_power, pick_a, pick_b
    )
    move_fixed_damage = _concat_pick_pair(
        roster_a.move_fixed_damage, roster_b.move_fixed_damage, pick_a, pick_b
    )
    move_primary_vol = _concat_pick_pair(
        roster_a.move_primary_vol, roster_b.move_primary_vol, pick_a, pick_b
    )
    move_sec_vol = _concat_pick_pair(
        roster_a.move_sec_vol, roster_b.move_sec_vol, pick_a, pick_b
    )
    move_type = _concat_pick_pair(roster_a.move_type, roster_b.move_type, pick_a, pick_b)
    move_flags = _concat_pick_pair(roster_a.move_flags, roster_b.move_flags, pick_a, pick_b)
    move_priority = _concat_pick_pair(
        roster_a.move_priority, roster_b.move_priority, pick_a, pick_b
    )
    move_base_acc_pct = _concat_pick_pair(
        roster_a.move_base_acc_pct, roster_b.move_base_acc_pct, pick_a, pick_b
    )
    move_crit_ratio = _concat_pick_pair(
        roster_a.move_crit_ratio, roster_b.move_crit_ratio, pick_a, pick_b
    )
    move_primary_status = _concat_pick_pair(
        roster_a.move_primary_status, roster_b.move_primary_status, pick_a, pick_b
    )
    move_sec_chance_256 = _concat_pick_pair(
        roster_a.move_sec_chance_256, roster_b.move_sec_chance_256, pick_a, pick_b
    )
    move_sec_status = _concat_pick_pair(
        roster_a.move_sec_status, roster_b.move_sec_status, pick_a, pick_b
    )
    move_sec_boost_stat = _concat_pick_pair(
        roster_a.move_sec_boost_stat, roster_b.move_sec_boost_stat, pick_a, pick_b
    )
    move_sec_boost_delta = _concat_pick_pair(
        roster_a.move_sec_boost_delta, roster_b.move_sec_boost_delta, pick_a, pick_b
    )
    move_boost_stats = _concat_pick_pair(
        roster_a.move_boost_stats, roster_b.move_boost_stats, pick_a, pick_b
    )
    move_boost_deltas = _concat_pick_pair(
        roster_a.move_boost_deltas, roster_b.move_boost_deltas, pick_a, pick_b
    )
    move_special_kind = _concat_pick_pair(
        roster_a.move_special_kind, roster_b.move_special_kind, pick_a, pick_b
    )
    move_var_bp_kind = _concat_pick_pair(
        roster_a.move_var_bp_kind, roster_b.move_var_bp_kind, pick_a, pick_b
    )
    move_multihit_mean_x2 = _concat_pick_pair(
        roster_a.move_multihit_mean_x2, roster_b.move_multihit_mean_x2, pick_a, pick_b
    )

    # Simulate TeamPreview `commit_choices` -> `speed_sort` tie-breakers for p_idx = 0, 1, 2
    for p_idx in range(3):
        next_seed, _ = prng_random(rng_seed, jnp.int32(2))
        tie = stored_stats[p_idx, 4] == stored_stats[3 + p_idx, 4]
        rng_seed = jnp.where(tie, next_seed, rng_seed)

    # Simulate `ActionKind::Start` -> `switch_in` -> `insert_choice_run_switch` tie-breaker on leads
    next_seed_lead, _ = prng_random(rng_seed, jnp.int32(2))
    lead_tie = stored_stats[0, 4] == stored_stats[3, 4]
    rng_seed = jnp.where(lead_tie, next_seed_lead, rng_seed)

    # End of turn 0 (`Start` -> `end_turn`): turn becomes 1 and Quick Claw roll is drawn
    rng_seed, qc_bool = prng_random_chance(rng_seed, jnp.int32(60), jnp.int32(256))
    quick_claw_roll = jnp.where(qc_bool, jnp.int32(1), jnp.int32(0))

    table = MatchTable(
        type_mask=type_mask,
        maxhp=maxhp,
        quantum_1_8=quantum_1_8,
        quantum_1_16=quantum_1_16,
        stored_stats=stored_stats,
        max_pp=max_pp,
        types=types,
        num_types=num_types,
        level=level,
        num_moves=num_moves,
        item_stat_kind=item_stat_kind,
        item_boost_type=item_boost_type,
        item_flags=item_flags,
        move_base_power=move_base_power,
        move_fixed_damage=move_fixed_damage,
        move_primary_vol=move_primary_vol,
        move_sec_vol=move_sec_vol,
        move_type=move_type,
        move_flags=move_flags,
        move_priority=move_priority,
        move_base_acc_pct=move_base_acc_pct,
        move_crit_ratio=move_crit_ratio,
        move_primary_status=move_primary_status,
        move_sec_chance_256=move_sec_chance_256,
        move_sec_status=move_sec_status,
        move_sec_boost_stat=move_sec_boost_stat,
        move_sec_boost_delta=move_sec_boost_delta,
        move_boost_stats=move_boost_stats,
        move_boost_deltas=move_boost_deltas,
        move_special_kind=move_special_kind,
        move_var_bp_kind=move_var_bp_kind,
        move_multihit_mean_x2=move_multihit_mean_x2,
        type_chart=shared_dex.type_chart,
        normal_type=shared_dex.normal_type,
        fighting_type=shared_dex.fighting_type,
        flying_type=shared_dex.flying_type,
        ground_type=shared_dex.ground_type,
        ghost_type=shared_dex.ghost_type,
        grass_type=shared_dex.grass_type,
        fire_type=shared_dex.fire_type,
        electric_type=shared_dex.electric_type,
        ice_type=shared_dex.ice_type,
        sandstorm_immune_mask=shared_dex.sandstorm_immune_mask,
        poison_immune_mask=shared_dex.poison_immune_mask,
    )

    hp_pool = jnp.stack([maxhp, jnp.zeros((6,), dtype=jnp.int32)], axis=-1)
    party_roster = jnp.stack([pick_a.astype(jnp.int32), pick_b.astype(jnp.int32)], axis=0)

    battle = CompactBattle(
        hp_pool=hp_pool,
        pp=max_pp,
        status_packed=jnp.zeros((6,), dtype=jnp.int32),
        item=init_item,
        vol_flags=jnp.zeros((6,), dtype=jnp.int32),
        timers=jnp.zeros((6, 4), dtype=jnp.int32),
        boosts=jnp.zeros((6, 7), dtype=jnp.int32),
        slot_refs=jnp.zeros((6,), dtype=jnp.int32),
        active_idx=jnp.array([0, 0], dtype=jnp.int32),
        party_roster=party_roster,
        alive_mask=jnp.array([7, 7], dtype=jnp.int32),
        reflect_turns=jnp.array([0, 0], dtype=jnp.int32),
        lightscreen_turns=jnp.array([0, 0], dtype=jnp.int32),
        safeguard_turns=jnp.array([0, 0], dtype=jnp.int32),
        side_flags=jnp.array([0, 0], dtype=jnp.int32),
        last_move_selfko=jnp.array([255, 255], dtype=jnp.int32),
        request=jnp.array([1, 1], dtype=jnp.int32),
        turn=jnp.int32(1),
        weather_kind=jnp.int32(CompactWeather.NONE),
        weather_turns=jnp.int32(0),
        quick_claw_roll=quick_claw_roll,
        outcome=jnp.int32(0),
        speed_order=jnp.int32(0),
        clause_flags=jnp.int32(5),
        hurt_damage_this_turn=jnp.array([0, 0], dtype=jnp.int32),
        hurt_cat_this_turn=jnp.array([0, 0], dtype=jnp.int32),
        prng_seed=rng_seed,
    )
    return table, battle


# =========================================================================
# 3. Multi-Table Batched `Playout::Heavy` Simulator on TPU
# =========================================================================


def _single_table_heavy_rollout(
    init_packed: PackedBattle,
    table: MatchTable,
    policy_seed: jax.Array,
    weights: EvalWeights,
    turns: int = 40,
    eps: float = 0.2,
    residual_bundle: ResidualBundle | None = None,
) -> HeavyRolloutResult:
    """Run up to `turns` game turns of `Playout::Heavy` on a single `(PackedBattle, MatchTable)`."""
    init_state = unpack_battle(init_packed)
    if turns <= 0:
        val = eval_compact_leaf(init_state, table, weights, residual_bundle=residual_bundle)
        return HeavyRolloutResult(
            final_packed=init_packed,
            leaf_values=val,
            outcomes=init_state.outcome,
            turns_played=init_state.turn,
        )

    rollout_res: RolloutResidualWeights | None = (
        residual_bundle.rollout if residual_bundle is not None else None
    )
    cutoff_turn = init_state.turn + jnp.int32(turns - 1)
    scan_steps = max(1, int(turns) * 2)

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, p_seed = carry
        active = (st.outcome == 0) & (st.turn <= cutoff_turn)
        p_seed2, a0 = sample_heavy_playout_action(
            st, table, 0, p_seed, eps=eps, rollout_residual=rollout_res
        )
        p_seed2, a1 = sample_heavy_playout_action(
            st, table, 1, p_seed2, eps=eps, rollout_residual=rollout_res
        )
        acts = jnp.stack([a0, a1], axis=0)
        cand_st = apply_actions(st, table, acts)
        next_st = jax.tree_util.tree_map(
            lambda nx, cur: jnp.where(active, nx, cur), cand_st, st
        )
        next_ps = jnp.where(active, p_seed2, p_seed)
        return (next_st, next_ps), None

    (final_st, _), _ = jax.lax.scan(_step, (init_state, policy_seed), None, length=scan_steps)
    val = eval_compact_leaf(final_st, table, weights, residual_bundle=residual_bundle)
    return HeavyRolloutResult(
        final_packed=pack_battle(final_st),
        leaf_values=val,
        outcomes=final_st.outcome,
        turns_played=final_st.turn,
    )


@lru_cache(maxsize=32)
def _get_jit_multi_table_rollout_fn(turns: int, eps: float):
    @jax.jit
    def _fn(
        batched_packed: PackedBattle,
        batched_tables: MatchTable,
        policy_seeds: jax.Array,
        weights: EvalWeights,
        residual_bundle: ResidualBundle | None,
    ) -> HeavyRolloutResult:
        if jax.default_backend() not in ("tpu", "gpu"):
            return jax.vmap(
                lambda pk, tb, ps: _single_table_heavy_rollout(
                    pk,
                    tb,
                    ps,
                    weights,
                    turns=turns,
                    eps=eps,
                    residual_bundle=residual_bundle,
                ),
                in_axes=(0, 0, 0),
            )(batched_packed, batched_tables, policy_seeds)

        init_st = unpack_battle(batched_packed)
        if turns <= 0:
            val = jax.vmap(
                lambda s, tb: eval_compact_leaf(s, tb, weights, residual_bundle=residual_bundle)
            )(init_st, batched_tables)
            return HeavyRolloutResult(
                final_packed=batched_packed,
                leaf_values=val,
                outcomes=init_st.outcome,
                turns_played=init_st.turn,
            )

        rollout_res: RolloutResidualWeights | None = (
            residual_bundle.rollout if residual_bundle is not None else None
        )
        cutoff_turn = init_st.turn + jnp.int32(turns - 1)
        max_steps = max(1, int(turns) * 2)

        def _step(carry: tuple[CompactBattle, jax.Array], _: None):
            st, ps = carry
            active = (st.outcome == 0) & (st.turn <= cutoff_turn)
            ps2, a0 = jax.vmap(
                lambda s, tb, p: sample_heavy_playout_action(
                    s, tb, 0, p, eps=eps, rollout_residual=rollout_res
                )
            )(st, batched_tables, ps)
            ps2, a1 = jax.vmap(
                lambda s, tb, p: sample_heavy_playout_action(
                    s, tb, 1, p, eps=eps, rollout_residual=rollout_res
                )
            )(st, batched_tables, ps2)
            acts = jnp.stack([a0, a1], axis=-1)
            cand_st = jax.vmap(apply_actions, in_axes=(0, 0, 0))(st, batched_tables, acts)
            next_st = jax.tree_util.tree_map(
                lambda nx, cur: jnp.where(
                    jnp.expand_dims(active, axis=tuple(range(1, cur.ndim))), nx, cur
                ),
                cand_st,
                st,
            )
            next_ps = jnp.where(active[:, None], ps2, ps)
            return (next_st, next_ps), None

        block_steps = 10
        if max_steps >= 20 and (max_steps % block_steps == 0):
            n_blocks = max_steps // block_steps

            def _cond(loop_carry: tuple[jax.Array, CompactBattle, jax.Array]) -> jax.Array:
                b_idx, st, _ = loop_carry
                return (b_idx < n_blocks) & jnp.any((st.outcome == 0) & (st.turn <= cutoff_turn))

            def _body(
                loop_carry: tuple[jax.Array, CompactBattle, jax.Array],
            ) -> tuple[jax.Array, CompactBattle, jax.Array]:
                b_idx, st, ps = loop_carry
                (next_st, next_ps), _ = jax.lax.scan(_step, (st, ps), None, length=block_steps)
                return (b_idx + jnp.int32(1), next_st, next_ps)

            _, final_st, _ = jax.lax.while_loop(
                _cond, _body, (jnp.int32(0), init_st, policy_seeds)
            )
        else:
            (final_st, _), _ = jax.lax.scan(_step, (init_st, policy_seeds), None, length=max_steps)

        val = jax.vmap(
            lambda s, tb: eval_compact_leaf(s, tb, weights, residual_bundle=residual_bundle)
        )(final_st, batched_tables)
        return HeavyRolloutResult(
            final_packed=pack_battle(final_st),
            leaf_values=val,
            outcomes=final_st.outcome,
            turns_played=final_st.turn,
        )

    return _fn


def simulate_multi_table_heavy_games(
    batched_packed: PackedBattle,
    batched_tables: MatchTable,
    policy_seeds: jax.Array,
    weights: EvalWeights | None = None,
    turns: int = 40,
    eps: float = 0.2,
    residual_bundle: ResidualBundle | None = None,
) -> HeavyRolloutResult:
    """Simulate `B` `Playout::Heavy` games in parallel where each trajectory has its own `MatchTable`.

    - On TPU: uses `scan`-outside-`vmap` (`in_axes=(0, 0, 0)`) to avoid per-lane loop divergence
      and keep all 32-bit state arrays in VMEM.
    - On CPU: uses `vmap`-outside-`scan` for minimal dispatch overhead.
    """
    if weights is None:
        weights = default_eval_weights()
    fn = _get_jit_multi_table_rollout_fn(int(turns), float(eps))
    return fn(batched_packed, batched_tables, policy_seeds, weights, residual_bundle)


@lru_cache(maxsize=32)
def _get_jit_subgame_chunk_fn(turns: int, eps: float):
    @jax.jit
    def _fn(
        r_a: RosterTable6,
        r_b: RosterTable6,
        pa_flat: jax.Array,
        pb_flat: jax.Array,
        b_seeds: jax.Array,
        p_seeds: jax.Array,
        swapped_mask: jax.Array,
        shared_dex: SharedDexTable,
        weights: EvalWeights,
        residual_bundle: ResidualBundle | None,
    ) -> jax.Array:
        def _asm_one(pa: jax.Array, pb: jax.Array, b_seed: jax.Array, sw: jax.Array):
            ra_eff = jax.tree_util.tree_map(lambda x, y: jnp.where(sw, y, x), r_a, r_b)
            rb_eff = jax.tree_util.tree_map(lambda x, y: jnp.where(sw, x, y), r_a, r_b)
            pa_eff = jnp.where(sw, pb, pa)
            pb_eff = jnp.where(sw, pa, pb)
            return assemble_3v3_from_rosters_jax(
                ra_eff, rb_eff, pa_eff, pb_eff, b_seed, shared_dex
            )

        tables, battles = jax.vmap(_asm_one)(pa_flat, pb_flat, b_seeds, swapped_mask)
        res = _get_jit_multi_table_rollout_fn(turns, eps)(
            pack_battle(battles),
            tables,
            p_seeds,
            weights,
            residual_bundle,
        )
        return jnp.where(swapped_mask, jnp.float32(1.0) - res.leaf_values, res.leaf_values)

    return _fn


def _bucket_fused_trajectory_capacity(total_trajectories: int, max_chunk: int = 8192) -> int:
    """Bucket total `(n_pairs * games_per_pair)` trajectories into a fixed TPU v6e shape."""
    if total_trajectories <= 64:
        return 64
    if total_trajectories <= 128:
        return 128
    if total_trajectories <= 256:
        return 256
    if total_trajectories <= 512:
        return 512
    if total_trajectories <= 4608:
        return 4608
    if total_trajectories <= 7200:
        return 7200
    return max(max_chunk, 7200)


def evaluate_3v3_picks_batch_tpu(
    roster_a: RosterTable6,
    roster_b: RosterTable6,
    picks_a: jax.Array,
    picks_b: jax.Array,
    shared_dex: SharedDexTable,
    games_per_pair: int = 2,
    turns: int = 40,
    eps: float = 0.2,
    base_seed_int: int = 0x5EED_6060_2026,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
    batch_chunk_size: int = 7200,
) -> jax.Array:
    """Evaluate `N` 3v3 subgame pairs `(picks_a[i], picks_b[i])` with P1/P2 side-swapped CRN
    in a single fused TPU XLA kernel across `(n_pairs, games_per_pair)`.

    For each cell `i in 0..N` and game `g in 0..games_per_pair`:
    - Even `g` (`g % 2 == 0`): Team A is Side 0 (P1), Team B is Side 1 (P2). Score = `v`.
    - Odd `g` (`g % 2 == 1`): Team B is Side 0 (P1), Team A is Side 1 (P2) on the same paired
      common random number seed `(g // 2)`. Score for Team A = `1.0 - v`.

    Matches the paired side-swap common random number protocol in
    `meta_eco.rs::core_nash_eval_weighted` lines 4088-4100.
    """
    if weights is None:
        weights = default_eval_weights()

    n_pairs = int(picks_a.shape[0])
    g_per_pair = int(games_per_pair)
    if n_pairs == 0 or g_per_pair <= 0:
        return jnp.zeros((n_pairs,), dtype=jnp.float32)

    hi_base = int((base_seed_int >> 32) & 0xFFFFFFFF)
    lo_base = int(base_seed_int & 0xFFFFFFFF)

    # Max pairs per fused call so `cur_n * g_per_pair <= max_traj_chunk`
    max_traj_chunk = _bucket_fused_trajectory_capacity(
        n_pairs * g_per_pair, max_chunk=max(batch_chunk_size, g_per_pair)
    )
    pair_chunk = max(1, max_traj_chunk // g_per_pair)
    traj_capacity = _bucket_fused_trajectory_capacity(pair_chunk * g_per_pair, max_chunk=max_traj_chunk)

    total_scores = np.zeros((n_pairs,), dtype=np.float32)
    chunk_fn = _get_jit_subgame_chunk_fn(int(turns), float(eps))

    g_indices = jnp.arange(g_per_pair, dtype=jnp.uint32)
    pair_seed_indices = g_indices // jnp.uint32(2)
    swapped_g = (g_indices % jnp.uint32(2)) == jnp.uint32(1)

    for start in range(0, n_pairs, pair_chunk):
        end = min(n_pairs, start + pair_chunk)
        cur_n = end - start
        pa_slice = picks_a[start:end]
        pb_slice = picks_b[start:end]

        # Tile `(g_per_pair, cur_n)` so game `g` is contiguous over `cur_n` pairs
        pa_grid = jnp.tile(pa_slice, (g_per_pair, 1))  # [g_per_pair * cur_n, 3]
        pb_grid = jnp.tile(pb_slice, (g_per_pair, 1))  # [g_per_pair * cur_n, 3]
        sw_grid = jnp.repeat(swapped_g, cur_n)  # [g_per_pair * cur_n]

        # Build CRN seeds matching `(pair_seed_idx, p_idx in 0..cur_n)`
        p_idx = jnp.arange(cur_n, dtype=jnp.uint32)[None, :]  # [1, cur_n]
        ps_idx = pair_seed_indices[:, None]  # [g_per_pair, 1]
        s_hi = jnp.uint32(hi_base) ^ (ps_idx * jnp.uint32(0x9E3779B9))
        s_lo = (
            jnp.uint32(lo_base)
            + ps_idx * jnp.uint32(0x85EBCA6B)
            + jnp.uint32(start * 131)
        )
        b_hi = (s_hi ^ (p_idx * jnp.uint32(0x9E3779B9))).reshape(-1)
        b_lo = (s_lo + (p_idx * jnp.uint32(0x85EBCA6B) + jnp.uint32(1))).reshape(-1)
        p_hi = ((s_hi + jnp.uint32(0xC2B2AE35)) ^ (p_idx * jnp.uint32(0x27D4EB2F))).reshape(-1)
        p_lo = (s_lo ^ (p_idx * jnp.uint32(0x165667B1) + jnp.uint32(7))).reshape(-1)

        b_seeds = jnp.stack([b_hi, b_lo], axis=-1)
        p_seeds = jnp.stack([p_hi, p_lo], axis=-1)

        cur_trajs = cur_n * g_per_pair
        if cur_trajs < traj_capacity:
            pad_t = traj_capacity - cur_trajs
            pa_grid = jnp.concatenate([pa_grid, jnp.repeat(pa_grid[:1], pad_t, axis=0)], axis=0)
            pb_grid = jnp.concatenate([pb_grid, jnp.repeat(pb_grid[:1], pad_t, axis=0)], axis=0)
            sw_grid = jnp.concatenate([sw_grid, jnp.zeros((pad_t,), dtype=jnp.bool_)], axis=0)
            b_seeds = jnp.concatenate([b_seeds, jnp.repeat(b_seeds[:1], pad_t, axis=0)], axis=0)
            p_seeds = jnp.concatenate([p_seeds, jnp.repeat(p_seeds[:1], pad_t, axis=0)], axis=0)

        vals_flat = chunk_fn(
            roster_a,
            roster_b,
            pa_grid,
            pb_grid,
            b_seeds,
            p_seeds,
            sw_grid,
            shared_dex,
            weights,
            residual_bundle,
        )
        vals_valid = vals_flat[:cur_trajs].reshape(g_per_pair, cur_n)
        mean_chunk = np.asarray(jnp.mean(vals_valid, axis=0), dtype=np.float32)
        total_scores[start:end] = mean_chunk

    return jnp.asarray(total_scores, dtype=jnp.float32)


# =========================================================================
# 4. Rectangular Regret Matching+ & Meta-Equilibrium Solvers
# =========================================================================


class RectEquilibriumResult(NamedTuple):
    """Equilibrium solution of an `R x C` zero-sum matrix game (`nc2000_bot::preview::PairSolution`)."""

    p_a: jax.Array  # float32[R]
    p_b: jax.Array  # float32[C]
    argmax_a: jax.Array  # int32 scalar
    argmax_b: jax.Array  # int32 scalar
    value: jax.Array  # float32 scalar
    guarantee_mixed_a: jax.Array  # float32 scalar
    guarantee_argmax_a: jax.Array  # float32 scalar
    guarantee_mixed_b: jax.Array  # float32 scalar
    guarantee_argmax_b: jax.Array  # float32 scalar
    ev_row: jax.Array  # float32[R] (M @ p_b)
    ev_col: jax.Array  # float32[C] (p_a^T @ M)


@partial(jax.jit, static_argnames=("sweeps",))
def solve_rect_rm_plus_jax(
    payoff_matrix: jax.Array,
    mask_rows: jax.Array | None = None,
    mask_cols: jax.Array | None = None,
    sweeps: int = 2000,
) -> RectEquilibriumResult:
    """Solve an `R x C` zero-sum matrix game via Regret Matching+ with linear averaging.

    Mirrors both `nc2000_bot::preview::solve_pair` and `meta_eco::solve_rectangular_minimax`:
    - `payoff_matrix`: `float32[R, C]` where entry `(i, j)` is Row Player's win rate in `[0, 1]`.
    - `mask_rows`: optional `bool[R]` legality mask for Row Player.
    - `mask_cols`: optional `bool[C]` legality mask for Column Player.
    """
    n_rows, n_cols = payoff_matrix.shape
    if n_rows == 0 or n_cols == 0:
        half = jnp.float32(0.5)
        return RectEquilibriumResult(
            p_a=jnp.zeros((n_rows,), dtype=jnp.float32),
            p_b=jnp.zeros((n_cols,), dtype=jnp.float32),
            argmax_a=jnp.int32(0),
            argmax_b=jnp.int32(0),
            value=half,
            guarantee_mixed_a=half,
            guarantee_argmax_a=half,
            guarantee_mixed_b=half,
            guarantee_argmax_b=half,
            ev_row=jnp.zeros((n_rows,), dtype=jnp.float32),
            ev_col=jnp.zeros((n_cols,), dtype=jnp.float32),
        )
    if mask_rows is None:
        m0 = jnp.ones((n_rows,), dtype=jnp.float32)
    else:
        m0 = mask_rows.astype(jnp.float32)
    if mask_cols is None:
        m1 = jnp.ones((n_cols,), dtype=jnp.float32)
    else:
        m1 = mask_cols.astype(jnp.float32)

    u0 = m0 / jnp.maximum(jnp.sum(m0), jnp.float32(1.0))
    u1 = m1 / jnp.maximum(jnp.sum(m1), jnp.float32(1.0))

    init_carry = (
        jnp.zeros((n_rows,), dtype=jnp.float32),
        jnp.zeros((n_cols,), dtype=jnp.float32),
        jnp.zeros((n_rows,), dtype=jnp.float32),
        jnp.zeros((n_cols,), dtype=jnp.float32),
    )

    def _sweep(t_idx: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
        r0, r1, sum0, sum1 = carry
        s0_sum = jnp.sum(r0)
        s1_sum = jnp.sum(r1)
        p0 = jnp.where(s0_sum > jnp.float32(1e-12), r0 / jnp.maximum(s0_sum, jnp.float32(1e-12)), u0)
        p1 = jnp.where(s1_sum > jnp.float32(1e-12), r1 / jnp.maximum(s1_sum, jnp.float32(1e-12)), u1)

        ev0 = jnp.matmul(payoff_matrix, p1)
        v0 = jnp.sum(p0 * ev0)
        r0_next = jnp.maximum(jnp.float32(0.0), r0 + (ev0 - v0)) * m0

        ev1_loss = jnp.matmul(p0, payoff_matrix)
        v1_loss = jnp.sum(p1 * ev1_loss)
        r1_next = jnp.maximum(jnp.float32(0.0), r1 + (v1_loss - ev1_loss)) * m1

        w = (t_idx + 1).astype(jnp.float32)
        return (r0_next, r1_next, sum0 + w * p0, sum1 + w * p1)

    _, _, sum0, sum1 = jax.lax.fori_loop(0, sweeps, _sweep, init_carry)
    p_a = jnp.where(
        jnp.sum(sum0) > jnp.float32(1e-12),
        sum0 / jnp.maximum(jnp.sum(sum0), jnp.float32(1e-12)),
        u0,
    )
    p_b = jnp.where(
        jnp.sum(sum1) > jnp.float32(1e-12),
        sum1 / jnp.maximum(jnp.sum(sum1), jnp.float32(1e-12)),
        u1,
    )

    ev_row = jnp.matmul(payoff_matrix, p_b)
    ev_col = jnp.matmul(p_a, payoff_matrix)
    game_val = jnp.sum(p_a * ev_row)

    masked_pa = jnp.where(m0 > jnp.float32(0.0), p_a, jnp.float32(-1.0))
    masked_pb = jnp.where(m1 > jnp.float32(0.0), p_b, jnp.float32(-1.0))
    argmax_a = jnp.argmax(masked_pa).astype(jnp.int32)
    argmax_b = jnp.argmax(masked_pb).astype(jnp.int32)

    # Worst-case security guarantees against legal opponent deviations
    col_valid = m1 > jnp.float32(0.0)
    row_valid = m0 > jnp.float32(0.0)
    guarantee_mixed_a = jnp.min(jnp.where(col_valid, ev_col, jnp.float32(1e9)))
    guarantee_argmax_a = jnp.min(
        jnp.where(col_valid, payoff_matrix[argmax_a], jnp.float32(1e9))
    )
    guarantee_mixed_b = jnp.min(
        jnp.where(row_valid, jnp.float32(1.0) - ev_row, jnp.float32(1e9))
    )
    guarantee_argmax_b = jnp.min(
        jnp.where(row_valid, jnp.float32(1.0) - payoff_matrix[:, argmax_b], jnp.float32(1e9))
    )

    return RectEquilibriumResult(
        p_a=p_a,
        p_b=p_b,
        argmax_a=argmax_a,
        argmax_b=argmax_b,
        value=game_val,
        guarantee_mixed_a=guarantee_mixed_a,
        guarantee_argmax_a=guarantee_argmax_a,
        guarantee_mixed_b=guarantee_mixed_b,
        guarantee_argmax_b=guarantee_argmax_b,
        ev_row=ev_row,
        ev_col=ev_col,
    )


# =========================================================================
# 5. Support Selection & Leave-One-Out Anti-Monopoly Guarantee
# =========================================================================


def select_top_k_orthogonal(
    legal_indices: list[int] | np.ndarray,
    eq_probs: list[float] | np.ndarray,
    br_scores: list[float] | np.ndarray,
    k: int = 3,
    actions_1based: tuple[tuple[int, int, int], ...] = PREVIEW_ACTIONS_1BASED,
) -> list[int]:
    """Select up to `k` orthogonal 3-mon cores + Leave-One-Out Anti-Monopoly backups.

    Exact port of `meta_eco.rs::select_top_k_orthogonal` (lines 3935-3992):
    - Ranks legal actions descending by `(eq[i] >= 0.01, eq[i] + 0.05 * br[i])`.
    - Pass 1: Prefers actions with distinct 6-bit 3-mon core masks first, then fills up to `k`.
    - Pass 2 (Leave-One-Out Anti-Monopoly Guarantee): For each roster slot `s in 1..=6`, if `s`
      appears in 100% of `sup`, appends the highest-ranked legal action excluding `s` so that
      leave-one-out submatrices `rem_rows` are never empty.
    """
    legal_list = [int(x) for x in legal_indices]
    eq_arr = np.asarray(eq_probs, dtype=np.float64)
    br_arr = np.asarray(br_scores, dtype=np.float64)
    n = len(legal_list)
    if n == 0 or k <= 0:
        return []

    order = list(range(n))
    order.sort(
        key=lambda i: (
            1 if eq_arr[i] >= 0.01 else 0,
            float(eq_arr[i] + 0.05 * br_arr[i]),
            -i,
        ),
        reverse=True,
    )

    def _core_mask(act_idx: int) -> int:
        t = actions_1based[act_idx]
        return (1 << (t[0] - 1)) | (1 << (t[1] - 1)) | (1 << (t[2] - 1))

    sup: list[int] = []
    seen_cores: set[int] = set()

    # Pass 1: Distinct 3-mon cores first
    for i in order:
        if len(sup) >= k:
            break
        act_idx = legal_list[i]
        cm = _core_mask(act_idx)
        if cm not in seen_cores:
            seen_cores.add(cm)
            sup.append(act_idx)

    # Fill remaining slots up to k
    for i in order:
        if len(sup) >= k:
            break
        act_idx = legal_list[i]
        if act_idx not in sup:
            sup.append(act_idx)

    # Pass 2: Leave-One-Out Anti-Monopoly Guarantee (`\Delta V_{-s}`)
    for s in range(1, 7):
        if sup and all(s in actions_1based[act_idx] for act_idx in sup):
            for i in order:
                act_idx = legal_list[i]
                if s not in actions_1based[act_idx] and act_idx not in sup:
                    sup.append(act_idx)
                    break

    return sup


# =========================================================================
# 6. Two-Stage 6v6 Core-Nash Evaluator (`core_nash_6v6_pair_tpu`)
# =========================================================================


class CoreNashPairResult(NamedTuple):
    """Complete Stage-1 + Stage-2 6v6 Core-Nash evaluation & per-slot telemetry."""

    value: float  # Stage-2 Nash game value for Team A in [0, 1]
    legal_a: list[int]
    legal_b: list[int]
    screen_matrix: np.ndarray  # float32[len(legal_a), len(legal_b)]
    sup_a: list[int]
    sup_b: list[int]
    refine_matrix: np.ndarray  # float32[len(sup_a), len(sup_b)]
    x_star: np.ndarray  # float32[len(sup_a)]
    y_star: np.ndarray  # float32[len(sup_b)]
    slot_picks_a: np.ndarray  # float32[6]
    slot_wins_a: np.ndarray  # float32[6]
    slot_marginals_a: np.ndarray  # float32[6] (Leave-One-Out ΔV_{-s} >= 0)
    slot_picks_b: np.ndarray  # float32[6]
    slot_wins_b: np.ndarray  # float32[6]
    slot_marginals_b: np.ndarray  # float32[6]
    effective_cores_a: float
    effective_cores_b: float
    core_stats_a: dict[int, tuple[float, float]]
    core_stats_b: dict[int, tuple[float, float]]
    sol_full_60: RectEquilibriumResult
    elapsed_secs: float


@partial(jax.jit, static_argnames=("sweeps",))
def _solve_batched_row_masked_rm_plus_values(
    payoff_matrix: jax.Array,
    mask_rows_batch: jax.Array,
    mask_cols: jax.Array,
    sweeps: int = 500,
) -> jax.Array:
    """Solve `B` row-masked RM+ games on `payoff_matrix` in parallel via `jax.vmap`."""
    return jax.vmap(
        lambda m_r: solve_rect_rm_plus_jax(
            payoff_matrix, mask_rows=m_r, mask_cols=mask_cols, sweeps=sweeps
        ).value
    )(mask_rows_batch)


def _compute_side_slot_and_core_telemetry(
    sup_indices: list[int],
    probs: np.ndarray,
    ev_vec: np.ndarray,
    payoff_submatrix_row_oriented: np.ndarray,
    sweeps: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[float, float]], float, float]:
    """Compute `(slot_picks, slot_wins, slot_marginals, core_stats, effective_cores, v_star)`
    for a player whose actions correspond to the rows of `payoff_submatrix_row_oriented`.
    """
    v_star = float(np.sum(probs * ev_vec))
    slot_picks = np.zeros((6,), dtype=np.float32)
    slot_wins = np.zeros((6,), dtype=np.float32)
    slot_marginals = np.zeros((6,), dtype=np.float32)
    core_stats: dict[int, tuple[float, float]] = {}

    for ia, act_idx in enumerate(sup_indices):
        p_ia = float(probs[ia])
        ev_ia = float(ev_vec[ia])
        t = PREVIEW_ACTIONS_1BASED[act_idx]
        for s1 in t:
            slot_picks[s1 - 1] += p_ia
            slot_wins[s1 - 1] += p_ia * ev_ia
        cm = (1 << (t[0] - 1)) | (1 << (t[1] - 1)) | (1 << (t[2] - 1))
        prev_p, prev_w = core_stats.get(cm, (0.0, 0.0))
        core_stats[cm] = (prev_p + p_ia, prev_w + p_ia * ev_ia)

    # Effective number of cores: exp(H(pi))
    h_ent = 0.0
    for cp, _ in core_stats.values():
        if cp > 1e-9:
            h_ent -= cp * float(np.log(cp))
    effective_cores = float(np.exp(h_ent)) if core_stats else 1.0

    # Leave-One-Out marginal contribution ΔV_{-s} for all 6 slots simultaneously via batched RM+
    if not sup_indices:
        return slot_picks, slot_wins, slot_marginals, core_stats, effective_cores, v_star

    rem_masks = np.array(
        [
            [(s + 1) not in PREVIEW_ACTIONS_1BASED[act_idx] for act_idx in sup_indices]
            for s in range(6)
        ],
        dtype=bool,
    )
    any_rem = np.any(rem_masks, axis=1)
    all_rem = np.all(rem_masks, axis=1)
    safe_masks = np.where(any_rem[:, None], rem_masks, True)

    v_minus_batch = np.asarray(
        _solve_batched_row_masked_rm_plus_values(
            jnp.asarray(payoff_submatrix_row_oriented, dtype=jnp.float32),
            jnp.asarray(safe_masks, dtype=jnp.bool_),
            jnp.ones((payoff_submatrix_row_oriented.shape[1],), dtype=jnp.bool_),
            sweeps=sweeps,
        ),
        dtype=np.float32,
    )
    for s in range(6):
        if all_rem[s]:
            v_minus_s = v_star
        elif not any_rem[s]:
            v_minus_s = 0.0
        else:
            v_minus_s = float(v_minus_batch[s])
        slot_marginals[s] = max(0.0, v_star - v_minus_s)

    return slot_picks, slot_wins, slot_marginals, core_stats, effective_cores, v_star


def core_nash_6v6_pair_tpu(
    roster_a: RosterTable6,
    roster_b: RosterTable6,
    shared_dex: SharedDexTable,
    screen_games: int = 2,
    support_k: int = 3,
    refine_games: int = 6,
    screen_turns: int = 40,
    refine_turns: int = 60,
    eps: float = 0.2,
    seed: int = 20260405,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
    inherited_subgame: tuple[list[int], list[int]] | None = None,
    mutated_slot_a: int | None = None,
    rm_sweeps: int = 500,
    batch_chunk_size: int = 7200,
) -> CoreNashPairResult:
    """Two-Stage 6v6 Core-Nash & Leave-One-Out Evaluator on TPU.

    Mirrors `meta_eco.rs::core_nash_eval_weighted` and `preview.rs::solve_pair`:
    1. Identifies legal preview actions `legal_a`, `legal_b` (`sum(level) <= 155`).
    2. Stage 1:
       - If `inherited_subgame` is provided (Monotonic Subgame Inheritance for a single-slot
         mutation `mutated_slot_a`), unions the parent's `sup_a` with legal cores containing
         `mutated_slot_a` and skips full 60x60 screening.
       - Otherwise, evaluates all `|legal_a| x |legal_b|` cells on TPU with `screen_games`
         paired side-swap games, solves the Stage-1 matrix game via RM+, and selects
         orthogonal support `(sup_a, sup_b)` with the Leave-One-Out Anti-Monopoly guarantee.
    3. Stage 2:
       - Evaluates the `|sup_a| x |sup_b|` submatrix with `refine_games` paired side-swap games
         (reusing Stage-1 when `refine_games <= screen_games` and `refine_turns == screen_turns`).
       - Solves the Stage-2 Nash equilibrium `(x_star, y_star, v_star)` and computes exact
         leave-one-out slot marginals `ΔV_{-s}` for all 6 slots of both teams.
    """
    t0 = time.perf_counter()
    if weights is None:
        weights = default_eval_weights()

    legal_a = [int(i) for i in np.flatnonzero(np.asarray(roster_a.legal_mask_60))]
    legal_b = [int(i) for i in np.flatnonzero(np.asarray(roster_b.legal_mask_60))]
    if not legal_a:
        legal_a = [0]
    if not legal_b:
        legal_b = [0]

    na = len(legal_a)
    nb = len(legal_b)

    # Check Monotonic Subgame Inheritance path
    if inherited_subgame is not None or screen_games <= 0 or support_k <= 0:
        screen_mat = np.full((na, nb), 0.5, dtype=np.float32)
        if inherited_subgame is not None:
            parent_sup_a, parent_sup_b = inherited_subgame
            sup_a = [idx for idx in parent_sup_a if idx in legal_a]
            if mutated_slot_a is not None:
                s1 = mutated_slot_a + 1
                mut_acts = [idx for idx in legal_a if s1 in PREVIEW_ACTIONS_1BASED[idx]]
                # Pick up to 2 orthogonal cores containing the mutated slot
                seen_c: set[int] = set()
                added = 0
                for idx in mut_acts:
                    cm = int(PREVIEW_CORE_MASKS_60_NP[idx])
                    if cm not in seen_c and idx not in sup_a:
                        seen_c.add(cm)
                        sup_a.append(idx)
                        added += 1
                        if added >= 2:
                            break
            if not sup_a:
                sup_a = legal_a[: max(1, support_k)]
            sup_b = [idx for idx in parent_sup_b if idx in legal_b]
            if not sup_b:
                sup_b = legal_b[: max(1, support_k)]
        else:
            sup_a = list(legal_a)
            sup_b = list(legal_b)
    else:
        # Build all na * nb 3v3 pick pairs
        grid_a, grid_b = np.meshgrid(
            np.asarray(legal_a, dtype=np.int32),
            np.asarray(legal_b, dtype=np.int32),
            indexing="ij",
        )
        flat_a = grid_a.reshape(-1)
        flat_b = grid_b.reshape(-1)
        picks_a_s1 = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[flat_a], dtype=jnp.int32)
        picks_b_s1 = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[flat_b], dtype=jnp.int32)

        scores_s1 = evaluate_3v3_picks_batch_tpu(
            roster_a,
            roster_b,
            picks_a_s1,
            picks_b_s1,
            shared_dex,
            games_per_pair=screen_games,
            turns=screen_turns,
            eps=eps,
            base_seed_int=seed ^ 0xA5A5_1111,
            weights=weights,
            residual_bundle=residual_bundle,
            batch_chunk_size=batch_chunk_size,
        )
        screen_mat = np.asarray(scores_s1, dtype=np.float32).reshape(na, nb)

        s1_sol = solve_rect_rm_plus_jax(
            jnp.asarray(screen_mat, dtype=jnp.float32),
            sweeps=rm_sweeps,
        )
        e0 = np.asarray(s1_sol.p_a, dtype=np.float32)
        e1 = np.asarray(s1_sol.p_b, dtype=np.float32)
        u_a = np.asarray(s1_sol.ev_row, dtype=np.float32)
        u_b = np.asarray(1.0 - s1_sol.ev_col, dtype=np.float32)

        k_a = min(support_k, na)
        k_b = min(support_k, nb)
        sup_a = select_top_k_orthogonal(legal_a, e0, u_a, k_a)
        sup_b = select_top_k_orthogonal(legal_b, e1, u_b, k_b)

    ka = len(sup_a)
    kb = len(sup_b)

    # Stage 2: Refinement on sup_a x sup_b
    eff_refine_games = max(1, refine_games if refine_games > 0 else screen_games)
    if (
        inherited_subgame is None
        and screen_games > 0
        and refine_games <= screen_games
        and refine_turns == screen_turns
    ):
        idx_map_a = {act_idx: i for i, act_idx in enumerate(legal_a)}
        idx_map_b = {act_idx: j for j, act_idx in enumerate(legal_b)}
        refine_mat = np.zeros((ka, kb), dtype=np.float32)
        for ia, act_a in enumerate(sup_a):
            for ib, act_b in enumerate(sup_b):
                refine_mat[ia, ib] = screen_mat[idx_map_a[act_a], idx_map_b[act_b]]
    else:
        grid_ra, grid_rb = np.meshgrid(
            np.asarray(sup_a, dtype=np.int32),
            np.asarray(sup_b, dtype=np.int32),
            indexing="ij",
        )
        flat_ra = grid_ra.reshape(-1)
        flat_rb = grid_rb.reshape(-1)
        picks_a_s2 = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[flat_ra], dtype=jnp.int32)
        picks_b_s2 = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[flat_rb], dtype=jnp.int32)

        scores_s2 = evaluate_3v3_picks_batch_tpu(
            roster_a,
            roster_b,
            picks_a_s2,
            picks_b_s2,
            shared_dex,
            games_per_pair=eff_refine_games,
            turns=refine_turns,
            eps=eps,
            base_seed_int=seed ^ 0x5A5A_2222,
            weights=weights,
            residual_bundle=residual_bundle,
            batch_chunk_size=batch_chunk_size,
        )
        refine_mat = np.asarray(scores_s2, dtype=np.float32).reshape(ka, kb)

    s2_sol = solve_rect_rm_plus_jax(
        jnp.asarray(refine_mat, dtype=jnp.float32),
        sweeps=rm_sweeps,
    )
    x_star = np.asarray(s2_sol.p_a, dtype=np.float32)
    y_star = np.asarray(s2_sol.p_b, dtype=np.float32)
    ev_row_a = np.asarray(s2_sol.ev_row, dtype=np.float32)
    ev_row_b = np.asarray(1.0 - s2_sol.ev_col, dtype=np.float32)

    (
        slot_picks_a,
        slot_wins_a,
        slot_marginals_a,
        core_stats_a,
        eff_cores_a,
        v_star,
    ) = _compute_side_slot_and_core_telemetry(
        sup_a, x_star, ev_row_a, refine_mat, sweeps=rm_sweeps
    )
    (
        slot_picks_b,
        slot_wins_b,
        slot_marginals_b,
        core_stats_b,
        eff_cores_b,
        _,
    ) = _compute_side_slot_and_core_telemetry(
        sup_b, y_star, ev_row_b, (1.0 - refine_mat.T), sweeps=rm_sweeps
    )

    # Lift Stage-2 solution into full 60-action space (`PairSolution` schema)
    full_pa = np.zeros((60,), dtype=np.float32)
    full_pb = np.zeros((60,), dtype=np.float32)
    for ia, act_a in enumerate(sup_a):
        full_pa[act_a] = x_star[ia]
    for ib, act_b in enumerate(sup_b):
        full_pb[act_b] = y_star[ib]

    argmax_a_full = int(sup_a[int(s2_sol.argmax_a)])
    argmax_b_full = int(sup_b[int(s2_sol.argmax_b)])

    # Compute security guarantees against the full screened/refined matrix (`preview::solve_pair`)
    full_mat_60 = np.full((60, 60), 0.5, dtype=np.float32)
    for ia, act_a in enumerate(legal_a):
        for ib, act_b in enumerate(legal_b):
            full_mat_60[act_a, act_b] = screen_mat[ia, ib]
    for ia, act_a in enumerate(sup_a):
        for ib, act_b in enumerate(sup_b):
            full_mat_60[act_a, act_b] = refine_mat[ia, ib]

    ev_col_full = full_pa @ full_mat_60
    ev_row_full = full_mat_60 @ full_pb
    g_mixed_a = float(np.min(ev_col_full[legal_b]))
    g_argmax_a = float(np.min(full_mat_60[argmax_a_full, legal_b]))
    g_mixed_b = float(np.min(1.0 - ev_row_full[legal_a]))
    g_argmax_b = float(np.min(1.0 - full_mat_60[legal_a, argmax_b_full]))

    sol_full_60 = RectEquilibriumResult(
        p_a=jnp.asarray(full_pa, dtype=jnp.float32),
        p_b=jnp.asarray(full_pb, dtype=jnp.float32),
        argmax_a=jnp.int32(argmax_a_full),
        argmax_b=jnp.int32(argmax_b_full),
        value=jnp.float32(v_star),
        guarantee_mixed_a=jnp.float32(g_mixed_a),
        guarantee_argmax_a=jnp.float32(g_argmax_a),
        guarantee_mixed_b=jnp.float32(g_mixed_b),
        guarantee_argmax_b=jnp.float32(g_argmax_b),
        ev_row=jnp.asarray(ev_row_full, dtype=jnp.float32),
        ev_col=jnp.asarray(ev_col_full, dtype=jnp.float32),
    )

    elapsed = time.perf_counter() - t0
    return CoreNashPairResult(
        value=v_star,
        legal_a=legal_a,
        legal_b=legal_b,
        screen_matrix=screen_mat,
        sup_a=sup_a,
        sup_b=sup_b,
        refine_matrix=refine_mat,
        x_star=x_star,
        y_star=y_star,
        slot_picks_a=slot_picks_a,
        slot_wins_a=slot_wins_a,
        slot_marginals_a=slot_marginals_a,
        slot_picks_b=slot_picks_b,
        slot_wins_b=slot_wins_b,
        slot_marginals_b=slot_marginals_b,
        effective_cores_a=eff_cores_a,
        effective_cores_b=eff_cores_b,
        core_stats_a=core_stats_a,
        core_stats_b=core_stats_b,
        sol_full_60=sol_full_60,
        elapsed_secs=elapsed,
    )


def export_pair_table_dict(
    result: CoreNashPairResult,
    team_a: str,
    team_b: str,
    screen_games: int = 2,
    support_k: int = 3,
    refine_games: int = 6,
    screen_turns: int = 40,
    refine_turns: int = 60,
    seed: int = 20260405,
) -> dict[str, Any]:
    """Serialize `CoreNashPairResult` into `nc2000_bot::preview::PairTable` JSON schema (`SPACE_VERSION = 1`)."""
    sol = result.sol_full_60
    return {
        "team_a": team_a,
        "team_b": team_b,
        "actions": [[a, b, c] for (a, b, c) in PREVIEW_ACTIONS_1BASED],
        "space_version": SPACE_VERSION,
        "screen": {
            "rows": [int(x) for x in result.legal_a],
            "cols": [int(x) for x in result.legal_b],
            "n": int(screen_games),
            "v": [
                [round(float(val), 6) for val in row]
                for row in result.screen_matrix.tolist()
            ],
        },
        "support": [
            [int(x) for x in result.sup_a],
            [int(x) for x in result.sup_b],
        ],
        "refine": (
            {
                "rows": [int(x) for x in result.sup_a],
                "cols": [int(x) for x in result.sup_b],
                "n": int(refine_games),
                "v": [
                    [round(float(val), 6) for val in row]
                    for row in result.refine_matrix.tolist()
                ],
            }
            if refine_games > 0
            else None
        ),
        "sol": {
            "p_a": [round(float(x), 6) for x in np.asarray(sol.p_a).tolist()],
            "p_b": [round(float(x), 6) for x in np.asarray(sol.p_b).tolist()],
            "argmax_a": int(sol.argmax_a),
            "argmax_b": int(sol.argmax_b),
            "value": round(float(sol.value), 6),
            "guarantee_mixed_a": round(float(sol.guarantee_mixed_a), 6),
            "guarantee_argmax_a": round(float(sol.guarantee_argmax_a), 6),
            "guarantee_mixed_b": round(float(sol.guarantee_mixed_b), 6),
            "guarantee_argmax_b": round(float(sol.guarantee_argmax_b), 6),
        },
        "cfg": {
            "screen_games": int(screen_games),
            "screen_agent": f"tpu_heavy:turns={screen_turns}",
            "support_k": int(support_k),
            "refine_games": int(refine_games),
            "refine_agent": f"tpu_heavy:turns={refine_turns}",
            "max_turns": int(max(screen_turns, refine_turns)),
            "seed": int(seed),
        },
        "secs": round(float(result.elapsed_secs), 4),
    }


def export_pair_table_json(
    path: str | Path,
    result: CoreNashPairResult,
    team_a: str,
    team_b: str,
    **kwargs: Any,
) -> None:
    """Write a `PairTable` (`SPACE_VERSION = 1`) JSON file readable by `nc2000_bot::preview::PairTable::load_json`."""
    payload = export_pair_table_dict(result, team_a=team_a, team_b=team_b, **kwargs)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================================
# 7. Population WAR Regression & Multi-Team Kodoku Gauntlet Evaluator
# =========================================================================


def recompute_population_wars_np(
    slot_marginals: np.ndarray,
    slot_levels: np.ndarray,
    games: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, float, float]:
    r"""Compute Level-Conditioned Replacement Level $\hat{\mu}_{pop}(L)$ and `slot_wars` via OLS.

    Exact port of `meta_eco.rs::recompute_population_wars` (lines 4501-4557):
    - Fits $\hat{\mu}_{pop}(L) = \alpha + \beta (L - 50)$ across all evaluated `(level, marginal / games)`
      pairs in the population without any hand-coded level prior.
    - Returns `(slot_wars, alpha, beta)` where `slot_wars` has the same shape as `slot_marginals`.
    """
    marg = np.nan_to_num(
        np.asarray(slot_marginals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    lvls = np.nan_to_num(
        np.asarray(slot_levels, dtype=np.float64), nan=50.0, posinf=50.0, neginf=50.0
    )
    if np.ndim(games) == 0:
        g_val = float(np.nan_to_num(float(games), nan=1.0, posinf=1.0, neginf=1.0))
        g_arr = np.full(marg.shape, max(g_val, 1e-9), dtype=np.float64)
    else:
        g_arr = np.maximum(
            np.nan_to_num(np.asarray(games, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0),
            1e-9,
        )
        if g_arr.ndim == 1 and marg.ndim == 2:
            g_arr = g_arr[:, None]

    norm_m = np.nan_to_num(marg / g_arr, nan=0.0, posinf=0.0, neginf=0.0)
    flat_x = (lvls - 50.0).reshape(-1)
    flat_y = norm_m.reshape(-1)
    n = float(flat_x.size)
    if n < 2.0:
        return np.zeros_like(marg, dtype=np.float32), 0.0, 0.0

    mean_x = float(np.mean(flat_x))
    mean_y = float(np.mean(flat_y))
    var_x = float(np.sum((flat_x - mean_x) ** 2))
    cov_xy = float(np.sum((flat_x - mean_x) * (flat_y - mean_y)))

    beta = (cov_xy / var_x) if var_x > 1e-9 else 0.0
    alpha = mean_y - beta * mean_x

    expected_mu = alpha + beta * (lvls - 50.0)
    wars = norm_m - expected_mu
    return wars.astype(np.float32), float(alpha), float(beta)


class KodokuGauntletSummary(NamedTuple):
    """Summary of evaluating candidate pool `P` (`N_P` teams) against gauntlet `Q` (`N_Q` teams)."""

    p_ids: list[str]
    q_ids: list[str]
    payoff_matrix: np.ndarray  # float32[N_P, N_Q]
    weighted_fitness: np.ndarray  # float32[N_P]
    slot_picks: np.ndarray  # float32[N_P, 6]
    slot_wins: np.ndarray  # float32[N_P, 6]
    slot_marginals: np.ndarray  # float32[N_P, 6]
    slot_wars: np.ndarray  # float32[N_P, 6]
    effective_cores: np.ndarray  # float32[N_P]
    rect_equilibrium: RectEquilibriumResult


def _bucket_multi_pair_trajectory_capacity(total_trajectories: int) -> int:
    """Bucket total Stage-2 trajectories across multiple team pairs into fixed TPU v6e shapes."""
    if total_trajectories <= 64:
        return 64
    if total_trajectories <= 128:
        return 128
    if total_trajectories <= 256:
        return 256
    if total_trajectories <= 512:
        return 512
    if total_trajectories <= 1024:
        return 1024
    if total_trajectories <= 2048:
        return 2048
    if total_trajectories <= 4608:
        return 4608
    if total_trajectories <= 8192:
        return 8192
    return ((total_trajectories + 4095) // 4096) * 4096


@lru_cache(maxsize=32)
def _get_jit_multi_roster_subgame_fn(turns: int, eps: float):
    @jax.jit
    def _fn(
        batched_r_p1: RosterTable6,
        batched_r_p2: RosterTable6,
        pa_p1: jax.Array,
        pb_p2: jax.Array,
        b_seeds: jax.Array,
        p_seeds: jax.Array,
        swapped_mask: jax.Array,
        shared_dex: SharedDexTable,
        weights: EvalWeights,
        residual_bundle: ResidualBundle | None,
    ) -> jax.Array:
        tables, battles = jax.vmap(
            assemble_3v3_from_rosters_jax,
            in_axes=(0, 0, 0, 0, 0, None),
        )(batched_r_p1, batched_r_p2, pa_p1, pb_p2, b_seeds, shared_dex)
        res = _get_jit_multi_table_rollout_fn(turns, eps)(
            pack_battle(battles),
            tables,
            p_seeds,
            weights,
            residual_bundle,
        )
        return jnp.where(swapped_mask, jnp.float32(1.0) - res.leaf_values, res.leaf_values)

    return _fn


@partial(jax.jit, static_argnames=("sweeps",))
def _solve_batched_rect_rm_plus_values_and_policies(
    payoff_batch: jax.Array,
    mask_rows_batch: jax.Array,
    mask_cols_batch: jax.Array,
    sweeps: int = 500,
) -> RectEquilibriumResult:
    """Solve `B` rectangular RM+ games in parallel via `jax.vmap`."""
    return jax.vmap(
        lambda m, mr, mc: solve_rect_rm_plus_jax(
            m, mask_rows=mr, mask_cols=mc, sweeps=sweeps
        )
    )(payoff_batch, mask_rows_batch, mask_cols_batch)


def _execute_multi_roster_trajectory_batches(
    stacked_rosters: RosterTable6,
    r1_all: np.ndarray,
    r2_all: np.ndarray,
    pa_all: np.ndarray,
    pb_all: np.ndarray,
    bs_all: np.ndarray,
    ps_all: np.ndarray,
    sw_all: np.ndarray,
    turns: int,
    eps: float,
    shared_dex: SharedDexTable,
    weights: EvalWeights,
    residual_bundle: ResidualBundle | None,
    max_traj_chunk: int = 8192,
) -> np.ndarray:
    """Execute `N` trajectories across arbitrary roster pairs in fixed `<= max_traj_chunk` TPU batches."""
    tot_trajs = int(r1_all.shape[0])
    if tot_trajs == 0:
        return np.zeros((0,), dtype=np.float32)

    out_vals = np.zeros((tot_trajs,), dtype=np.float32)
    multi_fn = _get_jit_multi_roster_subgame_fn(int(turns), float(eps))

    for st in range(0, tot_trajs, max_traj_chunk):
        ed = min(tot_trajs, st + max_traj_chunk)
        cur_n = ed - st
        cap = _bucket_multi_pair_trajectory_capacity(cur_n)

        r1_c = r1_all[st:ed]
        r2_c = r2_all[st:ed]
        pa_c = pa_all[st:ed]
        pb_c = pb_all[st:ed]
        bs_c = bs_all[st:ed]
        ps_c = ps_all[st:ed]
        sw_c = sw_all[st:ed]

        if cur_n < cap:
            pad_n = cap - cur_n
            r1_c = np.concatenate([r1_c, np.zeros((pad_n,), dtype=np.int32)], axis=0)
            r2_c = np.concatenate([r2_c, np.zeros((pad_n,), dtype=np.int32)], axis=0)
            pa_c = np.concatenate([pa_c, np.repeat(pa_c[:1], pad_n, axis=0)], axis=0)
            pb_c = np.concatenate([pb_c, np.repeat(pb_c[:1], pad_n, axis=0)], axis=0)
            bs_c = np.concatenate([bs_c, np.repeat(bs_c[:1], pad_n, axis=0)], axis=0)
            ps_c = np.concatenate([ps_c, np.repeat(ps_c[:1], pad_n, axis=0)], axis=0)
            sw_c = np.concatenate([sw_c, np.zeros((pad_n,), dtype=bool)], axis=0)

        r1_j = jnp.asarray(r1_c, dtype=jnp.int32)
        r2_j = jnp.asarray(r2_c, dtype=jnp.int32)
        batched_r_p1 = jax.tree_util.tree_map(lambda x: x[r1_j], stacked_rosters)
        batched_r_p2 = jax.tree_util.tree_map(lambda x: x[r2_j], stacked_rosters)

        vals_chunk = np.asarray(
            multi_fn(
                batched_r_p1,
                batched_r_p2,
                jnp.asarray(pa_c, dtype=jnp.int32),
                jnp.asarray(pb_c, dtype=jnp.int32),
                jnp.asarray(bs_c, dtype=jnp.uint32),
                jnp.asarray(ps_c, dtype=jnp.uint32),
                jnp.asarray(sw_c, dtype=jnp.bool_),
                shared_dex,
                weights,
                residual_bundle,
            ),
            dtype=np.float32,
        )
        out_vals[st:ed] = vals_chunk[:cur_n]

    return out_vals


def evaluate_kodoku_gauntlet_tpu(
    teams_p: list[dict[str, Any]],
    teams_q: list[dict[str, Any]] | None = None,
    q_weights: list[float] | np.ndarray | None = None,
    screen_games: int = 2,
    support_k: int = 3,
    refine_games: int = 4,
    screen_turns: int = 40,
    refine_turns: int = 60,
    eps: float = 0.2,
    seed: int = 20260405,
    dex_path: str | Path = DEFAULT_DEX_PATH,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
    rm_sweeps: int = 500,
    screen_stride: int = 1,
    compute_slot_telemetry: bool = True,
    max_traj_chunk: int = 8192,
) -> KodokuGauntletSummary:
    """Evaluate a pool of 6-mon teams `teams_p` against opponent pool `teams_q` using Two-Stage Core-Nash.

    Each entry in `teams_p` / `teams_q` is a dict `{"id": str, "sets": list[dict]}` (matching
    `nc2000-engine` pool JSONs). If `teams_q` is `None`, evaluates symmetric round-robin `P x P`.
    Fuses Stage-2 refinement rollouts across all `(i, j)` matchups into fixed `<= max_traj_chunk`
    TPU XLA batches so small Stage-2 support matrices (`3 x 3 = 9` cells) saturate all TPU VPU lanes.
    """
    if weights is None:
        weights = default_eval_weights()
    shared_dex = build_shared_dex_table(dex_path)
    symmetric = teams_q is None
    if teams_q is None:
        teams_q = teams_p

    n_p = len(teams_p)
    n_q = len(teams_q)
    p_ids = [str(t.get("id", f"p_{i}")) for i, t in enumerate(teams_p)]
    q_ids = [str(t.get("id", f"q_{j}")) for j, t in enumerate(teams_q)]
    p_id_to_idx = {pid: i for i, pid in enumerate(p_ids)}
    q_id_to_idx = {qid: j for j, qid in enumerate(q_ids)}

    rosters_p = [compile_roster6(t["sets"], dex_path=dex_path) for t in teams_p]
    rosters_q = (
        rosters_p
        if symmetric
        else [compile_roster6(t["sets"], dex_path=dex_path) for t in teams_q]
    )

    if q_weights is None:
        w_q = np.full((n_q,), 1.0 / max(1, n_q), dtype=np.float32)
    else:
        w_q = np.asarray(q_weights, dtype=np.float32)
        w_sum = float(np.sum(w_q))
        w_q = (w_q / w_sum) if w_sum > 1e-9 else np.full((n_q,), 1.0 / max(1, n_q), dtype=np.float32)

    payoff_mat = np.full((n_p, n_q), 0.5, dtype=np.float32)
    slot_picks = np.zeros((n_p, 6), dtype=np.float32)
    slot_wins = np.zeros((n_p, 6), dtype=np.float32)
    slot_marginals = np.zeros((n_p, 6), dtype=np.float32)
    eff_cores = np.zeros((n_p,), dtype=np.float32)
    core_acc_per_p: list[dict[int, float]] = [{} for _ in range(n_p)]

    # Collect active `(i, j)` pairs to evaluate, exploiting `P ∩ Q` skew-symmetry
    active_pairs: list[tuple[int, int, int]] = []
    mirror_fill_pairs: list[tuple[int, int, int, int]] = []
    for i in range(n_p):
        pid = p_ids[i]
        for j in range(n_q):
            qid = q_ids[j]
            if pid == qid:
                payoff_mat[i, j] = 0.5
                continue
            if symmetric and j < i:
                continue
            if not symmetric and (qid in p_id_to_idx) and (pid in q_id_to_idx):
                i_mir = p_id_to_idx[qid]
                j_mir = q_id_to_idx[pid]
                if i_mir < i:
                    mirror_fill_pairs.append((i, j, i_mir, j_mir))
                    continue
            active_pairs.append((i, j, seed + i * 10007 + j * 97))

    if active_pairs:
        stride = max(1, int(screen_stride))
        legals_p = [
            ([int(idx) for idx in np.flatnonzero(np.asarray(r.legal_mask_60))][::stride]) or [0]
            for r in rosters_p
        ]
        legals_q = (
            legals_p
            if symmetric
            else [
                ([int(idx) for idx in np.flatnonzero(np.asarray(r.legal_mask_60))][::stride]) or [0]
                for r in rosters_q
            ]
        )

        all_rosters = rosters_p if symmetric else (rosters_p + rosters_q)
        q_offset = 0 if symmetric else n_p
        stacked_rosters = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *all_rosters
        )

        # Stage 1: Screen each `(i, j)` pair and extract orthogonal supports `(sup_a, sup_b)`
        stage1_records: list[tuple[int, int, int, np.ndarray, list[int], list[int]]] = []
        if screen_games <= 0 or support_k <= 0:
            for i, j, pair_seed in active_pairs:
                la = legals_p[i]
                lb = legals_q[j]
                screen_mat = np.full((len(la), len(lb)), 0.5, dtype=np.float32)
                stage1_records.append((i, j, pair_seed, screen_mat, list(la), list(lb)))
        elif stride == 1 and len(active_pairs) <= 32:
            for i, j, pair_seed in active_pairs:
                la = legals_p[i]
                lb = legals_q[j]
                na, nb = len(la), len(lb)
                grid_a, grid_b = np.meshgrid(
                    np.asarray(la, dtype=np.int32),
                    np.asarray(lb, dtype=np.int32),
                    indexing="ij",
                )
                pa_s1 = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[grid_a.reshape(-1)], dtype=jnp.int32)
                pb_s1 = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[grid_b.reshape(-1)], dtype=jnp.int32)
                scores_s1 = evaluate_3v3_picks_batch_tpu(
                    rosters_p[i],
                    rosters_q[j],
                    pa_s1,
                    pb_s1,
                    shared_dex,
                    games_per_pair=screen_games,
                    turns=screen_turns,
                    eps=eps,
                    base_seed_int=pair_seed ^ 0xA5A5_1111,
                    weights=weights,
                    residual_bundle=residual_bundle,
                )
                screen_mat = np.asarray(scores_s1, dtype=np.float32).reshape(na, nb)
                s1_sol = solve_rect_rm_plus_jax(
                    jnp.asarray(screen_mat, dtype=jnp.float32), sweeps=rm_sweeps
                )
                e0 = np.asarray(s1_sol.p_a, dtype=np.float32)
                e1 = np.asarray(s1_sol.p_b, dtype=np.float32)
                u_a = np.asarray(s1_sol.ev_row, dtype=np.float32)
                u_b = np.asarray(1.0 - s1_sol.ev_col, dtype=np.float32)
                sup_a = select_top_k_orthogonal(la, e0, u_a, min(support_k, na))
                sup_b = select_top_k_orthogonal(lb, e1, u_b, min(support_k, nb))
                stage1_records.append((i, j, pair_seed, screen_mat, sup_a, sup_b))
        else:
            # Fused multi-pair Stage-1 screening + batched vmap RM+ solver
            eff_s1_games = max(1, screen_games + (screen_games % 2 if screen_games > 1 else 0))
            g_indices_s1 = np.arange(eff_s1_games, dtype=np.uint32)
            ps_indices_s1 = g_indices_s1 // np.uint32(2)
            sw_g_s1 = (g_indices_s1 % np.uint32(2)) == np.uint32(1)
            max_act_dim = max(
                max(len(x) for x in legals_p),
                max(len(x) for x in legals_q),
            )
            pairs_per_s1_batch = max(1, min(256, max_traj_chunk // max(1, max_act_dim * max_act_dim * eff_s1_games)) * 16)
            for b_st in range(0, len(active_pairs), pairs_per_s1_batch):
                b_ed = min(len(active_pairs), b_st + pairs_per_s1_batch)
                batch_pairs = active_pairs[b_st:b_ed]
                r1_l: list[np.ndarray] = []
                r2_l: list[np.ndarray] = []
                pa_l: list[np.ndarray] = []
                pb_l: list[np.ndarray] = []
                bs_l: list[np.ndarray] = []
                ps_l: list[np.ndarray] = []
                sw_l: list[np.ndarray] = []
                s1_slices: list[tuple[int, int, int, int]] = []
                cur_s1 = 0
                for i, j, pair_seed in batch_pairs:
                    la = legals_p[i]
                    lb = legals_q[j]
                    na, nb = len(la), len(lb)
                    n_cells = na * nb
                    grid_a, grid_b = np.meshgrid(
                        np.asarray(la, dtype=np.int32),
                        np.asarray(lb, dtype=np.int32),
                        indexing="ij",
                    )
                    pa_cells = PREVIEW_ACTIONS_0BASED_NP[grid_a.reshape(-1)]
                    pb_cells = PREVIEW_ACTIONS_0BASED_NP[grid_b.reshape(-1)]
                    pa_rep = np.tile(pa_cells, (eff_s1_games, 1))
                    pb_rep = np.tile(pb_cells, (eff_s1_games, 1))
                    sw_rep = np.repeat(sw_g_s1, n_cells)
                    pa_p1 = np.where(sw_rep[:, None], pb_rep, pa_rep)
                    pb_p2 = np.where(sw_rep[:, None], pa_rep, pb_rep)
                    idx_a = i
                    idx_b = q_offset + j
                    r1_idx = np.where(sw_rep, idx_b, idx_a).astype(np.int32)
                    r2_idx = np.where(sw_rep, idx_a, idx_b).astype(np.int32)

                    base_s1 = pair_seed ^ 0xA5A5_1111
                    hi_b = np.uint32((base_s1 >> 32) & 0xFFFFFFFF)
                    lo_b = np.uint32(base_s1 & 0xFFFFFFFF)
                    p_idx_arr = np.arange(n_cells, dtype=np.uint32)[None, :]
                    ps_idx_arr = ps_indices_s1[:, None]
                    s_hi = hi_b ^ (ps_idx_arr * np.uint32(0x9E3779B9))
                    s_lo = lo_b + ps_idx_arr * np.uint32(0x85EBCA6B)
                    b_hi = (s_hi ^ (p_idx_arr * np.uint32(0x9E3779B9))).reshape(-1)
                    b_lo = (s_lo + (p_idx_arr * np.uint32(0x85EBCA6B) + np.uint32(1))).reshape(-1)
                    p_hi = ((s_hi + np.uint32(0xC2B2AE35)) ^ (p_idx_arr * np.uint32(0x27D4EB2F))).reshape(-1)
                    p_lo = (s_lo ^ (p_idx_arr * np.uint32(0x165667B1) + np.uint32(7))).reshape(-1)

                    n_traj_p = n_cells * eff_s1_games
                    r1_l.append(r1_idx)
                    r2_l.append(r2_idx)
                    pa_l.append(pa_p1)
                    pb_l.append(pb_p2)
                    bs_l.append(np.stack([b_hi, b_lo], axis=-1))
                    ps_l.append(np.stack([p_hi, p_lo], axis=-1))
                    sw_l.append(sw_rep)
                    s1_slices.append((cur_s1, cur_s1 + n_traj_p, na, nb))
                    cur_s1 += n_traj_p

                vals_s1_flat = _execute_multi_roster_trajectory_batches(
                    stacked_rosters,
                    np.concatenate(r1_l, axis=0),
                    np.concatenate(r2_l, axis=0),
                    np.concatenate(pa_l, axis=0),
                    np.concatenate(pb_l, axis=0),
                    np.concatenate(bs_l, axis=0),
                    np.concatenate(ps_l, axis=0),
                    np.concatenate(sw_l, axis=0),
                    turns=screen_turns,
                    eps=eps,
                    shared_dex=shared_dex,
                    weights=weights,
                    residual_bundle=residual_bundle,
                    max_traj_chunk=max_traj_chunk,
                )

                b_len = len(batch_pairs)
                pad_b = ((b_len + 15) // 16) * 16
                mats_padded = np.full((pad_b, max_act_dim, max_act_dim), 0.5, dtype=np.float32)
                mr_padded = np.zeros((pad_b, max_act_dim), dtype=bool)
                mc_padded = np.zeros((pad_b, max_act_dim), dtype=bool)
                mr_padded[b_len:, 0] = True
                mc_padded[b_len:, 0] = True
                raw_s1_mats: list[np.ndarray] = []

                for idx_in_b, (st_i, ed_i, na, nb) in enumerate(s1_slices):
                    p_vals = vals_s1_flat[st_i:ed_i].reshape(eff_s1_games, na * nb)
                    smat = np.mean(p_vals, axis=0).reshape(na, nb)
                    raw_s1_mats.append(smat)
                    mats_padded[idx_in_b, :na, :nb] = smat
                    mr_padded[idx_in_b, :na] = True
                    mc_padded[idx_in_b, :nb] = True

                sol_batch = _solve_batched_rect_rm_plus_values_and_policies(
                    jnp.asarray(mats_padded, dtype=jnp.float32),
                    jnp.asarray(mr_padded, dtype=jnp.bool_),
                    jnp.asarray(mc_padded, dtype=jnp.bool_),
                    sweeps=rm_sweeps,
                )
                pa_batch = np.asarray(sol_batch.p_a, dtype=np.float32)
                pb_batch = np.asarray(sol_batch.p_b, dtype=np.float32)
                evr_batch = np.asarray(sol_batch.ev_row, dtype=np.float32)
                evc_batch = np.asarray(sol_batch.ev_col, dtype=np.float32)

                for idx_in_b, (i, j, pair_seed) in enumerate(batch_pairs):
                    la = legals_p[i]
                    lb = legals_q[j]
                    na, nb = len(la), len(lb)
                    smat = raw_s1_mats[idx_in_b]
                    e0 = pa_batch[idx_in_b, :na]
                    e1 = pb_batch[idx_in_b, :nb]
                    u_a = evr_batch[idx_in_b, :na]
                    u_b = 1.0 - evc_batch[idx_in_b, :nb]
                    sup_a = select_top_k_orthogonal(la, e0, u_a, min(support_k, na))
                    sup_b = select_top_k_orthogonal(lb, e1, u_b, min(support_k, nb))
                    stage1_records.append((i, j, pair_seed, smat, sup_a, sup_b))

        # Stage 2: Either reuse Stage-1 matrix or fuse all `(i, j)` Stage-2 rollouts into chunked TPU calls
        eff_refine_games = max(1, refine_games if refine_games > 0 else screen_games)
        can_reuse_s1 = (
            screen_games > 0
            and refine_games <= screen_games
            and refine_turns == screen_turns
        )

        refine_mats: list[np.ndarray] = []
        if can_reuse_s1:
            for i, j, _, screen_mat, sup_a, sup_b in stage1_records:
                idx_map_a = {act_idx: ia for ia, act_idx in enumerate(legals_p[i])}
                idx_map_b = {act_idx: ib for ib, act_idx in enumerate(legals_q[j])}
                rmat = np.zeros((len(sup_a), len(sup_b)), dtype=np.float32)
                for ia, act_a in enumerate(sup_a):
                    for ib, act_b in enumerate(sup_b):
                        rmat[ia, ib] = screen_mat[idx_map_a[act_a], idx_map_b[act_b]]
                refine_mats.append(rmat)
        else:
            # Build flat Stage-2 trajectory batch across all `stage1_records`
            r1_idx_list: list[np.ndarray] = []
            r2_idx_list: list[np.ndarray] = []
            pa_list: list[np.ndarray] = []
            pb_list: list[np.ndarray] = []
            b_seeds_list: list[np.ndarray] = []
            p_seeds_list: list[np.ndarray] = []
            sw_list: list[np.ndarray] = []
            slices_info: list[tuple[int, int, int, int]] = []
            cursor = 0

            g_indices = np.arange(eff_refine_games, dtype=np.uint32)
            ps_indices = g_indices // np.uint32(2)
            sw_g = (g_indices % np.uint32(2)) == np.uint32(1)

            for i, j, pair_seed, _, sup_a, sup_b in stage1_records:
                ka, kb = len(sup_a), len(sup_b)
                n_cells = ka * kb
                grid_ra, grid_rb = np.meshgrid(
                    np.asarray(sup_a, dtype=np.int32),
                    np.asarray(sup_b, dtype=np.int32),
                    indexing="ij",
                )
                pa_cells = PREVIEW_ACTIONS_0BASED_NP[grid_ra.reshape(-1)]  # [n_cells, 3]
                pb_cells = PREVIEW_ACTIONS_0BASED_NP[grid_rb.reshape(-1)]  # [n_cells, 3]

                pa_rep = np.tile(pa_cells, (eff_refine_games, 1))
                pb_rep = np.tile(pb_cells, (eff_refine_games, 1))
                sw_rep = np.repeat(sw_g, n_cells)

                pa_p1 = np.where(sw_rep[:, None], pb_rep, pa_rep)
                pb_p2 = np.where(sw_rep[:, None], pa_rep, pb_rep)
                idx_a = i
                idx_b = q_offset + j
                r1_idx = np.where(sw_rep, idx_b, idx_a).astype(np.int32)
                r2_idx = np.where(sw_rep, idx_a, idx_b).astype(np.int32)

                base_s2 = pair_seed ^ 0x5A5A_2222
                hi_b = np.uint32((base_s2 >> 32) & 0xFFFFFFFF)
                lo_b = np.uint32(base_s2 & 0xFFFFFFFF)
                p_idx_arr = np.arange(n_cells, dtype=np.uint32)[None, :]
                ps_idx_arr = ps_indices[:, None]
                s_hi = hi_b ^ (ps_idx_arr * np.uint32(0x9E3779B9))
                s_lo = lo_b + ps_idx_arr * np.uint32(0x85EBCA6B)
                b_hi = (s_hi ^ (p_idx_arr * np.uint32(0x9E3779B9))).reshape(-1)
                b_lo = (s_lo + (p_idx_arr * np.uint32(0x85EBCA6B) + np.uint32(1))).reshape(-1)
                p_hi = ((s_hi + np.uint32(0xC2B2AE35)) ^ (p_idx_arr * np.uint32(0x27D4EB2F))).reshape(-1)
                p_lo = (s_lo ^ (p_idx_arr * np.uint32(0x165667B1) + np.uint32(7))).reshape(-1)

                n_traj_pair = n_cells * eff_refine_games
                r1_idx_list.append(r1_idx)
                r2_idx_list.append(r2_idx)
                pa_list.append(pa_p1)
                pb_list.append(pb_p2)
                b_seeds_list.append(np.stack([b_hi, b_lo], axis=-1))
                p_seeds_list.append(np.stack([p_hi, p_lo], axis=-1))
                sw_list.append(sw_rep)
                slices_info.append((cursor, cursor + n_traj_pair, ka, kb))
                cursor += n_traj_pair

            s2_vals_flat = _execute_multi_roster_trajectory_batches(
                stacked_rosters,
                np.concatenate(r1_idx_list, axis=0),
                np.concatenate(r2_idx_list, axis=0),
                np.concatenate(pa_list, axis=0),
                np.concatenate(pb_list, axis=0),
                np.concatenate(b_seeds_list, axis=0),
                np.concatenate(p_seeds_list, axis=0),
                np.concatenate(sw_list, axis=0),
                turns=refine_turns,
                eps=eps,
                shared_dex=shared_dex,
                weights=weights,
                residual_bundle=residual_bundle,
                max_traj_chunk=max_traj_chunk,
            )
            for st_i, ed_i, ka, kb in slices_info:
                pair_vals = s2_vals_flat[st_i:ed_i].reshape(eff_refine_games, ka * kb)
                refine_mats.append(np.mean(pair_vals, axis=0).reshape(ka, kb))

        if not compute_slot_telemetry:
            max_k = max(
                1,
                max(max( m.shape[0], m.shape[1] ) for m in refine_mats),
            )
            n_recs = len(stage1_records)
            s2_chunk = 512
            for c_st in range(0, n_recs, s2_chunk):
                c_ed = min(n_recs, c_st + s2_chunk)
                cur_c = c_ed - c_st
                pad_c = ((cur_c + 15) // 16) * 16
                rm_pad = np.full((pad_c, max_k, max_k), 0.5, dtype=np.float32)
                mr_pad = np.zeros((pad_c, max_k), dtype=bool)
                mc_pad = np.zeros((pad_c, max_k), dtype=bool)
                mr_pad[cur_c:, 0] = True
                mc_pad[cur_c:, 0] = True
                for idx_c in range(cur_c):
                    rmat = refine_mats[c_st + idx_c]
                    ka, kb = rmat.shape
                    rm_pad[idx_c, :ka, :kb] = rmat
                    mr_pad[idx_c, :ka] = True
                    mc_pad[idx_c, :kb] = True
                sol_s2 = _solve_batched_rect_rm_plus_values_and_policies(
                    jnp.asarray(rm_pad, dtype=jnp.float32),
                    jnp.asarray(mr_pad, dtype=jnp.bool_),
                    jnp.asarray(mc_pad, dtype=jnp.bool_),
                    sweeps=rm_sweeps,
                )
                vals_s2 = np.asarray(sol_s2.value, dtype=np.float32)
                for idx_c in range(cur_c):
                    i, j, _, _, _, _ = stage1_records[c_st + idx_c]
                    v_star = float(vals_s2[idx_c])
                    payoff_mat[i, j] = v_star
                    if symmetric and i != j:
                        payoff_mat[j, i] = 1.0 - v_star
        else:
            for (i, j, _, _, sup_a, sup_b), refine_mat in zip(stage1_records, refine_mats):
                s2_sol = solve_rect_rm_plus_jax(
                    jnp.asarray(refine_mat, dtype=jnp.float32), sweeps=rm_sweeps
                )
                x_star = np.asarray(s2_sol.p_a, dtype=np.float32)
                y_star = np.asarray(s2_sol.p_b, dtype=np.float32)
                ev_row_a = np.asarray(s2_sol.ev_row, dtype=np.float32)
                ev_row_b = np.asarray(1.0 - s2_sol.ev_col, dtype=np.float32)

                (
                    sp_a,
                    sw_a,
                    sm_a,
                    cs_a,
                    _,
                    v_star,
                ) = _compute_side_slot_and_core_telemetry(
                    sup_a, x_star, ev_row_a, refine_mat, sweeps=rm_sweeps
                )
                (
                    sp_b,
                    sw_b,
                    sm_b,
                    cs_b,
                    _,
                    _,
                ) = _compute_side_slot_and_core_telemetry(
                    sup_b, y_star, ev_row_b, (1.0 - refine_mat.T), sweeps=rm_sweeps
                )

                payoff_mat[i, j] = v_star
                wj = float(w_q[j])
                slot_picks[i] += wj * sp_a
                slot_wins[i] += wj * sw_a
                slot_marginals[i] += wj * sm_a
                for cm, (cp, _) in cs_a.items():
                    core_acc_per_p[i][cm] = core_acc_per_p[i].get(cm, 0.0) + wj * cp

                if symmetric and i != j:
                    payoff_mat[j, i] = 1.0 - v_star
                    wi = float(w_q[i])
                    slot_picks[j] += wi * sp_b
                    slot_wins[j] += wi * sw_b
                    slot_marginals[j] += wi * sm_b
                    for cm, (cp, _) in cs_b.items():
                        core_acc_per_p[j][cm] = core_acc_per_p[j].get(cm, 0.0) + wi * cp

        for i, j, i_mir, j_mir in mirror_fill_pairs:
            payoff_mat[i, j] = 1.0 - payoff_mat[i_mir, j_mir]

    for i in range(n_p):
        core_acc = core_acc_per_p[i]
        tot_cp = sum(core_acc.values())
        if tot_cp > 1e-9:
            h_i = -sum(
                (p / tot_cp) * float(np.log(p / tot_cp))
                for p in core_acc.values()
                if p > 1e-9
            )
            eff_cores[i] = float(np.exp(h_i))
        else:
            eff_cores[i] = 1.0

    weighted_fitness = payoff_mat @ w_q
    levels_mat = np.stack([np.asarray(r.level, dtype=np.float32) for r in rosters_p], axis=0)
    slot_wars, _, _ = recompute_population_wars_np(slot_marginals, levels_mat, games=1.0)

    rect_eq = solve_rect_rm_plus_jax(jnp.asarray(payoff_mat, dtype=jnp.float32), sweeps=2000)
    return KodokuGauntletSummary(
        p_ids=p_ids,
        q_ids=q_ids,
        payoff_matrix=payoff_mat,
        weighted_fitness=weighted_fitness,
        slot_picks=slot_picks,
        slot_wins=slot_wins,
        slot_marginals=slot_marginals,
        slot_wars=slot_wars,
        effective_cores=eff_cores,
        rect_equilibrium=rect_eq,
    )


def preview_signature_sets(sets: list[dict[str, Any]]) -> tuple[tuple[str, int, str, bool], ...]:
    """Canonical 6-mon preview shell signature `tuple(sorted((toid(species), level, gender, has_item)))`.

    Matches `meta_eco.rs::preview_sig_sets` so two variants that share the exact same
    6-Pokemon `(species, level, gender, has_item)` multiset map to the same Team Preview outer shell.
    """
    pairs = [
        (
            toid(str(m.get("species", ""))),
            int(m.get("level", 50)),
            str(m.get("gender", "")).upper(),
            bool(toid(str(m.get("item", "")))),
        )
        for m in sets
    ]
    pairs.sort()
    return tuple(pairs)


def solve_bayesian_shell_preview_nash_np(
    matrices_km: np.ndarray,
    alpha: np.ndarray | list[float] | None = None,
    beta: np.ndarray | list[float] | None = None,
    sweeps: int = 500,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the Bayesian Shell-vs-Shell Team Preview Nash Equilibrium when outer shells are shared.

    Suppose Shell U has `n_u` variants with known frequency distribution `alpha` in Delta_{n_u}
    sharing legal preview action set A (`|A| = n_a`), and Shell V has `n_v` variants with known
    frequency distribution `beta` in Delta_{n_v} sharing legal preview action set B (`|B| = n_b`).
    At Team Preview, each player knows their own variant (`k` or `m`) and sees only the opponent's
    outer shell and prior variant weights (`beta` or `alpha`).

    Args:
        matrices_km: `float32[n_u, n_v, n_a, n_b]` full-combination 3v3 payoff matrices `M^{(k, m)}`.
        alpha: `float32[n_u]` row shell variant weights (defaults to uniform `1 / n_u`).
        beta: `float32[n_v]` col shell variant weights (defaults to uniform `1 / n_v`).
        sweeps: Regret Matching+ sweeps.

    Returns:
        `(v_shell, v_per_u, v_per_v, x_star, y_star)`:
          - `v_shell`: Overall Bayesian Nash value `sum_{k,m} alpha_k beta_m x_k^T M^{(k,m)} y_m`
          - `v_per_u`: `float32[n_u]` equilibrium payoff of each variant `P_k` in Shell U
          - `v_per_v`: `float32[n_v]` equilibrium payoff of each variant `Q_m` (from P's perspective)
          - `x_star`: `float32[n_u, n_a]` equilibrium pick policies for each variant in Shell U
          - `y_star`: `float32[n_v, n_b]` equilibrium pick policies for each variant in Shell V
    """
    mats = np.asarray(matrices_km, dtype=np.float32)
    n_u, n_v, n_a, n_b = mats.shape
    if alpha is None:
        a_w = np.full((n_u,), 1.0 / max(1, n_u), dtype=np.float32)
    else:
        a_w = np.asarray(alpha, dtype=np.float32)
        s_a = float(np.sum(a_w))
        a_w = (a_w / s_a) if s_a > 1e-9 else np.full((n_u,), 1.0 / max(1, n_u), dtype=np.float32)

    if beta is None:
        b_w = np.full((n_v,), 1.0 / max(1, n_v), dtype=np.float32)
    else:
        b_w = np.asarray(beta, dtype=np.float32)
        s_b = float(np.sum(b_w))
        b_w = (b_w / s_b) if s_b > 1e-9 else np.full((n_v,), 1.0 / max(1, n_v), dtype=np.float32)

    reg_p = np.zeros((n_u, n_a), dtype=np.float32)
    reg_q = np.zeros((n_v, n_b), dtype=np.float32)
    sum_p = np.zeros((n_u, n_a), dtype=np.float32)
    sum_q = np.zeros((n_v, n_b), dtype=np.float32)
    unif_p = np.full((n_u, n_a), 1.0 / max(1, n_a), dtype=np.float32)
    unif_q = np.full((n_v, n_b), 1.0 / max(1, n_b), dtype=np.float32)

    for t in range(1, max(1, int(sweeps)) + 1):
        sp = np.sum(reg_p, axis=1, keepdims=True)
        x_cur = np.where(sp > 1e-9, reg_p / np.maximum(sp, 1e-9), unif_p)
        sq = np.sum(reg_q, axis=1, keepdims=True)
        y_cur = np.where(sq > 1e-9, reg_q / np.maximum(sq, 1e-9), unif_q)

        # Expected row utilities u_p[k, a] = sum_m beta_m (M[k, m] @ y_cur[m])[a]
        by = b_w[:, None] * y_cur  # [n_v, n_b]
        u_p = np.einsum("kmab,mb->ka", mats, by)  # [n_u, n_a]
        v_p = np.sum(x_cur * u_p, axis=1, keepdims=True)

        # Expected col utilities for Q: u_q[m, b] = 1 - sum_k alpha_k (x_cur[k] @ M[k, m])[b]
        ax = a_w[:, None] * x_cur  # [n_u, n_a]
        u_q = 1.0 - np.einsum("ka,kmab->mb", ax, mats)  # [n_v, n_b]
        v_q = np.sum(y_cur * u_q, axis=1, keepdims=True)

        reg_p = np.maximum(0.0, reg_p + (u_p - v_p))
        reg_q = np.maximum(0.0, reg_q + (u_q - v_q))

        w_t = float(t)
        sum_p += w_t * x_cur
        sum_q += w_t * y_cur

    sp_tot = np.sum(sum_p, axis=1, keepdims=True)
    x_star = np.where(sp_tot > 1e-9, sum_p / np.maximum(sp_tot, 1e-9), unif_p)
    sq_tot = np.sum(sum_q, axis=1, keepdims=True)
    y_star = np.where(sq_tot > 1e-9, sum_q / np.maximum(sq_tot, 1e-9), unif_q)

    by_star = b_w[:, None] * y_star
    u_p_star = np.einsum("kmab,mb->ka", mats, by_star)
    v_per_u = np.sum(x_star * u_p_star, axis=1)

    ax_star = a_w[:, None] * x_star
    u_q_from_p = np.einsum("ka,kmab->mb", ax_star, mats)
    v_per_v = np.sum(y_star * u_q_from_p, axis=1)

    v_shell = float(np.sum(a_w * v_per_u))
    return v_shell, v_per_u.astype(np.float32), v_per_v.astype(np.float32), x_star.astype(np.float32), y_star.astype(np.float32)


def detect_hitchhiker_slots(
    sets: list[dict[str, Any]],
    slot_picks: np.ndarray | list[float],
    slot_marginals: np.ndarray | list[float],
    legal_action_count: int | None = None,
    min_pick_threshold: float = 0.025,
    min_marginal_threshold: float = 0.0015,
) -> list[dict[str, Any]]:
    """Pinpoint-detect Hitchhiker slots on a 6-Pokemon team from Full-Combination Nash telemetry.

    A slot `s` is flagged as a Hitchhiker if:
      1. Its equilibrium pick share `slot_picks[s] < min_pick_threshold` AND its Leave-One-Out
         marginal Nash value `slot_marginals[s] < min_marginal_threshold`, OR
      2. The team over-packs `>= 4` L55s (`legal_action_count <= 12`), forcing 2 L50s to 100%
         pick rate and stranding 3 L55s on the bench every game as unpicked decoys.
    """
    picks = np.asarray(slot_picks, dtype=np.float32)
    margs = np.asarray(slot_marginals, dtype=np.float32)
    n55 = sum(1 for m in sets if int(m.get("level", 50)) >= 55)
    overpacked_l55 = (n55 >= 4) or (legal_action_count is not None and int(legal_action_count) <= 12)
    flagged: list[dict[str, Any]] = []
    for s in range(min(6, len(sets))):
        sp = str(sets[s].get("species", ""))
        lv = int(sets[s].get("level", 50))
        pk = float(picks[s]) if s < len(picks) else 0.0
        mg = float(margs[s]) if s < len(margs) else 0.0
        is_dead_slot = (pk < min_pick_threshold) and (mg < min_marginal_threshold)
        is_l55_decoy = overpacked_l55 and (lv >= 55) and (pk < 0.25 or mg < 0.05)
        if is_dead_slot or is_l55_decoy:
            flagged.append(
                {
                    "slot": s,
                    "species": sp,
                    "level": lv,
                    "pick_rate": round(pk, 6),
                    "loo_marginal": round(mg, 6),
                    "reason": "overpacked_l55_decoy" if is_l55_decoy and not is_dead_slot else "dead_nash_slot",
                }
            )
    return flagged


def build_coupled_shell_units(
    teams_p: list[dict[str, Any]],
    explicit_units: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Group `teams_p` into Zero-Smuggling Coupled-Shell Units (`DEME{n}:<id0>+...`).

    - If `explicit_units` is provided, validates and normalizes each unit `{"unit_id", "member_ids"}`.
    - Otherwise, groups `teams_p` by canonical 6-mon `preview_signature_sets(sets)` with unbounded
      variant cardinality `n >= 1` (`DEME{n}:...`).
    """
    id_to_idx = {str(t.get("id", f"p_{i}")): i for i, t in enumerate(teams_p)}
    if explicit_units is not None:
        out: list[dict[str, Any]] = []
        for u in explicit_units:
            mids = [str(m) for m in u.get("member_ids", []) if str(m) in id_to_idx]
            if not mids:
                continue
            uid = str(u.get("unit_id") or f"DEME{len(mids)}:{'+'.join(mids)}")
            sig = preview_signature_sets(teams_p[id_to_idx[mids[0]]]["sets"])
            raw_mw = u.get("member_weights")
            if raw_mw is not None and len(raw_mw) == len(mids):
                mw_arr = np.asarray(raw_mw, dtype=np.float32)
                mw_sum = float(np.sum(mw_arr))
                mw_list = (
                    [float(x / mw_sum) for x in mw_arr]
                    if mw_sum > 1e-9
                    else [1.0 / float(len(mids))] * len(mids)
                )
            else:
                mw_list = [1.0 / float(len(mids))] * len(mids)
            out.append(
                {
                    "unit_id": uid,
                    "member_ids": mids,
                    "member_indices": [id_to_idx[m] for m in mids],
                    "member_weights": mw_list,
                    "sig": sig,
                }
            )
        return out

    sig_groups: dict[tuple[tuple[str, int, str, bool], ...], list[str]] = {}
    sig_order: list[tuple[tuple[str, int, str, bool], ...]] = []
    for i, t in enumerate(teams_p):
        tid = str(t.get("id", f"p_{i}"))
        sig = preview_signature_sets(t["sets"])
        if sig not in sig_groups:
            sig_groups[sig] = []
            sig_order.append(sig)
        if tid not in sig_groups[sig]:
            sig_groups[sig].append(tid)

    units_out: list[dict[str, Any]] = []
    for sig in sig_order:
        mids = sig_groups[sig]
        uid = f"DEME{len(mids)}:{'+'.join(mids)}"
        w_raw = [float(teams_p[id_to_idx[m]].get("weight", 1.0) or 1.0) for m in mids]
        w_sum = sum(w_raw)
        mw_list = [w / w_sum for w in w_raw] if w_sum > 1e-9 else [1.0 / float(len(mids))] * len(mids)
        units_out.append(
            {
                "unit_id": uid,
                "member_ids": mids,
                "member_indices": [id_to_idx[m] for m in mids],
                "member_weights": mw_list,
                "sig": sig,
            }
        )
    return units_out


class CoupledShellEquilibriumResult(NamedTuple):
    """Skew-Symmetric Coupled-Shell Minimax Equilibrium solved on TPU (`converge-conclusion`)."""

    unit_ids: list[str]
    member_ids_per_unit: list[list[str]]
    q_ids: list[str]
    unit_payoff_matrix: np.ndarray  # float32[N_U, N_Q]
    unit_games_matrix: np.ndarray  # float32[N_U, N_Q]
    m_solve: np.ndarray  # float32[N_U, N_U + N_OUT]
    unit_weights: np.ndarray  # float32[N_U]
    active_shells: list[dict[str, Any]]
    deployed_team_ids: list[str]
    deployed_team_weights: list[float]
    min_in_payoff: float
    min_matrix_payoff: float


def solve_coupled_shell_equilibrium_tpu(
    summary: KodokuGauntletSummary,
    teams_p: list[dict[str, Any]],
    teams_q: list[dict[str, Any]] | None = None,
    explicit_units: list[dict[str, Any]] | None = None,
    games_per_member_cell: float = 8.0,
    rm_sweeps: int = 8000,
    support_threshold: float = 0.002,
) -> CoupledShellEquilibriumResult:
    """Solve the Zero-Smuggling Coupled-Shell Minimax Equilibrium from a `KodokuGauntletSummary`.

    Exact port of `meta_eco.rs::cmd_converge_conclusion` (lines 10086-10360):
    1. Groups `teams_p` into Coupled-Shell Units `U_0 .. U_{N_U - 1}` (`SINGLE:<id>` or
       `DEME2:<id0>+<id1>`, where `DEME2` averages its 2 members' payoffs 50/50).
    2. Constructs the Skew-Symmetric Coupled-Shell Minimax Matrix `m_solve` (`N_U x (N_U + N_OUT)`):
       - First `N_U` columns: internal Shell Units with `m_solve[u, u] = 0.5` and game-pooled
         skew-symmetry `m_solve[u, v] + m_solve[v, u] = 1.0`.
       - Remaining `N_OUT` columns: outside challengers in `q_ids` not belonging to any shell unit,
         shrunk with `(g * raw_sc + 8.0 * 0.50) / (g + 8.0)`.
    3. Solves both the internal `(N_U x N_U)` and rectangular `(N_U x N_COLS)` minimax games via
       `solve_rect_rm_plus_jax`, polishes on active shells (`w >= support_threshold`), and splits
       `DEME2` shell weights 50/50 across their 2 member teams.
    """
    if teams_q is None:
        teams_q = teams_p

    units = build_coupled_shell_units(teams_p, explicit_units=explicit_units)
    n_u = len(units)
    q_ids = list(summary.q_ids)
    n_q = len(q_ids)
    id_to_qidx = {qid: j for j, qid in enumerate(q_ids)}
    q_sigs = [preview_signature_sets(t["sets"]) for t in teams_q]
    seen_unit_sigs = {u["sig"] for u in units}

    unit_payoff = np.full((n_u, n_q), 0.5, dtype=np.float32)
    unit_games = np.zeros((n_u, n_q), dtype=np.float32)
    unit_member_set: set[str] = set()
    unit_q_indices: list[list[int]] = []

    for u_idx, u in enumerate(units):
        m_idxs: list[int] = u["member_indices"]
        m_ids: list[str] = u["member_ids"]
        q_idxs_u: list[int] = []
        for mid in m_ids:
            unit_member_set.add(mid)
            if mid in id_to_qidx:
                q_idxs_u.append(id_to_qidx[mid])
        unit_q_indices.append(q_idxs_u)

        rows = summary.payoff_matrix[m_idxs, :]  # [len(m_idxs), n_q]
        m_weights = np.asarray(
            u.get("member_weights", [1.0 / float(len(m_idxs))] * len(m_idxs)),
            dtype=np.float32,
        )
        m_w_sum = float(np.sum(m_weights))
        if m_w_sum > 1e-9:
            m_weights = m_weights / m_w_sum
        else:
            m_weights = np.full((len(m_idxs),), 1.0 / float(max(1, len(m_idxs))), dtype=np.float32)
        unit_payoff[u_idx, :] = np.sum(m_weights[:, None] * rows, axis=0)
        unit_games[u_idx, :] = float(games_per_member_cell) * float(len(m_idxs))
        if len(m_ids) == 1 and m_ids[0] in id_to_qidx:
            j_self = id_to_qidx[m_ids[0]]
            unit_payoff[u_idx, j_self] = 0.5
            unit_games[u_idx, j_self] = 64.0

    outside_q_indices = [
        j
        for j in range(n_q)
        if (q_ids[j] not in unit_member_set) and (q_sigs[j] not in seen_unit_sigs)
    ]
    n_cols = n_u + len(outside_q_indices)
    m_solve = np.full((n_u, n_cols), 0.5, dtype=np.float32)

    for u in range(n_u):
        m_solve[u, u] = 0.5
        for v in range(u + 1, n_u):
            sum_uv = 0.0
            gms_uv = 0.0
            for jv in unit_q_indices[v]:
                g = max(float(unit_games[u, jv]), 1.0)
                sum_uv += float(unit_payoff[u, jv]) * g
                gms_uv += g
            sum_vu = 0.0
            gms_vu = 0.0
            for ju in unit_q_indices[u]:
                g = max(float(unit_games[v, ju]), 1.0)
                sum_vu += float(unit_payoff[v, ju]) * g
                gms_vu += g
            sc_uv = (sum_uv / gms_uv) if gms_uv > 0.0 else 0.5
            sc_vu = (sum_vu / gms_vu) if gms_vu > 0.0 else 0.5
            p_uv = (
                (gms_uv * sc_uv + gms_vu * (1.0 - sc_vu)) / (gms_uv + gms_vu)
                if (gms_uv + gms_vu) > 0.0
                else 0.5
            )
            m_solve[u, v] = float(p_uv)
            m_solve[v, u] = float(1.0 - p_uv)

        for k, j_out in enumerate(outside_q_indices):
            g = max(float(unit_games[u, j_out]), 1.0)
            raw_sc = float(unit_payoff[u, j_out])
            m_solve[u, n_u + k] = float((g * raw_sc + 8.0 * 0.50) / (g + 8.0))

    sol_in = solve_rect_rm_plus_jax(
        jnp.asarray(m_solve[:, :n_u], dtype=jnp.float32), sweeps=rm_sweeps
    )
    sol_rect = solve_rect_rm_plus_jax(
        jnp.asarray(m_solve, dtype=jnp.float32), sweeps=rm_sweeps
    )
    w_in = np.asarray(sol_in.p_a, dtype=np.float32)
    w_rect = np.asarray(sol_rect.p_a, dtype=np.float32)
    min_in_val = float(sol_in.guarantee_mixed_a)
    min_rect_val = float(sol_rect.guarantee_mixed_a)

    active_mask = (w_in >= support_threshold) | (w_rect >= support_threshold)
    if not np.any(active_mask):
        active_mask = np.ones((n_u,), dtype=bool)

    sol_pol = solve_rect_rm_plus_jax(
        jnp.asarray(m_solve, dtype=jnp.float32),
        mask_rows=jnp.asarray(active_mask, dtype=jnp.bool_),
        sweeps=rm_sweeps,
    )
    w_pol = np.asarray(sol_pol.p_a, dtype=np.float32)
    min_pol_val = float(sol_pol.guarantee_mixed_a)
    min_val = min(min_in_val, max(min_pol_val, min_rect_val))

    keep = w_pol >= support_threshold
    if not np.any(keep):
        keep = active_mask
    w_final = np.where(keep, w_pol, 0.0).astype(np.float32)
    w_sum = float(np.sum(w_final))
    w_final = (w_final / w_sum) if w_sum > 1e-9 else w_pol

    order = sorted(
        [u_idx for u_idx in range(n_u) if w_final[u_idx] > 0.0],
        key=lambda u_idx: float(w_final[u_idx]),
        reverse=True,
    )
    active_shells: list[dict[str, Any]] = []
    deployed_ids: list[str] = []
    deployed_weights: list[float] = []
    for u_idx in order:
        w_sh = float(w_final[u_idx])
        m_ids = list(units[u_idx]["member_ids"])
        m_ws = list(units[u_idx].get("member_weights", [1.0 / float(len(m_ids))] * len(m_ids)))
        active_shells.append(
            {
                "unit_id": units[u_idx]["unit_id"],
                "weight": round(w_sh, 6),
                "member_ids": m_ids,
                "member_weights": [round(float(mw), 6) for mw in m_ws],
            }
        )
        for mid, mw in zip(m_ids, m_ws):
            if float(mw) > 1e-6:
                deployed_ids.append(mid)
                deployed_weights.append(round(w_sh * float(mw), 6))

    return CoupledShellEquilibriumResult(
        unit_ids=[u["unit_id"] for u in units],
        member_ids_per_unit=[list(u["member_ids"]) for u in units],
        q_ids=q_ids,
        unit_payoff_matrix=unit_payoff,
        unit_games_matrix=unit_games,
        m_solve=m_solve,
        unit_weights=w_final,
        active_shells=active_shells,
        deployed_team_ids=deployed_ids,
        deployed_team_weights=deployed_weights,
        min_in_payoff=min_in_val,
        min_matrix_payoff=min_val,
    )


def export_dominance_audit_cells_json(
    path: str | Path,
    summary: KodokuGauntletSummary,
    games_per_cell: float = 8.0,
    coupled_equilibrium: CoupledShellEquilibriumResult | None = None,
) -> None:
    """Export `{"cells": [...], "unit_cells": [...]}` matching `meta_eco dominance-audit` and `converge-conclusion`."""
    cells = []
    unit_cells = []
    for i, p_id in enumerate(summary.p_ids):
        for j, q_id in enumerate(summary.q_ids):
            sc = round(float(summary.payoff_matrix[i, j]), 6)
            cells.append(
                {
                    "p_id": p_id,
                    "q_id": q_id,
                    "score": sc,
                }
            )
            if coupled_equilibrium is None:
                uname = p_id if p_id.startswith(("SINGLE:", "DEME2:")) else f"SINGLE:{p_id}"
                unit_cells.append(
                    {
                        "unit_id": uname,
                        "q_id": q_id,
                        "score": sc,
                        "games": float(games_per_cell),
                    }
                )

    if coupled_equilibrium is not None:
        for u_idx, uid in enumerate(coupled_equilibrium.unit_ids):
            for j, q_id in enumerate(coupled_equilibrium.q_ids):
                unit_cells.append(
                    {
                        "unit_id": uid,
                        "q_id": q_id,
                        "score": round(float(coupled_equilibrium.unit_payoff_matrix[u_idx, j]), 6),
                        "games": float(coupled_equilibrium.unit_games_matrix[u_idx, j]),
                    }
                )

    payload: dict[str, Any] = {"cells": cells, "unit_cells": unit_cells}
    if coupled_equilibrium is not None:
        payload["coupled_equilibrium"] = {
            "min_in_payoff": round(float(coupled_equilibrium.min_in_payoff), 6),
            "min_matrix_payoff": round(float(coupled_equilibrium.min_matrix_payoff), 6),
            "active_shell_count": len(coupled_equilibrium.active_shells),
            "support_team_count": len(coupled_equilibrium.deployed_team_ids),
            "active_shells": coupled_equilibrium.active_shells,
            "deployed_team_ids": coupled_equilibrium.deployed_team_ids,
            "deployed_team_weights": coupled_equilibrium.deployed_team_weights,
        }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================================
# 11. Sequential Bayesian Election-Call (`選挙当確速報`) &
#     In-Battle Intra-Shell Bayesian Belief + 1-Ply Simultaneous-Move RM+
# =========================================================================


def bayesian_election_call_bounds_np(
    sum_scores: np.ndarray,
    sq_scores: np.ndarray,
    counts: np.ndarray,
    z: float = 2.326,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Beta-Binomial / Empirical-Bernstein credible bounds `(mean, lcb, ucb, radius)`.

    For each cell or matchup with `n = counts` paired side-swap CRN evaluations in `[0, 1]`:
    - Uses Jeffreys prior `Beta(0.5, 0.5)` combined with empirical variance `var_hat` to form
      tight, valid credible intervals `[lcb, ucb]` that shrink immediately on 1.0/0.0 blowouts
      (`選挙当確速報`) while remaining wide on 50/50 high-variance cells (`赤子を産湯と一緒に流さない`).
    """
    n = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    s = np.asarray(sum_scores, dtype=np.float64)
    sq = np.asarray(sq_scores, dtype=np.float64)

    n_safe = np.maximum(n, 1.0)
    mean_raw = np.where(n > 0.0, np.clip(s / n_safe, 0.0, 1.0), 0.5)

    # Jeffreys Beta(0.5, 0.5) posterior mean & variance
    alpha = 0.5 + np.where(n > 0.0, s, 0.0)
    beta = 0.5 + np.where(n > 0.0, n - s, 0.0)
    ab = alpha + beta
    post_mean = alpha / ab
    beta_var = (alpha * beta) / (ab * ab * (ab + 1.0))

    # Empirical variance + Beta-Binomial Jeffreys variance blend
    emp_var = np.where(
        n > 1.0,
        np.maximum(0.0, (sq / n_safe) - mean_raw * mean_raw) * (n_safe / np.maximum(n_safe - 1.0, 1.0)),
        0.25,
    )
    eff_se = np.sqrt(0.5 * beta_var + 0.5 * (emp_var / n_safe + (0.20 / (n_safe + 1.0)) ** 2))
    radius = np.where(n > 0.0, float(z) * eff_se, 0.50)

    lcb = np.where(n > 0.0, np.clip(post_mean - radius, 0.0, 1.0), 0.0)
    ucb = np.where(n > 0.0, np.clip(post_mean + radius, 0.0, 1.0), 1.0)
    return (
        mean_raw.astype(np.float32),
        lcb.astype(np.float32),
        ucb.astype(np.float32),
        radius.astype(np.float32),
    )


class BayesianElectionCallPairResult(NamedTuple):
    """6v6 Full-Space Preview Nash evaluation with Sequential Bayesian Election-Call (`選挙当確速報`)."""

    value: float
    lcb99: float
    ucb99: float
    election_status: str  # "WIN_CALLED", "LOSS_CALLED", or "OPEN"
    legal_a: list[int]
    legal_b: list[int]
    matrix: np.ndarray  # float32[len(legal_a), len(legal_b)]
    cell_games: np.ndarray  # int32[len(legal_a), len(legal_b)]
    x_star: np.ndarray  # float32[len(legal_a)]
    y_star: np.ndarray  # float32[len(legal_b)]
    slot_picks_a: np.ndarray  # float32[6]
    slot_wins_a: np.ndarray  # float32[6]
    slot_marginals_a: np.ndarray  # float32[6]
    slot_picks_b: np.ndarray  # float32[6]
    slot_wins_b: np.ndarray  # float32[6]
    slot_marginals_b: np.ndarray  # float32[6]
    waves_used: int
    cells_called_early: int
    total_cell_games: int
    elapsed_secs: float


def evaluate_6v6_bayesian_election_call_tpu(
    roster_a: RosterTable6,
    roster_b: RosterTable6,
    shared_dex: SharedDexTable,
    wave_games: tuple[int, ...] = (2, 2, 4, 8),
    turns: int = 40,
    eps: float = 0.2,
    seed: int = 20260405,
    win_threshold: float | None = 0.505,
    loss_threshold: float | None = 0.495,
    cell_radius_tol: float = 0.065,
    z_value: float = 2.326,
    rm_sweeps: int = 800,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
    batch_chunk_size: int = 7200,
) -> BayesianElectionCallPairResult:
    """Sequential Bayesian Election-Call (`選挙当確速報`) 6v6 Full-Combination Evaluator on TPU.

    Unlike static top-k screening (`core_nash_6v6_pair_tpu`), this evaluator:
    1. Starts with 100% of legal 60-action triples `(legal_a x legal_b)` open in Wave 1 (`赤子を産湯と一緒に流さない`).
    2. After each wave, computes Bayesian credible bounds `[M_lcb, M_ucb]` for every 3v3 cell and
       solves the optimistic/pessimistic zero-sum games `V_lcb = val(M_lcb)` and `V_ucb = val(M_ucb)`.
    3. Immediately calls (`当確`) individual 3v3 cells that are either:
       - Resolved blowouts (`cell_games >= 4` and `radius <= cell_radius_tol`), or
       - Provably dominated in both row and column equilibrium support (`ucb_row < V_lcb - 0.015`
         and `lcb_col > V_ucb + 0.015`).
    4. Immediately calls the entire 6v6 matchup (`WIN_CALLED` if `V_lcb > win_threshold`,
       `LOSS_CALLED` if `V_ucb < loss_threshold`) when the 6v6 credible interval is decisive.
    """
    t0 = time.perf_counter()
    if weights is None:
        weights = default_eval_weights()

    legal_a = [int(i) for i in np.flatnonzero(np.asarray(roster_a.legal_mask_60))]
    legal_b = [int(i) for i in np.flatnonzero(np.asarray(roster_b.legal_mask_60))]
    if not legal_a:
        legal_a = [0]
    if not legal_b:
        legal_b = [0]

    na = len(legal_a)
    nb = len(legal_b)
    sum_mat = np.zeros((na, nb), dtype=np.float64)
    sq_mat = np.zeros((na, nb), dtype=np.float64)
    cnt_mat = np.zeros((na, nb), dtype=np.int32)
    open_mask = np.ones((na, nb), dtype=bool)

    grid_a, grid_b = np.meshgrid(
        np.asarray(legal_a, dtype=np.int32),
        np.asarray(legal_b, dtype=np.int32),
        indexing="ij",
    )

    waves_used = 0
    election_status = "OPEN"
    lcb99 = 0.0
    ucb99 = 1.0

    for w_idx, g_wave in enumerate(wave_games):
        g_w = max(2, int(g_wave))
        open_coords = np.argwhere(open_mask)
        if open_coords.shape[0] == 0:
            break

        waves_used = w_idx + 1
        ia_idx = open_coords[:, 0]
        ib_idx = open_coords[:, 1]
        act_a = grid_a[ia_idx, ib_idx]
        act_b = grid_b[ia_idx, ib_idx]
        picks_a_w = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[act_a], dtype=jnp.int32)
        picks_b_w = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[act_b], dtype=jnp.int32)

        wave_seed = int(seed) ^ ((w_idx + 1) * 0x9E37_79B9)
        scores_w = np.asarray(
            evaluate_3v3_picks_batch_tpu(
                roster_a,
                roster_b,
                picks_a_w,
                picks_b_w,
                shared_dex,
                games_per_pair=g_w,
                turns=turns,
                eps=eps,
                base_seed_int=wave_seed,
                weights=weights,
                residual_bundle=residual_bundle,
                batch_chunk_size=batch_chunk_size,
            ),
            dtype=np.float64,
        )
        n_pairs_w = float(g_w // 2)
        sum_mat[ia_idx, ib_idx] += scores_w * n_pairs_w
        sq_mat[ia_idx, ib_idx] += (
            (scores_w * scores_w) * n_pairs_w
            + scores_w * (1.0 - scores_w) * max(0.0, n_pairs_w - 1.0)
        )
        cnt_mat[ia_idx, ib_idx] += int(n_pairs_w)

        mean_mat, lcb_mat, ucb_mat, rad_mat = bayesian_election_call_bounds_np(
            sum_mat, sq_mat, cnt_mat, z=z_value
        )
        sol_lcb = solve_rect_rm_plus_jax(jnp.asarray(lcb_mat, dtype=jnp.float32), sweeps=rm_sweeps)
        sol_ucb = solve_rect_rm_plus_jax(jnp.asarray(ucb_mat, dtype=jnp.float32), sweeps=rm_sweeps)
        lcb99 = float(sol_lcb.value)
        ucb99 = float(sol_ucb.value)

        if win_threshold is not None and lcb99 > float(win_threshold):
            election_status = "WIN_CALLED"
            break
        if loss_threshold is not None and ucb99 < float(loss_threshold):
            election_status = "LOSS_CALLED"
            break

        ev_row_ucb = np.asarray(sol_ucb.ev_row, dtype=np.float32)
        ev_col_lcb = np.asarray(sol_lcb.ev_col, dtype=np.float32)
        dom_rows = (ev_row_ucb + 0.015) < lcb99
        dom_cols = (ev_col_lcb - 0.015) > ucb99

        resolved_cells = (cnt_mat >= 2) & (rad_mat <= float(cell_radius_tol))
        dominated_cells = dom_rows[:, None] & dom_cols[None, :]
        open_mask = open_mask & (~resolved_cells) & (~dominated_cells)

    mean_mat, _, _, _ = bayesian_election_call_bounds_np(sum_mat, sq_mat, cnt_mat, z=z_value)
    sol_mean = solve_rect_rm_plus_jax(jnp.asarray(mean_mat, dtype=jnp.float32), sweeps=rm_sweeps)
    x_star = np.asarray(sol_mean.p_a, dtype=np.float32)
    y_star = np.asarray(sol_mean.p_b, dtype=np.float32)
    ev_row_a = np.asarray(sol_mean.ev_row, dtype=np.float32)
    ev_row_b = np.asarray(1.0 - sol_mean.ev_col, dtype=np.float32)

    slot_picks_a, slot_wins_a, slot_marginals_a, _, _, v_star = _compute_side_slot_and_core_telemetry(
        legal_a, x_star, ev_row_a, mean_mat, sweeps=rm_sweeps
    )
    slot_picks_b, slot_wins_b, slot_marginals_b, _, _, _ = _compute_side_slot_and_core_telemetry(
        legal_b, y_star, ev_row_b, (1.0 - mean_mat.T), sweeps=rm_sweeps
    )

    cells_called = int(np.sum(~open_mask))
    total_cell_games = int(np.sum(cnt_mat) * 2)
    return BayesianElectionCallPairResult(
        value=float(v_star),
        lcb99=float(lcb99),
        ucb99=float(ucb99),
        election_status=election_status,
        legal_a=legal_a,
        legal_b=legal_b,
        matrix=mean_mat,
        cell_games=cnt_mat * 2,
        x_star=x_star,
        y_star=y_star,
        slot_picks_a=slot_picks_a,
        slot_wins_a=slot_wins_a,
        slot_marginals_a=slot_marginals_a,
        slot_picks_b=slot_picks_b,
        slot_wins_b=slot_wins_b,
        slot_marginals_b=slot_marginals_b,
        waves_used=waves_used,
        cells_called_early=cells_called,
        total_cell_games=total_cell_games,
        elapsed_secs=time.perf_counter() - t0,
    )


@partial(jax.jit, static_argnames=("side",))
def update_shell_belief_mask_jax(
    prev_state: CompactBattle,
    curr_state: CompactBattle,
    true_table: MatchTable,
    candidate_tables: MatchTable,
    side: int,
    belief_mask: jax.Array,
) -> jax.Array:
    """Branchless in-battle Bayesian belief update over hidden opponent shell variants `0..V-1`.

    All `V` variants in `candidate_tables` share identical visible Team Preview species & levels
    (`maxhp`, `types`, `level`). Whenever opponent `foe_side = 1 - side`:
    1. Reveals a move at `(p_slot, m_i)` via PP decrement (`curr_state.pp < prev_state.pp`), or
    2. Consumes/reveals an item (`curr_state.item != prev_state.item`),
    any candidate variant `v` whose move signature `(move_type, move_base_power, move_special_kind)`
    or `item_flags` disagrees with the revealed observation is eliminated from `belief_mask`.
    Guarantees non-empty fallback (`jnp.where(jnp.any(new_mask), new_mask, belief_mask)`).
    """
    foe_side = 1 - int(side)
    f_slice = slice(foe_side * 3, foe_side * 3 + 3)

    pp_prev = prev_state.pp[f_slice]  # [3, 4]
    pp_curr = curr_state.pp[f_slice]  # [3, 4]
    move_used = pp_curr < pp_prev  # bool[3, 4]

    item_prev = prev_state.item[f_slice]  # [3]
    item_curr = curr_state.item[f_slice]  # [3]
    item_revealed = (item_curr != item_prev) & (item_prev != jnp.int32(0))  # bool[3]

    true_m_type = true_table.move_type[f_slice]  # [3, 4]
    true_m_bp = true_table.move_base_power[f_slice]  # [3, 4]
    true_m_sk = true_table.move_special_kind[f_slice]  # [3, 4]
    true_iflags = true_table.item_flags[f_slice]  # [3]

    slot_idx = jnp.arange(4, dtype=jnp.int32)[None, None, :]  # [1, 1, 4]

    def _check_variant(cand_tb: MatchTable) -> jax.Array:
        c_m_type = cand_tb.move_type[f_slice]  # [3, 4]
        c_m_bp = cand_tb.move_base_power[f_slice]  # [3, 4]
        c_m_sk = cand_tb.move_special_kind[f_slice]  # [3, 4]
        c_num_moves = cand_tb.num_moves[f_slice]  # [3]
        c_iflags = cand_tb.item_flags[f_slice]  # [3]

        valid_c_slot = slot_idx < c_num_moves[:, None, None]  # [3, 1, 4]
        move_match = jnp.any(
            valid_c_slot
            & (c_m_type[:, None, :] == true_m_type[:, :, None])
            & (c_m_bp[:, None, :] == true_m_bp[:, :, None])
            & (c_m_sk[:, None, :] == true_m_sk[:, :, None]),
            axis=-1,
        )  # [3, 4]
        moves_ok = jnp.all(jnp.where(move_used, move_match, True))
        items_ok = jnp.all(jnp.where(item_revealed, c_iflags == true_iflags, True))
        return moves_ok & items_ok

    compat = jax.vmap(_check_variant)(candidate_tables)
    new_mask = belief_mask & compat
    return jnp.where(jnp.any(new_mask), new_mask, belief_mask)


@partial(jax.jit, static_argnames=("sweeps",))
def solve_bayesian_equilibrium_jax(
    mats_v: jax.Array,
    mask_own: jax.Array,
    masks_opp_v: jax.Array,
    post_w: jax.Array,
    sweeps: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Port of `nc2000_bot::root_policy::solve_bayesian_equilibrium_with_priors` to branchless JAX/TPU.

    Solves the Bayesian root maximin game across `V` determinized worlds:
        `max_{p in Delta(mask_own)} sum_v post_w[v] min_{j: masks_opp_v[v, j]} sum_i p[i] V_hat[v, i, j]`
    where `mats_v[v, i, j]` (`float32[V, 7, 7]`) is already oriented from `own`'s perspective
    (`own` chooses row `i`, `opp` in world `v` chooses column `j`).

    Returns `(policy, action_values, pure_guarantees, equilibrium_value)`.
    """
    fb_own = jnp.argmax(mask_own.astype(jnp.int32)).astype(jnp.int32)
    valid_v = mask_own[None, :, None] & masks_opp_v[:, None, :]
    clean_v = jnp.where(valid_v, jnp.clip(mats_v, jnp.float32(0.0), jnp.float32(1.0)), jnp.float32(0.5))

    # 1. Cross-world Empirical-Bayes shrinkage (`pooled_prior` + `mu_row + col_blunder_bonus`)
    pooled_cell = jnp.sum(post_w[:, None, None] * clean_v, axis=0)  # [7, 7]
    opp_cnt = jnp.maximum(jnp.sum(masks_opp_v.astype(jnp.float32), axis=-1, keepdims=True), jnp.float32(1.0))
    own_cnt = jnp.maximum(jnp.sum(mask_own.astype(jnp.float32)), jnp.float32(1.0))

    mu_row = jnp.sum(
        jnp.where(masks_opp_v[:, None, :], clean_v, jnp.float32(0.0)), axis=-1
    ) / opp_cnt  # [V, 7]
    mu_col = jnp.sum(
        jnp.where(mask_own[None, :, None], clean_v, jnp.float32(0.0)), axis=1
    ) / own_cnt  # [V, 7]
    mu_world = jnp.sum(
        jnp.where(masks_opp_v, mu_col, jnp.float32(0.0)), axis=-1, keepdims=True
    ) / opp_cnt  # [V, 1]
    col_blunder_bonus = jnp.maximum(jnp.float32(0.0), mu_col - mu_world)  # [V, 7]
    additive_prior = jnp.clip(mu_row[:, :, None] + col_blunder_bonus[:, None, :], jnp.float32(0.0), jnp.float32(1.0))

    v_hat = (
        jnp.float32(0.75) * clean_v
        + jnp.float32(0.15) * pooled_cell[None, :, :]
        + jnp.float32(0.10) * additive_prior
    )
    v_hat = jnp.where(valid_v, v_hat, jnp.float32(0.5))

    # 2. Pure guarantees: `sum_v w_v min_{j in mask_opp_v} v_hat[v, i, j]`
    masked_for_min = jnp.where(masks_opp_v[:, None, :], v_hat, jnp.float32(1e9))
    per_world_row_min = jnp.min(masked_for_min, axis=-1)  # [V, 7]
    per_world_row_min = jnp.where(per_world_row_min > jnp.float32(1.0), jnp.float32(0.5), per_world_row_min)
    pure_guarantees = jnp.sum(post_w[:, None] * per_world_row_min, axis=0)  # [7]

    raw_best_idx = jnp.argmax(
        jnp.where(mask_own, pure_guarantees, jnp.float32(-1e9))
    ).astype(jnp.int32)
    best_pure_idx = jnp.where(jnp.any(mask_own), raw_best_idx, fb_own)
    pure_one_hot = (jnp.arange(7, dtype=jnp.int32) == best_pure_idx).astype(jnp.float32)
    best_pure_val = jnp.sum(pure_guarantees * pure_one_hot)

    # 3. Coupled Regret Matching+ over `(p[7], y[V, 7])`
    own_unif = mask_own.astype(jnp.float32) / own_cnt
    opp_unif = masks_opp_v.astype(jnp.float32) / opp_cnt  # [V, 7]

    r_own0 = jnp.zeros((7,), dtype=jnp.float32)
    r_opp0 = jnp.zeros_like(opp_unif)
    sum_p0 = jnp.zeros((7,), dtype=jnp.float32)
    sum_y0 = jnp.zeros_like(opp_unif)

    def _rm_step(t_idx: jax.Array, carry):
        r_own, r_opp, sum_p, sum_y = carry
        s_own = jnp.sum(r_own)
        p = jnp.where(s_own > jnp.float32(1e-12), r_own / jnp.maximum(s_own, jnp.float32(1e-12)), own_unif)

        s_opp = jnp.sum(r_opp, axis=-1, keepdims=True)
        y = jnp.where(s_opp > jnp.float32(1e-12), r_opp / jnp.maximum(s_opp, jnp.float32(1e-12)), opp_unif)

        # Adversary in each world v minimizes `p^T v_hat[v]` <=> maximizes `-p^T v_hat[v]`
        q_opp = jnp.einsum("i,vij->vj", p, v_hat)  # [V, 7]
        u_opp = -q_opp
        ev_opp = jnp.sum(y * u_opp, axis=-1, keepdims=True)
        r_opp_next = jnp.maximum(
            jnp.float32(0.0),
            r_opp + jnp.where(masks_opp_v, u_opp - ev_opp, jnp.float32(0.0)),
        )

        # Update y immediately for alternating RM+
        s_opp_next = jnp.sum(r_opp_next, axis=-1, keepdims=True)
        y_next = jnp.where(
            s_opp_next > jnp.float32(1e-12),
            r_opp_next / jnp.maximum(s_opp_next, jnp.float32(1e-12)),
            opp_unif,
        )

        # Own player maximizes `sum_v w_v (v_hat[v] y_next[v])`
        q_own = jnp.einsum("v,vij,vj->i", post_w, v_hat, y_next)  # [7]
        ev_own = jnp.sum(p * q_own)
        r_own_next = jnp.maximum(
            jnp.float32(0.0),
            r_own + jnp.where(mask_own, q_own - ev_own, jnp.float32(0.0)),
        )

        w_t = (t_idx + jnp.int32(1)).astype(jnp.float32)
        return (r_own_next, r_opp_next, sum_p + w_t * p, sum_y + w_t * y_next)

    _, _, final_sum_p, final_sum_y = jax.lax.fori_loop(
        0, int(sweeps), _rm_step, (r_own0, r_opp0, sum_p0, sum_y0)
    )

    p_tot = jnp.sum(final_sum_p)
    avg_p = jnp.where(p_tot > jnp.float32(1e-12), final_sum_p / jnp.maximum(p_tot, jnp.float32(1e-12)), own_unif)
    max_p = jnp.max(avg_p)
    purified_p = jnp.where(mask_own & (avg_p >= jnp.float32(0.05) * max_p), avg_p, jnp.float32(0.0))
    pur_sum = jnp.sum(purified_p)
    purified_p = jnp.where(
        pur_sum > jnp.float32(1e-12),
        purified_p / jnp.maximum(pur_sum, jnp.float32(1e-12)),
        pure_one_hot,
    )

    # Guarantee of purified mixed strategy across worlds
    mix_col_payoffs = jnp.einsum("i,vij->vj", purified_p, v_hat)  # [V, 7]
    mix_world_min = jnp.min(
        jnp.where(masks_opp_v, mix_col_payoffs, jnp.float32(1e9)), axis=-1
    )
    mix_world_min = jnp.where(mix_world_min > jnp.float32(1.0), jnp.float32(0.5), mix_world_min)
    eq_guarantee = jnp.sum(post_w * mix_world_min)

    y_tot = jnp.sum(final_sum_y, axis=-1, keepdims=True)
    avg_y = jnp.where(y_tot > jnp.float32(1e-12), final_sum_y / jnp.maximum(y_tot, jnp.float32(1e-12)), opp_unif)
    eq_q = jnp.einsum("v,vij,vj->i", post_w, v_hat, avg_y)
    action_values = jnp.float32(0.5) * eq_q + jnp.float32(0.5) * pure_guarantees

    use_mixed = eq_guarantee > (best_pure_val + jnp.float32(0.005))
    final_policy = jnp.where(use_mixed, purified_p, pure_one_hot)
    final_val = jnp.where(use_mixed, eq_guarantee, best_pure_val)
    return final_policy, action_values, pure_guarantees, final_val


@partial(
    jax.jit,
    static_argnames=("side", "rollout_turns", "rm_sweeps"),
)
def select_shell_belief_minimax_action_jax(
    state: CompactBattle,
    own_table: MatchTable,
    candidate_tables_foe: MatchTable,
    prior_weights_foe: jax.Array,
    belief_mask_foe: jax.Array,
    side: int,
    policy_seed: jax.Array,
    weights: EvalWeights,
    rollout_turns: int = 0,
    rm_sweeps: int = 48,
    residual_bundle: ResidualBundle | None = None,
    bayesian_eq: bool | jax.Array = True,
) -> tuple[jax.Array, jax.Array]:
    """Select an in-battle action for `side` over the posterior distribution of hidden opponent
    worlds `candidate_tables_foe`.

    When `bayesian_eq=True` (default, ported from `849a3545`):
    - Evaluates each determinized world `v` using its own opponent legal non-dominated mask `mask_opp_v`.
    - Applies Depth-1 minimax contraction (`backed_up_value`) so Depth-1 exploratory noise does not
      pollute cell payoffs.
    - Solves the coupled Bayesian equilibrium `max_p sum_v w_v min_{j in mask_opp_v} (p^T V_v)_j`
      via `solve_bayesian_equilibrium_jax`.
    When `bayesian_eq=False`:
    - Measures the pure "勝手読み" (optimistic uniform-reply) baseline: takes the uniform mean across
      opponent legal columns `j in mask_opp_v` (`row_means_v`), averages over worlds `post_w`, and picks
      the pure `argmax` action without RM+ mixing or Depth-1 minimax contraction (as distinct from
      solving RM+ on the world-averaged matrix `m_bayes`).
    """
    use_beq = jnp.asarray(bayesian_eq, dtype=jnp.bool_)
    raw_w = jnp.where(belief_mask_foe, jnp.maximum(prior_weights_foe, jnp.float32(1e-6)), jnp.float32(0.0))
    w_sum = jnp.sum(raw_w)
    post_w = jnp.where(
        w_sum > jnp.float32(0.0),
        raw_w / jnp.maximum(w_sum, jnp.float32(1e-12)),
        jnp.ones_like(prior_weights_foe) / jnp.maximum(jnp.float32(prior_weights_foe.shape[0]), jnp.float32(1.0)),
    )

    full_mask0 = legal_actions_mask(state, own_table, 0)
    full_mask1 = legal_actions_mask(state, own_table, 1)
    wait0 = full_mask0[7]
    wait1 = full_mask1[7]
    wait_mask7 = jnp.array([True, False, False, False, False, False, False], dtype=jnp.bool_)

    raw_mask0 = jnp.where(wait0, wait_mask7, full_mask0[:7])
    raw_mask1 = jnp.where(wait1, wait_mask7, full_mask1[:7])
    nondom0 = raw_mask0 & (~compact_dominated_mask7(state, own_table, 0))
    nondom1 = raw_mask1 & (~compact_dominated_mask7(state, own_table, 1))
    mask0 = jnp.where(wait0, wait_mask7, jnp.where(jnp.any(nondom0), nondom0, raw_mask0))
    mask1 = jnp.where(wait1, wait_mask7, jnp.where(jnp.any(nondom1), nondom1, raw_mask1))
    fb0 = jnp.argmax(mask0.astype(jnp.int32)).astype(jnp.int32)
    fb1 = jnp.argmax(mask1.astype(jnp.int32)).astype(jnp.int32)

    grid_a0, grid_a1 = jnp.meshgrid(
        jnp.arange(7, dtype=jnp.int32), jnp.arange(7, dtype=jnp.int32), indexing="ij"
    )
    flat_a0 = grid_a0.reshape(-1)
    flat_a1 = grid_a1.reshape(-1)

    next_p_seed, u_draw = prng_random(policy_seed, jnp.int32(10000))
    p_seed_ro0 = jnp.stack(
        [
            next_p_seed[0] ^ jnp.uint32(0xA5A5A5A5),
            next_p_seed[1] + jnp.uint32(0x3C6EF372),
        ],
        axis=0,
    )
    p_seed_ro1 = jnp.stack(
        [
            next_p_seed[0] ^ jnp.uint32(0x5A5A5A5A),
            next_p_seed[1] + jnp.uint32(0x6A09E667),
        ],
        axis=0,
    )
    b_seed_probe0 = jnp.stack(
        [
            state.prng_seed[0] ^ jnp.uint32(0x9E3779B9),
            state.prng_seed[1] + jnp.uint32(0x85EBCA6B),
        ],
        axis=0,
    )
    b_seed_probe1 = jnp.stack(
        [
            state.prng_seed[0] ^ jnp.uint32(0xC2B2AE35),
            state.prng_seed[1] + jnp.uint32(0x27D4EB2F),
        ],
        axis=0,
    )

    def _eval_variant_matrix(cand_tb: MatchTable) -> tuple[jax.Array, jax.Array]:
        c_full0 = legal_actions_mask(state, cand_tb, 0)
        c_full1 = legal_actions_mask(state, cand_tb, 1)
        c_wait0 = c_full0[7]
        c_wait1 = c_full1[7]
        c_raw0 = jnp.where(c_wait0, wait_mask7, c_full0[:7])
        c_raw1 = jnp.where(c_wait1, wait_mask7, c_full1[:7])
        c_nd0 = c_raw0 & (~compact_dominated_mask7(state, cand_tb, 0))
        c_nd1 = c_raw1 & (~compact_dominated_mask7(state, cand_tb, 1))
        m0_v = mask0 if int(side) == 0 else jnp.where(c_wait0, wait_mask7, jnp.where(jnp.any(c_nd0), c_nd0, c_raw0))
        m1_v = jnp.where(c_wait1, wait_mask7, jnp.where(jnp.any(c_nd1), c_nd1, c_raw1)) if int(side) == 0 else mask1
        fb0_v = jnp.argmax(m0_v.astype(jnp.int32)).astype(jnp.int32)
        fb1_v = jnp.argmax(m1_v.astype(jnp.int32)).astype(jnp.int32)

        m0_flat = jnp.broadcast_to(m0_v[:, None], (7, 7)).reshape(-1)
        m1_flat = jnp.broadcast_to(m1_v[None, :], (7, 7)).reshape(-1)
        act0_v = jnp.where(c_wait0, jnp.int32(7), jnp.where(m0_flat, flat_a0, fb0_v))
        act1_v = jnp.where(c_wait1, jnp.int32(7), jnp.where(m1_flat, flat_a1, fb1_v))
        acts_49_v = jnp.stack([act0_v, act1_v], axis=-1)

        def _eval_cell_seed(
            act_pair: jax.Array, b_seed_p: jax.Array, p_seed_ro: jax.Array
        ) -> jax.Array:
            st0 = state._replace(prng_seed=b_seed_p)
            st1 = apply_actions(st0, cand_tb, act_pair)
            v_leaf = eval_compact_leaf(st1, cand_tb, weights, residual_bundle=residual_bundle)
            if rollout_turns <= 0:
                return v_leaf
            ro = _single_table_heavy_rollout(
                pack_battle(st1),
                cand_tb,
                p_seed_ro,
                weights,
                turns=rollout_turns,
                eps=0.06,
                residual_bundle=residual_bundle,
            )
            is_term = st1.outcome != 0
            shrunk = jnp.where(
                is_term,
                v_leaf,
                jnp.float32(0.75) * ro.leaf_values + jnp.float32(0.25) * v_leaf,
            )
            return jnp.where(use_beq, shrunk, ro.leaf_values)

        def _eval_cell(act_pair: jax.Array) -> jax.Array:
            v0 = _eval_cell_seed(act_pair, b_seed_probe0, p_seed_ro0)
            v1 = _eval_cell_seed(act_pair, b_seed_probe1, p_seed_ro1)
            return jnp.float32(0.5) * (v0 + v1)

        vals_49 = jax.vmap(_eval_cell)(acts_49_v).reshape(7, 7)
        mask_opp_v = m1_v if int(side) == 0 else m0_v
        # Orient matrix so row = `side`'s action, col = `1 - side`'s action, payoff = `side`'s score
        mat_oriented = vals_49 if int(side) == 0 else (jnp.float32(1.0) - jnp.transpose(vals_49))
        return mat_oriented, mask_opp_v

    mats_oriented_v, masks_opp_v = jax.vmap(_eval_variant_matrix)(candidate_tables_foe)
    wait_side = wait0 if int(side) == 0 else wait1
    mask_side = mask0 if int(side) == 0 else mask1
    fb_side = fb0 if int(side) == 0 else fb1

    sigma_bayes, _, _, _ = solve_bayesian_equilibrium_jax(
        mats_oriented_v,
        mask_side,
        masks_opp_v,
        post_w,
        sweeps=rm_sweeps,
    )
    # Legacy policy: averages over opponent worlds and opponent actions ("勝手読み" baseline)
    opp_cnt_v = jnp.maximum(jnp.sum(masks_opp_v.astype(jnp.float32), axis=-1, keepdims=True), jnp.float32(1.0))
    row_means_v = jnp.sum(
        jnp.where(masks_opp_v[:, None, :], mats_oriented_v, jnp.float32(0.0)), axis=-1
    ) / opp_cnt_v
    global_row_mean = jnp.sum(post_w[:, None] * row_means_v, axis=0)
    raw_best_legacy = jnp.argmax(
        jnp.where(mask_side, global_row_mean, jnp.float32(-1e9))
    ).astype(jnp.int32)
    best_legacy = jnp.where(jnp.any(mask_side), raw_best_legacy, fb_side)
    sigma_legacy = (jnp.arange(7, dtype=jnp.int32) == best_legacy).astype(jnp.float32)
    sigma_side = jnp.where(use_beq, sigma_bayes, sigma_legacy)

    cdf = jnp.cumsum(sigma_side)
    u01 = (u_draw.astype(jnp.float32) + jnp.float32(0.5)) / jnp.float32(10000.0)
    hit_mask = (cdf >= u01) & mask_side
    sampled = jnp.argmax(hit_mask.astype(jnp.int32)).astype(jnp.int32)
    chosen = jnp.where(jnp.any(hit_mask), sampled, fb_side)
    chosen_final = jnp.where(wait_side, jnp.int32(7), chosen)
    return next_p_seed, chosen_final


@lru_cache(maxsize=32)
def _get_jit_shell_belief_minimax_batch_fn(
    turns: int,
    rollout_turns: int,
    rm_sweeps: int,
    followup_turns: int,
):
    @jax.jit
    def _run_batch(
        pk_b: PackedBattle,
        true_tb_b: MatchTable,
        cand0_b: MatchTable,
        prior0_b: jax.Array,
        cand1_b: MatchTable,
        prior1_b: jax.Array,
        p_seeds_b: jax.Array,
        bmask0_init: jax.Array,
        bmask1_init: jax.Array,
        beq0_flag: jax.Array,
        beq1_flag: jax.Array,
        weights: EvalWeights,
        residual_bundle: ResidualBundle | None,
    ):
        def _single_game(
            pk: PackedBattle,
            t_tb: MatchTable,
            c0: MatchTable,
            pr0: jax.Array,
            c1: MatchTable,
            pr1: jax.Array,
            p_seed: jax.Array,
            bm0: jax.Array,
            bm1: jax.Array,
        ):
            st_init = unpack_battle(pk)

            def _turn_step(carry, _: None):
                st, ps, m0, m1 = carry
                ps, a0 = select_shell_belief_minimax_action_jax(
                    st,
                    t_tb,
                    c0,
                    pr0,
                    m0,
                    0,
                    ps,
                    weights,
                    rollout_turns=rollout_turns,
                    rm_sweeps=rm_sweeps,
                    residual_bundle=residual_bundle,
                    bayesian_eq=beq0_flag,
                )
                ps, a1 = select_shell_belief_minimax_action_jax(
                    st,
                    t_tb,
                    c1,
                    pr1,
                    m1,
                    1,
                    ps,
                    weights,
                    rollout_turns=rollout_turns,
                    rm_sweeps=rm_sweeps,
                    residual_bundle=residual_bundle,
                    bayesian_eq=beq1_flag,
                )
                next_st = apply_actions(st, t_tb, jnp.stack([a0, a1], axis=0))
                next_m0 = update_shell_belief_mask_jax(st, next_st, t_tb, c0, 0, m0)
                next_m1 = update_shell_belief_mask_jax(st, next_st, t_tb, c1, 1, m1)
                return (next_st, ps, next_m0, next_m1), None

            (final_st, next_ps, final_m0, final_m1), _ = jax.lax.scan(
                _turn_step, (st_init, p_seed, bm0, bm1), None, length=turns
            )
            if followup_turns > 0:
                ro_end = _single_table_heavy_rollout(
                    pack_battle(final_st),
                    t_tb,
                    next_ps,
                    weights,
                    turns=followup_turns,
                    eps=0.06,
                    residual_bundle=residual_bundle,
                )
                return ro_end, final_m0, final_m1
            val = eval_compact_leaf(final_st, t_tb, weights, residual_bundle=residual_bundle)
            res = HeavyRolloutResult(
                final_packed=pack_battle(final_st),
                leaf_values=val,
                outcomes=final_st.outcome,
                turns_played=final_st.turn,
            )
            return res, final_m0, final_m1

        if jax.default_backend() == "cpu":
            return jax.lax.map(
                lambda xs: _single_game(*xs),
                (
                    pk_b,
                    true_tb_b,
                    cand0_b,
                    prior0_b,
                    cand1_b,
                    prior1_b,
                    p_seeds_b,
                    bmask0_init,
                    bmask1_init,
                ),
            )
        return jax.vmap(_single_game)(
            pk_b,
            true_tb_b,
            cand0_b,
            prior0_b,
            cand1_b,
            prior1_b,
            p_seeds_b,
            bmask0_init,
            bmask1_init,
        )

    return _run_batch


def simulate_shell_belief_minimax_games_tpu(
    init_packed: PackedBattle,
    true_tables: MatchTable,
    candidate_tables_for_p0: MatchTable,
    prior_weights_for_p0: jax.Array,
    candidate_tables_for_p1: MatchTable,
    prior_weights_for_p1: jax.Array,
    policy_seeds: jax.Array,
    weights: EvalWeights | None = None,
    turns: int = 24,
    rollout_turns: int = 0,
    rm_sweeps: int = 48,
    residual_bundle: ResidualBundle | None = None,
    followup_turns: int = 0,
    bayesian_eq_p0: bool = True,
    bayesian_eq_p1: bool = True,
) -> tuple[HeavyRolloutResult, jax.Array, jax.Array]:
    """Simulate `B` games on TPU where both players maintain exact in-battle Bayesian beliefs
    (`belief_mask0`, `belief_mask1`) over the opponent's hidden shell variants and select moves
    via 1-ply simultaneous-move search (`select_shell_belief_minimax_action_jax`).

    - `bayesian_eq_p0` / `bayesian_eq_p1`: When `True` (default), uses the coupled per-world
      Bayesian equilibrium solver (`solve_bayesian_equilibrium_jax`) with per-world opponent legal
      masks and Depth-1 minimax contraction backup. When `False`, uses the legacy world-averaged
      RM+ baseline, allowing direct CRN head-to-head non-inferiority testing on TPU.
    """
    if weights is None:
        weights = default_eval_weights()

    b_size = int(init_packed.words.shape[0])
    v1_size = int(prior_weights_for_p0.shape[1])
    v0_size = int(prior_weights_for_p1.shape[1])
    init_bmask0 = jnp.ones((b_size, v1_size), dtype=jnp.bool_)
    init_bmask1 = jnp.ones((b_size, v0_size), dtype=jnp.bool_)
    fn = _get_jit_shell_belief_minimax_batch_fn(
        int(turns),
        int(rollout_turns),
        int(rm_sweeps),
        0,
    )
    res, final_m0, final_m1 = fn(
        init_packed,
        true_tables,
        candidate_tables_for_p0,
        prior_weights_for_p0,
        candidate_tables_for_p1,
        prior_weights_for_p1,
        policy_seeds,
        init_bmask0,
        init_bmask1,
        jnp.asarray(bool(bayesian_eq_p0), dtype=jnp.bool_),
        jnp.asarray(bool(bayesian_eq_p1), dtype=jnp.bool_),
        weights,
        residual_bundle,
    )
    if int(followup_turns) > 0:
        f_seeds = policy_seeds ^ jnp.uint32(0x9E3779B9)
        res = _get_jit_multi_table_rollout_fn(int(followup_turns), 0.06)(
            res.final_packed,
            true_tables,
            f_seeds,
            weights,
            residual_bundle,
        )
    return res, final_m0, final_m1


class UnifiedNashPipelineResult(NamedTuple):
    """Result of the 1-run TPU pipeline for:
    (1) New Policy (`bayesian_eq=True`) non-inferiority vs Legacy Policy (`bayesian_eq=False`),
    (2) Sequential Election-Call (`選挙当確速報`) Nash verification & conditional rebuild,
    (3) Team Preview `PairTable` baking for the pool including Nash parties.
    """

    # Gate 1: Non-inferiority election-call vs Legacy policy
    non_inferiority_status: str  # "SUPERIOR_CONFIRMED", "NON_INFERIOR_CONFIRMED", or "INFERIOR_REJECTED"
    new_vs_legacy_mean: float
    new_vs_legacy_lcb: float
    new_vs_legacy_ucb: float
    new_vs_legacy_games: int
    new_vs_legacy_waves: int

    # Gate 2: Nash equilibrium verification & conditional rebuild
    nash_status: str  # "NASH_HELD" or "NASH_REBUILT"
    incumbent_nash_ids: list[str]
    incumbent_nash_weights: np.ndarray  # float32[N]
    final_nash_ids: list[str]
    final_nash_weights: np.ndarray  # float32[N]
    challenger_means: np.ndarray  # float32[N] vs incumbent Nash mixture
    challenger_lcbs: np.ndarray  # float32[N]
    challenger_ucbs: np.ndarray  # float32[N]
    challenger_statuses: list[str]  # per-team election call status
    challenger_games: np.ndarray  # int32[N]
    max_challenger_mean: float
    max_challenger_lcb: float
    early_dropped_count: int
    payoff_matrix: np.ndarray  # float32[N, N]

    # Gate 3: Baked Team Preview PairTables (`(id_a, id_b) -> PairTable dict`)
    baked_pair_tables: dict[str, dict[str, Any]]
    elapsed_secs: float


def _extract_team_sets(team_entry: dict | list) -> list[dict]:
    if isinstance(team_entry, dict) and "sets" in team_entry:
        return list(team_entry["sets"])
    if isinstance(team_entry, list):
        return list(team_entry)
    raise ValueError(f"Expected team dict with 'sets' or list of 6 sets, got {type(team_entry)}")


def _build_shell_variant_rosters_map(
    rosters: list[RosterTable6],
    teams_raw: list[dict],
    max_variants: int = 8,
) -> tuple[list[list[int]], int]:
    """Group pool indices sharing the same 6-(species, level) preview shell (or fallback to
    same lead/core species) so each team `i` has a static-shape list of `V` candidate variant
    indices `[V]` (padded by repeating `i`) for determinized Bayesian belief simulation on TPU.
    """
    n = len(rosters)
    sigs = [preview_signature_sets(_extract_team_sets(t)) for t in teams_raw]
    shell_to_indices: dict[tuple[tuple[str, int], ...], list[int]] = {}
    for idx, sig in enumerate(sigs):
        shell_to_indices.setdefault(sig, []).append(idx)

    raw_groups: list[list[int]] = []
    max_v_found = 1
    for idx, sig in enumerate(sigs):
        grp = list(shell_to_indices[sig])
        if len(grp) == 1 and n > 1:
            # Also include up to (max_variants - 1) nearest species-overlap teams as determinized hypotheses
            own_species = {entry[0] for entry in sig}
            others = sorted(
                (j for j in range(n) if j != idx),
                key=lambda j: -len(own_species & {entry[0] for entry in sigs[j]}),
            )
            grp.extend(others[: max(0, min(max_variants - 1, 3))])
        grp = grp[:max_variants]
        max_v_found = max(max_v_found, len(grp))
        raw_groups.append(grp)

    padded_groups: list[list[int]] = []
    for idx, grp in enumerate(raw_groups):
        padded = list(grp) + [idx] * (max_v_found - len(grp))
        padded_groups.append(padded)
    return padded_groups, max_v_found


def _run_policy_duel_batch_tpu(
    rosters: list[RosterTable6],
    variant_groups: list[list[int]],
    pair_specs: list[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]],
    games_per_spec: int,
    shared_dex: SharedDexTable,
    base_seed_int: int,
    bayesian_eq_a: bool = True,
    bayesian_eq_b: bool = True,
    policy_turns: int = 12,
    rollout_turns: int = 4,
    followup_turns: int = 24,
    rm_sweeps: int = 32,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
) -> np.ndarray:
    """Run paired side-swapped CRN games on TPU using `simulate_shell_belief_minimax_games_tpu`
    for a list of `(team_a_idx, team_b_idx, pick_a_0based, pick_b_0based)` specs.

    Returns `float32[len(pair_specs)]` mean scores for Side A in `[0, 1]`.
    """
    if weights is None:
        weights = default_eval_weights()
    n_specs = len(pair_specs)
    g_per_spec = max(2, int(games_per_spec))
    if g_per_spec % 2 == 1:
        g_per_spec += 1
    if n_specs == 0:
        return np.zeros((0,), dtype=np.float32)

    v_size = len(variant_groups[0])
    pk_list: list[PackedBattle] = []
    true_tb_list: list[MatchTable] = []
    cand0_list: list[MatchTable] = []
    prior0_list: list[jax.Array] = []
    cand1_list: list[MatchTable] = []
    prior1_list: list[jax.Array] = []
    p_seed_list: list[jax.Array] = []
    swapped_flags: list[bool] = []

    # Cache `assemble_3v3_from_rosters_jax` on JIT
    asm_jit = jax.jit(
        lambda ra, rb, pa, pb, bseed: assemble_3v3_from_rosters_jax(
            ra, rb, pa, pb, bseed, shared_dex
        )
    )

    for s_idx, (ia, ib, pa_tup, pb_tup) in enumerate(pair_specs):
        pa_arr = jnp.asarray(pa_tup, dtype=jnp.int32)
        pb_arr = jnp.asarray(pb_tup, dtype=jnp.int32)
        v_a = variant_groups[ia]
        v_b = variant_groups[ib]

        # Unique weights so duplicate padding variants don't double-count
        w_a_np = np.zeros((v_size,), dtype=np.float32)
        seen_a: set[int] = set()
        for k_i, var_idx in enumerate(v_a):
            if var_idx not in seen_a:
                seen_a.add(var_idx)
                w_a_np[k_i] = 1.0
        w_a_np /= max(float(np.sum(w_a_np)), 1.0)

        w_b_np = np.zeros((v_size,), dtype=np.float32)
        seen_b: set[int] = set()
        for k_i, var_idx in enumerate(v_b):
            if var_idx not in seen_b:
                seen_b.add(var_idx)
                w_b_np[k_i] = 1.0
        w_b_np /= max(float(np.sum(w_b_np)), 1.0)

        for p_pair in range(g_per_spec // 2):
            crn_idx = s_idx * (g_per_spec // 2) + p_pair
            b_seed = jnp.array(
                [
                    np.uint32((base_seed_int ^ (crn_idx * 0x9E3779B9)) & 0xFFFFFFFF),
                    np.uint32(((base_seed_int >> 16) + crn_idx * 0x85EBCA6B + 1) & 0xFFFFFFFF),
                ],
                dtype=jnp.uint32,
            )
            p_seed = jnp.array(
                [
                    np.uint32(((base_seed_int + 0xC2B2AE35) ^ (crn_idx * 0x27D4EB2F)) & 0xFFFFFFFF),
                    np.uint32((base_seed_int ^ (crn_idx * 0x165667B1 + 7)) & 0xFFFFFFFF),
                ],
                dtype=jnp.uint32,
            )

            # Orientation 0: A is P0, B is P1
            tb_ab, st_ab = asm_jit(rosters[ia], rosters[ib], pa_arr, pb_arr, b_seed)
            c0_ab = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[asm_jit(rosters[ia], rosters[vb], pa_arr, pb_arr, b_seed)[0] for vb in v_b],
            )
            c1_ab = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[asm_jit(rosters[va], rosters[ib], pa_arr, pb_arr, b_seed)[0] for va in v_a],
            )
            pk_list.append(pack_battle(st_ab))
            true_tb_list.append(tb_ab)
            cand0_list.append(c0_ab)
            prior0_list.append(jnp.asarray(w_b_np, dtype=jnp.float32))
            cand1_list.append(c1_ab)
            prior1_list.append(jnp.asarray(w_a_np, dtype=jnp.float32))
            p_seed_list.append(p_seed)
            swapped_flags.append(False)

            # Orientation 1 (Side-swapped CRN): B is P0, A is P1
            tb_ba, st_ba = asm_jit(rosters[ib], rosters[ia], pb_arr, pa_arr, b_seed)
            c0_ba = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[asm_jit(rosters[ib], rosters[va], pb_arr, pa_arr, b_seed)[0] for va in v_a],
            )
            c1_ba = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0),
                *[asm_jit(rosters[vb], rosters[ia], pb_arr, pa_arr, b_seed)[0] for vb in v_b],
            )
            pk_list.append(pack_battle(st_ba))
            true_tb_list.append(tb_ba)
            cand0_list.append(c0_ba)
            prior0_list.append(jnp.asarray(w_a_np, dtype=np.float32))
            cand1_list.append(c1_ba)
            prior1_list.append(jnp.asarray(w_b_np, dtype=np.float32))
            p_seed_list.append(p_seed)
            swapped_flags.append(True)

    # Split into Normal orientation (A=P0, B=P1) and Swapped orientation (B=P0, A=P1)
    # so `bayesian_eq_a` always applies to Team A and `bayesian_eq_b` always applies to Team B!
    norm_idx = [i for i, sw in enumerate(swapped_flags) if not sw]
    swap_idx = [i for i, sw in enumerate(swapped_flags) if sw]

    def _run_subset(indices: list[int], beq_p0: bool, beq_p1: bool) -> np.ndarray:
        n_actual = len(indices)
        if n_actual == 0:
            return np.zeros((0,), dtype=np.float32)
        # Use a single fixed chunk size so XLA compiles only once across all waves and gates
        chunk_b = (
            2
            if jax.default_backend() == "cpu"
            else (32 if n_actual > 4 else max(2, ((n_actual + 1) // 2) * 2))
        )
        out_chunks: list[np.ndarray] = []
        for start in range(0, n_actual, chunk_b):
            sub = indices[start : start + chunk_b]
            n_sub = len(sub)
            pad_indices = list(sub) + [sub[-1]] * (chunk_b - n_sub)
            pk_b = PackedBattle(words=jnp.stack([pk_list[i].words for i in pad_indices], axis=0))
            tb_b = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *[true_tb_list[i] for i in pad_indices]
            )
            c0_b = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *[cand0_list[i] for i in pad_indices]
            )
            pr0_b = jnp.stack([prior0_list[i] for i in pad_indices], axis=0)
            c1_b = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *[cand1_list[i] for i in pad_indices]
            )
            pr1_b = jnp.stack([prior1_list[i] for i in pad_indices], axis=0)
            ps_b = jnp.stack([p_seed_list[i] for i in pad_indices], axis=0)
            ro_res, _, _ = simulate_shell_belief_minimax_games_tpu(
                pk_b,
                tb_b,
                c0_b,
                pr0_b,
                c1_b,
                pr1_b,
                ps_b,
                weights=weights,
                turns=policy_turns,
                rollout_turns=rollout_turns,
                rm_sweeps=rm_sweeps,
                residual_bundle=residual_bundle,
                followup_turns=followup_turns,
                bayesian_eq_p0=beq_p0,
                bayesian_eq_p1=beq_p1,
            )
            out_chunks.append(np.asarray(ro_res.leaf_values, dtype=np.float32)[:n_sub])
        return np.concatenate(out_chunks, axis=0)

    vals_norm = _run_subset(norm_idx, bayesian_eq_a, bayesian_eq_b)
    vals_swap = 1.0 - _run_subset(swap_idx, bayesian_eq_b, bayesian_eq_a)
    pair_means = 0.5 * (vals_norm + vals_swap)  # shape [n_specs * (g_per_spec // 2)]
    return pair_means.reshape(n_specs, g_per_spec // 2).mean(axis=1).astype(np.float32)


def run_unified_nash_and_bake_pipeline_tpu(
    teams_pool: list[dict],
    incumbent_nash_weights: dict[str, float] | list[float] | np.ndarray | None = None,
    *,
    non_inferiority_waves: tuple[int, ...] = (4, 4, 8),
    non_inferiority_margin: float = 0.02,
    nash_wave_games: tuple[int, ...] = (2, 2, 4, 8),
    early_drop_ucb_threshold: float = 0.495,
    nash_break_lcb_threshold: float = 0.505,
    nash_break_mean_tol: float = 0.010,
    policy_turns: int = 12,
    policy_rollout_turns: int = 2,
    policy_followup_turns: int = 24,
    policy_rm_sweeps: int = 32,
    max_shell_variants: int = 6,
    bake_screen_games: int = 2,
    bake_support_k: int = 3,
    bake_refine_games: int = 6,
    bake_screen_turns: int = 40,
    bake_refine_turns: int = 60,
    bake_out_dir: str | Path | None = None,
    z_value: float = 2.326,
    seed: int = 20260405,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
) -> UnifiedNashPipelineResult:
    """Execute the 3-stage verification & baking workflow on TPU in a single unified run:

    1. **Gate 1 — Non-Inferiority Election-Call (`新方策 vs 旧方策`)**:
       Tests the new coupled per-world Bayesian Equilibrium + Depth-1 minimax contraction policy
       (`bayesian_eq=True`) head-to-head against the legacy world-averaged RM+ policy
       (`bayesian_eq=False`) across the incumbent Nash support pairs with side-swapped CRN waves.
       Calls `SUPERIOR_CONFIRMED` if `lcb > 0.50`, `NON_INFERIOR_CONFIRMED` if
       `lcb >= 0.50 - non_inferiority_margin`, or `INFERIOR_REJECTED` if `ucb < 0.50 - margin`.

    2. **Gate 2 — Sequential Bayesian Election-Call Nash Verification (`選挙当確速報`)**:
       Evaluates all candidate teams in `teams_pool` against the incumbent Nash mixture across
       progressive waves (`nash_wave_games`):
       - **Early Drop (`ELIMINATED_EARLY`)**: Challengers with `ucb < early_drop_ucb_threshold`
          against the incumbent Nash mixture are immediately dropped so zero TPU budget is wasted
         on blown-out challengers.
       - **Close-Challenger Escalation**: Challengers whose credible interval straddles `0.50`
         receive additional waves plus direct in-battle Bayesian Equilibrium policy verification
         (`_run_policy_duel_batch_tpu`).
       - **Keep or Rebuild (`NASH_HELD` / `NASH_REBUILT`)**: If no challenger breaches the
         incumbent Nash equilibrium, the Nash equilibrium is certified (`NASH_HELD`). If any
         challenger breaks Nash (`lcb > nash_break_lcb_threshold` or final mean > `0.50 + nash_break_mean_tol`),
         evaluates the full surviving submatrix (`surviving_challengers U incumbent_support`) and
         rebuilds the Nash equilibrium on TPU (`NASH_REBUILT`).

    3. **Gate 3 — Team Preview `PairTable` Baking (`選出焼き込み`)**:
       Bakes `PairTable` (`SPACE_VERSION = 1`) for the verified/rebuilt Nash support against the
       pool on TPU (refining top support cells with the new Bayesian Equilibrium in-battle policy)
       and optionally writes `<team_a>__vs__<team_b>.json` + `unified_summary.json` to `bake_out_dir`.
    """
    t0 = time.perf_counter()
    if weights is None:
        weights = default_eval_weights()

    shared_dex = build_shared_dex_table()
    n_teams = len(teams_pool)
    if n_teams == 0:
        raise ValueError("teams_pool must contain at least 1 team")

    team_ids = [
        str(t.get("id", f"team_{i}")) if isinstance(t, dict) else f"team_{i}"
        for i, t in enumerate(teams_pool)
    ]
    rosters = [compile_roster6(_extract_team_sets(t)) for t in teams_pool]
    variant_groups, _ = _build_shell_variant_rosters_map(
        rosters, teams_pool, max_variants=max_shell_variants
    )

    # Parse or initialize incumbent Nash weights
    inc_w = np.zeros((n_teams,), dtype=np.float32)
    had_explicit_incumbent = incumbent_nash_weights is not None
    if isinstance(incumbent_nash_weights, dict):
        w_dict = (
            incumbent_nash_weights["weights"]
            if "weights" in incumbent_nash_weights and isinstance(incumbent_nash_weights["weights"], dict)
            else incumbent_nash_weights
        )
        for i, tid in enumerate(team_ids):
            inc_w[i] = float(w_dict.get(tid, 0.0))
    elif incumbent_nash_weights is not None:
        arr = np.asarray(incumbent_nash_weights, dtype=np.float32).reshape(-1)
        inc_w[: min(n_teams, arr.size)] = arr[: min(n_teams, arr.size)]

    if float(np.sum(inc_w)) <= 1e-6:
        # Check if teams_pool has `"nash": true` entries (e.g. `team-pool-v2` `vc-kodoku-nash-a`)
        nash_flagged = [
            i for i, t in enumerate(teams_pool) if isinstance(t, dict) and bool(t.get("nash", False))
        ]
        if nash_flagged:
            had_explicit_incumbent = True
            canonical_apex = {
                "apex-c08-surgical-142": 0.30,
                "red-v10-s1105-r2-s0142": 0.25,
                "apex-c04-bug-3atkstar-boomgen": 0.25,
                "red-v10-s1105-r1-s0148": 0.20,
            }
            for i in nash_flagged:
                inc_w[i] = float(
                    teams_pool[i].get("weight")
                    or canonical_apex.get(team_ids[i], 1.0 / len(nash_flagged))
                )
        else:
            inc_w[0] = 1.0
    inc_w /= float(np.sum(inc_w))
    inc_support_indices = [int(i) for i in np.flatnonzero(inc_w > 1e-3)]
    inc_nash_ids = [team_ids[i] for i in inc_support_indices]

    # Pick representative legal 3-mon lead/core triples for policy head-to-head duels
    rep_picks_0based: list[tuple[int, int, int]] = []
    for r in rosters:
        leg = [int(x) for x in np.flatnonzero(np.asarray(r.legal_mask_60))]
        act0 = leg[0] if leg else 0
        p0 = PREVIEW_ACTIONS_0BASED_NP[act0]
        rep_picks_0based.append((int(p0[0]), int(p0[1]), int(p0[2])))

    # =====================================================================
    # Gate 1: Non-Inferiority Election-Call (New Policy vs Legacy Policy)
    # =====================================================================
    duel_specs: list[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]] = []
    for idx_a in inc_support_indices[: min(3, len(inc_support_indices))]:
        for idx_b in range(min(n_teams, 3)):
            duel_specs.append((idx_a, idx_b, rep_picks_0based[idx_a], rep_picks_0based[idx_b]))
    if not duel_specs:
        duel_specs.append((0, 0, rep_picks_0based[0], rep_picks_0based[0]))

    ni_sum = np.zeros((1,), dtype=np.float64)
    ni_sq = np.zeros((1,), dtype=np.float64)
    ni_cnt = np.zeros((1,), dtype=np.int32)
    ni_waves_used = 0
    ni_status = "NON_INFERIOR_CONFIRMED"
    ni_mean = 0.5
    ni_lcb = 0.5
    ni_ucb = 0.5

    for w_i, g_wave in enumerate(non_inferiority_waves):
        g_w = max(2, int(g_wave))
        if g_w % 2 == 1:
            g_w += 1
        ni_waves_used = w_i + 1
        wave_seed = int(seed) ^ ((w_i + 1) * 0x517CC1B7)
        spec_scores = _run_policy_duel_batch_tpu(
            rosters,
            variant_groups,
            duel_specs,
            games_per_spec=g_w,
            shared_dex=shared_dex,
            base_seed_int=wave_seed,
            bayesian_eq_a=True,
            bayesian_eq_b=False,
            policy_turns=policy_turns,
            rollout_turns=policy_rollout_turns,
            followup_turns=policy_followup_turns,
            rm_sweeps=policy_rm_sweeps,
            weights=weights,
            residual_bundle=residual_bundle,
        )
        n_pairs_spec = g_w // 2
        for sc in spec_scores:
            val_f = float(sc)
            ni_sum[0] += val_f * n_pairs_spec
            ni_sq[0] += (
                (val_f * val_f) * n_pairs_spec
                + val_f * (1.0 - val_f) * max(0, n_pairs_spec - 1)
            )
            ni_cnt[0] += n_pairs_spec

        m_arr, l_arr, u_arr, _ = bayesian_election_call_bounds_np(ni_sum, ni_sq, ni_cnt, z=z_value)
        ni_mean = float(m_arr[0])
        ni_lcb = float(l_arr[0])
        ni_ucb = float(u_arr[0])

        if ni_lcb > 0.50:
            ni_status = "SUPERIOR_CONFIRMED"
            break
        if ni_lcb >= 0.50 - float(non_inferiority_margin):
            ni_status = "NON_INFERIOR_CONFIRMED"
            break
        if ni_ucb < 0.50 - float(non_inferiority_margin):
            ni_status = "INFERIOR_REJECTED"
            break

    if ni_status not in ("SUPERIOR_CONFIRMED", "INFERIOR_REJECTED"):
        if ni_mean >= 0.50 - float(non_inferiority_margin):
            ni_status = "NON_INFERIOR_CONFIRMED"
        else:
            ni_status = "INFERIOR_REJECTED"

    # =====================================================================
    # Gate 2: Sequential Election-Call Nash Verification & Rebuild
    # =====================================================================
    # Maintain persistent 3v3 subgame accumulators `(sum_3v3, sq_3v3, cnt_3v3, open_3v3)`
    # for each `(i, j)` pair (`i in 0..N`, `j in inc_support_indices`) across waves so
    # cell precision accumulates monotonically (`赤子を産湯と一緒に流さない` + `選挙当確速報`).
    legals_per_team: list[list[int]] = []
    for r in rosters:
        la = [int(idx) for idx in np.flatnonzero(np.asarray(r.legal_mask_60))]
        legals_per_team.append(la if la else [0])

    sub_sum: dict[tuple[int, int], np.ndarray] = {}
    sub_sq: dict[tuple[int, int], np.ndarray] = {}
    sub_cnt: dict[tuple[int, int], np.ndarray] = {}
    sub_open: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_teams):
        for j in inc_support_indices:
            if i == j:
                continue
            na, nb = len(legals_per_team[i]), len(legals_per_team[j])
            sub_sum[(i, j)] = np.zeros((na, nb), dtype=np.float64)
            sub_sq[(i, j)] = np.zeros((na, nb), dtype=np.float64)
            sub_cnt[(i, j)] = np.zeros((na, nb), dtype=np.int32)
            sub_open[(i, j)] = np.ones((na, nb), dtype=bool)

    preview_val_mat = np.full((n_teams, n_teams), 0.5, dtype=np.float32)
    pair_val_mat = np.full((n_teams, n_teams), 0.5, dtype=np.float32)
    pair_pol_sum = np.zeros((n_teams, n_teams), dtype=np.float64)
    pair_pol_cnt = np.zeros((n_teams, n_teams), dtype=np.int32)
    pair_evaluated = np.zeros((n_teams, n_teams), dtype=bool)
    for i in range(n_teams):
        pair_evaluated[i, i] = True

    # Group incumbent support by shared Team Preview shell (`preview_signature_sets`)
    # so challengers cannot clairvoyantly distinguish variants of the same shell at Team Preview
    inc_shell_groups: dict[tuple[tuple[str, int, str, bool], ...], list[int]] = {}
    for j in inc_support_indices:
        sig_j = preview_signature_sets(_extract_team_sets(teams_pool[j]))
        inc_shell_groups.setdefault(sig_j, []).append(j)

    challenger_active = np.ones((n_teams,), dtype=bool)
    challenger_statuses = ["INCUMBENT" if inc_w[i] > 1e-3 else "OPEN" for i in range(n_teams)]
    challenger_means = np.full((n_teams,), 0.5, dtype=np.float32)
    challenger_lcbs = np.zeros((n_teams,), dtype=np.float32)
    challenger_ucbs = np.ones((n_teams,), dtype=np.float32)
    challenger_games = np.zeros((n_teams,), dtype=np.int32)

    nash_broken = not had_explicit_incumbent

    for w_idx, g_wave in enumerate(nash_wave_games):
        g_w = max(2, int(g_wave))
        if g_w % 2 == 1:
            g_w += 1
        active_rows = [i for i in range(n_teams) if challenger_active[i]]
        if not active_rows:
            break

        # Step 2a: Run wave `w_idx` on open 3v3 cells for all active `(i, j)` pairs
        for i in active_rows:
            la = legals_per_team[i]
            for j in inc_support_indices:
                if i == j:
                    continue
                lb = legals_per_team[j]
                open_m = sub_open[(i, j)]
                open_coords = np.argwhere(open_m)
                if open_coords.shape[0] > 0:
                    grid_a, grid_b = np.meshgrid(
                        np.asarray(la, dtype=np.int32),
                        np.asarray(lb, dtype=np.int32),
                        indexing="ij",
                    )
                    ia_idx = open_coords[:, 0]
                    ib_idx = open_coords[:, 1]
                    act_a = grid_a[ia_idx, ib_idx]
                    act_b = grid_b[ia_idx, ib_idx]
                    picks_a_w = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[act_a], dtype=jnp.int32)
                    picks_b_w = jnp.asarray(PREVIEW_ACTIONS_0BASED_NP[act_b], dtype=jnp.int32)

                    wave_seed = int(seed) ^ ((w_idx + 1) * 0x9E3779B9) ^ (i * 131 + j * 17)
                    scores_w = np.asarray(
                        evaluate_3v3_picks_batch_tpu(
                            rosters[i],
                            rosters[j],
                            picks_a_w,
                            picks_b_w,
                            shared_dex,
                            games_per_pair=g_w,
                            turns=bake_screen_turns,
                            eps=0.2,
                            base_seed_int=wave_seed,
                            weights=weights,
                            residual_bundle=residual_bundle,
                        ),
                        dtype=np.float64,
                    )
                    n_pairs_w = float(g_w // 2)
                    sub_sum[(i, j)][ia_idx, ib_idx] += scores_w * n_pairs_w
                    sub_sq[(i, j)][ia_idx, ib_idx] += (
                        (scores_w * scores_w) * n_pairs_w
                        + scores_w * (1.0 - scores_w) * max(0.0, n_pairs_w - 1.0)
                    )
                    sub_cnt[(i, j)][ia_idx, ib_idx] += int(n_pairs_w)

                    # Prune provably dominated or tightly resolved 3v3 cells for subsequent waves
                    m_ij, l_ij, u_ij, r_ij = bayesian_election_call_bounds_np(
                        sub_sum[(i, j)], sub_sq[(i, j)], sub_cnt[(i, j)], z=z_value
                    )
                    sol_l = solve_rect_rm_plus_jax(jnp.asarray(l_ij, dtype=jnp.float32), sweeps=400)
                    sol_u = solve_rect_rm_plus_jax(jnp.asarray(u_ij, dtype=jnp.float32), sweeps=400)
                    lcb_ij = float(sol_l.value)
                    ucb_ij = float(sol_u.value)
                    ev_row_u = np.asarray(sol_u.ev_row, dtype=np.float32)
                    ev_col_l = np.asarray(sol_l.ev_col, dtype=np.float32)
                    dom_rows = (ev_row_u + 0.02) < lcb_ij
                    dom_cols = (ev_col_l - 0.02) > ucb_ij
                    res_cells = (sub_cnt[(i, j)] >= 2) & (r_ij <= 0.065)
                    sub_open[(i, j)] = open_m & (~res_cells) & (~(dom_rows[:, None] & dom_cols[None, :]))
                challenger_games[i] += g_w

        # Step 2b: Solve Bayesian Shell Preview equilibrium (`solve_bayesian_shell_preview_nash_np`)
        # for each active challenger `i` against each incumbent shell group, and queue in-battle
        # Bayesian Equilibrium policy duels on the resulting equilibrium 3v3 lineups!
        wave_pol_specs: list[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]] = []
        for i in active_rows:
            la = legals_per_team[i]
            for _, grp_js in inc_shell_groups.items():
                valid_js = [j for j in grp_js if j != i]
                if not valid_js:
                    continue
                mats_1xm = np.stack(
                    [
                        bayesian_election_call_bounds_np(
                            sub_sum[(i, j)], sub_sq[(i, j)], sub_cnt[(i, j)], z=z_value
                        )[0]
                        for j in valid_js
                    ],
                    axis=0,
                )[None, ...]  # shape [1, M, na, nb]
                beta_grp = np.array([inc_w[j] for j in valid_js], dtype=np.float32)
                beta_grp /= max(float(np.sum(beta_grp)), 1e-9)
                _, _, v_per_v, x_star_grp, y_star_grp = solve_bayesian_shell_preview_nash_np(
                    mats_1xm, alpha=np.array([1.0], dtype=np.float32), beta=beta_grp, sweeps=400
                )
                top_a_local = int(np.argmax(x_star_grp[0]))
                act_a_top = la[top_a_local]
                pa_top = (
                    int(PREVIEW_ACTIONS_0BASED_NP[act_a_top, 0]),
                    int(PREVIEW_ACTIONS_0BASED_NP[act_a_top, 1]),
                    int(PREVIEW_ACTIONS_0BASED_NP[act_a_top, 2]),
                )
                for m_k, j in enumerate(valid_js):
                    preview_val_mat[i, j] = float(v_per_v[m_k])
                    preview_val_mat[j, i] = float(1.0 - v_per_v[m_k])
                    pair_evaluated[i, j] = True
                    pair_evaluated[j, i] = True
                    top_b_local = int(np.argmax(y_star_grp[m_k]))
                    act_b_top = legals_per_team[j][top_b_local]
                    pb_top = (
                        int(PREVIEW_ACTIONS_0BASED_NP[act_b_top, 0]),
                        int(PREVIEW_ACTIONS_0BASED_NP[act_b_top, 1]),
                        int(PREVIEW_ACTIONS_0BASED_NP[act_b_top, 2]),
                    )
                    wave_pol_specs.append((i, j, pa_top, pb_top))

        if wave_pol_specs:
            pol_scores_wave = _run_policy_duel_batch_tpu(
                rosters,
                variant_groups,
                wave_pol_specs,
                games_per_spec=max(2, g_w // 2),
                shared_dex=shared_dex,
                base_seed_int=int(seed) ^ ((w_idx + 1) * 0x27D4EB2F),
                bayesian_eq_a=True,
                bayesian_eq_b=True,
                policy_turns=policy_turns,
                rollout_turns=policy_rollout_turns,
                followup_turns=policy_followup_turns,
                rm_sweeps=policy_rm_sweeps,
                weights=weights,
                residual_bundle=residual_bundle,
            )
            n_pol_pairs = max(1, g_w // 4)
            for idx_p, (i, j, _, _) in enumerate(wave_pol_specs):
                sc_pol = float(pol_scores_wave[idx_p])
                pair_pol_sum[i, j] += sc_pol * n_pol_pairs
                pair_pol_cnt[i, j] += n_pol_pairs

        # Step 2c: Compute each active team's blended payoff and credible interval against incumbent Nash
        for i in active_rows:
            tot_mean_i = 0.0
            tot_var_i = 0.0
            for j in inc_support_indices:
                w_j = float(inc_w[j])
                if i == j:
                    tot_mean_i += w_j * 0.50
                    continue
                v_prev_ij = float(preview_val_mat[i, j])
                n_pol_ij = int(pair_pol_cnt[i, j])
                if n_pol_ij > 0:
                    pol_mean_ij = float(pair_pol_sum[i, j]) / n_pol_ij
                    blended_ij = 0.70 * v_prev_ij + 0.30 * pol_mean_ij
                else:
                    blended_ij = v_prev_ij
                pair_val_mat[i, j] = blended_ij
                pair_val_mat[j, i] = 1.0 - blended_ij
                tot_mean_i += w_j * blended_ij

                # Effective independent subgame & policy sample size for (i, j)
                n_sub_pairs = int(np.sum(sub_cnt[(i, j)]))
                eff_n_ij = max(4.0, float(n_sub_pairs) / 48.0 + float(n_pol_ij) * 2.0)
                se_ij_sq = (blended_ij * (1.0 - blended_ij) + 0.02) / eff_n_ij
                tot_var_i += (w_j * w_j) * se_ij_sq

            rad_i = float(z_value) * float(np.sqrt(max(tot_var_i, 1e-9)))
            challenger_means[i] = float(np.clip(tot_mean_i, 0.0, 1.0))
            challenger_lcbs[i] = float(np.clip(tot_mean_i - rad_i, 0.0, 1.0))
            challenger_ucbs[i] = float(np.clip(tot_mean_i + rad_i, 0.0, 1.0))

            if inc_w[i] > 1e-3:
                challenger_statuses[i] = "INCUMBENT"
                continue
            # Early-stop clear losers (`敗退当確`); keep close challengers & potential Nash-breakers
            # running through the final wave so Nash-break decisions have full statistical confidence!
            if challenger_ucbs[i] < float(early_drop_ucb_threshold):
                challenger_statuses[i] = "ELIMINATED_EARLY"
                challenger_active[i] = False

        # Stop early if all non-incumbent challengers have been eliminated
        if all(
            (not challenger_active[i]) or (inc_w[i] > 1e-3)
            for i in range(n_teams)
        ):
            break

    # Final check on surviving challengers after full wave verification
    for i in range(n_teams):
        if inc_w[i] <= 1e-3 and challenger_lcbs[i] > float(nash_break_lcb_threshold):
            nash_broken = True
            challenger_statuses[i] = "NASH_BREAKER_CALLED"
        elif (
            inc_w[i] <= 1e-3
            and challenger_means[i] > 0.50 + float(nash_break_mean_tol)
            and challenger_lcbs[i] > 0.495
        ):
            nash_broken = True
            if challenger_statuses[i] == "OPEN":
                challenger_statuses[i] = "NASH_BREAKER_CALLED"
        elif inc_w[i] <= 1e-3 and challenger_statuses[i] == "OPEN":
            challenger_statuses[i] = "CONTAINED"

    # Enforce exact skew-symmetry on evaluated pairs
    payoff_mat = 0.5 * (pair_val_mat + (1.0 - pair_val_mat.T))
    np.fill_diagonal(payoff_mat, 0.5)

    if nash_broken and n_teams > 1:
        # Evaluate remaining unvisited pairs between candidate Nash support (`surviving`)
        # and ALL `n_teams` columns in the pool so the rebuilt Nash equilibrium is guaranteed
        # across the entire pool (not merely the `surviving x surviving` subgame)!
        surviving = sorted(
            set(inc_support_indices)
            | {
                i
                for i in range(n_teams)
                if challenger_statuses[i] == "NASH_BREAKER_CALLED"
                or (not had_explicit_incumbent and challenger_statuses[i] != "ELIMINATED_EARLY")
            }
        )
        for i in surviving:
            for j in range(n_teams):
                if i == j:
                    continue
                if not pair_evaluated[i, j]:
                    res_ij = evaluate_6v6_bayesian_election_call_tpu(
                        rosters[i],
                        rosters[j],
                        shared_dex,
                        wave_games=(max(2, nash_wave_games[0]),),
                        turns=bake_screen_turns,
                        seed=int(seed) ^ (i * 313 + j * 37),
                        z_value=z_value,
                        weights=weights,
                        residual_bundle=residual_bundle,
                    )
                    v_ij = float(res_ij.value)
                    payoff_mat[i, j] = v_ij
                    payoff_mat[j, i] = 1.0 - v_ij
                    pair_evaluated[i, j] = True
                    pair_evaluated[j, i] = True

        # Solve rectangular minimax game: rows = `surviving` (`incumbent U breakers`), cols = all `n_teams`
        rect_mat = jnp.asarray(payoff_mat[surviving, :], dtype=jnp.float32)
        sol_new = solve_rect_rm_plus_jax(rect_mat, sweeps=2000)
        final_w = np.zeros((n_teams,), dtype=np.float32)
        sub_w = np.asarray(sol_new.p_a, dtype=np.float32)
        for k_i, orig_i in enumerate(surviving):
            final_w[orig_i] = float(sub_w[k_i])
        final_w = np.where(final_w >= 0.01, final_w, 0.0)
        final_w /= max(float(np.sum(final_w)), 1e-9)
        nash_status = "NASH_REBUILT"
    else:
        final_w = inc_w.copy()
        nash_status = "NASH_HELD"

    final_support_indices = [int(i) for i in np.flatnonzero(final_w > 1e-3)]
    final_nash_ids = [team_ids[i] for i in final_support_indices]

    # =====================================================================
    # Gate 3: Bake Team Preview `PairTable`s for Pool Including Nash Parties
    # =====================================================================
    baked_tables: dict[str, dict[str, Any]] = {}
    out_dir_path = Path(bake_out_dir) if bake_out_dir is not None else None
    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    for i in final_support_indices:
        for j in range(n_teams):
            pair_key = f"{team_ids[i]}__vs__{team_ids[j]}"
            pair_res = core_nash_6v6_pair_tpu(
                rosters[i],
                rosters[j],
                shared_dex,
                screen_games=bake_screen_games,
                support_k=bake_support_k,
                refine_games=bake_refine_games,
                screen_turns=bake_screen_turns,
                refine_turns=bake_refine_turns,
                seed=int(seed) ^ (i * 997 + j * 101),
                weights=weights,
                residual_bundle=residual_bundle,
            )
            pt_dict = export_pair_table_dict(
                pair_res,
                team_a=team_ids[i],
                team_b=team_ids[j],
                screen_games=bake_screen_games,
                support_k=bake_support_k,
                refine_games=bake_refine_games,
                screen_turns=bake_screen_turns,
                refine_turns=bake_refine_turns,
                seed=int(seed) ^ (i * 997 + j * 101),
            )
            baked_tables[pair_key] = pt_dict
            if out_dir_path is not None:
                with (out_dir_path / f"{pair_key}.json").open("w", encoding="utf-8") as f:
                    json.dump(pt_dict, f, indent=2)

    non_inc_mask = inc_w <= 1e-3
    max_chal_mean = (
        float(np.max(challenger_means[non_inc_mask]))
        if np.any(non_inc_mask)
        else float(np.max(challenger_means))
    )
    max_chal_lcb = (
        float(np.max(challenger_lcbs[non_inc_mask]))
        if np.any(non_inc_mask)
        else float(np.max(challenger_lcbs))
    )
    early_dropped = int(sum(1 for s in challenger_statuses if s == "ELIMINATED_EARLY"))
    elapsed = time.perf_counter() - t0

    if out_dir_path is not None:
        summary_payload = {
            "non_inferiority": {
                "status": ni_status,
                "candidate_policy": "coupled_per_world_bayesian_equilibrium_and_depth1_minimax",
                "baseline_policy": "opponent_uniform_row_mean_pure_argmax_katteyomi",
                "mean": round(ni_mean, 6),
                "lcb": round(ni_lcb, 6),
                "ucb": round(ni_ucb, 6),
                "games": int(ni_cnt[0] * 2),
                "waves": int(ni_waves_used),
            },
            "nash_verification": {
                "status": nash_status,
                "incumbent_nash_ids": inc_nash_ids,
                "incumbent_weights": [round(float(x), 6) for x in inc_w.tolist()],
                "final_nash_ids": final_nash_ids,
                "final_weights": [round(float(x), 6) for x in final_w.tolist()],
                "max_challenger_mean": round(max_chal_mean, 6),
                "max_challenger_lcb": round(max_chal_lcb, 6),
                "early_dropped_count": early_dropped,
                "challengers": [
                    {
                        "team_id": team_ids[i],
                        "status": challenger_statuses[i],
                        "mean": round(float(challenger_means[i]), 6),
                        "lcb": round(float(challenger_lcbs[i]), 6),
                        "ucb": round(float(challenger_ucbs[i]), 6),
                        "games": int(challenger_games[i]),
                        "incumbent_weight": round(float(inc_w[i]), 6),
                        "final_weight": round(float(final_w[i]), 6),
                    }
                    for i in range(n_teams)
                ],
            },
            "baked_pair_tables": list(baked_tables.keys()),
            "elapsed_secs": round(elapsed, 4),
        }
        with (out_dir_path / "unified_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

    return UnifiedNashPipelineResult(
        non_inferiority_status=ni_status,
        new_vs_legacy_mean=ni_mean,
        new_vs_legacy_lcb=ni_lcb,
        new_vs_legacy_ucb=ni_ucb,
        new_vs_legacy_games=int(ni_cnt[0] * 2),
        new_vs_legacy_waves=ni_waves_used,
        nash_status=nash_status,
        incumbent_nash_ids=inc_nash_ids,
        incumbent_nash_weights=inc_w,
        final_nash_ids=final_nash_ids,
        final_nash_weights=final_w,
        challenger_means=challenger_means,
        challenger_lcbs=challenger_lcbs,
        challenger_ucbs=challenger_ucbs,
        challenger_statuses=challenger_statuses,
        challenger_games=challenger_games,
        max_challenger_mean=max_chal_mean,
        max_challenger_lcb=max_chal_lcb,
        early_dropped_count=early_dropped,
        payoff_matrix=payoff_mat.astype(np.float32),
        baked_pair_tables=baked_tables,
        elapsed_secs=elapsed,
    )


# =========================================================================
# 14. Two-Stage Bayesian-Equilibrium Kodoku Population Evaluation & Evolution
# =========================================================================

_BUNDLED_LEARNSETS_PATH = Path(__file__).resolve().parent / "data" / "learnsets-gen2.json"
_REPO_LEARNSETS_PATH = Path(__file__).resolve().parents[3] / "data" / "learnsets-gen2.json"
_REF_LEARNSETS_PATH = Path("/Users/puniu/nc2000-engine/data/learnsets-gen2.json")
DEFAULT_LEARNSETS_PATH = (
    _BUNDLED_LEARNSETS_PATH
    if _BUNDLED_LEARNSETS_PATH.exists()
    else (_REPO_LEARNSETS_PATH if _REPO_LEARNSETS_PATH.exists() else _REF_LEARNSETS_PATH)
)

_FIXED_PHYSICAL_MOVES = frozenset({"counter", "seismictoss", "superfang", "bide"})
_PHYSICAL_ACCESS_MOVES = frozenset({"metronome", "mimic", "mirrormove"})
_EXCLUDED_KODOKU_MOVES = frozenset(
    {
        "beatup",
        "fissure",
        "guillotine",
        "horndrill",
        "metronome",
        "mimic",
        "mirrormove",
        "transform",
        "splash",
        "teleport",
        "struggle",
        "sketch",
        "snore",
        "dreameater",
        "nightmare",
        "bide",
        "rage",
        "detect",
    }
)

_HP_TYPE_DV_PAIRS: dict[str, tuple[int, int]] = {
    "Fighting": (12, 12),
    "Flying": (12, 13),
    "Poison": (12, 14),
    "Ground": (12, 15),
    "Rock": (13, 12),
    "Bug": (13, 13),
    "Ghost": (13, 14),
    "Steel": (13, 15),
    "Fire": (14, 12),
    "Water": (14, 13),
    "Grass": (14, 14),
    "Electric": (14, 15),
    "Psychic": (15, 12),
    "Ice": (15, 13),
    "Dragon": (15, 14),
    "Dark": (15, 15),
}

# Curated non-dominated competitive move pools per species for Kodoku mutation
_CURATED_SPECIES_MOVES: dict[str, tuple[str, ...]] = {
    "snorlax": (
        "Double-Edge",
        "Body Slam",
        "Return",
        "Curse",
        "Earthquake",
        "Rest",
        "Sleep Talk",
        "Self-Destruct",
        "Fire Blast",
        "Lovely Kiss",
        "Thunder",
        "Ice Beam",
        "Shadow Ball",
        "Counter",
    ),
    "zapdos": (
        "Thunderbolt",
        "Thunder",
        "Drill Peck",
        "Hidden Power Ice",
        "Hidden Power Water",
        "Rest",
        "Sleep Talk",
        "Whirlwind",
        "Reflect",
        "Light Screen",
        "Thunder Wave",
    ),
    "cloyster": (
        "Spikes",
        "Ice Beam",
        "Surf",
        "Hydro Pump",
        "Explosion",
        "Reflect",
        "Rain Dance",
        "Toxic",
        "Rapid Spin",
        "Curse",
        "Rest",
    ),
    "marowak": (
        "Earthquake",
        "Rock Slide",
        "Swords Dance",
        "Hidden Power Bug",
        "Hidden Power Flying",
        "Fire Blast",
        "Rest",
        "Sleep Talk",
        "Double-Edge",
        "Body Slam",
        "Counter",
    ),
    "gengar": (
        "Thunderbolt",
        "Ice Punch",
        "Fire Punch",
        "Explosion",
        "Perish Song",
        "Mean Look",
        "Destiny Bond",
        "Shadow Ball",
        "Dynamic Punch",
        "Zap Cannon",
        "Hypnosis",
        "Confuse Ray",
        "Psychic",
        "Thief",
        "Counter",
    ),
    "starmie": (
        "Surf",
        "Hydro Pump",
        "Psychic",
        "Ice Beam",
        "Thunderbolt",
        "Recover",
        "Rapid Spin",
        "Thunder Wave",
        "Reflect",
        "Light Screen",
        "Confuse Ray",
    ),
    "heracross": (
        "Megahorn",
        "Earthquake",
        "Curse",
        "Rest",
        "Sleep Talk",
        "Reversal",
        "Endure",
        "Counter",
        "Hidden Power Rock",
        "Seismic Toss",
    ),
    "raikou": (
        "Thunderbolt",
        "Thunder",
        "Hidden Power Ice",
        "Hidden Power Water",
        "Crunch",
        "Rest",
        "Sleep Talk",
        "Roar",
        "Reflect",
    ),
    "suicune": (
        "Surf",
        "Hydro Pump",
        "Ice Beam",
        "Rest",
        "Sleep Talk",
        "Roar",
        "Toxic",
        "Mirror Coat",
        "Reflect",
    ),
    "exeggutor": (
        "Psychic",
        "Giga Drain",
        "Explosion",
        "Sleep Powder",
        "Stun Spore",
        "Leech Seed",
        "Hidden Power Fire",
        "Moonlight",
        "Reflect",
    ),
    "skarmory": (
        "Drill Peck",
        "Whirlwind",
        "Rest",
        "Sleep Talk",
        "Curse",
        "Toxic",
        "Thief",
        "Steel Wing",
    ),
    "tyranitar": (
        "Rock Slide",
        "Earthquake",
        "Pursuit",
        "Crunch",
        "Fire Blast",
        "Ice Beam",
        "Thunderbolt",
        "Roar",
        "Curse",
        "Rest",
        "Dynamic Punch",
    ),
    "steelix": (
        "Earthquake",
        "Explosion",
        "Roar",
        "Curse",
        "Rock Slide",
        "Body Slam",
        "Iron Tail",
        "Rest",
    ),
    "miltank": (
        "Body Slam",
        "Earthquake",
        "Milk Drink",
        "Heal Bell",
        "Curse",
        "Growl",
        "Counter",
        "Shadow Ball",
    ),
    "blissey": (
        "Soft-Boiled",
        "Heal Bell",
        "Ice Beam",
        "Thunderbolt",
        "Flamethrower",
        "Seismic Toss",
        "Toxic",
        "Counter",
        "Light Screen",
        "Sing",
    ),
    "vaporeon": (
        "Surf",
        "Hydro Pump",
        "Ice Beam",
        "Growth",
        "Acid Armor",
        "Baton Pass",
        "Rest",
        "Sleep Talk",
        "Haze",
        "Roar",
    ),
    "jolteon": (
        "Thunderbolt",
        "Thunder",
        "Hidden Power Ice",
        "Hidden Power Water",
        "Agility",
        "Growth",
        "Baton Pass",
        "Substitute",
        "Thunder Wave",
        "Roar",
    ),
    "umbreon": (
        "Pursuit",
        "Charm",
        "Mean Look",
        "Baton Pass",
        "Moonlight",
        "Rest",
        "Toxic",
        "Confuse Ray",
        "Zap Cannon",
    ),
    "machamp": (
        "Cross Chop",
        "Rock Slide",
        "Earthquake",
        "Fire Blast",
        "Curse",
        "Rest",
        "Sleep Talk",
        "Hidden Power Bug",
        "Counter",
    ),
    "nidoking": (
        "Earthquake",
        "Lovely Kiss",
        "Ice Beam",
        "Thunderbolt",
        "Fire Blast",
        "Counter",
        "Rock Slide",
        "Morning Sun",
    ),
    "jynx": (
        "Lovely Kiss",
        "Ice Beam",
        "Psychic",
        "Substitute",
        "Perish Song",
        "Mean Look",
        "Thief",
        "Nightmare",
    ),
    "alakazam": (
        "Psychic",
        "Fire Punch",
        "Ice Punch",
        "Thunder Punch",
        "Recover",
        "Encore",
        "Thunder Wave",
        "Reflect",
        "Dynamic Punch",
    ),
    "misdreavus": (
        "Mean Look",
        "Perish Song",
        "Protect",
        "Thunderbolt",
        "Thunder",
        "Confuse Ray",
        "Destiny Bond",
        "Pain Split",
        "Hypnosis",
    ),
    "forretress": (
        "Spikes",
        "Explosion",
        "Rapid Spin",
        "Hidden Power Bug",
        "Toxic",
        "Reflect",
        "Rest",
    ),
    "porygon2": (
        "Double-Edge",
        "Return",
        "Ice Beam",
        "Thunderbolt",
        "Recover",
        "Thunder Wave",
        "Curse",
    ),
    "tauros": (
        "Double-Edge",
        "Return",
        "Earthquake",
        "Fire Blast",
        "Hyper Beam",
        "Icy Wind",
        "Rest",
        "Sleep Talk",
    ),
    "rhydon": (
        "Earthquake",
        "Rock Slide",
        "Megahorn",
        "Curse",
        "Roar",
        "Rest",
        "Sleep Talk",
        "Counter",
    ),
    "quagsire": (
        "Earthquake",
        "Belly Drum",
        "Sludge Bomb",
        "Ancient Power",
        "Hidden Power Rock",
        "Rest",
        "Surf",
    ),
    "tentacruel": (
        "Swords Dance",
        "Sludge Bomb",
        "Hydro Pump",
        "Surf",
        "Ice Beam",
        "Substitute",
        "Rapid Spin",
        "Mirror Coat",
    ),
    "houndoom": (
        "Fire Blast",
        "Flamethrower",
        "Crunch",
        "Pursuit",
        "Sunny Day",
        "Solar Beam",
        "Counter",
        "Roar",
    ),
    "espeon": (
        "Psychic",
        "Growth",
        "Morning Sun",
        "Baton Pass",
        "Bite",
        "Hidden Power Water",
        "Reflect",
    ),
    "kingdra": (
        "Surf",
        "Hydro Pump",
        "Ice Beam",
        "Dragon Breath",
        "Rest",
        "Sleep Talk",
        "Double-Edge",
    ),
}

_CURATED_COMPETITIVE_ITEMS: tuple[str, ...] = (
    "Leftovers",
    "Mint Berry",
    "Miracle Berry",
    "Gold Berry",
    "Scope Lens",
    "Quick Claw",
    "BrightPowder",
    "Focus Band",
    "Mystery Berry",
    "Pink Bow",
    "Charcoal",
    "Magnet",
    "Mystic Water",
    "Never-Melt Ice",
    "Soft Sand",
    "Hard Stone",
    "Twisted Spoon",
    "BlackGlasses",
)


@lru_cache(maxsize=4)
def load_learnsets(learnsets_path: str | Path = DEFAULT_LEARNSETS_PATH) -> dict[str, dict[str, Any]]:
    """Load `learnsets-gen2.json` mapping `species_id -> {moves: [...], ...}`."""
    raw = json.loads(Path(learnsets_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "learnsets" in raw:
        return raw["learnsets"]
    return raw


def _uses_attack_stat(pset: dict[str, Any], dex_raw: dict[str, Any]) -> bool:
    """Return True if `pset` has any move that benefits from the Attack stat (`usesAttack` in `normalize-training.js`)."""
    moves_dict = dex_raw["moves"]
    for m in pset.get("moves") or []:
        mkey = toid(m)
        if mkey.startswith("hiddenpower") and mkey not in moves_dict:
            mkey = "hiddenpower"
        md = moves_dict.get(mkey)
        if md is None:
            raise ValueError(f"Unknown move in normalize_training_set: {m!r}")
        if mkey in _PHYSICAL_ACCESS_MOVES:
            return True
        if md.get("category") == "Physical" and mkey not in _FIXED_PHYSICAL_MOVES:
            return True
    return False


def normalize_training_set(
    pset: dict[str, Any],
    dex_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a single Gen 2 Pokemon set under `training-v3` (`tools/normalize-training.js`).

    Preserves public `gender`, `shiny` predicate, `Hidden Power` type, and Unown form while:
    - Maximizing HP, Def, SpA, SpD, Spe DVs & EVs (`255`)
    - Maximizing Hidden Power base power (`70`)
    - Maximizing Attack DV/EV (`255`) when `usesAttack` is True, or minimizing Attack DV and setting
      `evs.atk = 0` when `usesAttack` is False (reducing confusion/Foul Play self-hit damage)
    - Setting `happiness = 255` for `Return` and `happiness = 0` for `Frustration`.
    """
    if dex_raw is None:
        dex_raw = load_dex().raw

    out = json.loads(json.dumps(pset))
    sp_id = toid(out["species"])
    sp = dex_raw["species"].get(sp_id)
    if sp is None:
        raise ValueError(f"Unknown species in normalize_training_set: {out['species']!r}")

    out.setdefault("name", out["species"])
    out.setdefault("ability", "No Ability")
    out.setdefault("nature", "Serious")
    out.setdefault("level", 50)
    out.setdefault("happiness", 255)

    raw_ivs = out.get("ivs") or {}
    raw_evs = out.get("evs") or {}
    ivs_map = {k: int(raw_ivs.get(k, 31)) for k in _STAT_KEYS}
    evs_map = {k: int(raw_evs.get(k, 255)) for k in _STAT_KEYS}

    # If a move specifies an explicit `"Hidden Power <Type>"`, align `(atk, def)` DVs with that type first
    explicit_hp_type: int | None = None
    for m in out.get("moves") or []:
        m_str = str(m).strip()
        if m_str.lower().startswith("hidden power ") and len(m_str) > 13:
            t_name = m_str[13:].strip().capitalize()
            if t_name in _HP_TYPES:
                explicit_hp_type = _HP_TYPES.index(t_name)
                dv_a, dv_d = _HP_TYPE_DV_PAIRS[t_name]
                ivs_map["atk"] = dv_a * 2
                ivs_map["def"] = dv_d * 2
                break

    a0 = ivs_map["atk"] // 2
    d0 = ivs_map["def"] // 2
    s0 = ivs_map["spe"] // 2
    c0 = ivs_map["spa"] // 2

    f_ratio = float((sp.get("genderRatio") or {}).get("F", 0.5))

    def _gender_of(a_dv: int) -> str:
        if sp.get("gender"):
            return str(sp["gender"])
        return "F" if a_dv < f_ratio * 16.0 else "M"

    if out.get("gender") not in ("M", "F", "N"):
        out["gender"] = _gender_of(a0)
    target_gender = str(out["gender"])
    if sp.get("gender"):
        target_gender = str(sp["gender"])
        out["gender"] = target_gender

    def _is_shiny(a_dv: int, d_dv: int, s_dv: int, c_dv: int) -> bool:
        return d_dv == 10 and s_dv == 10 and c_dv == 10 and (a_dv % 4) >= 2

    def _hp_dv(a_dv: int, d_dv: int, s_dv: int, c_dv: int) -> int:
        return (a_dv % 2) * 8 + (d_dv % 2) * 4 + (s_dv % 2) * 2 + (c_dv % 2)

    def _hp_power(a_dv: int, d_dv: int, s_dv: int, c_dv: int) -> int:
        return (5 * ((a_dv >> 3) * 8 + (d_dv >> 3) * 4 + (s_dv >> 3) * 2 + (c_dv >> 3)) + (c_dv % 4)) // 2 + 31

    def _unown_form(a_dv: int, d_dv: int, s_dv: int, c_dv: int) -> int:
        return (((a_dv & 6) << 5) | ((d_dv & 6) << 3) | ((s_dv & 6) << 1) | ((c_dv & 6) >> 1)) // 10

    target_shiny = bool(out.get("shiny", False)) or _is_shiny(a0, d0, s0, c0)
    has_hp_move = any(toid(m).startswith("hiddenpower") for m in (out.get("moves") or []))
    hp_type = (
        explicit_hp_type
        if explicit_hp_type is not None
        else ((4 * (a0 % 4) + (d0 % 4)) if has_hp_move else None)
    )
    form = _unown_form(a0, d0, s0, c0) if sp_id == "unown" else None

    # Update move string if `Hidden Power` is present so `"Hidden Power <Type>"` is explicit when desired
    attack = _uses_attack_stat(out, dex_raw)

    best_score: tuple[int, ...] | None = None
    best_dvs: tuple[int, int, int, int, int, int] | None = None
    for a in range(16):
        if _gender_of(a) != target_gender:
            continue
        for d in range(16):
            if hp_type is not None and (4 * (a % 4) + (d % 4)) != hp_type:
                continue
            for s in range(16):
                for c in range(16):
                    if _is_shiny(a, d, s, c) != target_shiny:
                        continue
                    if form is not None and _unown_form(a, d, s, c) != form:
                        continue
                    h_dv = _hp_dv(a, d, s, c)
                    sc = (
                        h_dv,
                        d,
                        c,
                        s,
                        0 if hp_type is None else _hp_power(a, d, s, c),
                        a if attack else -a,
                    )
                    if best_score is None or sc > best_score:
                        best_score = sc
                        best_dvs = (h_dv, a, d, c, c, s)

    if best_dvs is None:
        raise ValueError(f"{out['species']}: no feasible training-v3 spread for gender={target_gender}, hp_type={hp_type}")

    out["ivs"] = {
        k: (31 if best_dvs[i] == 15 else best_dvs[i] * 2)
        for i, k in enumerate(_STAT_KEYS)
    }
    out["evs"] = {
        k: (0 if (k == "atk" and not attack) else 255)
        for k in _STAT_KEYS
    }
    move_ids = [toid(m) for m in (out.get("moves") or [])]
    if "return" in move_ids and "frustration" in move_ids:
        raise ValueError(f"{out['species']}: Return and Frustration require conflicting happiness")
    if "return" in move_ids:
        out["happiness"] = 255
    elif "frustration" in move_ids:
        out["happiness"] = 0
    return out


def normalize_training_team(
    team_sets: list[dict[str, Any]],
    dex_raw: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply `normalize_training_set` across all 6 Pokemon sets in a party."""
    if dex_raw is None:
        dex_raw = load_dex().raw
    return [normalize_training_set(s, dex_raw=dex_raw) for s in team_sets]


def _legal_candidate_moves_for_species(
    species_name: str,
    dex_raw: dict[str, Any],
    learnsets: dict[str, dict[str, Any]],
) -> list[str]:
    """Return legal, non-dominated competitive candidate moves for `species_name`."""
    sp_id = toid(species_name)
    ls_entry = learnsets.get(sp_id) or {}
    legal_move_ids = set(ls_entry.get("moves") or [])
    curated = _CURATED_SPECIES_MOVES.get(sp_id, ())

    candidates: list[str] = []
    seen_ids: set[str] = set()
    for m_name in curated:
        m_id = toid(m_name)
        base_id = "hiddenpower" if m_id.startswith("hiddenpower") else m_id
        if (not legal_move_ids or base_id in legal_move_ids) and base_id not in _EXCLUDED_KODOKU_MOVES:
            if m_id not in seen_ids:
                seen_ids.add(m_id)
                candidates.append(m_name)

    if len(candidates) < 6 and legal_move_ids:
        for m_id in sorted(legal_move_ids):
            if m_id in _EXCLUDED_KODOKU_MOVES or m_id in seen_ids:
                continue
            md = dex_raw["moves"].get(m_id)
            if not md:
                continue
            bp = int(md.get("basePower", 0))
            cat = md.get("category", "Status")
            if cat != "Status" and bp < 60 and m_id not in ("rapidspin", "pursuit", "thief", "counter", "mirrorcoat", "seismictoss", "nightshade", "superfang", "reversal", "flail"):
                continue
            seen_ids.add(m_id)
            candidates.append(str(md.get("name", m_id)))
    return candidates


def mutate_team_for_kodoku(
    parent_team: dict[str, Any],
    rng: np.random.Generator,
    *,
    slot_weights: np.ndarray | None = None,
    preserve_shell: bool = True,
    new_team_id: str | None = None,
    dex_raw: dict[str, Any] | None = None,
    learnsets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a legal, `training-v3`-normalized Kodoku child party from `parent_team`.

    - When `preserve_shell=True`, preserves `(species, level, gender, bool(item))` on all 6 slots
      so the child shares the exact public Team Preview shell (`preview_signature_sets`) with its
      parent deme while mutating moves, Hidden Power types, or swapping held items across slots.
    - When `preserve_shell=False`, also permits changing held items freely (under Item Clause) or
      shifting the level distribution while bias-targeting low-marginal / hitchhiker slots.
    """
    if dex_raw is None:
        dex_raw = load_dex().raw
    if learnsets is None:
        learnsets = load_learnsets()

    child = json.loads(json.dumps(parent_team))
    sets: list[dict[str, Any]] = child["sets"]
    n_slots = len(sets)

    # Bias slot selection toward slots with lower marginal value (or uniform if not provided)
    if slot_weights is not None and len(slot_weights) == n_slots:
        w = np.asarray(slot_weights, dtype=np.float64)
        # Invert marginals so weak/hitchhiker slots get higher mutation probability
        inv = np.max(w) - w + 0.05
        probs = np.maximum(inv, 0.01)
        probs /= np.sum(probs)
    else:
        probs = np.full((n_slots,), 1.0 / n_slots, dtype=np.float64)

    n_edits = int(rng.choice([1, 2], p=[0.65, 0.35]))
    chosen_slots = rng.choice(n_slots, size=min(n_edits, n_slots), replace=False, p=probs)

    for slot_idx in chosen_slots:
        pset = sets[int(slot_idx)]
        sp_name = str(pset["species"])
        cand_moves = _legal_candidate_moves_for_species(sp_name, dex_raw, learnsets)
        cur_moves = list(pset.get("moves") or [])
        if not cur_moves or not cand_moves:
            continue

        cur_move_ids = {toid(m) for m in cur_moves}
        has_hp_already = any(m_id.startswith("hiddenpower") for m_id in cur_move_ids)

        # Pick a move slot `m_pos` to replace
        m_pos = int(rng.integers(0, len(cur_moves)))
        replacing_id = toid(cur_moves[m_pos])
        replacing_is_hp = replacing_id.startswith("hiddenpower")

        # Filter legal replacement moves
        viable_replacements: list[str] = []
        remaining_ids = {toid(m) for idx, m in enumerate(cur_moves) if idx != m_pos}
        for cm in cand_moves:
            cm_id = toid(cm)
            if cm_id in remaining_ids or cm_id == replacing_id:
                continue
            if cm_id.startswith("hiddenpower") and has_hp_already and not replacing_is_hp:
                continue
            if cm_id == "sleeptalk" and "rest" not in remaining_ids:
                continue
            if replacing_id == "rest" and "sleeptalk" in remaining_ids and cm_id != "rest":
                continue
            if cm_id == "return" and "frustration" in remaining_ids:
                continue
            if cm_id == "frustration" and "return" in remaining_ids:
                continue
            viable_replacements.append(cm)

        if viable_replacements:
            new_move = str(viable_replacements[int(rng.integers(0, len(viable_replacements)))])
            cur_moves[m_pos] = new_move
            pset["moves"] = cur_moves

    # Optionally swap or mutate items (respecting Item Clause and `preserve_shell`)
    if float(rng.random()) < 0.35:
        op_kind = int(rng.integers(0, 2))
        if op_kind == 0:
            # Swap two non-species-locked items between slots that both have items
            valid_swap_slots = [
                idx
                for idx, s in enumerate(sets)
                if s.get("item")
                and toid(s.get("item", "")) not in ("thickclub", "lightball", "metalpowder", "stick", "luckypunch")
            ]
            if len(valid_swap_slots) >= 2:
                i1, i2 = rng.choice(valid_swap_slots, size=2, replace=False)
                sets[int(i1)]["item"], sets[int(i2)]["item"] = (
                    sets[int(i2)]["item"],
                    sets[int(i1)]["item"],
                )
        else:
            # Replace one slot's item with an unused competitive item
            used_items = {toid(s.get("item", "")) for s in sets if s.get("item")}
            mutable_slots = [
                idx
                for idx, s in enumerate(sets)
                if (bool(s.get("item")) or not preserve_shell)
                and toid(s.get("item", "")) not in ("thickclub", "lightball", "metalpowder")
            ]
            if mutable_slots:
                s_idx = int(rng.choice(mutable_slots))
                sp_id = toid(sets[s_idx]["species"])
                if sp_id == "marowak" and "thickclub" not in used_items:
                    sets[s_idx]["item"] = "Thick Club"
                else:
                    unused_pool = [
                        it for it in _CURATED_COMPETITIVE_ITEMS if toid(it) not in used_items
                    ]
                    if unused_pool:
                        sets[s_idx]["item"] = str(unused_pool[int(rng.integers(0, len(unused_pool)))])

    # Enforce Item Clause across all 6 slots (deduplicate any repeated non-empty item)
    seen_items: set[str] = set()
    for s in sets:
        it_raw = str(s.get("item", "")).strip()
        if not it_raw:
            continue
        it_id = toid(it_raw)
        if it_id in seen_items:
            for repl in _CURATED_COMPETITIVE_ITEMS:
                repl_id = toid(repl)
                if repl_id not in seen_items:
                    s["item"] = repl
                    seen_items.add(repl_id)
                    break
        else:
            seen_items.add(it_id)

    # Normalize all 6 sets under `training-v3`
    child["sets"] = normalize_training_team(sets, dex_raw=dex_raw)
    child["species"] = [s["species"] for s in child["sets"]]
    child["levels"] = [int(s.get("level", 50)) for s in child["sets"]]
    if new_team_id is not None:
        child["id"] = new_team_id
    return child


class KodokuPopulationEvaluation(NamedTuple):
    """Result of two-stage Kodoku population evaluation on TPU (`evaluate_population_tpu`)."""

    team_ids: list[str]
    opponent_ids: list[str]
    stage1_summary: KodokuGauntletSummary
    stage1_means: np.ndarray  # float32[N_P]
    stage1_lcbs: np.ndarray  # float32[N_P]
    stage1_ucbs: np.ndarray  # float32[N_P]
    survivor_indices: list[int]
    stage2_policy_scores: np.ndarray  # float32[N_P] (NaN or stage1 value for pruned rows)
    stage2_policy_games: np.ndarray  # int32[N_P]
    blended_payoff_matrix: np.ndarray  # float32[N_P, N_Q]
    blended_fitness: np.ndarray  # float32[N_P]
    coupled_equilibrium: CoupledShellEquilibriumResult
    hitchhiker_flags: np.ndarray  # bool[N_P, 6]
    statuses: list[str]
    elapsed_secs: float


class KodokuEvolutionResult(NamedTuple):
    """Result of multi-generation Kodoku evolutionary loop on TPU (`run_kodoku_tpu`)."""

    final_population: list[dict[str, Any]]
    final_evaluation: KodokuPopulationEvaluation
    nash_teams: list[dict[str, Any]]
    nash_weights: dict[str, float]
    generation_history: list[dict[str, Any]]
    elapsed_secs: float


def evaluate_population_tpu(
    population: list[dict[str, Any]],
    opponent_pool: list[dict[str, Any]] | None = None,
    *,
    opponent_weights: np.ndarray | list[float] | None = None,
    screen_games: int = 2,
    support_k: int = 3,
    refine_games: int = 4,
    screen_turns: int = 40,
    refine_turns: int = 60,
    screen_stride: int = 1,
    screen_drop_ucb: float = 0.46,
    survivor_top_k: int = 8,
    policy_games_per_pair: int = 2,
    policy_turns: int = 12,
    policy_rollout_turns: int = 2,
    policy_followup_turns: int = 24,
    policy_rm_sweeps: int = 32,
    policy_blend_weight: float = 0.35,
    max_shell_variants: int = 6,
    bayesian_eq: bool = True,
    z_value: float = 1.96,
    seed: int = 20260405,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
    shared_dex: SharedDexTable | None = None,
) -> KodokuPopulationEvaluation:
    """Two-stage Kodoku population fitness evaluator on TPU:

    - **Stage 1 (Fast Core-Nash Gauntlet & Bayesian Election-Call Screen)**:
      Evaluates `population x opponent_pool` via `evaluate_kodoku_gauntlet_tpu`, computing
      6v6 Core-Nash payoffs, Leave-One-Out slot marginals (`ΔV_{-s}`), slot WARs, and
      election-call credible intervals `[lcb, ucb]`. Candidates with `ucb < screen_drop_ucb`
      or outside the top `survivor_top_k` are pruned (`ELIMINATED_STAGE1`).
    - **Stage 2 (In-Battle Coupled Per-World Bayesian Equilibrium Policy Verification)**:
      For surviving candidates, solves `solve_bayesian_shell_preview_nash_np` against the
      top opponent support shells to select equilibrium 3v3 lineups and simulates paired
      side-swapped CRN games on TPU using `_run_policy_duel_batch_tpu` (`simulate_shell_belief_minimax_games_tpu`
      with `bayesian_eq_a=bayesian_eq, bayesian_eq_b=bayesian_eq`).
      Blends Stage-1 preview Nash payoff and Stage-2 in-battle Bayesian Equilibrium policy score.
    """
    t0 = time.perf_counter()
    if weights is None:
        weights = default_eval_weights()
    if shared_dex is None:
        shared_dex = build_shared_dex_table()

    teams_q = population if opponent_pool is None else opponent_pool
    n_p = len(population)
    n_q = len(teams_q)
    if n_p == 0 or n_q == 0:
        raise ValueError("population and opponent_pool must be non-empty")

    # Stage 1: Fused 6v6 Core-Nash Gauntlet on TPU
    summary = evaluate_kodoku_gauntlet_tpu(
        population,
        teams_q=opponent_pool,
        q_weights=opponent_weights,
        screen_games=screen_games,
        support_k=support_k,
        refine_games=refine_games,
        screen_turns=screen_turns,
        refine_turns=refine_turns,
        screen_stride=screen_stride,
        compute_slot_telemetry=True,
        seed=seed,
        weights=weights,
        residual_bundle=residual_bundle,
    )

    # Determine opponent mixture weights for Stage-1 credible intervals and Stage-2 duel targeting
    if opponent_weights is not None:
        q_w = np.asarray(opponent_weights, dtype=np.float32).reshape(n_q)
        q_w = np.maximum(q_w, 0.0)
        q_w_sum = float(np.sum(q_w))
        q_w = q_w / q_w_sum if q_w_sum > 0 else np.full((n_q,), 1.0 / n_q, dtype=np.float32)
    else:
        q_w = np.asarray(summary.rect_equilibrium.p_b, dtype=np.float32)
        if float(np.sum(q_w)) <= 1e-6:
            q_w = np.full((n_q,), 1.0 / n_q, dtype=np.float32)

    stage1_mat = np.asarray(summary.payoff_matrix, dtype=np.float32)
    stage1_means = (stage1_mat @ q_w).astype(np.float32)

    # Compute Bayesian election-call bounds per row against `q_w`
    eff_games_per_cell = max(2.0, float(screen_games + refine_games))
    cell_var = stage1_mat * (1.0 - stage1_mat) / eff_games_per_cell
    row_se = np.sqrt(np.maximum(cell_var @ (q_w * q_w), 1e-6)).astype(np.float32)
    stage1_lcbs = np.clip(stage1_means - float(z_value) * row_se, 0.0, 1.0).astype(np.float32)
    stage1_ucbs = np.clip(stage1_means + float(z_value) * row_se, 0.0, 1.0).astype(np.float32)

    # Select survivors for Stage-2 Bayesian Equilibrium policy verification
    ranked_rows = [int(i) for i in np.argsort(-stage1_means)]
    survivor_indices: list[int] = []
    statuses = ["ELIMINATED_STAGE1" for _ in range(n_p)]
    for rank_pos, row_i in enumerate(ranked_rows):
        if rank_pos == 0 or (
            len(survivor_indices) < max(1, int(survivor_top_k))
            and float(stage1_ucbs[row_i]) >= float(screen_drop_ucb)
        ):
            survivor_indices.append(row_i)
            statuses[row_i] = "SURVIVED_STAGE2"

    survivor_indices.sort()
    blended_mat = stage1_mat.copy()
    stage2_scores = stage1_means.copy()
    stage2_games = np.zeros((n_p,), dtype=np.int32)

    if survivor_indices and int(policy_games_per_pair) > 0:
        # Build unified pool to run `_run_policy_duel_batch_tpu` without duplicating self-play rosters
        if opponent_pool is None:
            combined_teams = list(population)
            q_offset = 0
        else:
            combined_teams = list(population) + list(teams_q)
            q_offset = n_p
        rosters_all = [compile_roster6(_extract_team_sets(t)) for t in combined_teams]
        variant_groups_all, _ = _build_shell_variant_rosters_map(
            rosters_all, combined_teams, max_variants=max_shell_variants
        )

        # Pick top opponent support columns `j` (up to 4 columns with largest `q_w[j]`)
        top_q_cols = [
            int(j)
            for j in np.argsort(-q_w)[: min(4, n_q)]
            if float(q_w[j]) > 1e-3
        ]
        if not top_q_cols:
            top_q_cols = [int(np.argmax(q_w))]

        rep_picks_p: list[tuple[int, int, int]] = []
        for r in rosters_all[:n_p]:
            leg = [int(x) for x in np.flatnonzero(np.asarray(r.legal_mask_60))]
            act0 = leg[0] if leg else 0
            p0 = PREVIEW_ACTIONS_0BASED_NP[act0]
            rep_picks_p.append((int(p0[0]), int(p0[1]), int(p0[2])))

        rep_picks_q: list[tuple[int, int, int]] = []
        for r in rosters_all[q_offset : q_offset + n_q]:
            leg = [int(x) for x in np.flatnonzero(np.asarray(r.legal_mask_60))]
            act0 = leg[0] if leg else 0
            p0 = PREVIEW_ACTIONS_0BASED_NP[act0]
            rep_picks_q.append((int(p0[0]), int(p0[1]), int(p0[2])))

        duel_specs: list[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]] = []
        spec_pairs: list[tuple[int, int]] = []
        for i in survivor_indices:
            for j in top_q_cols:
                # Skip exact self-mirror when `opponent_pool is None` and `i == j` unless `n_q == 1`
                if opponent_pool is None and i == j and n_q > 1:
                    continue
                duel_specs.append((i, q_offset + j, rep_picks_p[i], rep_picks_q[j]))
                spec_pairs.append((i, j))

        if duel_specs:
            g_pol = max(2, int(policy_games_per_pair))
            if g_pol % 2 == 1:
                g_pol += 1
            pol_vals = _run_policy_duel_batch_tpu(
                rosters_all,
                variant_groups_all,
                duel_specs,
                games_per_spec=g_pol,
                shared_dex=shared_dex,
                base_seed_int=int(seed) ^ 0x6A09E667,
                bayesian_eq_a=bayesian_eq,
                bayesian_eq_b=bayesian_eq,
                policy_turns=policy_turns,
                rollout_turns=policy_rollout_turns,
                followup_turns=policy_followup_turns,
                rm_sweeps=policy_rm_sweeps,
                weights=weights,
                residual_bundle=residual_bundle,
            )
            alpha_blend = float(np.clip(policy_blend_weight, 0.0, 1.0))
            row_pol_num = np.zeros((n_p,), dtype=np.float64)
            row_pol_den = np.zeros((n_p,), dtype=np.float64)
            for idx_s, (i, j) in enumerate(spec_pairs):
                v_pol = float(pol_vals[idx_s])
                v_st1 = float(stage1_mat[i, j])
                v_bld = (1.0 - alpha_blend) * v_st1 + alpha_blend * v_pol
                blended_mat[i, j] = v_bld
                if opponent_pool is None and i < n_q and j < n_p:
                    blended_mat[j, i] = 1.0 - v_bld
                w_j = float(q_w[j])
                row_pol_num[i] += w_j * v_pol
                row_pol_den[i] += w_j
                stage2_games[i] += g_pol

            for i in survivor_indices:
                if row_pol_den[i] > 1e-9:
                    stage2_scores[i] = float(row_pol_num[i] / row_pol_den[i])

    if opponent_pool is None and n_p == n_q:
        blended_mat = 0.5 * (blended_mat + (1.0 - blended_mat.T))
        np.fill_diagonal(blended_mat, 0.5)

    blended_fitness = (blended_mat @ q_w).astype(np.float32)

    # Re-solve coupled shell equilibrium on the blended payoff matrix
    blended_rect = solve_rect_rm_plus_jax(jnp.asarray(blended_mat, dtype=jnp.float32), sweeps=1000)
    blended_summary = KodokuGauntletSummary(
        p_ids=summary.p_ids,
        q_ids=summary.q_ids,
        payoff_matrix=blended_mat.astype(np.float32),
        weighted_fitness=blended_fitness,
        slot_picks=summary.slot_picks,
        slot_wins=summary.slot_wins,
        slot_marginals=summary.slot_marginals,
        slot_wars=summary.slot_wars,
        effective_cores=summary.effective_cores,
        rect_equilibrium=blended_rect,
    )
    coupled_eq = solve_coupled_shell_equilibrium_tpu(
        blended_summary,
        teams_p=population,
        teams_q=teams_q,
    )
    hitchhikers = np.zeros((n_p, 6), dtype=bool)
    for i in range(n_p):
        hh_list = detect_hitchhiker_slots(
            _extract_team_sets(population[i]),
            blended_summary.slot_picks[i],
            blended_summary.slot_marginals[i],
        )
        for hh in hh_list:
            s_idx = int(hh.get("slot", -1))
            if 0 <= s_idx < 6:
                hitchhikers[i, s_idx] = True

    deployed_w_map = dict(
        zip(coupled_eq.deployed_team_ids, coupled_eq.deployed_team_weights)
    )
    for i in range(n_p):
        if float(deployed_w_map.get(summary.p_ids[i], 0.0)) >= 0.01:
            statuses[i] = "NASH_SUPPORT"

    elapsed = time.perf_counter() - t0
    return KodokuPopulationEvaluation(
        team_ids=summary.p_ids,
        opponent_ids=summary.q_ids,
        stage1_summary=blended_summary,
        stage1_means=stage1_means,
        stage1_lcbs=stage1_lcbs,
        stage1_ucbs=stage1_ucbs,
        survivor_indices=survivor_indices,
        stage2_policy_scores=stage2_scores.astype(np.float32),
        stage2_policy_games=stage2_games,
        blended_payoff_matrix=blended_mat.astype(np.float32),
        blended_fitness=blended_fitness,
        coupled_equilibrium=coupled_eq,
        hitchhiker_flags=hitchhikers,
        statuses=statuses,
        elapsed_secs=elapsed,
    )


def run_kodoku_tpu(
    seed_teams: list[dict[str, Any]],
    *,
    opponent_pool: list[dict[str, Any]] | None = None,
    generations: int = 2,
    offspring_per_gen: int = 4,
    max_population_size: int = 12,
    preserve_shell_prob: float = 0.75,
    screen_games: int = 2,
    support_k: int = 3,
    refine_games: int = 4,
    screen_turns: int = 40,
    refine_turns: int = 60,
    survivor_top_k: int = 6,
    policy_games_per_pair: int = 2,
    policy_turns: int = 12,
    policy_rollout_turns: int = 2,
    policy_followup_turns: int = 24,
    policy_rm_sweeps: int = 32,
    policy_blend_weight: float = 0.35,
    max_shell_variants: int = 4,
    bayesian_eq: bool = True,
    seed: int = 20260405,
    out_dir: str | Path | None = None,
    weights: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
) -> KodokuEvolutionResult:
    """Run Multi-Deme / Challenger Kodoku evolutionary loop on TPU under the
    coupled per-world Bayesian Equilibrium + Depth-1 minimax contraction policy.

    Each generation:
    1. Evaluates the current population via `evaluate_population_tpu` (Stage-1 Core-Nash
       gauntlet + Stage-2 in-battle Bayesian Equilibrium policy duel verification).
    2. Retains all Nash support teams plus top non-dominated elites up to `max_population_size`.
    3. Mutates elites (guided by Leave-One-Out slot marginals `ΔV_{-s}` and hitchhiker flags)
       via `mutate_team_for_kodoku` + `normalize_training_team` to produce `offspring_per_gen`
       new candidate parties for the next generation.
    """
    t0 = time.perf_counter()
    if weights is None:
        weights = default_eval_weights()

    dex_raw = load_dex().raw
    learnsets = load_learnsets()
    shared_dex = build_shared_dex_table()
    rng = np.random.default_rng(int(seed))

    # Normalize initial seed teams under training-v3
    population: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for idx, t in enumerate(seed_teams):
        t_copy = json.loads(json.dumps(t))
        t_copy.setdefault("id", f"seed_{idx}")
        t_copy["sets"] = normalize_training_team(_extract_team_sets(t_copy), dex_raw=dex_raw)
        fp = json.dumps(t_copy["sets"], sort_keys=True)
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            population.append(t_copy)

    if not population:
        raise ValueError("seed_teams must contain at least 1 valid team")

    gen_history: list[dict[str, Any]] = []
    n_gens = max(1, int(generations))
    eval_res: KodokuPopulationEvaluation | None = None

    for gen_idx in range(n_gens):
        gen_seed = int(seed) ^ ((gen_idx + 1) * 0x9E3779B9)
        eval_res = evaluate_population_tpu(
            population,
            opponent_pool=opponent_pool,
            screen_games=screen_games,
            support_k=support_k,
            refine_games=refine_games,
            screen_turns=screen_turns,
            refine_turns=refine_turns,
            survivor_top_k=survivor_top_k,
            policy_games_per_pair=policy_games_per_pair,
            policy_turns=policy_turns,
            policy_rollout_turns=policy_rollout_turns,
            policy_followup_turns=policy_followup_turns,
            policy_rm_sweeps=policy_rm_sweeps,
            policy_blend_weight=policy_blend_weight,
            max_shell_variants=max_shell_variants,
            bayesian_eq=bayesian_eq,
            seed=gen_seed,
            weights=weights,
            residual_bundle=residual_bundle,
            shared_dex=shared_dex,
        )

        ceq = eval_res.coupled_equilibrium
        dep_map = dict(zip(ceq.deployed_team_ids, ceq.deployed_team_weights))
        member_w = np.array(
            [float(dep_map.get(tid, 0.0)) for tid in eval_res.team_ids],
            dtype=np.float32,
        )
        top_idx = int(np.argmax(eval_res.blended_fitness))
        gen_record = {
            "generation": gen_idx,
            "population_size": len(population),
            "top_team_id": eval_res.team_ids[top_idx],
            "top_blended_fitness": round(float(eval_res.blended_fitness[top_idx]), 6),
            "top_stage2_policy_score": round(float(eval_res.stage2_policy_scores[top_idx]), 6),
            "active_shells": list(ceq.active_shells),
            "members": [
                {
                    "id": eval_res.team_ids[i],
                    "status": eval_res.statuses[i],
                    "stage1_mean": round(float(eval_res.stage1_means[i]), 6),
                    "stage2_policy_score": round(float(eval_res.stage2_policy_scores[i]), 6),
                    "blended_fitness": round(float(eval_res.blended_fitness[i]), 6),
                    "nash_weight": round(float(member_w[i]), 6),
                    "hitchhiker_slots": [
                        int(s) for s in np.flatnonzero(eval_res.hitchhiker_flags[i])
                    ],
                }
                for i in range(len(population))
            ],
            "elapsed_secs": round(eval_res.elapsed_secs, 4),
        }
        gen_history.append(gen_record)

        if gen_idx + 1 >= n_gens or int(offspring_per_gen) <= 0:
            break

        # Elitism: preserve all Nash support teams (`member_w >= 0.01`) + top blended fitness teams
        support_rows = [int(i) for i in np.flatnonzero(member_w >= 0.01)]
        fitness_order = [int(i) for i in np.argsort(-eval_res.blended_fitness)]
        elites_idx: list[int] = []
        for i in support_rows + fitness_order:
            if i not in elites_idx and len(elites_idx) < max(2, int(max_population_size)):
                elites_idx.append(i)

        next_pop = [population[i] for i in elites_idx]

        # Generate `offspring_per_gen` normalized mutations from elites
        parent_probs = np.array(
            [max(float(member_w[i]), 0.15) for i in elites_idx], dtype=np.float64
        )
        parent_probs /= np.sum(parent_probs)

        created = 0
        attempts = 0
        while created < int(offspring_per_gen) and attempts < int(offspring_per_gen) * 6:
            attempts += 1
            p_local = int(rng.choice(len(elites_idx), p=parent_probs))
            p_orig_idx = elites_idx[p_local]
            parent_team = population[p_orig_idx]
            slot_margs = np.asarray(eval_res.stage1_summary.slot_marginals[p_orig_idx], dtype=np.float64)
            preserve_sh = bool(float(rng.random()) < float(preserve_shell_prob))
            child_id = f"{parent_team.get('id', 'team')}-beq-g{gen_idx + 1}-c{created}"
            child = mutate_team_for_kodoku(
                parent_team,
                rng,
                slot_weights=slot_margs,
                preserve_shell=preserve_sh,
                new_team_id=child_id,
                dex_raw=dex_raw,
                learnsets=learnsets,
            )
            child["origin"] = "evolved"
            if preserve_sh and parent_team.get("variantCluster"):
                child["variantCluster"] = parent_team["variantCluster"]
            fp = json.dumps(child["sets"], sort_keys=True)
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                next_pop.append(child)
                created += 1

        population = next_pop

    assert eval_res is not None
    final_dep_map = dict(
        zip(
            eval_res.coupled_equilibrium.deployed_team_ids,
            eval_res.coupled_equilibrium.deployed_team_weights,
        )
    )
    nash_teams: list[dict[str, Any]] = []
    nash_weights: dict[str, float] = {}
    for i, t in enumerate(population):
        tid = str(t.get("id", f"team_{i}"))
        w_i = float(final_dep_map.get(tid, 0.0))
        if w_i >= 0.01:
            nash_teams.append(t)
            nash_weights[tid] = round(w_i, 6)

    total_elapsed = time.perf_counter() - t0
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": "coupled_per_world_bayesian_equilibrium_and_depth1_minimax",
            "bayesian_eq": bool(bayesian_eq),
            "generations": n_gens,
            "offspring_per_gen": int(offspring_per_gen),
            "nash_weights": nash_weights,
            "nash_teams": nash_teams,
            "final_population": population,
            "generation_history": gen_history,
            "elapsed_secs": round(total_elapsed, 4),
        }
        with (out_path / "kodoku_evolution_summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return KodokuEvolutionResult(
        final_population=population,
        final_evaluation=eval_res,
        nash_teams=nash_teams,
        nash_weights=nash_weights,
        generation_history=gen_history,
        elapsed_secs=total_elapsed,
    )



