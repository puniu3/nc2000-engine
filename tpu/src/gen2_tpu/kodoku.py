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
from .kernel import apply_actions
from .prng import prng_random, prng_random_chance, seed_from_str
from .residual import ResidualBundle, RolloutResidualWeights
from .search import (
    EvalWeights,
    HeavyRolloutResult,
    default_eval_weights,
    eval_compact_leaf,
    sample_heavy_playout_action,
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
            if ikey == "focusband":
                iflags |= ItemFlag.FOCUSBAND
            if ikey == "mintberry":
                iflags |= ItemFlag.MINTBERRY
            if ikey == "miracleberry":
                iflags |= ItemFlag.MIRACLEBERRY
            if ikey == "goldberry":
                iflags |= ItemFlag.GOLDBERRY
            if ikey in ("berry", "berryjuice"):
                iflags |= ItemFlag.BERRY
            if ikey == "mysteryberry":
                iflags |= ItemFlag.MYSTERYBERRY
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
    """Run a `turns`-step `Playout::Heavy` game on a single `(PackedBattle, MatchTable)`."""
    init_state = unpack_battle(init_packed)
    rollout_res: RolloutResidualWeights | None = (
        residual_bundle.rollout if residual_bundle is not None else None
    )

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, p_seed = carry
        p_seed, a0 = sample_heavy_playout_action(
            st, table, 0, p_seed, eps=eps, rollout_residual=rollout_res
        )
        p_seed, a1 = sample_heavy_playout_action(
            st, table, 1, p_seed, eps=eps, rollout_residual=rollout_res
        )
        acts = jnp.stack([a0, a1], axis=0)
        next_st = apply_actions(st, table, acts)
        return (next_st, p_seed), None

    (final_st, _), _ = jax.lax.scan(_step, (init_state, policy_seed), None, length=turns)
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
        if jax.default_backend() != "tpu":
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
        rollout_res: RolloutResidualWeights | None = (
            residual_bundle.rollout if residual_bundle is not None else None
        )

        def _step(carry: tuple[CompactBattle, jax.Array], _: None):
            st, ps = carry
            ps, a0 = jax.vmap(
                lambda s, tb, p: sample_heavy_playout_action(
                    s, tb, 0, p, eps=eps, rollout_residual=rollout_res
                )
            )(st, batched_tables, ps)
            ps, a1 = jax.vmap(
                lambda s, tb, p: sample_heavy_playout_action(
                    s, tb, 1, p, eps=eps, rollout_residual=rollout_res
                )
            )(st, batched_tables, ps)
            acts = jnp.stack([a0, a1], axis=-1)
            next_st = jax.vmap(apply_actions, in_axes=(0, 0, 0))(st, batched_tables, acts)
            return (next_st, ps), None

        block_turns = 5
        if turns >= 10 and (turns % block_turns == 0):
            n_blocks = turns // block_turns

            def _cond(loop_carry: tuple[jax.Array, CompactBattle, jax.Array]) -> jax.Array:
                b_idx, st, _ = loop_carry
                return (b_idx < n_blocks) & jnp.any(st.outcome == 0)

            def _body(
                loop_carry: tuple[jax.Array, CompactBattle, jax.Array],
            ) -> tuple[jax.Array, CompactBattle, jax.Array]:
                b_idx, st, ps = loop_carry
                (next_st, next_ps), _ = jax.lax.scan(_step, (st, ps), None, length=block_turns)
                return (b_idx + jnp.int32(1), next_st, next_ps)

            _, final_st, _ = jax.lax.while_loop(
                _cond, _body, (jnp.int32(0), init_st, policy_seeds)
            )
        else:
            (final_st, _), _ = jax.lax.scan(_step, (init_st, policy_seeds), None, length=turns)

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


def _make_subgame_seeds(
    n_trajectories: int,
    seed_hi: jax.Array,
    seed_lo: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Generate deterministic 32-bit `(battle_seeds, policy_seeds)` of shape `uint32[N, 2]`."""
    idx = jnp.arange(n_trajectories, dtype=jnp.uint32)
    b_hi = seed_hi ^ (idx * jnp.uint32(0x9E3779B9))
    b_lo = seed_lo + (idx * jnp.uint32(0x85EBCA6B) + jnp.uint32(1))
    p_hi = (seed_hi + jnp.uint32(0xC2B2AE35)) ^ (idx * jnp.uint32(0x27D4EB2F))
    p_lo = seed_lo ^ (idx * jnp.uint32(0x165667B1) + jnp.uint32(7))
    return jnp.stack([b_hi, b_lo], axis=-1), jnp.stack([p_hi, p_lo], axis=-1)


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


def replicator_dynamics_jax(
    payoff_matrix: jax.Array,
    init_weights: jax.Array | None = None,
    eta: float = 2.0,
    steps: int = 1000,
) -> tuple[jax.Array, jax.Array]:
    r"""Continuous-abundance replicator dynamics $w_i^{(t+1)} = w_i^{(t)} \exp(\eta (u_i - \bar{u})) / Z$.

    Matches `gcp-batch-psro.py::replicator_weights` and `meta_eco.rs` symmetric meta-Nash updates.
    Returns `(final_weights, fitness_vector)`.
    """
    n = payoff_matrix.shape[0]
    if init_weights is None:
        w0 = jnp.full((n,), 1.0 / max(1, n), dtype=jnp.float32)
    else:
        w0 = init_weights.astype(jnp.float32)
        w0 = w0 / jnp.maximum(jnp.sum(w0), jnp.float32(1e-12))

    eta_f = jnp.float32(eta)

    def _body(_: int, w: jax.Array) -> jax.Array:
        u = jnp.matmul(payoff_matrix, w)
        u_bar = jnp.sum(w * u)
        w_next = w * jnp.exp(eta_f * (u - u_bar))
        return w_next / jnp.maximum(jnp.sum(w_next), jnp.float32(1e-12))

    w_star = jax.lax.fori_loop(0, steps, _body, w0)
    u_star = jnp.matmul(payoff_matrix, w_star)
    return w_star, u_star


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
    marg = np.asarray(slot_marginals, dtype=np.float64)
    lvls = np.asarray(slot_levels, dtype=np.float64)
    if np.ndim(games) == 0:
        g_arr = np.full(marg.shape, max(float(games), 1e-9), dtype=np.float64)
    else:
        g_arr = np.maximum(np.asarray(games, dtype=np.float64), 1e-9)
        if g_arr.ndim == 1 and marg.ndim == 2:
            g_arr = g_arr[:, None]

    norm_m = marg / g_arr
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


def preview_signature_sets(sets: list[dict[str, Any]]) -> tuple[tuple[str, int], ...]:
    """Canonical 6-mon preview shell signature `tuple(sorted((toid(species), level)))`.

    Matches `nc2000_bot::preview::preview_sig_sets` so two variants that share the exact same
    6-Pokemon `(species, level)` multiset map to the same Team Preview outer shell.
    """
    pairs = [(toid(str(m.get("species", ""))), int(m.get("level", 50))) for m in sets]
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
            out.append(
                {
                    "unit_id": uid,
                    "member_ids": mids,
                    "member_indices": [id_to_idx[m] for m in mids],
                    "sig": sig,
                }
            )
        return out

    sig_groups: dict[tuple[tuple[str, int], ...], list[str]] = {}
    sig_order: list[tuple[tuple[str, int], ...]] = []
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
        units_out.append(
            {
                "unit_id": uid,
                "member_ids": mids,
                "member_indices": [id_to_idx[m] for m in mids],
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
        unit_payoff[u_idx, :] = np.mean(rows, axis=0)
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
        active_shells.append(
            {
                "unit_id": units[u_idx]["unit_id"],
                "weight": round(w_sh, 6),
                "member_ids": m_ids,
            }
        )
        w_mem = w_sh / float(len(m_ids))
        for mid in m_ids:
            deployed_ids.append(mid)
            deployed_weights.append(round(w_mem, 6))

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
