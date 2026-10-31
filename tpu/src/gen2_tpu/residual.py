r"""Drop-in residual evaluation & rollout neural network (`ResidualBundle`) and JAX trainer.

Mirrors `nc2000_bot::eval::{EvalResidualWeights, RolloutResidualWeights, ResidualBundle}` and
`tools/train-residual-nn.py` with 100% drop-in JSON compatibility (`NC2000_RESIDUAL_JSON`):
1. `extract_eval_features_with_eff` & `extract_eval_features`: Branchless 32-dimensional
   multi-body synergy (`x[0..16] >= 0`) and vulnerability (`x[16..32] >= 0`) feature
   extractor (`float32[32]`).
2. `EvalResidualWeights` (`32 -> 16 -> 1`) & `compact_eval_residual_sym`: Antisymmetric
   residual evaluation $\Delta_{\text{sym}}(s) = f_\theta(x(s, 0)) - f_\theta(x(s, 1))$
   with exact structural guarantees ($f_\theta(0) = 0$, non-decreasing in synergy
   features `0..16`, non-increasing in vulnerability features `16..32`).
3. `RolloutResidualWeights` (`12 x 6 -> 4`) & `compute_rollout_action_residuals`:
   Rank-4 bilinear action capability adjuster $\Delta Q_\psi(s, c)$ for `Playout::Heavy`.
4. `train_residual_bundle_jax`: Pure-JAX AdamW trainer enforcing the exact positivity,
   zero-baseline, and 1v1 regularization constraints of `tools/train-residual-nn.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import _get1, _get2, best_hit_fraction, mon_stat
from .state import CompactBattle, MatchTable, MoveFlag, Status, VolBit

EVAL_FEAT_DIM = 32
EVAL_HIDDEN_DIM = 16
ROLLOUT_STATE_DIM = 12
ROLLOUT_ACT_DIM = 6
ROLLOUT_RANK = 4


def _popcount_3bit(mask: jax.Array) -> jax.Array:
    m = mask.astype(jnp.int32)
    return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1)


def _mon_action_speed(
    state: CompactBattle,
    table: MatchTable,
    side: int | jax.Array,
    party_slot: int | jax.Array,
) -> jax.Array:
    return mon_stat(state, table, side, party_slot, 4, False, False)


class EvalResidualWeights(NamedTuple):
    """2-layer constrained residual MLP (`32 -> 16 -> 1`) matching `nc2000_bot::eval::EvalResidualWeights`."""

    active: jax.Array  # bool
    w1: jax.Array  # float32[16, 32]
    b1: jax.Array  # float32[16]
    w2: jax.Array  # float32[16]


class RolloutResidualWeights(NamedTuple):
    """Rank-4 bilinear rollout residual (`12 x 6 -> 4`) matching `nc2000_bot::eval::RolloutResidualWeights`."""

    active: jax.Array  # bool
    u_proj: jax.Array  # float32[4, 12]
    v_proj: jax.Array  # float32[4, 6]
    act_bias: jax.Array  # float32[6]


class ResidualBundle(NamedTuple):
    """Combined residual bundle loadable by `nc2000-bot` via `NC2000_RESIDUAL_JSON`."""

    eval: EvalResidualWeights
    rollout: RolloutResidualWeights


def zero_eval_residual_weights() -> EvalResidualWeights:
    return EvalResidualWeights(
        active=jnp.bool_(False),
        w1=jnp.zeros((EVAL_HIDDEN_DIM, EVAL_FEAT_DIM), dtype=jnp.float32),
        b1=jnp.zeros((EVAL_HIDDEN_DIM,), dtype=jnp.float32),
        w2=jnp.zeros((EVAL_HIDDEN_DIM,), dtype=jnp.float32),
    )


def zero_rollout_residual_weights() -> RolloutResidualWeights:
    return RolloutResidualWeights(
        active=jnp.bool_(False),
        u_proj=jnp.zeros((ROLLOUT_RANK, ROLLOUT_STATE_DIM), dtype=jnp.float32),
        v_proj=jnp.zeros((ROLLOUT_RANK, ROLLOUT_ACT_DIM), dtype=jnp.float32),
        act_bias=jnp.zeros((ROLLOUT_ACT_DIM,), dtype=jnp.float32),
    )


def zero_residual_bundle() -> ResidualBundle:
    """Return an inactive all-zero `ResidualBundle` (`ResidualBundle::zero()`)."""
    return ResidualBundle(
        eval=zero_eval_residual_weights(),
        rollout=zero_rollout_residual_weights(),
    )


_DEFAULT_W1_LIST: list[list[float]] = [
    [
        0.1592635, 0.12946095, 0.1194611, 0.0003951387, 0.107727565, 0.0, 0.10432918, 0.13698481,
        0.05285247, 0.11506298, 0.0, 0.097680226, 0.13070732, 0.11265245, 0.12788852, 0.097194225,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.101189956, 0.1694334, 0.08344528, 0.08816895, 0.048799418, 0.10238286, 0.2126568, 0.0,
        0.07133045, 0.10579651, 0.09185278, 0.07481665, 0.028079467, 0.12867014, 0.1693474, 0.08667781,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.4301937, 0.17532508, 0.44588163, 0.0, 0.0, 0.0, 0.40826002, 0.09990849,
        0.0, 0.2664039, 0.0, 0.30883932, 0.012687975, 0.032227036, 0.16722296, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.4180432, 0.03375988, 0.36683336, 0.0, 0.0, 0.0, 0.4056989, 0.0,
        0.0, 0.299519, 0.0, 0.34608608, 0.15768145, 0.0, 0.24105823, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.107165806, 0.19891012, 0.18452336, 0.027144747, 0.0, 0.074771635, 0.123442635, 0.0,
        0.06029057, 0.23902446, 0.0, 0.18970487, 0.18352276, 0.1174403, 0.2622345, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.41934305, 0.0065284097, 0.33491388, 0.0, 0.0, 0.0, 0.31453362, 0.09637944,
        0.0, 0.33386287, 0.0, 0.23820871, 0.07165454, 0.12357317, 0.06650649, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.23348774, 0.19378921, 0.15567504, 0.042202853, 0.042812735, 0.074472606, 0.1357919, 0.06553856,
        0.07969123, 0.089954264, 0.0, 0.16780518, 0.0040651164, 0.18930735, 0.083820105, 0.021488046,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.49159515, 0.036362596, 0.48515597, 0.0, 0.0, 0.0, 0.4541916, 0.0,
        0.0, 0.3482166, 0.0, 0.39139307, 0.06412866, 0.09684929, 0.11244748, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.22959538, 0.0, 0.21355996, 0.12527631, 0.16826937, 0.23940413, 0.05320776,
        0.107458405, 0.08296397, 0.0, 0.07929242, 0.07782938, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.22885, 0.39440846, 0.0, 0.23830053, 0.185397, 0.35957575, 0.0, 0.0,
        0.39405563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.19171354, 0.019145502, 0.19165786, 0.12039654, 0.19017553, 0.2244097, 0.05693801,
        0.1494015, 0.0, 0.052529648, 0.017231552, 0.0, 0.0, 0.0, 0.04571331,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.180914, 0.34676903, 0.0, 0.28007132, 0.1692566, 0.22470985, 0.06791634, 0.0,
        0.28711328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.234466, 0.50558555, 0.0, 0.2485967, 0.19141845, 0.35332057, 0.01311815, 0.0,
        0.37858465, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.20037, 0.43232366, 0.0, 0.4189235, 0.22831236, 0.37606534, 0.11593896, 0.0,
        0.35738957, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.16187, 0.353993, 0.0, 0.29652765, 0.27149695, 0.24860895, 0.13369489, 0.0,
        0.3844487, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.16410282, 0.0, 0.14905632, 0.13443504, 0.28700623, 0.15320116, 0.060230475,
        0.14842583, 0.07988839, 0.08262625, 0.047094699, 0.09009234, 0.0, 0.0, 0.0,
    ],
]

_DEFAULT_W2_LIST: list[float] = [
    0.0,
    0.0,
    0.20958547,
    0.17367375,
    0.0,
    0.15544592,
    0.0,
    0.19376242,
    0.0,
    -0.27715674,
    0.0,
    -0.21910277,
    -0.28395805,
    -0.2426653,
    -0.19603783,
    0.0,
]


def default_residual_bundle() -> ResidualBundle:
    """Return `nc2000_bot::eval::DEFAULT_RESIDUAL_BUNDLE` (Gate 1 calibrated `eval` active)."""
    return ResidualBundle(
        eval=EvalResidualWeights(
            active=jnp.bool_(True),
            w1=jnp.array(_DEFAULT_W1_LIST, dtype=jnp.float32),
            b1=jnp.zeros((EVAL_HIDDEN_DIM,), dtype=jnp.float32),
            w2=jnp.array(_DEFAULT_W2_LIST, dtype=jnp.float32),
        ),
        rollout=zero_rollout_residual_weights(),
    )


def residual_bundle_to_dict(bundle: ResidualBundle) -> dict[str, Any]:
    """Serialize `ResidualBundle` to a JSON-compatible dictionary matching `NC2000_RESIDUAL_JSON`."""
    return {
        "eval": {
            "active": bool(np.asarray(bundle.eval.active).item()),
            "w1": np.asarray(bundle.eval.w1, dtype=np.float32).tolist(),
            "b1": np.asarray(bundle.eval.b1, dtype=np.float32).tolist(),
            "w2": np.asarray(bundle.eval.w2, dtype=np.float32).tolist(),
        },
        "rollout": {
            "active": bool(np.asarray(bundle.rollout.active).item()),
            "u_proj": np.asarray(bundle.rollout.u_proj, dtype=np.float32).tolist(),
            "v_proj": np.asarray(bundle.rollout.v_proj, dtype=np.float32).tolist(),
            "act_bias": np.asarray(bundle.rollout.act_bias, dtype=np.float32).tolist(),
        },
    }


def residual_bundle_from_dict(data: dict[str, Any]) -> ResidualBundle:
    """Deserialize `ResidualBundle` from a parsed `NC2000_RESIDUAL_JSON` dictionary."""
    ev = data["eval"]
    ro = data["rollout"]
    return ResidualBundle(
        eval=EvalResidualWeights(
            active=jnp.bool_(bool(ev["active"])),
            w1=jnp.array(ev["w1"], dtype=jnp.float32).reshape(EVAL_HIDDEN_DIM, EVAL_FEAT_DIM),
            b1=jnp.array(ev["b1"], dtype=jnp.float32).reshape(EVAL_HIDDEN_DIM),
            w2=jnp.array(ev["w2"], dtype=jnp.float32).reshape(EVAL_HIDDEN_DIM),
        ),
        rollout=RolloutResidualWeights(
            active=jnp.bool_(bool(ro["active"])),
            u_proj=jnp.array(ro["u_proj"], dtype=jnp.float32).reshape(
                ROLLOUT_RANK, ROLLOUT_STATE_DIM
            ),
            v_proj=jnp.array(ro["v_proj"], dtype=jnp.float32).reshape(
                ROLLOUT_RANK, ROLLOUT_ACT_DIM
            ),
            act_bias=jnp.array(ro["act_bias"], dtype=jnp.float32).reshape(ROLLOUT_ACT_DIM),
        ),
    )


def export_residual_bundle_json(bundle: ResidualBundle, path: str | Path) -> None:
    """Write `ResidualBundle` to a JSON file loadable by `nc2000-bot` (`NC2000_RESIDUAL_JSON`)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(residual_bundle_to_dict(bundle), indent=2) + "\n")


def load_residual_bundle_json(path: str | Path) -> ResidualBundle:
    """Load `ResidualBundle` from a JSON file (`NC2000_RESIDUAL_JSON`)."""
    data = json.loads(Path(path).read_text())
    return residual_bundle_from_dict(data)


# =========================================================================
# 2. Branchless 32-D Multi-Body Feature Extraction (`extract_eval_features`)
# =========================================================================


def extract_eval_features_with_eff(
    state: CompactBattle,
    table: MatchTable,
    side: int,
    act_eff: tuple[jax.Array, jax.Array] | jax.Array,
) -> jax.Array:
    """Branchless port of `nc2000_bot::eval::extract_eval_features_with_eff` (`float32[32]`).

    - `x[0..16]`: Non-negative Multi-Body Synergy Features (monotonic >= 0 for `side`)
    - `x[16..32]`: Non-negative Multi-Body Vulnerability / Discount Features (monotonic <= 0 for `side`)
    """
    foe = 1 - side
    own_alive_mask = _get1(state.alive_mask, side)
    foe_alive_mask = _get1(state.alive_mask, foe)
    own_alive_cnt = _popcount_3bit(own_alive_mask)
    foe_alive_cnt = _popcount_3bit(foe_alive_mask)

    own_act_slot = jnp.minimum(jnp.int32(2), _get1(state.active_idx, side))
    foe_act_slot = jnp.minimum(jnp.int32(2), _get1(state.active_idx, foe))
    own_act_idx = side * 3 + own_act_slot
    foe_act_idx = foe * 3 + foe_act_slot

    own_hp_val = _get2(state.hp_pool, own_act_idx, 0)
    foe_hp_val = _get2(state.hp_pool, foe_act_idx, 0)
    own_act_alive = ((own_alive_mask & (jnp.int32(1) << own_act_slot)) != 0) & (own_hp_val > 0)
    foe_act_alive = ((foe_alive_mask & (jnp.int32(1) << foe_act_slot)) != 0) & (foe_hp_val > 0)
    valid = (own_alive_cnt > 0) & (foe_alive_cnt > 0) & own_act_alive & foe_act_alive

    own_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, own_act_idx)).astype(jnp.float32)
    foe_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, foe_act_idx)).astype(jnp.float32)
    own_hp_frac = own_hp_val.astype(jnp.float32) / own_maxhp
    foe_hp_frac = foe_hp_val.astype(jnp.float32) / foe_maxhp

    eff_s = act_eff[side] if isinstance(act_eff, tuple) else _get1(act_eff, side)
    eff_f = act_eff[foe] if isinstance(act_eff, tuple) else _get1(act_eff, foe)
    own_eff = jnp.minimum(jnp.float32(1.5), eff_s.astype(jnp.float32))
    foe_eff = jnp.minimum(jnp.float32(1.5), eff_f.astype(jnp.float32))

    spd_own = _mon_action_speed(state, table, side, own_act_slot)
    spd_foe = _mon_action_speed(state, table, foe, foe_act_slot)
    faster = jnp.where(
        spd_own > spd_foe,
        jnp.float32(1.0),
        jnp.where(spd_own == spd_foe, jnp.float32(0.5), jnp.float32(0.0)),
    )

    own_b0 = _get2(state.boosts, own_act_idx, 0)
    own_b1 = _get2(state.boosts, own_act_idx, 1)
    own_b2 = _get2(state.boosts, own_act_idx, 2)
    own_b3 = _get2(state.boosts, own_act_idx, 3)
    own_b4 = _get2(state.boosts, own_act_idx, 4)
    own_b5 = _get2(state.boosts, own_act_idx, 5)
    own_b6 = _get2(state.boosts, own_act_idx, 6)

    pos_atk = jnp.maximum(jnp.int32(0), own_b0).astype(jnp.float32) / jnp.float32(6.0)
    pos_def = jnp.maximum(jnp.int32(0), own_b1).astype(jnp.float32) / jnp.float32(6.0)
    pos_spa = jnp.maximum(jnp.int32(0), own_b2).astype(jnp.float32) / jnp.float32(6.0)
    pos_spd = jnp.maximum(jnp.int32(0), own_b3).astype(jnp.float32) / jnp.float32(6.0)
    pos_spe = jnp.maximum(jnp.int32(0), own_b4).astype(jnp.float32) / jnp.float32(6.0)
    pos_eva = jnp.maximum(jnp.int32(0), own_b6).astype(jnp.float32) / jnp.float32(6.0)

    pos_off_boost = jnp.minimum(jnp.float32(1.5), pos_atk + pos_spa)
    pos_bulk_eva_boost = jnp.minimum(jnp.float32(1.5), pos_def + pos_spd + pos_eva)
    pos_boost_sum = jnp.minimum(
        jnp.float32(2.0), pos_off_boost + pos_spe + pos_bulk_eva_boost
    )

    neg_def_spd = (
        jnp.maximum(jnp.int32(0), -own_b1) + jnp.maximum(jnp.int32(0), -own_b3)
    ).astype(jnp.float32) / jnp.float32(6.0)
    neg_off_spe_acc = (
        jnp.maximum(jnp.int32(0), -own_b0)
        + jnp.maximum(jnp.int32(0), -own_b2)
        + jnp.maximum(jnp.int32(0), -own_b4)
        + jnp.maximum(jnp.int32(0), -own_b5)
    ).astype(jnp.float32) / jnp.float32(6.0)

    foe_pos_boosts_i = jnp.int32(0)
    for b_i in range(7):
        fb = _get2(state.boosts, foe_act_idx, b_i)
        foe_pos_boosts_i = foe_pos_boosts_i + jnp.maximum(jnp.int32(0), fb)
    foe_pos_boost_sum = jnp.minimum(
        jnp.float32(1.5), foe_pos_boosts_i.astype(jnp.float32) / jnp.float32(6.0)
    )

    own_vol = _get1(state.vol_flags, own_act_idx)
    foe_vol = _get1(state.vol_flags, foe_act_idx)

    own_sub_max = jnp.maximum(jnp.float32(1.0), own_maxhp * jnp.float32(0.25))
    sub_hp_frac = jnp.where(
        (own_vol & VolBit.SUBSTITUTE) != 0,
        jnp.clip(
            _get2(state.hp_pool, own_act_idx, 1).astype(jnp.float32) / own_sub_max,
            jnp.float32(0.0),
            jnp.float32(1.0),
        ),
        jnp.float32(0.0),
    )
    foe_sub_max = jnp.maximum(jnp.float32(1.0), foe_maxhp * jnp.float32(0.25))
    foe_sub_hp_frac = jnp.where(
        (foe_vol & VolBit.SUBSTITUTE) != 0,
        jnp.clip(
            _get2(state.hp_pool, foe_act_idx, 1).astype(jnp.float32) / foe_sub_max,
            jnp.float32(0.0),
            jnp.float32(1.0),
        ),
        jnp.float32(0.0),
    )

    own_st = _get1(state.status_packed, own_act_idx) & 0x07
    foe_st = _get1(state.status_packed, foe_act_idx) & 0x07
    is_asleep_or_frozen = jnp.where(
        (own_st == Status.SLP) | (own_st == Status.FRZ), jnp.float32(1.0), jnp.float32(0.0)
    )
    awake_factor = jnp.float32(1.0) - is_asleep_or_frozen

    # Generic move capabilities on own and foe active
    has_self_switch = jnp.float32(0.0)
    has_force_switch = jnp.float32(0.0)
    has_setup_off = jnp.float32(0.0)
    has_setup_spd_def = jnp.float32(0.0)
    has_sub_move = jnp.float32(0.0)
    has_heal_move = jnp.float32(0.0)
    has_sleep_move = jnp.float32(0.0)

    is_dis = (own_vol & VolBit.DISABLE) != 0
    dis_slot = (_get1(state.slot_refs, own_act_idx) >> 4) & 0x03
    ghost_mask = jnp.uint32(1) << (table.ghost_type.astype(jnp.uint32) & jnp.uint32(31))
    is_ghost = (_get1(table.type_mask, own_act_idx) & ghost_mask) != jnp.uint32(0)
    own_nm = _get1(table.num_moves, own_act_idx)

    for m_i in range(4):
        m_ok = (
            (jnp.int32(m_i) < own_nm)
            & (_get2(state.pp, own_act_idx, m_i) > 0)
            & (~(is_dis & (jnp.int32(m_i) == dis_slot)))
        )
        sk = _get2(table.move_special_kind, own_act_idx, m_i)
        mflags = _get2(table.move_flags, own_act_idx, m_i)
        p_st = _get2(table.move_primary_status, own_act_idx, m_i)
        base_acc = _get2(table.move_base_acc_pct, own_act_idx, m_i).astype(jnp.float32)

        has_self_switch = jnp.where(
            m_ok & (sk == 18) & (own_alive_cnt > 1), jnp.float32(1.0), has_self_switch
        )
        has_force_switch = jnp.where(
            m_ok & (sk == 7) & (foe_alive_cnt > 1), jnp.float32(1.0), has_force_switch
        )

        boosts_self = (mflags & MoveFlag.BOOSTS_SELF) != 0
        m_setup_off = jnp.float32(0.0)
        m_setup_spd_def = jnp.float32(0.0)
        for b_k in range(3):
            st_i = table.move_boost_stats[own_act_idx, m_i, b_k]
            d = table.move_boost_deltas[own_act_idx, m_i, b_k]
            b_val = jnp.minimum(jnp.float32(1.0), d.astype(jnp.float32) * jnp.float32(0.5))
            pos_d = boosts_self & (d > 0)
            is_off_stat = (st_i == 0) | (st_i == 2)
            is_other_stat = (~is_off_stat) & (st_i >= 0) & (st_i < 7)
            m_setup_off = jnp.where(pos_d & is_off_stat, jnp.maximum(m_setup_off, b_val), m_setup_off)
            m_setup_spd_def = jnp.where(
                pos_d & is_other_stat, jnp.maximum(m_setup_spd_def, b_val), m_setup_spd_def
            )

        # `else if ms.pseudo_kind == 5 && !is_ghost` (`sk == 5` is Curse)
        is_curse_non_ghost = (~boosts_self) & (sk == 5) & (~is_ghost)
        m_setup_off = jnp.where(
            is_curse_non_ghost, jnp.maximum(m_setup_off, jnp.float32(0.5)), m_setup_off
        )
        m_setup_spd_def = jnp.where(
            is_curse_non_ghost, jnp.maximum(m_setup_spd_def, jnp.float32(0.5)), m_setup_spd_def
        )
        # `else if ms.pseudo_kind == 8 && own_hp_frac > 0.5` (`sk == 6` is Belly Drum)
        is_bdrum = (~boosts_self) & (sk == 6) & (own_hp_frac > jnp.float32(0.5))
        m_setup_off = jnp.where(is_bdrum, jnp.maximum(m_setup_off, jnp.float32(1.0)), m_setup_off)

        has_setup_off = jnp.where(m_ok, jnp.maximum(has_setup_off, m_setup_off), has_setup_off)
        has_setup_spd_def = jnp.where(
            m_ok, jnp.maximum(has_setup_spd_def, m_setup_spd_def), has_setup_spd_def
        )

        is_sub = (mflags & MoveFlag.IS_SUBSTITUTE) != 0
        has_sub_move = jnp.where(
            m_ok & is_sub & (own_hp_frac > jnp.float32(0.25)), jnp.float32(1.0), has_sub_move
        )
        is_heal = (mflags & (MoveFlag.IS_HEAL_CYCLE | MoveFlag.IS_HALF_HEAL)) != 0
        has_heal_move = jnp.where(m_ok & is_heal, jnp.float32(1.0), has_heal_move)
        has_sleep_move = jnp.where(
            m_ok & (p_st == Status.SLP),
            jnp.maximum(has_sleep_move, base_acc / jnp.float32(100.0)),
            has_sleep_move,
        )

    has_setup_any = jnp.maximum(has_setup_off, has_setup_spd_def)

    foe_has_phaze = jnp.float32(0.0)
    foe_dis = (foe_vol & VolBit.DISABLE) != 0
    foe_dis_slot = (_get1(state.slot_refs, foe_act_idx) >> 4) & 0x03
    foe_nm = _get1(table.num_moves, foe_act_idx)
    for m_i in range(4):
        m_ok = (
            (jnp.int32(m_i) < foe_nm)
            & (_get2(state.pp, foe_act_idx, m_i) > 0)
            & (~(foe_dis & (jnp.int32(m_i) == foe_dis_slot)))
        )
        sk = _get2(table.move_special_kind, foe_act_idx, m_i)
        foe_has_phaze = jnp.where(
            m_ok & (sk == 7) & (own_alive_cnt > 1), jnp.float32(1.0), foe_has_phaze
        )

    # Bench metrics for own and foe
    flying_mask = jnp.uint32(1) << (table.flying_type.astype(jnp.uint32) & jnp.uint32(31))
    bench_hp_sum = jnp.float32(0.0)
    bench_best_eff = jnp.float32(0.0)
    bench_max_atk = jnp.float32(0.0)
    bench_max_spa = jnp.float32(0.0)
    bench_grounded_cnt = jnp.float32(0.0)

    for slot in range(3):
        b_idx = side * 3 + slot
        b_hp_val = _get2(state.hp_pool, b_idx, 0)
        b_ok = (
            (jnp.int32(slot) != own_act_slot)
            & ((own_alive_mask & (1 << slot)) != 0)
            & (b_hp_val > 0)
        )
        b_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, b_idx)).astype(jnp.float32)
        b_hp = b_hp_val.astype(jnp.float32) / b_maxhp
        bench_hp_sum = bench_hp_sum + jnp.where(b_ok, b_hp, jnp.float32(0.0))

        b_flying = (_get1(table.type_mask, b_idx) & flying_mask) != jnp.uint32(0)
        bench_grounded_cnt = bench_grounded_cnt + jnp.where(
            b_ok & (~b_flying), jnp.float32(0.5), jnp.float32(0.0)
        )
        eff = jnp.minimum(
            jnp.float32(1.5),
            best_hit_fraction(state, table, side, slot, foe, foe_act_slot, jnp.bool_(True)),
        )
        bench_best_eff = jnp.where(b_ok & (eff > bench_best_eff), eff, bench_best_eff)

        atk_stat = mon_stat(state, table, side, slot, 0, True, False).astype(jnp.float32) / jnp.float32(
            200.0
        )
        spa_stat = mon_stat(state, table, side, slot, 2, True, False).astype(jnp.float32) / jnp.float32(
            200.0
        )
        bench_max_atk = jnp.where(
            b_ok,
            jnp.maximum(bench_max_atk, jnp.minimum(jnp.float32(1.5), b_hp * atk_stat)),
            bench_max_atk,
        )
        bench_max_spa = jnp.where(
            b_ok,
            jnp.maximum(bench_max_spa, jnp.minimum(jnp.float32(1.5), b_hp * spa_stat)),
            bench_max_spa,
        )

    foe_bench_hp_sum = jnp.float32(0.0)
    for slot in range(3):
        b_idx = foe * 3 + slot
        b_hp_val = _get2(state.hp_pool, b_idx, 0)
        b_ok = (
            (jnp.int32(slot) != foe_act_slot)
            & ((foe_alive_mask & (1 << slot)) != 0)
            & (b_hp_val > 0)
        )
        b_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, b_idx)).astype(jnp.float32)
        foe_bench_hp_sum = foe_bench_hp_sum + jnp.where(
            b_ok, b_hp_val.astype(jnp.float32) / b_maxhp, jnp.float32(0.0)
        )

    pass_viability = (
        jnp.clip(
            own_hp_frac
            - foe_eff * (jnp.float32(1.0) - jnp.float32(0.5) * faster)
            + jnp.float32(0.25) * sub_hp_frac,
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        * awake_factor
    )
    safe_window = (
        jnp.clip(
            own_hp_frac - jnp.float32(1.2) * foe_eff + jnp.float32(0.3) * sub_hp_frac,
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        * awake_factor
    )

    own_screens_frac = jnp.minimum(
        jnp.float32(1.0),
        jnp.maximum(_get1(state.reflect_turns, side), _get1(state.lightscreen_turns, side)).astype(
            jnp.float32
        )
        / jnp.float32(5.0),
    )
    own_safeguard_frac = jnp.minimum(
        jnp.float32(1.0), _get1(state.safeguard_turns, side).astype(jnp.float32) / jnp.float32(5.0)
    )
    own_spikes_frac = (_get1(state.side_flags, side) & 0x03).astype(jnp.float32) / jnp.float32(3.0)
    foe_spikes_frac = (_get1(state.side_flags, foe) & 0x03).astype(jnp.float32) / jnp.float32(3.0)

    foe_status_any = jnp.where(foe_st != 0, jnp.float32(1.0), jnp.float32(0.0))
    foe_safeguard = jnp.where(_get1(state.safeguard_turns, foe) > 0, jnp.float32(1.0), jnp.float32(0.0))
    foe_seeded_or_cursed = jnp.where(
        (foe_vol & (VolBit.LEECH_SEED | VolBit.CURSE)) != 0, jnp.float32(1.0), jnp.float32(0.0)
    )

    # x[0..16]: Non-negative Multi-Body Synergy Features
    x0 = has_self_switch * pos_atk * bench_max_atk * pass_viability
    x1 = has_self_switch * pos_spa * bench_max_spa * pass_viability
    x2 = (
        has_self_switch
        * (pos_spe + jnp.float32(0.7) * pos_bulk_eva_boost)
        * bench_best_eff
        * pass_viability
    )
    x3 = has_self_switch * sub_hp_frac * bench_best_eff * awake_factor
    x4 = (
        has_self_switch
        * (
            has_setup_off * jnp.maximum(bench_max_atk, bench_max_spa)
            + (has_setup_spd_def + jnp.float32(0.5) * has_sub_move) * bench_best_eff
        )
        * safe_window
    )
    x5 = (
        pos_off_boost
        * own_eff
        * (jnp.float32(0.4) + jnp.float32(0.6) * faster + jnp.float32(0.5) * sub_hp_frac)
        * (jnp.float32(1.0) - jnp.float32(0.7) * foe_has_phaze)
        * awake_factor
    )
    x6 = (
        pos_bulk_eva_boost
        * own_hp_frac
        * (jnp.float32(0.4) + jnp.float32(0.6) * has_heal_move + jnp.float32(0.4) * sub_hp_frac)
        * (jnp.float32(1.0) - jnp.float32(0.7) * foe_has_phaze)
        * awake_factor
    )
    x7 = (
        sub_hp_frac
        * (own_eff + jnp.float32(0.5) * pos_boost_sum + jnp.float32(0.4) * has_setup_any)
        * awake_factor
    )
    x8 = (
        has_setup_any
        * safe_window
        * jnp.maximum(own_eff, jnp.float32(0.6) * bench_best_eff * has_self_switch)
        * (jnp.float32(1.0) - jnp.minimum(jnp.float32(1.0), pos_off_boost))
        * (jnp.float32(1.0) - jnp.float32(0.7) * foe_has_phaze)
    )
    x9 = (
        has_force_switch
        * (foe_pos_boost_sum + foe_sub_hp_frac)
        * jnp.clip(own_hp_frac - foe_eff + jnp.float32(0.35), jnp.float32(0.0), jnp.float32(1.0))
        * awake_factor
    )
    x10 = (
        has_force_switch
        * foe_spikes_frac
        * jnp.minimum(jnp.float32(1.0), jnp.float32(0.5) * foe_bench_hp_sum)
        * jnp.clip(own_hp_frac - foe_eff + jnp.float32(0.25), jnp.float32(0.0), jnp.float32(1.0))
        * awake_factor
    )
    x11 = own_screens_frac * jnp.minimum(
        jnp.float32(1.2), jnp.float32(0.5) * own_hp_frac + jnp.float32(0.35) * bench_hp_sum
    )
    x12 = own_safeguard_frac * (
        jnp.float32(0.5) * own_hp_frac + jnp.float32(0.5) * has_setup_any
    )
    x13 = (
        has_sleep_move
        * (jnp.float32(1.0) - foe_status_any)
        * (jnp.float32(1.0) - foe_safeguard)
        * (jnp.float32(1.0) - jnp.minimum(jnp.float32(1.0), jnp.ceil(foe_sub_hp_frac)))
        * (jnp.float32(0.4) + jnp.float32(0.6) * faster)
        * awake_factor
    )
    x14 = (
        foe_seeded_or_cursed
        * own_hp_frac
        * (jnp.float32(0.6) + jnp.float32(0.4) * sub_hp_frac + jnp.float32(0.4) * has_heal_move)
    )
    x15 = (
        faster
        * jnp.clip(own_eff - jnp.float32(0.45), jnp.float32(0.0), jnp.float32(1.0))
        * jnp.clip(own_hp_frac - foe_eff + jnp.float32(0.3), jnp.float32(0.0), jnp.float32(1.0))
    )

    # x[16..32]: Non-negative Multi-Body Vulnerability / Discount Features
    perish_turns = _get2(state.timers, own_act_idx, 3)
    perish_urgency = jnp.where(
        ((own_vol & VolBit.PERISH_SONG) != 0) & (perish_turns > 0),
        (jnp.float32(4.0) - jnp.minimum(jnp.int32(3), perish_turns).astype(jnp.float32))
        / jnp.float32(3.0),
        jnp.float32(0.0),
    )
    is_trapped = jnp.where(
        (own_vol & VolBit.TRAPPED_OR_LOCKED) != 0, jnp.float32(1.0), jnp.float32(0.0)
    )
    is_encored = jnp.where((own_vol & VolBit.ENCORE) != 0, jnp.float32(1.0), jnp.float32(0.0))
    is_disabled = jnp.where(is_dis, jnp.float32(1.0), jnp.float32(0.0))
    is_confused = jnp.where((own_vol & VolBit.CONFUSION) != 0, jnp.float32(1.0), jnp.float32(0.0))
    is_paralyzed = jnp.where(own_st == Status.PAR, jnp.float32(1.0), jnp.float32(0.0))
    has_escalating_residual = jnp.where(
        (own_st == Status.TOX)
        | ((own_vol & (VolBit.LEECH_SEED | VolBit.CURSE | VolBit.NIGHTMARE)) != 0),
        jnp.float32(1.0),
        jnp.float32(0.0),
    )
    foe_is_trapped = jnp.where(
        (foe_vol & VolBit.TRAPPED_OR_LOCKED) != 0, jnp.float32(1.0), jnp.float32(0.0)
    )

    x16 = perish_urgency * is_trapped
    x17 = (
        perish_urgency
        * (jnp.float32(1.0) - foe_is_trapped * (jnp.float32(1.0) - is_trapped))
        * (jnp.float32(0.6) * pos_boost_sum + jnp.float32(0.5) * sub_hp_frac)
    )
    x18 = (
        is_trapped
        * jnp.clip(foe_eff - own_eff, jnp.float32(0.0), jnp.float32(1.0))
        * jnp.minimum(jnp.float32(1.0), bench_hp_sum)
    )
    x19 = jnp.minimum(jnp.float32(1.0), is_encored + jnp.float32(0.6) * is_disabled) * (
        jnp.float32(0.3) * own_eff
        + jnp.float32(0.7) * pos_boost_sum
        + jnp.float32(0.3) * has_setup_any
    )
    slower_ko_risk = (jnp.float32(1.0) - faster) * jnp.clip(
        foe_eff - own_hp_frac + jnp.float32(0.15), jnp.float32(0.0), jnp.float32(1.0)
    )
    x20 = (
        pos_boost_sum
        * jnp.minimum(
            jnp.float32(1.2),
            is_asleep_or_frozen * (jnp.float32(1.0) - jnp.float32(0.5) * has_heal_move)
            + slower_ko_risk,
        )
        * (jnp.float32(1.0) - sub_hp_frac)
    )
    x21 = (
        (pos_boost_sum + sub_hp_frac)
        * foe_has_phaze
        * jnp.clip(foe_hp_frac - own_eff + jnp.float32(0.35), jnp.float32(0.0), jnp.float32(1.0))
    )
    x22 = own_eff * slower_ko_risk * (jnp.float32(1.0) - sub_hp_frac)
    forced_out_pressure = jnp.minimum(
        jnp.float32(1.5),
        jnp.clip(foe_eff - own_hp_frac + jnp.float32(0.2), jnp.float32(0.0), jnp.float32(1.0))
        + jnp.float32(0.5) * foe_has_phaze
        + jnp.float32(0.5) * perish_urgency,
    )
    x23 = own_spikes_frac * bench_grounded_cnt * forced_out_pressure
    x24 = is_confused * (
        jnp.float32(0.35) + jnp.float32(0.45) * is_paralyzed + jnp.float32(0.4) * pos_atk
    )
    x25 = has_escalating_residual * jnp.clip(
        jnp.float32(1.0) - jnp.float32(0.5) * own_eff, jnp.float32(0.2), jnp.float32(1.0)
    )
    x26 = neg_def_spd * foe_eff
    x27 = neg_off_spe_acc * (jnp.float32(0.3) + jnp.float32(0.7) * own_eff)
    x28 = foe_sub_hp_frac * (
        jnp.float32(1.0) - jnp.clip(own_eff - jnp.float32(0.25), jnp.float32(0.0), jnp.float32(1.0))
    )
    x29 = jnp.float32(0.0)
    x30 = jnp.float32(0.0)
    x31 = is_asleep_or_frozen * jnp.minimum(
        jnp.float32(1.2), jnp.float32(0.5) * foe_eff + jnp.float32(0.5) * foe_pos_boost_sum
    )

    feats = jnp.stack(
        [
            x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15,
            x16, x17, x18, x19, x20, x21, x22, x23,
            x24, x25, x26, x27, x28, x29, x30, x31,
        ],
        axis=0,
    )
    return jnp.where(valid, feats, jnp.zeros((EVAL_FEAT_DIM,), dtype=jnp.float32))


def extract_eval_features(
    state: CompactBattle,
    table: MatchTable,
    side: int,
) -> jax.Array:
    """Branchless port of `nc2000_bot::eval::extract_eval_features` (`float32[32]`)."""
    effs = []
    for s in (0, 1):
        act_slot = jnp.minimum(jnp.int32(2), _get1(state.active_idx, s))
        foe_s = 1 - s
        foe_slot = jnp.minimum(jnp.int32(2), _get1(state.active_idx, foe_s))
        s_alive = ((_get1(state.alive_mask, s) & (jnp.int32(1) << act_slot)) != 0) & (
            _get2(state.hp_pool, s * 3 + act_slot, 0) > 0
        )
        f_alive = ((_get1(state.alive_mask, foe_s) & (jnp.int32(1) << foe_slot)) != 0) & (
            _get2(state.hp_pool, foe_s * 3 + foe_slot, 0) > 0
        )
        raw_e = best_hit_fraction(state, table, s, act_slot, foe_s, foe_slot, jnp.bool_(True))
        effs.append(jnp.where(s_alive & f_alive, raw_e, jnp.float32(0.0)))
    return extract_eval_features_with_eff(state, table, side, (effs[0], effs[1]))


def eval_residual_forward(x: jax.Array, m: EvalResidualWeights) -> jax.Array:
    r"""Forward pass of the 2-layer residual MLP $\Delta_\theta(x) = W_2 \text{ReLU}(W_1 x + b_1)$."""
    h = jax.nn.relu(jnp.matmul(m.w1, x) + m.b1)
    out = jnp.sum(m.w2 * h)
    return jnp.where(m.active, out, jnp.float32(0.0))


def compact_eval_residual_sym(
    state: CompactBattle,
    table: MatchTable,
    act_eff: tuple[jax.Array, jax.Array] | jax.Array,
    m: EvalResidualWeights | None = None,
) -> jax.Array:
    r"""Strictly antisymmetric residual logit $\Delta_{\text{sym}}(s) = f_\theta(x(s, 0)) - f_\theta(x(s, 1))$."""
    if m is None:
        m = default_residual_bundle().eval
    both_alive = (_get1(state.alive_mask, 0) != 0) & (_get1(state.alive_mask, 1) != 0)
    x0 = extract_eval_features_with_eff(state, table, 0, act_eff)
    x1 = extract_eval_features_with_eff(state, table, 1, act_eff)
    diff = eval_residual_forward(x0, m) - eval_residual_forward(x1, m)
    return jnp.where(m.active & both_alive, diff, jnp.float32(0.0))


# =========================================================================
# 3. Rollout State & Action Capability Residual (`RolloutResidualWeights`)
# =========================================================================


def extract_rollout_state_and_actions(
    state: CompactBattle,
    table: MatchTable,
    side: int,
    legal_mask7: jax.Array,
    stay_best_dmg: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Branchless port of `nc2000_bot::eval::extract_rollout_state_and_actions`.

    Instead of variable-length `cs: &[u8]`, operates directly on fixed-shape
    action indices `0..7` masked by `legal_mask7: bool[7]`.
    Returns `(phi_state: float32[12], act_feats: float32[7, 6], any_nonzero: bool)`.
    """
    att_slot = jnp.minimum(jnp.int32(2), _get1(state.active_idx, side))
    att_idx = side * 3 + att_slot
    def_side = 1 - side
    def_slot = jnp.minimum(jnp.int32(2), _get1(state.active_idx, def_side))
    def_idx = def_side * 3 + def_slot

    me_hp = _get2(state.hp_pool, att_idx, 0)
    foe_hp = _get2(state.hp_pool, def_idx, 0)
    alive_both = (me_hp > 0) & (foe_hp > 0)

    maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, att_idx)).astype(jnp.float32)
    foe_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, def_idx)).astype(jnp.float32)
    own_hp_frac = me_hp.astype(jnp.float32) / maxhp
    foe_hp_frac = foe_hp.astype(jnp.float32) / foe_maxhp

    own_alive_mask = _get1(state.alive_mask, side)
    foe_alive_mask = _get1(state.alive_mask, def_side)
    bench_alive = _popcount_3bit(own_alive_mask) > 1
    foe_bench_alive = _popcount_3bit(foe_alive_mask) > 1

    own_pos_boosts_i = jnp.int32(0)
    foe_pos_boosts_i = jnp.int32(0)
    for b_i in range(7):
        own_pos_boosts_i = own_pos_boosts_i + jnp.maximum(
            jnp.int32(0), _get2(state.boosts, att_idx, b_i)
        )
        foe_pos_boosts_i = foe_pos_boosts_i + jnp.maximum(
            jnp.int32(0), _get2(state.boosts, def_idx, b_i)
        )
    own_pos_boosts = jnp.minimum(
        jnp.float32(1.5), own_pos_boosts_i.astype(jnp.float32) / jnp.float32(4.0)
    )
    foe_pos_boosts = jnp.minimum(
        jnp.float32(1.5), foe_pos_boosts_i.astype(jnp.float32) / jnp.float32(4.0)
    )

    me_vol = _get1(state.vol_flags, att_idx)
    foe_vol = _get1(state.vol_flags, def_idx)
    own_has_sub = jnp.where((me_vol & VolBit.SUBSTITUTE) != 0, jnp.float32(1.0), jnp.float32(0.0))
    foe_has_sub = jnp.where((foe_vol & VolBit.SUBSTITUTE) != 0, jnp.float32(1.0), jnp.float32(0.0))

    has_cap_move = jnp.bool_(False)
    own_has_self_switch = jnp.float32(0.0)
    for m_i in range(4):
        c_ok = legal_mask7[m_i]
        sk = _get2(table.move_special_kind, att_idx, m_i)
        mflags = _get2(table.move_flags, att_idx, m_i)
        p_st = _get2(table.move_primary_status, att_idx, m_i)
        is_bp = c_ok & (sk == 18) & bench_alive
        own_has_self_switch = jnp.where(is_bp, jnp.float32(1.0), own_has_self_switch)
        is_other_cap = c_ok & (
            ((sk == 7) & foe_bench_alive)
            | ((mflags & MoveFlag.BOOSTS_SELF) != 0)
            | (sk == 5)
            | (sk == 6)
            | ((mflags & MoveFlag.IS_SUBSTITUTE) != 0)
            | (((mflags & MoveFlag.IS_STATUS) != 0) & (p_st != 0))
        )
        has_cap_move = has_cap_move | is_bp | is_other_cap

    can_vol_switch = (
        (stay_best_dmg < jnp.float32(0.25))
        & (own_pos_boosts == jnp.float32(0.0))
        & (own_has_sub == jnp.float32(0.0))
        & bench_alive
    )

    foe_best_dmg = jnp.minimum(
        jnp.float32(1.5),
        best_hit_fraction(state, table, def_side, def_slot, side, att_slot, jnp.bool_(True)),
    )
    own_spd = _mon_action_speed(state, table, side, att_slot)
    foe_spd = _mon_action_speed(state, table, def_side, def_slot)
    speed_adv = jnp.where(
        own_spd > foe_spd,
        jnp.float32(1.0),
        jnp.where(own_spd < foe_spd, jnp.float32(-1.0), jnp.float32(0.0)),
    )

    bench_best_stat = jnp.float32(0.0)
    bench_best_eff = jnp.float32(0.0)
    for slot in range(3):
        b_idx = side * 3 + slot
        b_hp_val = _get2(state.hp_pool, b_idx, 0)
        b_ok = (
            bench_alive
            & (jnp.int32(slot) != att_slot)
            & ((own_alive_mask & (1 << slot)) != 0)
            & (b_hp_val > 0)
        )
        b_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, b_idx)).astype(jnp.float32)
        b_hp = b_hp_val.astype(jnp.float32) / b_maxhp
        atk_s = mon_stat(state, table, side, slot, 0, True, False).astype(jnp.float32)
        spa_s = mon_stat(state, table, side, slot, 2, True, False).astype(jnp.float32)
        cand_stat = jnp.minimum(
            jnp.float32(1.5), b_hp * jnp.maximum(atk_s, spa_s) / jnp.float32(200.0)
        )
        bench_best_stat = jnp.where(b_ok, jnp.maximum(bench_best_stat, cand_stat), bench_best_stat)
        eff = jnp.minimum(
            jnp.float32(1.5),
            best_hit_fraction(state, table, side, slot, def_side, def_slot, jnp.bool_(True)),
        )
        bench_best_eff = jnp.where(b_ok, jnp.maximum(bench_best_eff, eff), bench_best_eff)

    phi_state = jnp.stack(
        [
            jnp.float32(1.0),
            own_hp_frac,
            foe_hp_frac,
            jnp.minimum(jnp.float32(1.5), stay_best_dmg),
            foe_best_dmg,
            speed_adv,
            own_pos_boosts,
            foe_pos_boosts,
            own_has_sub,
            foe_has_sub,
            own_has_self_switch,
            bench_best_stat,
        ],
        axis=0,
    )

    ghost_mask = jnp.uint32(1) << (table.ghost_type.astype(jnp.uint32) & jnp.uint32(31))
    is_ghost = (_get1(table.type_mask, att_idx) & ghost_mask) != jnp.uint32(0)
    ground_mask = jnp.uint32(1) << (table.ground_type.astype(jnp.uint32) & jnp.uint32(31))
    def_is_ground = (_get1(table.type_mask, def_idx) & ground_mask) != jnp.uint32(0)

    foe_has_slp = jnp.bool_(False)
    for p_i in range(3):
        m_idx = def_side * 3 + p_i
        m_alive = ((foe_alive_mask & (1 << p_i)) != 0) & (_get2(state.hp_pool, m_idx, 0) > 0)
        m_slp = (_get1(state.status_packed, m_idx) & 0x07) == Status.SLP
        foe_has_slp = foe_has_slp | (m_alive & m_slp)

    foe_st_raw = _get1(state.status_packed, def_idx) & 0x07
    foe_sg_zero = _get1(state.safeguard_turns, def_side) == 0

    rows = []
    any_nonzero = jnp.bool_(False)
    for m_i in range(4):
        c_ok = legal_mask7[m_i]
        sk = _get2(table.move_special_kind, att_idx, m_i)
        mflags = _get2(table.move_flags, att_idx, m_i)
        p_st = _get2(table.move_primary_status, att_idx, m_i)
        m_type = _get2(table.move_type, att_idx, m_i)
        base_acc = _get2(table.move_base_acc_pct, att_idx, m_i).astype(jnp.float32)

        # a[0]: self_switch transfer
        cond0 = (
            c_ok
            & (sk == 18)
            & bench_alive
            & ((own_pos_boosts > jnp.float32(0.0)) | (own_has_sub > jnp.float32(0.0)))
        )
        receiver_edge = jnp.clip(
            jnp.maximum(bench_best_stat, bench_best_eff) - stay_best_dmg + jnp.float32(0.5),
            jnp.float32(0.1),
            jnp.float32(1.5),
        )
        a0 = jnp.where(
            cond0,
            (own_pos_boosts + jnp.float32(0.6) * own_has_sub) * receiver_edge,
            jnp.float32(0.0),
        )

        # a[1]: force_switch phaze against boosted/Substitute foe
        cond1 = (
            c_ok
            & (sk == 7)
            & foe_bench_alive
            & (own_hp_frac > foe_best_dmg)
            & ((foe_pos_boosts > jnp.float32(0.0)) | (foe_has_sub > jnp.float32(0.0)))
        )
        a1 = jnp.where(
            cond1, foe_pos_boosts + jnp.float32(0.75) * foe_has_sub, jnp.float32(0.0)
        )

        # a[2]: self-boost setup when healthy and un-maxed (< +2)
        cond2_win = (
            c_ok
            & (own_hp_frac > foe_best_dmg + jnp.float32(0.15))
            & (stay_best_dmg < jnp.float32(0.85))
        )
        boosts_self = (mflags & MoveFlag.BOOSTS_SELF) != 0
        b_gain_self = jnp.float32(0.0)
        for b_k in range(3):
            st_i = table.move_boost_stats[att_idx, m_i, b_k]
            d = table.move_boost_deltas[att_idx, m_i, b_k]
            valid_st = (st_i >= 0) & (st_i < 7) & (d > 0)
            cur_b = _get2(state.boosts, att_idx, jnp.clip(st_i, 0, 6))
            b_gain_self = b_gain_self + jnp.where(
                boosts_self & valid_st & (cur_b < 2),
                d.astype(jnp.float32) * jnp.float32(0.5),
                jnp.float32(0.0),
            )
        me_b0 = _get2(state.boosts, att_idx, 0)
        b_gain_curse = jnp.where(
            (~boosts_self) & (sk == 5) & (~is_ghost) & (me_b0 < 2),
            jnp.float32(0.75),
            jnp.float32(0.0),
        )
        b_gain_bdrum = jnp.where(
            (~boosts_self) & (sk == 6) & (me_b0 < 2) & (own_hp_frac > jnp.float32(0.6)),
            jnp.float32(1.5),
            jnp.float32(0.0),
        )
        b_gain = jnp.minimum(jnp.float32(1.5), b_gain_self + b_gain_curse + b_gain_bdrum)
        a2 = jnp.where(cond2_win & (b_gain > jnp.float32(0.0)), b_gain, jnp.float32(0.0))

        # a[3]: Substitute creation when faster or safe
        cond3 = (
            c_ok
            & ((mflags & MoveFlag.IS_SUBSTITUTE) != 0)
            & (own_has_sub == jnp.float32(0.0))
            & (own_hp_frac > jnp.float32(0.35))
            & ((speed_adv > jnp.float32(0.0)) | (foe_best_dmg < jnp.float32(0.25)))
            & (stay_best_dmg < jnp.float32(0.85))
        )
        a3 = jnp.where(cond3, jnp.float32(1.0), jnp.float32(0.0))

        # a[4]: Status infliction (Sleep / Paralysis / Toxic) on healthy foe
        slp_blocked = (p_st == Status.SLP) & foe_has_slp
        ground_blocked = (m_type == table.electric_type) & def_is_ground
        cond4 = (
            c_ok
            & ((mflags & MoveFlag.IS_STATUS) != 0)
            & (p_st != 0)
            & (foe_st_raw == 0)
            & (foe_has_sub == jnp.float32(0.0))
            & foe_sg_zero
            & (stay_best_dmg < jnp.float32(0.85))
            & (~slp_blocked)
            & (~ground_blocked)
        )
        w_st = jnp.where(p_st == Status.SLP, jnp.float32(1.0), jnp.float32(0.6))
        a4 = jnp.where(cond4, (base_acc / jnp.float32(100.0)) * w_st, jnp.float32(0.0))
        a5 = jnp.float32(0.0)

        row = jnp.stack([a0, a1, a2, a3, a4, a5], axis=0)
        rows.append(row)
        any_nonzero = any_nonzero | jnp.any(row > jnp.float32(0.0))

    for slot in range(3):
        c_ok = legal_mask7[4 + slot] & can_vol_switch
        cand_idx = side * 3 + slot
        cand_hp = _get2(state.hp_pool, cand_idx, 0).astype(jnp.float32) / jnp.maximum(
            jnp.int32(1), _get1(table.maxhp, cand_idx)
        ).astype(jnp.float32)
        cand_off = jnp.minimum(
            jnp.float32(1.2),
            best_hit_fraction(state, table, side, slot, def_side, def_slot, jnp.bool_(True)),
        )
        foe_vs_cand = jnp.minimum(
            jnp.float32(1.2),
            best_hit_fraction(state, table, def_side, def_slot, side, slot, jnp.bool_(True)),
        )
        cond5 = (
            c_ok
            & (cand_hp > foe_vs_cand + jnp.float32(0.2))
            & (cand_off > stay_best_dmg + jnp.float32(0.25))
        )
        a5 = jnp.where(
            cond5,
            jnp.maximum(jnp.float32(0.0), cand_off - stay_best_dmg - jnp.float32(0.5) * foe_vs_cand),
            jnp.float32(0.0),
        )
        z = jnp.float32(0.0)
        row = jnp.stack([z, z, z, z, z, a5], axis=0)
        rows.append(row)
        any_nonzero = any_nonzero | (a5 > jnp.float32(0.0))

    act_feats = jnp.stack(rows, axis=0)  # float32[7, 6]
    valid = alive_both & (has_cap_move | can_vol_switch) & any_nonzero
    return (
        jnp.where(valid, phi_state, jnp.zeros_like(phi_state)),
        jnp.where(valid, act_feats, jnp.zeros_like(act_feats)),
        valid,
    )


def compute_rollout_action_residuals(
    state: CompactBattle,
    table: MatchTable,
    side: int,
    legal_mask7: jax.Array,
    stay_best_dmg: jax.Array,
    m: RolloutResidualWeights,
) -> jax.Array:
    r"""Evaluate the rank-4 bilinear rollout action-residual $\Delta Q_\psi(s, c)$ for all `7` action slots."""
    phi_s, act_feats, valid = extract_rollout_state_and_actions(
        state, table, side, legal_mask7, stay_best_dmg
    )
    u = jnp.matmul(m.u_proj, phi_s)  # float32[4]
    w_eff = m.act_bias + jnp.matmul(u, m.v_proj)  # float32[6]
    deltas = jnp.matmul(act_feats, w_eff)  # float32[7]
    return jnp.where(m.active & valid & legal_mask7, deltas, jnp.zeros((7,), dtype=jnp.float32))


# =========================================================================
# 4. Pure-JAX Constrained Residual Trainer (`train_residual_bundle_jax`)
# =========================================================================


class EvalResidualTrainParams(NamedTuple):
    """Unconstrained raw parameters mapped to constrained `EvalResidualWeights`."""

    raw_w1_syn: jax.Array  # float32[8, 16]
    raw_w1_vul: jax.Array  # float32[8, 16]
    raw_b1: jax.Array  # float32[16]
    raw_w2_syn: jax.Array  # float32[8]
    raw_w2_vul: jax.Array  # float32[8]


def effective_eval_residual_weights(
    params: EvalResidualTrainParams, active: bool = True
) -> EvalResidualWeights:
    """Project raw parameters onto the monotonic, zero-preserving cone of `EvalResidualWeights`:
    - `w1[0..8, 0..16] >= 0`, `w1[0..8, 16..32] = 0`
    - `w1[8..16, 16..32] >= 0`, `w1[8..16, 0..16] = 0`
    - `w1[:, 29:31] = 0`
    - `b1 <= 0` so `ReLU(w1 @ 0 + b1) = 0 -> f(0) == 0` identically
    - `w2[0..8] >= 0`, `w2[8..16] <= 0`
    """
    w1_syn = jnp.maximum(jnp.float32(0.0), params.raw_w1_syn)
    w1_vul = jnp.maximum(jnp.float32(0.0), params.raw_w1_vul)
    # Mask reserved features x[29..31] (indices 13..15 within the 16-wide vulnerability slice)
    vul_mask = jnp.array(
        [1.0 if j not in (13, 14) else 0.0 for j in range(16)], dtype=jnp.float32
    )[None, :]
    w1_vul = w1_vul * vul_mask

    z8x16 = jnp.zeros((8, 16), dtype=jnp.float32)
    top = jnp.concatenate([w1_syn, z8x16], axis=1)
    bot = jnp.concatenate([z8x16, w1_vul], axis=1)
    w1 = jnp.concatenate([top, bot], axis=0)
    b1 = -jnp.maximum(jnp.float32(0.0), -params.raw_b1)
    w2 = jnp.concatenate(
        [
            jnp.maximum(jnp.float32(0.0), params.raw_w2_syn),
            -jnp.maximum(jnp.float32(0.0), params.raw_w2_vul),
        ],
        axis=0,
    )
    return EvalResidualWeights(active=jnp.bool_(active), w1=w1, b1=b1, w2=w2)


class ResidualTrainingMetrics(NamedTuple):
    """Validation metrics before and after `train_residual_bundle_jax`."""

    base_rms: float
    trained_rms: float
    base_mse: float
    trained_mse: float
    max_antisymmetry_err: float
    zero_input_out: float


def train_residual_bundle_jax(
    x0: jax.Array,
    x1: jax.Array,
    base_logit: jax.Array,
    target_v: jax.Array,
    is_1v1_unboosted: jax.Array | None = None,
    epochs: int = 200,
    lr: float = 3e-3,
    weight_decay: float = 2e-3,
    l2_out: float = 0.04,
    seed: int = 20261031,
    init_from_default: bool = True,
) -> tuple[ResidualBundle, ResidualTrainingMetrics]:
    """Train `EvalResidualWeights` in pure JAX with exact structural monotonicity and $f(0) = 0$.

    Mirrors `tools/train-residual-nn.py` (`MSE + 0.25 * BCE + l2_out * E[delta^2 * (1 + 2 * 1v1)]`)
    and returns a drop-in `ResidualBundle` ready to export via `export_residual_bundle_json`.
    """
    n = x0.shape[0]
    if is_1v1_unboosted is None:
        is_1v1_unboosted = jnp.zeros((n,), dtype=jnp.float32)

    if init_from_default:
        def_ev = default_residual_bundle().eval
        params = EvalResidualTrainParams(
            raw_w1_syn=def_ev.w1[:8, :16],
            raw_w1_vul=def_ev.w1[8:, 16:],
            raw_b1=def_ev.b1,
            raw_w2_syn=def_ev.w2[:8],
            raw_w2_vul=-def_ev.w2[8:],
        )
    else:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        params = EvalResidualTrainParams(
            raw_w1_syn=jax.random.uniform(k1, (8, 16), dtype=jnp.float32) * jnp.float32(0.20),
            raw_w1_vul=jax.random.uniform(k2, (8, 16), dtype=jnp.float32) * jnp.float32(0.20),
            raw_b1=jnp.zeros((16,), dtype=jnp.float32),
            raw_w2_syn=jnp.full((8,), 0.05, dtype=jnp.float32),
            raw_w2_vul=jnp.full((8,), 0.05, dtype=jnp.float32),
        )

    def _forward_sym_batch(p: EvalResidualTrainParams, bx0: jax.Array, bx1: jax.Array) -> jax.Array:
        ew = effective_eval_residual_weights(p, active=True)
        h0 = jax.nn.relu(jnp.matmul(bx0, ew.w1.T) + ew.b1[None, :])
        h1 = jax.nn.relu(jnp.matmul(bx1, ew.w1.T) + ew.b1[None, :])
        return jnp.matmul(h0 - h1, ew.w2)

    def _loss_fn(p: EvalResidualTrainParams) -> jax.Array:
        delta = _forward_sym_batch(p, x0, x1)
        logits = base_logit + delta
        preds = jax.nn.sigmoid(logits)
        mse = jnp.mean((preds - target_v) ** 2)
        # Stable BCE with logits: max(logits, 0) - logits * y + log(1 + exp(-|logits|))
        bce = jnp.mean(
            jnp.maximum(logits, jnp.float32(0.0))
            - logits * target_v
            + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        )
        reg = jnp.float32(l2_out) * jnp.mean(
            (delta**2) * (jnp.float32(1.0) + jnp.float32(2.0) * is_1v1_unboosted)
        )
        return mse + jnp.float32(0.25) * bce + reg

    grad_fn = jax.jit(jax.value_and_grad(_loss_fn))

    # Pure-JAX AdamW state
    m_state = jax.tree_util.tree_map(jnp.zeros_like, params)
    v_state = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1_adam = jnp.float32(0.9)
    b2_adam = jnp.float32(0.999)
    eps_adam = jnp.float32(1e-8)

    @jax.jit
    def _adamw_step(
        p: EvalResidualTrainParams,
        m_s: EvalResidualTrainParams,
        v_s: EvalResidualTrainParams,
        step_i: jax.Array,
    ):
        loss_val, grads = grad_fn(p)
        t_f = step_i.astype(jnp.float32)
        m_next = jax.tree_util.tree_map(
            lambda m_leaf, g_leaf: b1_adam * m_leaf + (jnp.float32(1.0) - b1_adam) * g_leaf,
            m_s,
            grads,
        )
        v_next = jax.tree_util.tree_map(
            lambda v_leaf, g_leaf: b2_adam * v_leaf + (jnp.float32(1.0) - b2_adam) * (g_leaf**2),
            v_s,
            grads,
        )
        m_hat_scale = jnp.float32(1.0) / (jnp.float32(1.0) - (b1_adam**t_f))
        v_hat_scale = jnp.float32(1.0) / (jnp.float32(1.0) - (b2_adam**t_f))

        def _update_leaf(p_leaf: jax.Array, m_leaf: jax.Array, v_leaf: jax.Array) -> jax.Array:
            m_hat = m_leaf * m_hat_scale
            v_hat = v_leaf * v_hat_scale
            decayed = p_leaf * (jnp.float32(1.0) - jnp.float32(lr * weight_decay))
            return decayed - jnp.float32(lr) * m_hat / (jnp.sqrt(v_hat) + eps_adam)

        p_next = jax.tree_util.tree_map(_update_leaf, p, m_next, v_next)
        return p_next, m_next, v_next, loss_val

    for ep in range(1, epochs + 1):
        params, m_state, v_state, _ = _adamw_step(
            params, m_state, v_state, jnp.int32(ep)
        )

    trained_eval = effective_eval_residual_weights(params, active=True)
    base_preds = jax.nn.sigmoid(base_logit)
    trained_delta = _forward_sym_batch(params, x0, x1)
    trained_preds = jax.nn.sigmoid(base_logit + trained_delta)

    base_mse = float(jnp.mean((base_preds - target_v) ** 2))
    trained_mse = float(jnp.mean((trained_preds - target_v) ** 2))
    rev_delta = _forward_sym_batch(params, x1, x0)
    max_asym = float(jnp.max(jnp.abs(trained_delta + rev_delta)))
    zero_out = float(eval_residual_forward(jnp.zeros((EVAL_FEAT_DIM,), dtype=jnp.float32), trained_eval))

    bundle = ResidualBundle(eval=trained_eval, rollout=zero_rollout_residual_weights())
    metrics = ResidualTrainingMetrics(
        base_rms=float(np.sqrt(base_mse)),
        trained_rms=float(np.sqrt(trained_mse)),
        base_mse=base_mse,
        trained_mse=trained_mse,
        max_antisymmetry_err=max_asym,
        zero_input_out=zero_out,
    )
    return bundle, metrics
