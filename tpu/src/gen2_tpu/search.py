"""TPU-accelerated offline search, `Playout::Heavy`, `EvalWeights`, and RM+ solver.

Mirrors `nc2000_bot::{eval, mcts, smmcts, fork, preview}` for offline TPU evaluation:
1. `EvalWeights` & `eval_compact_leaf`: Branchless 32-bit port of `nc2000_bot::eval::compact_eval_leaf`
   (material `hp`/`alive`, status penalties with sleep-clock scaling, active stat `boost`s,
   `substitute`, `threat` (`best_hit_fraction`), `pp` ratio, benched `spikes` hazard exposure,
   and 1v1 last-mon `race` margin).
2. `sample_heavy_playout_action`: Branchless port of `Playout::Heavy` (`eps=0.2`, `turns=8`)
   epsilon-greedy max-damage action picker (`compact_ordinary_pick` / `compact_greedy_pick`).
3. `sharded_heavy_rollout`: Matchup-grouped batch rollout operating directly on 120-byte
   `PackedBattle` (`uint32[B, 30]`) states with a single shared `MatchTable` per matchup.
4. `evaluate_root_payoff_matrix` & `solve_rm_plus_jax`: Counterfactual joint-action (`|A0| x |A1|`)
   fork evaluation and simultaneous-move Regret Matching+ equilibrium solver on TPU.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from .kernel import (
    _get1,
    _get2,
    apply_actions,
    best_hit_fraction,
    expected_hit_fraction,
    legal_actions_mask,
    mon_stat,
    step_switch,
)
from .prng import prng_random
from .residual import (
    ResidualBundle,
    RolloutResidualWeights,
    compact_eval_residual_sym,
    compute_rollout_action_residuals,
    default_residual_bundle,
    extract_eval_features_with_eff,
)
from .state import (
    CompactBattle,
    MatchTable,
    MoveFlag,
    PackedBattle,
    Status,
    VolBit,
    pack_battle,
    unpack_battle,
)
from .tpu_emu import make_tpu_mesh


def popcount_3bit(mask: jax.Array) -> jax.Array:
    """Count set bits in a 3-bit party `alive_mask` (`0..=7`)."""
    m = mask.astype(jnp.int32)
    return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1)


def mon_action_speed(
    state: CompactBattle,
    table: MatchTable,
    side: int | jax.Array,
    party_slot: int | jax.Array,
) -> jax.Array:
    """Return effective Speed stat (`stat_idx = 4`) of `(side, party_slot)`."""
    return mon_stat(state, table, side, party_slot, 4, False, False)


class EvalWeights(NamedTuple):
    """32-bit TPU static evaluation weights mirroring `nc2000_bot::eval::EvalWeights`."""

    hp: jax.Array
    alive: jax.Array
    brn: jax.Array
    par: jax.Array
    slp: jax.Array
    frz: jax.Array
    psn: jax.Array
    tox: jax.Array
    boost: jax.Array  # float32[7]
    threat: jax.Array
    pp: jax.Array
    scale: jax.Array
    couple_evasion: jax.Array  # bool
    slp_time_scale: jax.Array  # bool
    substitute: jax.Array
    race: jax.Array
    leaf_alpha: jax.Array
    spikes: jax.Array
    confusion: jax.Array
    residual_scale: jax.Array = jnp.float32(0.0)


def default_eval_weights(residual_scale: float = 0.0) -> EvalWeights:
    """Return the calibrated M17 `EvalWeights::default()` in `float32`."""
    return EvalWeights(
        hp=jnp.float32(1.0),
        alive=jnp.float32(0.5),
        brn=jnp.float32(0.35),
        par=jnp.float32(0.35),
        slp=jnp.float32(0.42),
        frz=jnp.float32(0.56),
        psn=jnp.float32(0.25),
        tox=jnp.float32(0.35),
        boost=jnp.array([0.15, 0.10, 0.15, 0.10, 0.15, 0.10, 0.10], dtype=jnp.float32),
        threat=jnp.float32(0.5),
        pp=jnp.float32(0.2),
        scale=jnp.float32(1.5),
        couple_evasion=jnp.bool_(True),
        slp_time_scale=jnp.bool_(True),
        substitute=jnp.float32(0.5),
        race=jnp.float32(3.0),
        leaf_alpha=jnp.float32(1.0),
        spikes=jnp.float32(1.5),
        confusion=jnp.float32(0.0),
        residual_scale=jnp.float32(residual_scale),
    )


def _side_score_with_eff(
    state: CompactBattle,
    table: MatchTable,
    w: EvalWeights,
    side: int,
) -> tuple[jax.Array, jax.Array]:
    """Branchless port of `compact_side_score_with_eff` for `side` in `{0, 1}`."""
    alive_m = _get1(state.alive_mask, side)
    active_slot = _get1(state.active_idx, side)
    flying_mask = jnp.uint32(1) << (table.flying_type.astype(jnp.uint32) & jnp.uint32(31))

    score = jnp.float32(0.0)
    pp_num = jnp.float32(0.0)
    pp_den = jnp.float32(0.0)
    spikes_exposed = jnp.int32(0)

    for slot in range(3):
        idx = side * 3 + slot
        hp_val = _get2(state.hp_pool, idx, 0)
        maxhp_val = jnp.maximum(jnp.int32(1), _get1(table.maxhp, idx))
        is_alive = ((alive_m & (1 << slot)) != 0) & (hp_val > 0)

        tmask = _get1(table.type_mask, idx)
        is_flying = (tmask & flying_mask) != jnp.uint32(0)
        exposed = is_alive & (jnp.int32(slot) != active_slot) & (~is_flying)
        spikes_exposed = spikes_exposed + jnp.where(exposed, jnp.int32(1), jnp.int32(0))

        hp_frac = hp_val.astype(jnp.float32) / maxhp_val.astype(jnp.float32)
        mon_mat = w.alive + w.hp * hp_frac

        st_pk = _get1(state.status_packed, idx)
        st_raw = st_pk & 0x07
        st_cnt = (st_pk >> 3) & 0x1F
        slp_pen = jnp.where(
            w.slp_time_scale,
            w.slp * jnp.clip(st_cnt, 0, 4).astype(jnp.float32) / jnp.float32(3.0),
            w.slp,
        )
        st_pen = jnp.where(
            st_raw == Status.BRN,
            w.brn,
            jnp.where(
                st_raw == Status.PAR,
                w.par,
                jnp.where(
                    st_raw == Status.SLP,
                    slp_pen,
                    jnp.where(
                        st_raw == Status.FRZ,
                        w.frz,
                        jnp.where(
                            st_raw == Status.PSN,
                            w.psn,
                            jnp.where(st_raw == Status.TOX, w.tox, jnp.float32(0.0)),
                        ),
                    ),
                ),
            ),
        )
        score = score + jnp.where(is_alive, mon_mat - st_pen, jnp.float32(0.0))

        nm = _get1(table.num_moves, idx)
        for m_i in range(4):
            valid_m = is_alive & (jnp.int32(m_i) < nm)
            pp_num = pp_num + jnp.where(
                valid_m, _get2(state.pp, idx, m_i).astype(jnp.float32), jnp.float32(0.0)
            )
            pp_den = pp_den + jnp.where(
                valid_m, _get2(table.max_pp, idx, m_i).astype(jnp.float32), jnp.float32(0.0)
            )

    score = score + jnp.where(
        pp_den > jnp.float32(0.0), w.pp * pp_num / jnp.maximum(pp_den, jnp.float32(1.0)), jnp.float32(0.0)
    )

    layers = _get1(state.side_flags, side) & 0x03
    spikes_amounts = jnp.array([0.0, 3.0, 4.0, 6.0], dtype=jnp.float32)
    spikes_frac = _get1(spikes_amounts, layers) / jnp.float32(24.0)
    score = score - w.spikes * spikes_frac * spikes_exposed.astype(jnp.float32)

    act_idx = side * 3 + active_slot
    act_hp = _get2(state.hp_pool, act_idx, 0)
    act_alive = ((alive_m & (jnp.int32(1) << active_slot)) != 0) & (act_hp > 0)

    act_boosts = _get1(state.boosts, act_idx).astype(jnp.float32)
    boost_score = jnp.sum(w.boost * act_boosts)

    act_vol = _get1(state.vol_flags, act_idx)
    has_sub = (act_vol & VolBit.SUBSTITUTE) != 0
    sub_score = jnp.where(has_sub, w.substitute, jnp.float32(0.0))

    has_conf = (act_vol & VolBit.CONFUSION) != 0
    conf_t = jnp.clip(_get2(state.timers, act_idx, 0), 0, 5).astype(jnp.float32)
    conf_att = jnp.maximum(jnp.float32(0.0), conf_t - jnp.float32(1.0))
    conf_pen = jnp.where(has_conf, w.confusion * jnp.float32(0.5) * conf_att, jnp.float32(0.0))

    foe_s = 1 - side
    foe_slot = _get1(state.active_idx, foe_s)
    foe_idx = foe_s * 3 + foe_slot
    foe_alive = ((_get1(state.alive_mask, foe_s) & (jnp.int32(1) << foe_slot)) != 0) & (
        _get2(state.hp_pool, foe_idx, 0) > 0
    )

    raw_eff = best_hit_fraction(state, table, side, active_slot, foe_s, foe_slot, w.couple_evasion)
    act_eff = jnp.where(act_alive & foe_alive, raw_eff, jnp.float32(0.0))
    threat_score = w.threat * jnp.minimum(jnp.float32(1.0), act_eff)

    score = score + jnp.where(
        act_alive, boost_score + sub_score - conf_pen + threat_score, jnp.float32(0.0)
    )
    return score, act_eff


def compact_race_margin(
    state: CompactBattle,
    table: MatchTable,
    w: EvalWeights,
) -> jax.Array:
    """Branchless port of `nc2000_bot::eval::compact_race_margin` (last-mon 1v1 KO race)."""
    alive0 = popcount_3bit(_get1(state.alive_mask, 0))
    alive1 = popcount_3bit(_get1(state.alive_mask, 1))
    is_1v1 = (alive0 == 1) & (alive1 == 1)

    slot0 = _get1(state.active_idx, 0)
    slot1 = _get1(state.active_idx, 1)
    idx0 = slot0
    idx1 = 3 + slot1
    hp0 = _get2(state.hp_pool, idx0, 0)
    hp1 = _get2(state.hp_pool, idx1, 0)
    both_pos = is_1v1 & (hp0 > 0) & (hp1 > 0)

    spd0 = mon_action_speed(state, table, 0, slot0)
    spd1 = mon_action_speed(state, table, 1, slot1)

    def _turns_to_ko(
        att_s: int,
        att_slot: jax.Array,
        def_s: int,
        def_slot: jax.Array,
        att_faster: jax.Array,
    ) -> jax.Array:
        e = best_hit_fraction(state, table, att_s, att_slot, def_s, def_slot, w.couple_evasion)
        att_idx = att_s * 3 + att_slot
        def_idx = def_s * 3 + def_slot
        def_hp = _get2(state.hp_pool, def_idx, 0).astype(jnp.float32)
        def_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, def_idx)).astype(jnp.float32)

        def_vol = _get1(state.vol_flags, def_idx)
        def_dis = (def_vol & VolBit.DISABLE) != 0
        def_dis_slot = (_get1(state.slot_refs, def_idx) >> 4) & 0x03
        def_nm = _get1(table.num_moves, def_idx)

        has_heal = jnp.bool_(False)
        for m_i in range(4):
            m_ok = (
                (jnp.int32(m_i) < def_nm)
                & (_get2(state.pp, def_idx, m_i) > 0)
                & (~(def_dis & (jnp.int32(m_i) == def_dis_slot)))
            )
            is_hc = (_get2(table.move_flags, def_idx, m_i) & MoveFlag.IS_HEAL_CYCLE) != 0
            has_heal = has_heal | (m_ok & is_hc)

        dmg_frac_max = e * def_hp / def_maxhp
        kill_now = (e >= jnp.float32(1.0)) & att_faster
        stalled_by_heal = has_heal & (dmg_frac_max < jnp.float32(0.5)) & (~kill_now)

        base_t = jnp.ceil(jnp.float32(1.0) / jnp.maximum(e, jnp.float32(1e-6)))
        has_sub = (def_vol & VolBit.SUBSTITUTE) != 0
        must_rech = (_get1(state.slot_refs, att_idx) & 0x80) != 0
        t = (
            base_t
            + jnp.where(has_sub, jnp.float32(1.0), jnp.float32(0.0))
            + jnp.where(must_rech, jnp.float32(1.0), jnp.float32(0.0))
        )

        st_pk = _get1(state.status_packed, att_idx)
        st_raw = st_pk & 0x07
        st_cnt = jnp.clip((st_pk >> 3) & 0x1F, 1, 4).astype(jnp.float32)
        att_nm = _get1(table.num_moves, att_idx)
        talks = jnp.bool_(False)
        for m_i in range(4):
            talks = talks | (
                (jnp.int32(m_i) < att_nm)
                & (_get2(state.pp, att_idx, m_i) > 0)
                & (_get2(table.move_special_kind, att_idx, m_i) == 10)
            )
        slp_add = jnp.where((st_raw == Status.SLP) & (~talks), st_cnt, jnp.float32(0.0))
        frz_add = jnp.where(st_raw == Status.FRZ, jnp.float32(4.0), jnp.float32(0.0))
        t = t + slp_add + frz_add

        inf = jnp.float32(1e6)
        return jnp.where((e <= jnp.float32(1e-6)) | stalled_by_heal, inf, t)

    def _surv_turns(side: int, slot: jax.Array) -> jax.Array:
        idx = side * 3 + slot
        st_raw = _get1(state.status_packed, idx) & 0x07
        ticks = (st_raw == Status.PSN) | (st_raw == Status.TOX) | (st_raw == Status.BRN)
        hp_f = _get2(state.hp_pool, idx, 0).astype(jnp.float32)
        maxhp_f = jnp.maximum(jnp.int32(1), _get1(table.maxhp, idx)).astype(jnp.float32)
        tick = jnp.maximum(jnp.float32(1.0), maxhp_f / jnp.float32(8.0))
        return jnp.where(ticks, jnp.ceil(hp_f / tick), jnp.float32(1e6))

    k0_raw = _turns_to_ko(0, slot0, 1, slot1, spd0 > spd1)
    k1_raw = _turns_to_ko(1, slot1, 0, slot0, spd1 > spd0)
    v0 = _surv_turns(0, slot0)
    v1 = _surv_turns(1, slot1)

    inf = jnp.float32(1e5)
    k0 = jnp.where(k0_raw <= v0, k0_raw, inf)
    k1 = jnp.where(k1_raw <= v1, k1_raw, inf)
    t0 = jnp.minimum(k0, v1)
    t1 = jnp.minimum(k1, v0)

    short_race = jnp.minimum(t0, t1) <= jnp.float32(3.0)
    f0 = t0 < inf
    f1 = t1 < inf
    spd_edge = jnp.where(
        spd0 > spd1,
        jnp.float32(0.5),
        jnp.where(spd0 < spd1, jnp.float32(-0.5), jnp.float32(0.0)),
    )
    raw_diff = jnp.where(
        f0 & (~f1),
        jnp.float32(2.0),
        jnp.where((~f0) & f1, jnp.float32(-2.0), (t1 - t0) + spd_edge),
    )
    clamped = jnp.clip(raw_diff, jnp.float32(-1.0), jnp.float32(1.0))
    return jnp.where(both_pos & short_race, clamped, jnp.float32(0.0))


def compact_base_logit_and_eff(
    state: CompactBattle,
    table: MatchTable,
    w: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Branchless port of `nc2000_bot::eval::compact_base_logit_and_eff`.

    Returns `(logit, (eff0, eff1))` where `logit = w.scale * (s0 - s1 + w.race * rm)`
    plus `w.residual_scale * compact_eval_residual_sym(...)` when `residual_bundle` is
    provided or `w.residual_scale != 0.0`.
    """
    if w is None:
        w = default_eval_weights()
    s0, eff0 = _side_score_with_eff(state, table, w, 0)
    s1, eff1 = _side_score_with_eff(state, table, w, 1)
    diff = s0 - s1
    rm = compact_race_margin(state, table, w)
    diff = diff + w.race * rm
    logit = w.scale * diff

    if residual_bundle is not None:
        res_sym = compact_eval_residual_sym(
            state, table, (eff0, eff1), residual_bundle.eval
        )
        logit = logit + w.residual_scale * res_sym
    return logit, (eff0, eff1)


def eval_compact_01(
    state: CompactBattle,
    table: MatchTable,
    w: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
) -> jax.Array:
    """Win-probability evaluation in `(0, 1)` from Side 0's perspective (`compact_eval01`)."""
    logit, _ = compact_base_logit_and_eff(state, table, w, residual_bundle=residual_bundle)
    return jax.nn.sigmoid(logit)


def eval_compact_leaf(
    state: CompactBattle,
    table: MatchTable,
    w: EvalWeights | None = None,
    residual_bundle: ResidualBundle | None = None,
) -> jax.Array:
    """Search leaf evaluation matching `nc2000_bot::eval::compact_eval_leaf` + terminal outcomes."""
    if w is None:
        w = default_eval_weights()
    p01 = eval_compact_01(state, table, w, residual_bundle=residual_bundle)
    shaped = jnp.float32(0.5) + w.leaf_alpha * (p01 - jnp.float32(0.5))
    return jnp.where(
        state.outcome == 1,
        jnp.float32(1.0),
        jnp.where(
            state.outcome == 2,
            jnp.float32(0.0),
            jnp.where(state.outcome == 3, jnp.float32(0.5), shaped),
        ),
    )


# =========================================================================
# 2. `Playout::Heavy` Epsilon-Greedy Max-Damage Action Picker
# =========================================================================


def sample_heavy_playout_action(
    state: CompactBattle,
    table: MatchTable,
    side: int,
    seed: jax.Array,
    eps: float = 0.2,
    rollout_residual: RolloutResidualWeights | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Branchless port of `Playout::Heavy`'s `compact_ordinary_pick` / `compact_greedy_pick`.

    - With probability `eps` (default `0.2`), picks uniformly among legal actions.
    - With probability `1 - eps`, picks the legal move maximizing `expected_hit_fraction`
      (plus optional rank-4 bilinear `rollout_residual` $\Delta Q_\psi(s, c)$, avoiding
      suicidal `Self-Destruct`/`Explosion` on the last alive Pokemon), or if all legal
      actions have `<= 0` score, samples uniformly among non-suicidal legal actions.
    - On forced/Baton-Pass switches (`request == 2`), scores candidate bench slots by
      passed-boost offense + HP fraction (or healthiest bench HP fraction when fainted).
    """
    mask = legal_actions_mask(state, table, side)  # bool[8]
    legal7 = mask[:7]
    n_legal = jnp.sum(legal7.astype(jnp.int32))

    # Draw 1: epsilon check (out of 10000)
    seed, r_eps = prng_random(seed, jnp.int32(10000))
    # Draw 2: uniform index draw
    seed, r_unif = prng_random(seed, jnp.maximum(jnp.int32(1), n_legal))

    # Map `r_unif` to the `r_unif`-th legal action in `0..7`
    prefix = jnp.cumsum(legal7.astype(jnp.int32)) - jnp.int32(1)
    unif_matches = legal7 & (prefix == r_unif)
    unif_act = jnp.where(
        n_legal > 0,
        jnp.argmax(unif_matches.astype(jnp.int32)).astype(jnp.int32),
        jnp.int32(7),
    )

    att_slot = _get1(state.active_idx, side)
    att_idx = side * 3 + att_slot
    def_side = 1 - side
    def_slot = _get1(state.active_idx, def_side)
    def_idx = def_side * 3 + def_slot

    alive_m = _get1(state.alive_mask, side)
    last_mon = popcount_3bit(alive_m) == 1

    # Non-suicidal uniform fallback when all moves have 0 expected damage
    is_sd = (_get1(table.move_flags, att_idx) & MoveFlag.IS_SELFDESTRUCT) != 0  # bool[4]
    suicidal7 = jnp.concatenate([last_mon & is_sd, jnp.zeros((3,), dtype=jnp.bool_)], axis=0)
    safe7 = legal7 & (~suicidal7)
    n_safe = jnp.sum(safe7.astype(jnp.int32))
    safe_mask = jnp.where(n_safe > 0, safe7, legal7)
    n_safe_eff = jnp.maximum(jnp.int32(1), jnp.sum(safe_mask.astype(jnp.int32)))
    r_safe = r_unif % n_safe_eff
    safe_prefix = jnp.cumsum(safe_mask.astype(jnp.int32)) - jnp.int32(1)
    safe_act = jnp.argmax((safe_mask & (safe_prefix == r_safe)).astype(jnp.int32)).astype(jnp.int32)

    # Score move slots 0..4 by expected_hit_fraction (unrolled to avoid nested while_loop under vmap)
    best_move_act = jnp.int32(0)
    best_move_score = jnp.float32(-1.0)
    move_base_scores = []
    for m_i in range(4):
        ok = legal7[m_i] & (~suicidal7[m_i])
        sc = expected_hit_fraction(
            state, table, side, att_slot, def_side, def_slot, m_i, jnp.bool_(True)
        )
        move_base_scores.append(sc)
        better = ok & (sc > best_move_score)
        best_move_act = jnp.where(better, jnp.int32(m_i), best_move_act)
        best_move_score = jnp.where(better, sc, best_move_score)

    if rollout_residual is not None:
        stay_dmg = jnp.maximum(jnp.float32(0.0), best_move_score)
        deltas = compute_rollout_action_residuals(
            state, table, side, legal7, stay_dmg, rollout_residual
        )
        base7 = jnp.concatenate(
            [jnp.stack(move_base_scores, axis=0), jnp.zeros((3,), dtype=jnp.float32)],
            axis=0,
        )
        for c_i in range(7):
            ok_c = legal7[c_i] & (~suicidal7[c_i]) & (deltas[c_i] > jnp.float32(0.0))
            tot_c = base7[c_i] + deltas[c_i]
            better_c = ok_c & (tot_c > best_move_score)
            best_move_act = jnp.where(better_c, jnp.int32(c_i), best_move_act)
            best_move_score = jnp.where(better_c, tot_c, best_move_score)

    greedy_move_act = jnp.where(best_move_score > jnp.float32(0.0), best_move_act, safe_act)

    # Score switch slots 4..7 (`slot = c - 4`) for request == 2 (or when only switches are legal)
    req = _get1(state.request, side)
    att_hp = _get2(state.hp_pool, att_idx, 0)
    att_b0 = _get2(state.boosts, att_idx, 0)
    att_b2 = _get2(state.boosts, att_idx, 2)
    foe_alive = ((_get1(state.alive_mask, def_side) & (jnp.int32(1) << def_slot)) != 0) & (
        _get2(state.hp_pool, def_idx, 0) > 0
    )
    passing_alive = (req == 2) & (att_hp > 0) & ((att_b0 > 0) | (att_b2 > 0)) & foe_alive

    best_sw_act = unif_act
    best_sw_score = jnp.float32(-1.0)
    for slot in range(3):
        act_code = jnp.int32(4 + slot)
        ok = legal7[4 + slot]
        c_idx = side * 3 + slot
        c_hp = _get2(state.hp_pool, c_idx, 0).astype(jnp.float32)
        c_maxhp = jnp.maximum(jnp.int32(1), _get1(table.maxhp, c_idx)).astype(jnp.float32)
        hp_f = c_hp / c_maxhp

        atk_s = mon_stat(state, table, side, slot, 0, False, False).astype(jnp.float32)
        spa_s = mon_stat(state, table, side, slot, 2, False, False).astype(jnp.float32)
        b0_mult = jnp.float32(1.0) + jnp.maximum(jnp.int32(0), att_b0).astype(jnp.float32) * jnp.float32(0.5)
        b2_mult = jnp.float32(1.0) + jnp.maximum(jnp.int32(0), att_b2).astype(jnp.float32) * jnp.float32(0.5)
        passed_stat = jnp.maximum(atk_s * b0_mult, spa_s * b2_mult) / jnp.float32(250.0)
        eff = best_hit_fraction(state, table, side, slot, def_side, def_slot, jnp.bool_(True))
        bp_sc = hp_f * (jnp.float32(0.4) + jnp.float32(0.6) * passed_stat + jnp.float32(0.8) * jnp.minimum(jnp.float32(1.5), eff))

        sc = jnp.where(passing_alive, bp_sc, hp_f)
        better = ok & (sc > best_sw_score)
        best_sw_act = jnp.where(better, act_code, best_sw_act)
        best_sw_score = jnp.where(better, sc, best_sw_score)

    has_legal_move = jnp.any(legal7[:4])
    greedy_act = jnp.where((req == 1) & has_legal_move, greedy_move_act, best_sw_act)

    eps_thresh = jnp.int32(round(eps * 10000.0))
    use_unif = (n_legal <= 1) | (r_eps < eps_thresh)
    chosen = jnp.where(req == 0, jnp.int32(7), jnp.where(use_unif, unif_act, greedy_act))
    return seed, chosen


# =========================================================================
# 3. Packed Matchup-Grouped Heavy Rollouts (`PackedBattle` + `Playout::Heavy`)
# =========================================================================


class HeavyRolloutResult(NamedTuple):
    """Result of a matchup-grouped `Playout::Heavy` rollout batch on `PackedBattle`."""

    final_packed: PackedBattle  # uint32[B, 30]
    leaf_values: jax.Array  # float32[B]
    outcomes: jax.Array  # int32[B]
    turns_played: jax.Array  # int32[B]


def heavy_rollout_packed_single(
    init_packed: PackedBattle,
    table: MatchTable,
    policy_seed: jax.Array,
    weights: EvalWeights,
    turns: int = 8,
    eps: float = 0.2,
) -> HeavyRolloutResult:
    """Run an `8`-turn `Playout::Heavy` rollout from a 120-byte `PackedBattle` state."""
    init_state = unpack_battle(init_packed)

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, p_seed = carry
        p_seed, a0 = sample_heavy_playout_action(st, table, 0, p_seed, eps=eps)
        p_seed, a1 = sample_heavy_playout_action(st, table, 1, p_seed, eps=eps)
        acts = jnp.stack([a0, a1], axis=0)
        next_st = apply_actions(st, table, acts)
        return (next_st, p_seed), None

    (final_st, _), _ = jax.lax.scan(
        _step, (init_state, policy_seed), None, length=turns
    )
    val = eval_compact_leaf(final_st, table, weights)
    return HeavyRolloutResult(
        final_packed=pack_battle(final_st),
        leaf_values=val,
        outcomes=final_st.outcome,
        turns_played=final_st.turn,
    )


def clone_packed_battle_batch(
    base_state: CompactBattle,
    batch_size: int,
    base_seed_int: int = 0x123456789ABCDEF0,
) -> tuple[PackedBattle, jax.Array]:
    """Pack a single `CompactBattle` into `PackedBattle` (`120 B`) and replicate into a
    single contiguous `[batch_size, 30]` `uint32` tensor with distinct per-trajectory seeds.
    """
    single_packed = pack_battle(base_state)  # uint32[30]
    words_batch = jnp.broadcast_to(single_packed.words, (batch_size, 30))

    idx = jnp.arange(batch_size, dtype=jnp.uint32)
    hi_base = jnp.uint32((base_seed_int >> 32) & 0xFFFFFFFF)
    lo_base = jnp.uint32(base_seed_int & 0xFFFFFFFF)

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
    words_batch = jnp.concatenate([words_batch[:, :28], battle_seeds], axis=-1)
    return PackedBattle(words=words_batch), policy_seeds


def _vmap_heavy_rollout_impl(
    batched_packed: PackedBattle,
    table: MatchTable,
    policy_seeds: jax.Array,
    weights: EvalWeights,
    turns: int = 8,
    eps: float = 0.2,
) -> HeavyRolloutResult:
    """Scan-outside-vmap `Playout::Heavy` rollout on TPU; vmap-outside-scan on CPU."""
    if jax.default_backend() != "tpu":
        return jax.vmap(
            lambda pk, ps: heavy_rollout_packed_single(
                pk, table, ps, weights, turns=turns, eps=eps
            ),
            in_axes=(0, 0),
        )(batched_packed, policy_seeds)

    init_st = unpack_battle(batched_packed)

    def _step(carry: tuple[CompactBattle, jax.Array], _: None):
        st, ps = carry
        ps, a0 = jax.vmap(
            lambda s, p: sample_heavy_playout_action(s, table, 0, p, eps=eps)
        )(st, ps)
        ps, a1 = jax.vmap(
            lambda s, p: sample_heavy_playout_action(s, table, 1, p, eps=eps)
        )(st, ps)
        acts = jnp.stack([a0, a1], axis=-1)
        next_st = jax.vmap(apply_actions, in_axes=(0, None, 0))(st, table, acts)
        return (next_st, ps), None

    (final_st, _), _ = jax.lax.scan(
        _step, (init_st, policy_seeds), None, length=turns
    )
    val = jax.vmap(eval_compact_leaf, in_axes=(0, None, None))(
        final_st, table, weights
    )
    return HeavyRolloutResult(
        final_packed=pack_battle(final_st),
        leaf_values=val,
        outcomes=final_st.outcome,
        turns_played=final_st.turn,
    )


@lru_cache(maxsize=16)
def _build_sharded_heavy_rollout_fn(mesh: Mesh, turns: int, eps: float):
    packed_spec = PackedBattle(words=P("core", None))
    table_spec = MatchTable(*([P()] * len(MatchTable._fields)))
    seed_spec = P("core", None)
    weights_spec = EvalWeights(*([P()] * len(EvalWeights._fields)))
    out_spec = HeavyRolloutResult(
        final_packed=packed_spec,
        leaf_values=P("core"),
        outcomes=P("core"),
        turns_played=P("core"),
    )

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(packed_spec, table_spec, seed_spec, weights_spec),
        out_specs=out_spec,
        check_vma=False,
    )
    def _per_core_fn(
        local_packed: PackedBattle,
        shared_table: MatchTable,
        local_seeds: jax.Array,
        shared_weights: EvalWeights,
    ):
        return _vmap_heavy_rollout_impl(
            local_packed, shared_table, local_seeds, shared_weights, turns=turns, eps=eps
        )

    return jax.jit(_per_core_fn)


def sharded_heavy_rollout(
    batched_packed: PackedBattle,
    table: MatchTable,
    policy_seeds: jax.Array,
    weights: EvalWeights | None = None,
    turns: int = 8,
    eps: float = 0.2,
    mesh: Mesh | None = None,
) -> HeavyRolloutResult:
    """Distribute matchup-grouped `Playout::Heavy` rollouts on `PackedBattle` (`uint32[B, 30]`)
    across all virtual TPU cores.
    """
    if weights is None:
        weights = default_eval_weights()
    if mesh is None:
        mesh = make_tpu_mesh(axis_names=("core",))
    fn = _build_sharded_heavy_rollout_fn(mesh, turns, eps)
    return fn(batched_packed, table, policy_seeds, weights)


# =========================================================================
# 4. Counterfactual Root Payoff Matrix & Regret Matching+ Equilibrium Solver
# =========================================================================


class RootEquilibriumResult(NamedTuple):
    """Simultaneous-move root evaluation and Nash equilibrium solved on TPU."""

    payoff_matrix: jax.Array  # float32[7, 7]
    legal_mask0: jax.Array  # bool[7]
    legal_mask1: jax.Array  # bool[7]
    sigma0: jax.Array  # float32[7]
    sigma1: jax.Array  # float32[7]
    game_value: jax.Array  # float32


def solve_rm_plus_jax(
    payoff_matrix: jax.Array,
    legal_mask0: jax.Array,
    legal_mask1: jax.Array,
    sweeps: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Branchless JAX port of `nc2000_bot::smmcts::solve_rm_plus` over a `7 x 7` payoff matrix.

    Computes the Nash equilibrium mixed strategies `(sigma0, sigma1)` and zero-sum
    `game_value` for Side 0 using Regret Matching+ with linear averaging.
    """
    m0 = legal_mask0.astype(jnp.float32)
    m1 = legal_mask1.astype(jnp.float32)
    u0 = m0 / jnp.maximum(jnp.sum(m0), jnp.float32(1.0))
    u1 = m1 / jnp.maximum(jnp.sum(m1), jnp.float32(1.0))

    init_carry = (
        jnp.zeros((7,), dtype=jnp.float32),  # r0
        jnp.zeros((7,), dtype=jnp.float32),  # r1
        jnp.zeros((7,), dtype=jnp.float32),  # sum0
        jnp.zeros((7,), dtype=jnp.float32),  # sum1
    )

    def _sweep(t_idx: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
        r0, r1, sum0, sum1 = carry
        s0_sum = jnp.sum(r0)
        s1_sum = jnp.sum(r1)
        p0 = jnp.where(s0_sum > jnp.float32(0.0), r0 / jnp.maximum(s0_sum, jnp.float32(1e-12)), u0)
        p1 = jnp.where(s1_sum > jnp.float32(0.0), r1 / jnp.maximum(s1_sum, jnp.float32(1e-12)), u1)

        # Side 0 action utilities against p1: u_a0 = M @ p1
        ev0 = jnp.matmul(payoff_matrix, p1)
        v0 = jnp.sum(p0 * ev0)
        r0_next = jnp.maximum(jnp.float32(0.0), r0 + (ev0 - v0)) * m0

        # Side 1 action utilities against p0 (maximizing 1 - payoff):
        ev1_loss = jnp.matmul(p0, payoff_matrix)
        v1_loss = jnp.sum(p1 * ev1_loss)
        # Regret for Side 1 is (1 - ev1_loss) - (1 - v1_loss) = v1_loss - ev1_loss
        r1_next = jnp.maximum(jnp.float32(0.0), r1 + (v1_loss - ev1_loss)) * m1

        w = (t_idx + 1).astype(jnp.float32)
        return (r0_next, r1_next, sum0 + w * p0, sum1 + w * p1)

    _, _, sum0, sum1 = jax.lax.fori_loop(0, sweeps, _sweep, init_carry)
    sigma0 = jnp.where(
        jnp.sum(sum0) > jnp.float32(0.0),
        sum0 / jnp.maximum(jnp.sum(sum0), jnp.float32(1e-12)),
        u0,
    )
    sigma1 = jnp.where(
        jnp.sum(sum1) > jnp.float32(0.0),
        sum1 / jnp.maximum(jnp.sum(sum1), jnp.float32(1e-12)),
        u1,
    )
    game_val = jnp.sum(sigma0[:, None] * payoff_matrix * sigma1[None, :])
    return sigma0, sigma1, game_val


@partial(
    jax.jit,
    static_argnames=("samples_per_cell", "rollout_turns", "rm_sweeps", "base_seed_int"),
)
def evaluate_root_payoff_matrix(
    root_state: CompactBattle,
    table: MatchTable,
    weights: EvalWeights | None = None,
    samples_per_cell: int = 32,
    rollout_turns: int = 8,
    rm_sweeps: int = 256,
    base_seed_int: int = 0xCAFEBABE12345678,
) -> RootEquilibriumResult:
    """Counterfactual root matrix probe (`fork.rs` / `preview.rs` / `root_policy.rs`).

    Evaluates all `7 x 7 = 49` joint root action pairs `(a0, a1)` across `samples_per_cell`
    paired chance seeds (`49 * samples_per_cell` parallel trajectories sharing one `MatchTable`),
    steps the forced root joint action `(a0, a1)`, runs `Playout::Heavy` for `rollout_turns`,
    scores leaves with `eval_compact_leaf`, and solves the Nash equilibrium via `solve_rm_plus_jax`.
    """
    if weights is None:
        weights = default_eval_weights()

    mask0 = legal_actions_mask(root_state, table, 0)[:7]
    mask1 = legal_actions_mask(root_state, table, 1)[:7]
    fb0 = jnp.argmax(mask0.astype(jnp.int32)).astype(jnp.int32)
    fb1 = jnp.argmax(mask1.astype(jnp.int32)).astype(jnp.int32)

    # Paired common random numbers across all 49 cells so every (a0, a1) cell faces
    # the exact same sequence of `samples_per_cell` battle & rollout PRNG seeds
    s_idx = jnp.arange(samples_per_cell, dtype=jnp.uint32)
    hi_base = jnp.uint32((base_seed_int >> 32) & 0xFFFFFFFF)
    lo_base = jnp.uint32(base_seed_int & 0xFFFFFFFF)
    b_seeds = jnp.stack(
        [
            hi_base ^ (s_idx * jnp.uint32(0x9E3779B9)),
            lo_base + (s_idx * jnp.uint32(0x85EBCA6B) + jnp.uint32(1)),
        ],
        axis=-1,
    )
    p_seeds = jnp.stack(
        [
            (hi_base + jnp.uint32(0xC2B2AE35)) ^ (s_idx * jnp.uint32(0x27D4EB2F)),
            lo_base ^ (s_idx * jnp.uint32(0x165667B1) + jnp.uint32(7)),
        ],
        axis=-1,
    )

    grid_a0, grid_a1 = jnp.meshgrid(
        jnp.arange(7, dtype=jnp.int32), jnp.arange(7, dtype=jnp.int32), indexing="ij"
    )
    flat_a0 = grid_a0.reshape(-1)  # [49]
    flat_a1 = grid_a1.reshape(-1)  # [49]

    # Flatten (49 cells, samples_per_cell) into a single batch [49 * samples_per_cell]
    # and run scan-outside-vmap so XLA TPU steps all trajectories together per turn
    rep_a0 = jnp.repeat(flat_a0, samples_per_cell)
    rep_a1 = jnp.repeat(flat_a1, samples_per_cell)
    tile_b_seeds = jnp.tile(b_seeds, (49, 1))
    tile_p_seeds = jnp.tile(p_seeds, (49, 1))

    if jax.default_backend() == "tpu":
        def _step_root_cell(
            a0: jax.Array, a1: jax.Array, b_seed: jax.Array
        ) -> PackedBattle:
            eff_a0 = _get1(mask0, a0)
            eff_a1 = _get1(mask1, a1)
            act0 = jnp.where(eff_a0, a0, fb0)
            act1 = jnp.where(eff_a1, a1, fb1)
            st0 = root_state._replace(prng_seed=b_seed)
            st1 = apply_actions(st0, table, jnp.stack([act0, act1], axis=0))
            return pack_battle(st1)

        packed_after_root = jax.vmap(_step_root_cell)(rep_a0, rep_a1, tile_b_seeds)
        rollout_res = _vmap_heavy_rollout_impl(
            packed_after_root,
            table,
            tile_p_seeds,
            weights,
            turns=rollout_turns,
            eps=0.2,
        )
        cell_sample_vals = rollout_res.leaf_values.reshape(49, samples_per_cell)
    else:
        def _eval_joint_sample(
            a0: jax.Array, a1: jax.Array, b_seed: jax.Array, p_seed: jax.Array
        ) -> jax.Array:
            act0 = jnp.where(_get1(mask0, a0), a0, fb0)
            act1 = jnp.where(_get1(mask1, a1), a1, fb1)
            st0 = root_state._replace(prng_seed=b_seed)
            st1 = apply_actions(st0, table, jnp.stack([act0, act1], axis=0))
            res = heavy_rollout_packed_single(
                pack_battle(st1),
                table,
                p_seed,
                weights,
                turns=rollout_turns,
                eps=0.2,
            )
            return res.leaf_values

        cell_sample_vals = jax.vmap(_eval_joint_sample)(
            rep_a0, rep_a1, tile_b_seeds, tile_p_seeds
        ).reshape(49, samples_per_cell)

    payoff_matrix = jnp.mean(cell_sample_vals, axis=-1).reshape(7, 7)
    valid_cells = mask0[:, None] & mask1[None, :]
    payoff_matrix = jnp.where(valid_cells, payoff_matrix, jnp.float32(0.5))

    sigma0, sigma1, game_val = solve_rm_plus_jax(payoff_matrix, mask0, mask1, sweeps=rm_sweeps)
    return RootEquilibriumResult(
        payoff_matrix=payoff_matrix,
        legal_mask0=mask0,
        legal_mask1=mask1,
        sigma0=sigma0,
        sigma1=sigma1,
        game_value=game_val,
    )


# =========================================================================
# 5. Depth-2 Simultaneous-Move Minimax / RM+ Oracle & Teacher Labeling
# =========================================================================


class Depth2OracleResult(NamedTuple):
    """Result of the Depth-2 counterfactual simultaneous-move oracle on TPU."""

    payoff_matrix: jax.Array  # float32[7, 7]
    legal_mask0: jax.Array  # bool[7]
    legal_mask1: jax.Array  # bool[7]
    sigma0: jax.Array  # float32[7]
    sigma1: jax.Array  # float32[7]
    game_value: jax.Array  # float32 (Depth-2 equilibrium value v_mcts)
    q_values0: jax.Array  # float32[7] (M @ sigma1)
    q_values1: jax.Array  # float32[7] (1 - sigma0^T @ M)
    ply1_game_value: jax.Array  # float32 (1-ply equilibrium value for comparison)


def _settle_forced_switch_if_needed(
    state: CompactBattle, table: MatchTable, seed: jax.Array
) -> tuple[CompactBattle, jax.Array]:
    """If either side is in `request == 2` (Baton Pass / faint replacement) after a turn step,
    settle the forced switch via greedy switch scoring (`eps = 0.0`) so the next ply starts
    at a simultaneous `request == 1` move stage.
    """
    needs_sw = ((_get1(state.request, 0) == 2) | (_get1(state.request, 1) == 2)) & (
        state.outcome == 0
    )
    seed1, sw0 = sample_heavy_playout_action(state, table, 0, seed, eps=0.0)
    seed2, sw1 = sample_heavy_playout_action(state, table, 1, seed1, eps=0.0)
    sw_state = step_switch(state, table, jnp.stack([sw0, sw1], axis=0))
    settled = jax.tree_util.tree_map(
        lambda a, b: jnp.where(needs_sw, a, b), sw_state, state
    )
    return settled, jnp.where(needs_sw, seed2, seed)


@jax.jit
def _step_and_settle_batch(
    states: CompactBattle,
    table: MatchTable,
    acts: jax.Array,
    b_seeds: jax.Array,
    p_seeds: jax.Array,
) -> CompactBattle:
    """Step a 1D batch of `(state, act_pair, b_seed, p_seed)` by 1 ply and settle any `request == 2`."""

    def _one(
        st: CompactBattle, act_pair: jax.Array, b_seed: jax.Array, p_seed: jax.Array
    ) -> CompactBattle:
        st0 = st._replace(prng_seed=b_seed)
        st1 = apply_actions(st0, table, act_pair)
        st_settled, _ = _settle_forced_switch_if_needed(st1, table, p_seed)
        return st_settled

    return jax.vmap(_one)(states, acts, b_seeds, p_seeds)


@partial(
    jax.jit,
    static_argnames=("top_k", "ply2_samples", "base_seed_int"),
)
def _prepare_ply1_candidates(
    root_state: CompactBattle,
    table: MatchTable,
    ply1: RootEquilibriumResult,
    top_k: int = 3,
    ply2_samples: int = 4,
    base_seed_int: int = 0xCAFEBABE12345678,
):
    mask0 = ply1.legal_mask0
    mask1 = ply1.legal_mask1
    fb0 = jnp.argmax(mask0.astype(jnp.int32)).astype(jnp.int32)
    fb1 = jnp.argmax(mask1.astype(jnp.int32)).astype(jnp.int32)

    q0_ply1 = jnp.where(mask0, jnp.matmul(ply1.payoff_matrix, ply1.sigma1), jnp.float32(-1e9))
    q1_ply1 = jnp.where(
        mask1, jnp.float32(1.0) - jnp.matmul(ply1.sigma0, ply1.payoff_matrix), jnp.float32(-1e9)
    )
    top_a0 = jnp.argsort(-q0_ply1)[:top_k]
    top_a1 = jnp.argsort(-q1_ply1)[:top_k]

    grid_i, grid_j = jnp.meshgrid(
        jnp.arange(top_k, dtype=jnp.int32),
        jnp.arange(top_k, dtype=jnp.int32),
        indexing="ij",
    )
    cand_a0 = top_a0[grid_i.reshape(-1)]  # [K*K]
    cand_a1 = top_a1[grid_j.reshape(-1)]  # [K*K]

    act0 = jnp.where(mask0[cand_a0], cand_a0, fb0)
    act1 = jnp.where(mask1[cand_a1], cand_a1, fb1)
    acts1 = jnp.stack([act0, act1], axis=-1)  # [K*K, 2]

    num_cands = top_k * top_k
    hi_base = jnp.uint32((base_seed_int >> 32) & 0xFFFFFFFF)
    lo_base = jnp.uint32(base_seed_int & 0xFFFFFFFF)
    ply1_b_seeds = jnp.broadcast_to(
        jnp.array(
            [hi_base ^ jnp.uint32(0x13579BDF), lo_base + jnp.uint32(0x2468ACE1)],
            dtype=jnp.uint32,
        ),
        (num_cands, 2),
    )
    ply1_p_seeds = jnp.broadcast_to(
        jnp.array(
            [hi_base ^ jnp.uint32(0xDEAD10CC), lo_base + jnp.uint32(0xBEEF2026)],
            dtype=jnp.uint32,
        ),
        (num_cands, 2),
    )
    root_batch = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_cands,) + x.shape), root_state
    )
    return cand_a0, cand_a1, root_batch, acts1, ply1_b_seeds, ply1_p_seeds


@partial(
    jax.jit,
    static_argnames=("top_k", "ply2_samples", "base_seed_int"),
)
def _prepare_ply2_batch(
    settled_cands: CompactBattle,
    table: MatchTable,
    top_k: int = 3,
    ply2_samples: int = 4,
    base_seed_int: int = 0xCAFEBABE12345678,
):
    num_cands = top_k * top_k
    m2_0_cands = jax.vmap(lambda s: legal_actions_mask(s, table, 0)[:7])(settled_cands)
    m2_1_cands = jax.vmap(lambda s: legal_actions_mask(s, table, 1)[:7])(settled_cands)
    fb2_0_cands = jnp.argmax(m2_0_cands.astype(jnp.int32), axis=-1).astype(jnp.int32)
    fb2_1_cands = jnp.argmax(m2_1_cands.astype(jnp.int32), axis=-1).astype(jnp.int32)

    hi_base = jnp.uint32((base_seed_int >> 32) & 0xFFFFFFFF)
    lo_base = jnp.uint32(base_seed_int & 0xFFFFFFFF)
    s2_idx = jnp.arange(ply2_samples, dtype=jnp.uint32)
    b2_seeds = jnp.stack(
        [
            (hi_base ^ jnp.uint32(0x55555555)) ^ (s2_idx * jnp.uint32(0x9E3779B9)),
            (lo_base ^ jnp.uint32(0x55555555)) + (s2_idx * jnp.uint32(0x85EBCA6B) + jnp.uint32(1)),
        ],
        axis=-1,
    )
    p2_seeds = jnp.stack(
        [
            (hi_base + jnp.uint32(0x7F4A7C15)) ^ (s2_idx * jnp.uint32(0x27D4EB2F)),
            (lo_base + jnp.uint32(0x31415926)) ^ (s2_idx * jnp.uint32(0x165667B1) + jnp.uint32(7)),
        ],
        axis=-1,
    )
    grid_b0, grid_b1 = jnp.meshgrid(
        jnp.arange(7, dtype=jnp.int32), jnp.arange(7, dtype=jnp.int32), indexing="ij"
    )
    flat_b0 = jnp.repeat(grid_b0.reshape(-1), ply2_samples)  # [49 * S2]
    flat_b1 = jnp.repeat(grid_b1.reshape(-1), ply2_samples)  # [49 * S2]

    n_per_cand = 49 * ply2_samples
    cand_idx_flat = jnp.repeat(jnp.arange(num_cands, dtype=jnp.int32), n_per_cand)
    b0_all = jnp.tile(flat_b0, (num_cands,))
    b1_all = jnp.tile(flat_b1, (num_cands,))
    b2_all = jnp.tile(b2_seeds, (num_cands * 49, 1))
    p2_all = jnp.tile(p2_seeds, (num_cands * 49, 1))

    st_in_batch = jax.tree_util.tree_map(lambda x: x[cand_idx_flat], settled_cands)
    m2_0_flat = m2_0_cands[cand_idx_flat]
    m2_1_flat = m2_1_cands[cand_idx_flat]
    eb0 = jnp.where(
        m2_0_flat[jnp.arange(num_cands * n_per_cand), b0_all],
        b0_all,
        fb2_0_cands[cand_idx_flat],
    )
    eb1 = jnp.where(
        m2_1_flat[jnp.arange(num_cands * n_per_cand), b1_all],
        b1_all,
        fb2_1_cands[cand_idx_flat],
    )
    acts2_all = jnp.stack([eb0, eb1], axis=-1)
    return m2_0_cands, m2_1_cands, st_in_batch, acts2_all, b2_all, p2_all


@partial(
    jax.jit,
    static_argnames=("top_k", "ply2_samples", "rm_sweeps"),
)
def _solve_depth2_from_leaves(
    ply1: RootEquilibriumResult,
    cand_a0: jax.Array,
    cand_a1: jax.Array,
    settled_outcomes: jax.Array,
    m2_0_cands: jax.Array,
    m2_1_cands: jax.Array,
    st2_batch: CompactBattle,
    table: MatchTable,
    weights: EvalWeights,
    top_k: int = 3,
    ply2_samples: int = 4,
    rm_sweeps: int = 256,
) -> Depth2OracleResult:
    num_cands = top_k * top_k
    all_vals = jax.vmap(lambda s: eval_compact_leaf(s, table, weights))(st2_batch).reshape(
        num_cands, 49, ply2_samples
    )
    mats2 = jnp.mean(all_vals, axis=-1).reshape(num_cands, 7, 7)

    def _solve_ply2_cand(
        mat2: jax.Array, m2_0: jax.Array, m2_1: jax.Array, outc: jax.Array
    ) -> jax.Array:
        valid2 = m2_0[:, None] & m2_1[None, :]
        mat2_clean = jnp.where(valid2, mat2, jnp.float32(0.5))
        _, _, v2 = solve_rm_plus_jax(mat2_clean, m2_0, m2_1, sweeps=rm_sweeps)
        return jnp.where(
            outc == 1,
            jnp.float32(1.0),
            jnp.where(
                outc == 2,
                jnp.float32(0.0),
                jnp.where(outc == 3, jnp.float32(0.5), v2),
            ),
        )

    ply2_vals = jax.vmap(_solve_ply2_cand)(
        mats2, m2_0_cands, m2_1_cands, settled_outcomes
    )
    mask0 = ply1.legal_mask0
    mask1 = ply1.legal_mask1
    refined_matrix = ply1.payoff_matrix.at[cand_a0, cand_a1].set(ply2_vals)
    valid_cells = mask0[:, None] & mask1[None, :]
    refined_matrix = jnp.where(valid_cells, refined_matrix, jnp.float32(0.5))

    sigma0, sigma1, game_val = solve_rm_plus_jax(refined_matrix, mask0, mask1, sweeps=rm_sweeps)
    q0 = jnp.where(mask0, jnp.matmul(refined_matrix, sigma1), jnp.float32(0.0))
    q1 = jnp.where(mask1, jnp.float32(1.0) - jnp.matmul(sigma0, refined_matrix), jnp.float32(0.0))
    return Depth2OracleResult(
        payoff_matrix=refined_matrix,
        legal_mask0=mask0,
        legal_mask1=mask1,
        sigma0=sigma0,
        sigma1=sigma1,
        game_value=game_val,
        q_values0=q0,
        q_values1=q1,
        ply1_game_value=ply1.game_value,
    )


def evaluate_depth2_counterfactual_oracle(
    root_state: CompactBattle,
    table: MatchTable,
    weights: EvalWeights | None = None,
    top_k: int = 3,
    ply1_samples: int = 16,
    ply2_samples: int = 4,
    rollout_turns: int = 6,
    ply2_rollout_turns: int = 0,
    rm_sweeps: int = 256,
    base_seed_int: int = 0xCAFEBABE12345678,
) -> Depth2OracleResult:
    """Strict 2-ply simultaneous-move minimax / RM+ oracle on TPU.

    1. Evaluates all `7 x 7 = 49` Ply-1 joint actions `(a0, a1)` across `ply1_samples`
       paired chance seeds via the JIT-cached `evaluate_root_payoff_matrix`.
    2. Selects the top-`top_k` legal actions per side (`top_k x top_k` critical cells).
    3. Steps Ply 1, settles any mid-turn `Baton Pass` or faint `request == 2` switch via
       `step_switch`, expands the `7 x 7 = 49` Ply-2 stage game across `ply2_samples`
       paired chance seeds using the shared `_step_and_settle_batch` kernel, and solves
       the refined root Nash equilibrium.
    """
    if weights is None:
        weights = default_eval_weights()

    ply1 = evaluate_root_payoff_matrix(
        root_state,
        table,
        weights=weights,
        samples_per_cell=ply1_samples,
        rollout_turns=rollout_turns,
        rm_sweeps=rm_sweeps,
        base_seed_int=base_seed_int,
    )
    cand_a0, cand_a1, root_batch, acts1, b1_seeds, p1_seeds = _prepare_ply1_candidates(
        root_state,
        table,
        ply1,
        top_k=top_k,
        ply2_samples=ply2_samples,
        base_seed_int=base_seed_int,
    )
    settled_cands = _step_and_settle_batch(root_batch, table, acts1, b1_seeds, p1_seeds)
    m2_0_cands, m2_1_cands, st_in_batch, acts2_all, b2_all, p2_all = _prepare_ply2_batch(
        settled_cands,
        table,
        top_k=top_k,
        ply2_samples=ply2_samples,
        base_seed_int=base_seed_int,
    )
    st2_batch = _step_and_settle_batch(st_in_batch, table, acts2_all, b2_all, p2_all)
    if ply2_rollout_turns > 0:
        ro = _vmap_heavy_rollout_impl(
            pack_battle(st2_batch),
            table,
            p2_all,
            weights,
            turns=ply2_rollout_turns,
            eps=0.2,
        )
        st2_batch = unpack_battle(ro.final_packed)
    return _solve_depth2_from_leaves(
        ply1,
        cand_a0,
        cand_a1,
        settled_cands.outcome,
        m2_0_cands,
        m2_1_cands,
        st2_batch,
        table,
        weights,
        top_k=top_k,
        ply2_samples=ply2_samples,
        rm_sweeps=rm_sweeps,
    )


class LabeledPositionBatch(NamedTuple):
    """Batch of TPU-labeled positions ready for `train_residual_bundle_jax`."""

    x0: jax.Array  # float32[N, 32]
    x1: jax.Array  # float32[N, 32]
    base_logit: jax.Array  # float32[N]
    base_eval: jax.Array  # float32[N]
    v_oracle: jax.Array  # float32[N] (Depth-2 counterfactual value)
    v_td: jax.Array  # float32[N] (multi-step rollout continuation value)
    is_1v1_unboosted: jax.Array  # float32[N]


def label_positions_batch_tpu(
    packed_positions: PackedBattle,
    table: MatchTable,
    weights: EvalWeights | None = None,
    top_k: int = 3,
    ply1_samples: int = 8,
    ply2_samples: int = 4,
    rollout_turns: int = 6,
    rm_sweeps: int = 128,
) -> LabeledPositionBatch:
    """Label a batch of `PackedBattle` (`uint32[N, 30]`) positions sharing `table` using
    the Depth-2 counterfactual oracle (`v_oracle`) and multi-step rollout (`v_td`),
    extracting `x0`, `x1`, `base_logit`, and `is_1v1_unboosted` for `train_residual_bundle_jax`.
    """
    if weights is None:
        weights = default_eval_weights(residual_scale=0.0)
    else:
        weights = weights._replace(residual_scale=jnp.float32(0.0))

    @jax.jit
    def _extract_batch_feats(pks: PackedBattle):
        def _feat_one(pk: PackedBattle):
            st = unpack_battle(pk)
            logit, effs = compact_base_logit_and_eff(st, table, weights)
            x0 = extract_eval_features_with_eff(st, table, 0, effs)
            x1 = extract_eval_features_with_eff(st, table, 1, effs)
            alive0 = popcount_3bit(_get1(st.alive_mask, 0))
            alive1 = popcount_3bit(_get1(st.alive_mask, 1))
            slot0 = _get1(st.active_idx, 0)
            slot1 = _get1(st.active_idx, 1)
            unboosted = jnp.all(_get1(st.boosts, slot0) == 0) & jnp.all(
                _get1(st.boosts, 3 + slot1) == 0
            )
            u1v1 = ((alive0 == 1) & (alive1 == 1) & unboosted).astype(jnp.float32)
            return x0, x1, logit, jax.nn.sigmoid(logit), u1v1

        return jax.vmap(_feat_one)(pks)

    x0_b, x1_b, logit_b, eval_b, u1v1_b = _extract_batch_feats(packed_positions)
    n_pos = int(packed_positions.words.shape[0])
    v_oracle_list = []
    v_td_list = []
    for i in range(n_pos):
        st_i = unpack_battle(PackedBattle(words=packed_positions.words[i]))
        d2 = evaluate_depth2_counterfactual_oracle(
            st_i,
            table,
            weights=weights,
            top_k=top_k,
            ply1_samples=ply1_samples,
            ply2_samples=ply2_samples,
            rollout_turns=rollout_turns,
            rm_sweeps=rm_sweeps,
        )
        v_oracle_list.append(d2.game_value)
        v_td_list.append(d2.ply1_game_value)

    return LabeledPositionBatch(
        x0=x0_b,
        x1=x1_b,
        base_logit=logit_b,
        base_eval=eval_b,
        v_oracle=jnp.stack(v_oracle_list, axis=0),
        v_td=jnp.stack(v_td_list, axis=0),
        is_1v1_unboosted=u1v1_b,
    )

