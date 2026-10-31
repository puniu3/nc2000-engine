"""TPU-native 32-bit state representations for Gen 2 (`CompactBattle` & `MatchTable`).

All fields are JAX arrays using 32-bit (`int32`, `uint32`) or narrower types so that:
1. `jax.vmap` automatically stacks a batch of `N` battles into coalesced Structure-of-Arrays
   (SoA) tensors `[N, ...]` aligned with TPU VPU `(8, 128)` vector lanes.
2. Zero 64-bit types (`int64` / `uint64` / `float64`) are used, matching strict TPU
   execution (`jax_enable_x64 = False`).
3. Every field and bitflag mirrors `nc2000_engine::compact::{CompactBattle, MatchTable}`.
"""

from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp


class Status:
    """Gen 2 primary status discriminants (`nc2000_engine::state::Status`)."""

    NONE = 0
    BRN = 1
    PAR = 2
    SLP = 3
    FRZ = 4
    PSN = 5
    TOX = 6
    FNT = 7


class VolBit:
    """Volatile status bitflags in `CompactBattle.vol_flags` (`nc2000_engine::compact::vol_bit`)."""

    CONFUSION = 1 << 0
    FLINCH = 1 << 1
    DESTINY_BOND = 1 << 2
    SUBSTITUTE = 1 << 3
    PAR_SPEED_DROP = 1 << 4
    BRN_ATK_DROP = 1 << 5
    LEECH_SEED = 1 << 6
    NIGHTMARE = 1 << 7
    CURSE = 1 << 8
    FORESIGHT = 1 << 9
    PROTECT = 1 << 10
    ENDURE = 1 << 11
    ENCORE = 1 << 12
    DISABLE = 1 << 13
    PERISH_SONG = 1 << 14
    TRAPPED_OR_LOCKED = 1 << 15


class CompactWeather:
    """Weather kinds in `CompactBattle.weather_kind`."""

    NONE = 0
    RAIN = 1
    SUN = 2
    SANDSTORM = 3


class ItemStatKind:
    """Species-specific stat-boosting held items (`nc2000_engine::compact::ItemStatKind`)."""

    NONE = 0
    THICK_CLUB = 1
    LIGHT_BALL = 2
    METAL_POWDER = 3


class ActionPhase:
    """Action phase ordering (`nc2000_engine::compact::ActionPhase`)."""

    RESIDUAL = 0
    MOVE = 1
    SWITCH = 2
    BEFORE_TURN_PURSUIT = 3
    INSTA_SWITCH = 4
    TEAM_PREVIEW = 5


class MatchTable(NamedTuple):
    """Immutable per-match reference table shared across a search tree or rollout batch.

    Shapes (unbatched):
      - Per-Pokemon `[6]` arrays (`0..3` = Side 0, `3..6` = Side 1):
        `type_mask`: `uint32[6]`
        `maxhp`: `int32[6]`
        `quantum_1_8`: `int32[6]`
        `quantum_1_16`: `int32[6]`
        `stored_stats`: `int32[6, 5]` (`[atk, def, spa, spd, spe]`)
        `max_pp`: `int32[6, 4]`
        `types`: `int32[6, 2]`
        `num_types`: `int32[6]`
        `level`: `int32[6]`
        `num_moves`: `int32[6]`
        `item_stat_kind`: `int32[6]`
        `item_boost_type`: `int32[6]`
        `item_flags`: `int32[6]` (bitmask of held item properties)
      - Per-Move `[6, 4]` arrays:
        `move_base_power`: `int32[6, 4]`
        `move_fixed_damage`: `int32[6, 4]`
        `move_primary_vol`: `int32[6, 4]`
        `move_sec_vol`: `int32[6, 4]`
        `move_type`: `int32[6, 4]`
        `move_flags`: `int32[6, 4]` (bitmask of boolean move properties)
        `move_priority`: `int32[6, 4]`
        `move_base_acc_pct`: `int32[6, 4]`
        `move_crit_ratio`: `int32[6, 4]`
        `move_primary_status`: `int32[6, 4]`
        `move_sec_chance_256`: `int32[6, 4]`
        `move_sec_status`: `int32[6, 4]`
        `move_sec_boost_stat`: `int32[6, 4]`
        `move_sec_boost_delta`: `int32[6, 4]`
        `move_boost_stats`: `int32[6, 4, 3]`
        `move_boost_deltas`: `int32[6, 4, 3]`
        `move_special_kind`: `int32[6, 4]`
        `move_var_bp_kind`: `int32[6, 4]`
        `move_multihit_mean_x2`: `int32[6, 4]`
      - Global Dex constants:
        `type_chart`: `int32[20, 20]`
        `normal_type`: `int32`
        `fighting_type`: `int32`
        `flying_type`: `int32`
        `ground_type`: `int32`
        `ghost_type`: `int32`
        `grass_type`: `int32`
        `fire_type`: `int32`
        `electric_type`: `int32`
        `ice_type`: `int32`
        `sandstorm_immune_mask`: `uint32`
        `poison_immune_mask`: `uint32`
    """

    type_mask: jax.Array
    maxhp: jax.Array
    quantum_1_8: jax.Array
    quantum_1_16: jax.Array
    stored_stats: jax.Array
    max_pp: jax.Array
    types: jax.Array
    num_types: jax.Array
    level: jax.Array
    num_moves: jax.Array
    item_stat_kind: jax.Array
    item_boost_type: jax.Array
    item_flags: jax.Array

    move_base_power: jax.Array
    move_fixed_damage: jax.Array
    move_primary_vol: jax.Array
    move_sec_vol: jax.Array
    move_type: jax.Array
    move_flags: jax.Array
    move_priority: jax.Array
    move_base_acc_pct: jax.Array
    move_crit_ratio: jax.Array
    move_primary_status: jax.Array
    move_sec_chance_256: jax.Array
    move_sec_status: jax.Array
    move_sec_boost_stat: jax.Array
    move_sec_boost_delta: jax.Array
    move_boost_stats: jax.Array
    move_boost_deltas: jax.Array
    move_special_kind: jax.Array
    move_var_bp_kind: jax.Array
    move_multihit_mean_x2: jax.Array

    type_chart: jax.Array
    normal_type: jax.Array
    fighting_type: jax.Array
    flying_type: jax.Array
    ground_type: jax.Array
    ghost_type: jax.Array
    grass_type: jax.Array
    fire_type: jax.Array
    electric_type: jax.Array
    ice_type: jax.Array
    sandstorm_immune_mask: jax.Array
    poison_immune_mask: jax.Array


class ItemFlag:
    """Bitflags packed into `MatchTable.item_flags`."""

    QUICK_CLAW = 1 << 0
    LEFTOVERS = 1 << 1
    BRIGHTPOWDER = 1 << 2
    FOCUSBAND = 1 << 3
    MINTBERRY = 1 << 4
    MIRACLEBERRY = 1 << 5
    GOLDBERRY = 1 << 6
    BERRY = 1 << 7
    MYSTERYBERRY = 1 << 8
    SCOPE_LENS = 1 << 9


class MoveFlag:
    """Bitflags packed into `MatchTable.move_flags`."""

    IS_PHYSICAL = 1 << 0
    IS_STATUS = 1 << 1
    ALWAYS_HIT = 1 << 2
    IGNORE_ACCURACY = 1 << 3
    IGNORE_EVASION = 1 << 4
    IGNORE_POS_EVASION = 1 << 5
    IS_SELFDESTRUCT = 1 << 6
    IS_WATER = 1 << 7
    IS_FIRE = 1 << 8
    IS_SOLARBEAM = 1 << 9
    BYPASSES_SUB = 1 << 10
    IS_REST = 1 << 11
    IS_HALF_HEAL = 1 << 12
    IS_HEAL_CYCLE = 1 << 13
    IS_SLEEP_USABLE = 1 << 14
    CAN_SLEEP_TALK = 1 << 15
    IS_PROTECT = 1 << 16
    IS_SUBSTITUTE = 1 << 17
    DRAIN_HALF = 1 << 18
    RECOIL_QUARTER = 1 << 19
    BOOSTS_SELF = 1 << 20


class CompactBattle(NamedTuple):
    """Mutable battle state mirroring `nc2000_engine::compact::CompactBattle`.

    Unbatched shapes:
      - `hp_pool`: `int32[6, 2]` (`[mon_idx, 0]` = main HP, `[mon_idx, 1]` = Substitute HP)
      - `pp`: `int32[6, 4]`
      - `status_packed`: `int32[6]` (low 3 bits = Status, bits 3..8 = counter)
      - `item`: `int32[6]` (`0` = None, `> 0` = held item present)
      - `vol_flags`: `int32[6]` (bitmask of `VolBit`)
      - `timers`: `int32[6, 4]` (`[conf_turns, enc_turns, dis_turns, perish_turns]`)
      - `boosts`: `int32[6, 7]` (`[atk, def, spa, spd, spe, acc, eva]` in `-6..=+6`)
      - `slot_refs`: `int32[6]`
      - `active_idx`: `int32[2]` (`0..=2`)
      - `party_roster`: `int32[2, 3]`
      - `alive_mask`: `int32[2]` (3-bit mask of alive party slots)
      - `reflect_turns`: `int32[2]`
      - `lightscreen_turns`: `int32[2]`
      - `safeguard_turns`: `int32[2]`
      - `side_flags`: `int32[2]`
      - `last_move_selfko`: `int32[2]` (`255` = None)
      - `request`: `int32[2]` (`0`=Wait, `1`=Move, `2`=Switch, `3`=TeamPreview)
      - `turn`: `int32`
      - `weather_kind`: `int32`
      - `weather_turns`: `int32`
      - `quick_claw_roll`: `int32`
      - `outcome`: `int32` (`0`=Ongoing, `1`=P1Win, `2`=P2Win, `3`=Tie)
      - `speed_order`: `int32`
      - `clause_flags`: `int32`
      - `hurt_damage_this_turn`: `int32[2]`
      - `hurt_cat_this_turn`: `int32[2]`
      - `prng_seed`: `uint32[2]` (`[hi32, lo32]` 64-bit Gen5 LCG state)
    """

    hp_pool: jax.Array
    pp: jax.Array
    status_packed: jax.Array
    item: jax.Array
    vol_flags: jax.Array
    timers: jax.Array
    boosts: jax.Array
    slot_refs: jax.Array

    active_idx: jax.Array
    party_roster: jax.Array
    alive_mask: jax.Array
    reflect_turns: jax.Array
    lightscreen_turns: jax.Array
    safeguard_turns: jax.Array
    side_flags: jax.Array
    last_move_selfko: jax.Array
    request: jax.Array

    turn: jax.Array
    weather_kind: jax.Array
    weather_turns: jax.Array
    quick_claw_roll: jax.Array
    outcome: jax.Array
    speed_order: jax.Array
    clause_flags: jax.Array
    hurt_damage_this_turn: jax.Array
    hurt_cat_this_turn: jax.Array
    prng_seed: jax.Array


class PackedBattle(NamedTuple):
    """Bit-packed 120-byte (`uint32[30]`) battle state for TPU VMEM/HBM transport.

    Compresses `CompactBattle` (129 x `int32` = 516 bytes) by 4.3x into a single
    contiguous `uint32[30]` array per battle (120 bytes):
      - `words[0..6]` (`hp_and_status`):
        `hp_main` (10b) | `hp_sub` (8b) | `status_packed` (8b) | `timer0` (3b) | `timer1` (3b)
      - `words[6..12]` (`pp_and_timer2`):
        `pp0` (7b) | `pp1` (7b) | `pp2` (7b) | `pp3` (7b) | `timer2` (3b)
      - `words[12..18]` (`vol_item_slots`):
        `vol_flags` (16b) | `item` (8b) | `slot_refs` (8b)
      - `words[18..24]` (`boosts_and_timer3`):
        7 x 4-bit `(boost + 6)` (28b) | `timer3` (3b)
      - `words[24..26]` (`side_words` for Side 0, Side 1):
        `active_idx` (2b) | `roster0..2` (3x3b=9b) | `alive_mask` (3b) |
        `reflect` (3b) | `lightscreen` (3b) | `safeguard` (3b) | `side_flags` (8b) = 31 bits
      - `words[26]` (`hurt_word`):
        `hurt_dmg0` (10b) | `hurt_dmg1` (10b) | `(hurt_cat0 + 1)` (2b) | `(hurt_cat1 + 1)` (2b) |
        `selfko_code0` (3b) | `selfko_code1` (3b) = 30 bits
      - `words[27]` (`global_word`):
        `turn` (10b) | `weather_kind` (2b) | `weather_turns` (4b) | `quick_claw_roll` (2b) |
        `outcome` (2b) | `speed_order` (2b) | `clause_flags` (4b) |
        `request0` (2b) | `request1` (2b) = 30 bits
      - `words[28..30]` (`prng_seed`):
        `[hi32, lo32]` (64-bit Gen5RNG state)
    """

    words: jax.Array  # uint32[..., 30]


PACKED_BATTLE_WORDS: int = 30
PACKED_BATTLE_BYTES: int = PACKED_BATTLE_WORDS * 4  # 120 bytes


def pack_battle(state: CompactBattle) -> PackedBattle:
    """Losslessly pack `CompactBattle` (516 B) into `PackedBattle` (`uint32[30]`, 120 B).

    Works on both unbatched `CompactBattle` (`words: uint32[30]`) and batched
    `CompactBattle` (`words: uint32[B, 30]`) via pure 32-bit bitwise shifts/masks.
    """
    hp_main = jnp.clip(state.hp_pool[..., :, 0], 0, 1023).astype(jnp.uint32)
    hp_sub = jnp.clip(state.hp_pool[..., :, 1], 0, 255).astype(jnp.uint32)
    st_pk = (state.status_packed.astype(jnp.uint32)) & jnp.uint32(0xFF)
    t0 = (state.timers[..., :, 0].astype(jnp.uint32)) & jnp.uint32(0x7)
    t1 = (state.timers[..., :, 1].astype(jnp.uint32)) & jnp.uint32(0x7)
    t2 = (state.timers[..., :, 2].astype(jnp.uint32)) & jnp.uint32(0x7)
    t3 = (state.timers[..., :, 3].astype(jnp.uint32)) & jnp.uint32(0x7)

    w_hp_st = (
        hp_main
        | (hp_sub << jnp.uint32(10))
        | (st_pk << jnp.uint32(18))
        | (t0 << jnp.uint32(26))
        | (t1 << jnp.uint32(29))
    )

    pp_u = jnp.clip(state.pp, 0, 127).astype(jnp.uint32)
    w_pp = (
        pp_u[..., :, 0]
        | (pp_u[..., :, 1] << jnp.uint32(7))
        | (pp_u[..., :, 2] << jnp.uint32(14))
        | (pp_u[..., :, 3] << jnp.uint32(21))
        | (t2 << jnp.uint32(28))
    )

    vol_u = (state.vol_flags.astype(jnp.uint32)) & jnp.uint32(0xFFFF)
    item_u = (state.item.astype(jnp.uint32)) & jnp.uint32(0xFF)
    sref_u = (state.slot_refs.astype(jnp.uint32)) & jnp.uint32(0xFF)
    w_vol = vol_u | (item_u << jnp.uint32(16)) | (sref_u << jnp.uint32(24))

    b_off = jnp.clip(state.boosts + jnp.int32(6), 0, 15).astype(jnp.uint32)
    w_boosts = (
        b_off[..., :, 0]
        | (b_off[..., :, 1] << jnp.uint32(4))
        | (b_off[..., :, 2] << jnp.uint32(8))
        | (b_off[..., :, 3] << jnp.uint32(12))
        | (b_off[..., :, 4] << jnp.uint32(16))
        | (b_off[..., :, 5] << jnp.uint32(20))
        | (b_off[..., :, 6] << jnp.uint32(24))
        | (t3 << jnp.uint32(28))
    )

    act_u = (state.active_idx.astype(jnp.uint32)) & jnp.uint32(0x3)
    r0 = (state.party_roster[..., :, 0].astype(jnp.uint32)) & jnp.uint32(0x7)
    r1 = (state.party_roster[..., :, 1].astype(jnp.uint32)) & jnp.uint32(0x7)
    r2 = (state.party_roster[..., :, 2].astype(jnp.uint32)) & jnp.uint32(0x7)
    alive_u = (state.alive_mask.astype(jnp.uint32)) & jnp.uint32(0x7)
    ref_u = (state.reflect_turns.astype(jnp.uint32)) & jnp.uint32(0x7)
    ls_u = (state.lightscreen_turns.astype(jnp.uint32)) & jnp.uint32(0x7)
    sg_u = (state.safeguard_turns.astype(jnp.uint32)) & jnp.uint32(0x7)
    sf_u = (state.side_flags.astype(jnp.uint32)) & jnp.uint32(0xFF)

    w_side = (
        act_u
        | (r0 << jnp.uint32(2))
        | (r1 << jnp.uint32(5))
        | (r2 << jnp.uint32(8))
        | (alive_u << jnp.uint32(11))
        | (ref_u << jnp.uint32(14))
        | (ls_u << jnp.uint32(17))
        | (sg_u << jnp.uint32(20))
        | (sf_u << jnp.uint32(23))
    )

    hd0 = jnp.clip(state.hurt_damage_this_turn[..., 0], 0, 1023).astype(jnp.uint32)
    hd1 = jnp.clip(state.hurt_damage_this_turn[..., 1], 0, 1023).astype(jnp.uint32)
    hc0 = jnp.clip(state.hurt_cat_this_turn[..., 0] + jnp.int32(1), 0, 3).astype(jnp.uint32)
    hc1 = jnp.clip(state.hurt_cat_this_turn[..., 1] + jnp.int32(1), 0, 3).astype(jnp.uint32)
    selfko_code = jnp.where(
        state.last_move_selfko == jnp.int32(255),
        jnp.uint32(7),
        (state.last_move_selfko.astype(jnp.uint32)) & jnp.uint32(0x7),
    )
    w_hurt = (
        hd0
        | (hd1 << jnp.uint32(10))
        | (hc0 << jnp.uint32(20))
        | (hc1 << jnp.uint32(22))
        | (selfko_code[..., 0] << jnp.uint32(24))
        | (selfko_code[..., 1] << jnp.uint32(27))
    )[..., None]

    turn_u = jnp.clip(state.turn, 0, 1023).astype(jnp.uint32)
    wk_u = (state.weather_kind.astype(jnp.uint32)) & jnp.uint32(0x3)
    wt_u = (state.weather_turns.astype(jnp.uint32)) & jnp.uint32(0xF)
    qc_u = (state.quick_claw_roll.astype(jnp.uint32)) & jnp.uint32(0x3)
    out_u = (state.outcome.astype(jnp.uint32)) & jnp.uint32(0x3)
    spo_u = (state.speed_order.astype(jnp.uint32)) & jnp.uint32(0x3)
    cf_u = (state.clause_flags.astype(jnp.uint32)) & jnp.uint32(0xF)
    req_u = (state.request.astype(jnp.uint32)) & jnp.uint32(0x3)
    w_global = (
        turn_u
        | (wk_u << jnp.uint32(10))
        | (wt_u << jnp.uint32(12))
        | (qc_u << jnp.uint32(16))
        | (out_u << jnp.uint32(18))
        | (spo_u << jnp.uint32(20))
        | (cf_u << jnp.uint32(22))
        | (req_u[..., 0] << jnp.uint32(26))
        | (req_u[..., 1] << jnp.uint32(28))
    )[..., None]

    words = jnp.concatenate(
        [
            w_hp_st,
            w_pp,
            w_vol,
            w_boosts,
            w_side,
            w_hurt,
            w_global,
            state.prng_seed.astype(jnp.uint32),
        ],
        axis=-1,
    )
    return PackedBattle(words=words)


def unpack_battle(packed: PackedBattle) -> CompactBattle:
    """Unpack `PackedBattle` (`uint32[..., 30]`) back into `CompactBattle`."""
    w = packed.words
    w_hp_st = w[..., 0:6]
    w_pp = w[..., 6:12]
    w_vol = w[..., 12:18]
    w_boosts = w[..., 18:24]
    w_side = w[..., 24:26]
    w_hurt = w[..., 26]
    w_global = w[..., 27]
    prng_seed = w[..., 28:30]

    hp_main = (w_hp_st & jnp.uint32(0x3FF)).astype(jnp.int32)
    hp_sub = ((w_hp_st >> jnp.uint32(10)) & jnp.uint32(0xFF)).astype(jnp.int32)
    hp_pool = jnp.stack([hp_main, hp_sub], axis=-1)

    status_packed = ((w_hp_st >> jnp.uint32(18)) & jnp.uint32(0xFF)).astype(jnp.int32)
    t0 = ((w_hp_st >> jnp.uint32(26)) & jnp.uint32(0x7)).astype(jnp.int32)
    t1 = ((w_hp_st >> jnp.uint32(29)) & jnp.uint32(0x7)).astype(jnp.int32)
    t2 = ((w_pp >> jnp.uint32(28)) & jnp.uint32(0x7)).astype(jnp.int32)
    t3 = ((w_boosts >> jnp.uint32(28)) & jnp.uint32(0x7)).astype(jnp.int32)
    timers = jnp.stack([t0, t1, t2, t3], axis=-1)

    pp0 = (w_pp & jnp.uint32(0x7F)).astype(jnp.int32)
    pp1 = ((w_pp >> jnp.uint32(7)) & jnp.uint32(0x7F)).astype(jnp.int32)
    pp2 = ((w_pp >> jnp.uint32(14)) & jnp.uint32(0x7F)).astype(jnp.int32)
    pp3 = ((w_pp >> jnp.uint32(21)) & jnp.uint32(0x7F)).astype(jnp.int32)
    pp = jnp.stack([pp0, pp1, pp2, pp3], axis=-1)

    vol_flags = (w_vol & jnp.uint32(0xFFFF)).astype(jnp.int32)
    item = ((w_vol >> jnp.uint32(16)) & jnp.uint32(0xFF)).astype(jnp.int32)
    slot_refs = ((w_vol >> jnp.uint32(24)) & jnp.uint32(0xFF)).astype(jnp.int32)

    b0 = ((w_boosts & jnp.uint32(0xF)).astype(jnp.int32)) - jnp.int32(6)
    b1 = (((w_boosts >> jnp.uint32(4)) & jnp.uint32(0xF)).astype(jnp.int32)) - jnp.int32(6)
    b2 = (((w_boosts >> jnp.uint32(8)) & jnp.uint32(0xF)).astype(jnp.int32)) - jnp.int32(6)
    b3 = (((w_boosts >> jnp.uint32(12)) & jnp.uint32(0xF)).astype(jnp.int32)) - jnp.int32(6)
    b4 = (((w_boosts >> jnp.uint32(16)) & jnp.uint32(0xF)).astype(jnp.int32)) - jnp.int32(6)
    b5 = (((w_boosts >> jnp.uint32(20)) & jnp.uint32(0xF)).astype(jnp.int32)) - jnp.int32(6)
    b6 = (((w_boosts >> jnp.uint32(24)) & jnp.uint32(0xF)).astype(jnp.int32)) - jnp.int32(6)
    boosts = jnp.stack([b0, b1, b2, b3, b4, b5, b6], axis=-1)

    active_idx = (w_side & jnp.uint32(0x3)).astype(jnp.int32)
    r0 = ((w_side >> jnp.uint32(2)) & jnp.uint32(0x7)).astype(jnp.int32)
    r1 = ((w_side >> jnp.uint32(5)) & jnp.uint32(0x7)).astype(jnp.int32)
    r2 = ((w_side >> jnp.uint32(8)) & jnp.uint32(0x7)).astype(jnp.int32)
    party_roster = jnp.stack([r0, r1, r2], axis=-1)
    alive_mask = ((w_side >> jnp.uint32(11)) & jnp.uint32(0x7)).astype(jnp.int32)
    reflect_turns = ((w_side >> jnp.uint32(14)) & jnp.uint32(0x7)).astype(jnp.int32)
    lightscreen_turns = ((w_side >> jnp.uint32(17)) & jnp.uint32(0x7)).astype(jnp.int32)
    safeguard_turns = ((w_side >> jnp.uint32(20)) & jnp.uint32(0x7)).astype(jnp.int32)
    side_flags = ((w_side >> jnp.uint32(23)) & jnp.uint32(0xFF)).astype(jnp.int32)

    hd0 = (w_hurt & jnp.uint32(0x3FF)).astype(jnp.int32)
    hd1 = ((w_hurt >> jnp.uint32(10)) & jnp.uint32(0x3FF)).astype(jnp.int32)
    hurt_damage_this_turn = jnp.stack([hd0, hd1], axis=-1)
    hc0 = (((w_hurt >> jnp.uint32(20)) & jnp.uint32(0x3)).astype(jnp.int32)) - jnp.int32(1)
    hc1 = (((w_hurt >> jnp.uint32(22)) & jnp.uint32(0x3)).astype(jnp.int32)) - jnp.int32(1)
    hurt_cat_this_turn = jnp.stack([hc0, hc1], axis=-1)
    sk0 = ((w_hurt >> jnp.uint32(24)) & jnp.uint32(0x7)).astype(jnp.int32)
    sk1 = ((w_hurt >> jnp.uint32(27)) & jnp.uint32(0x7)).astype(jnp.int32)
    selfko_code = jnp.stack([sk0, sk1], axis=-1)
    last_move_selfko = jnp.where(selfko_code == jnp.int32(7), jnp.int32(255), selfko_code)

    turn = (w_global & jnp.uint32(0x3FF)).astype(jnp.int32)
    weather_kind = ((w_global >> jnp.uint32(10)) & jnp.uint32(0x3)).astype(jnp.int32)
    weather_turns = ((w_global >> jnp.uint32(12)) & jnp.uint32(0xF)).astype(jnp.int32)
    quick_claw_roll = ((w_global >> jnp.uint32(16)) & jnp.uint32(0x3)).astype(jnp.int32)
    outcome = ((w_global >> jnp.uint32(18)) & jnp.uint32(0x3)).astype(jnp.int32)
    speed_order = ((w_global >> jnp.uint32(20)) & jnp.uint32(0x3)).astype(jnp.int32)
    clause_flags = ((w_global >> jnp.uint32(22)) & jnp.uint32(0xF)).astype(jnp.int32)
    req0 = ((w_global >> jnp.uint32(26)) & jnp.uint32(0x3)).astype(jnp.int32)
    req1 = ((w_global >> jnp.uint32(28)) & jnp.uint32(0x3)).astype(jnp.int32)
    request = jnp.stack([req0, req1], axis=-1)

    return CompactBattle(
        hp_pool=hp_pool,
        pp=pp,
        status_packed=status_packed,
        item=item,
        vol_flags=vol_flags,
        timers=timers,
        boosts=boosts,
        slot_refs=slot_refs,
        active_idx=active_idx,
        party_roster=party_roster,
        alive_mask=alive_mask,
        reflect_turns=reflect_turns,
        lightscreen_turns=lightscreen_turns,
        safeguard_turns=safeguard_turns,
        side_flags=side_flags,
        last_move_selfko=last_move_selfko,
        request=request,
        turn=turn,
        weather_kind=weather_kind,
        weather_turns=weather_turns,
        quick_claw_roll=quick_claw_roll,
        outcome=outcome,
        speed_order=speed_order,
        clause_flags=clause_flags,
        hurt_damage_this_turn=hurt_damage_this_turn,
        hurt_cat_this_turn=hurt_cat_this_turn,
        prng_seed=prng_seed,
    )

