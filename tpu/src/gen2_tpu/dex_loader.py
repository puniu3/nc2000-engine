"""Loader for `~/nc2000-engine/data/gen2stadium2.json` and builder for `(MatchTable, CompactBattle)`.

Constructs the exact `MatchTable` and initial `CompactBattle` state matching
`nc2000_engine::compact::CompactBattle::from_battle` after team preview (`team 1,2,3`).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .prng import prng_random, prng_random_chance, seed_from_str
from .state import (
    CompactBattle,
    CompactWeather,
    ItemFlag,
    ItemStatKind,
    MatchTable,
    MoveFlag,
    Status,
    VolBit,
)

_BUNDLED_DEX_PATH = Path(__file__).resolve().parent / "data" / "gen2stadium2.json"
_REPO_DEX_PATH = Path(__file__).resolve().parents[3] / "data" / "gen2stadium2.json"
_REF_DEX_PATH = Path("/Users/puniu/nc2000-engine/data/gen2stadium2.json")
DEFAULT_DEX_PATH = (
    _BUNDLED_DEX_PATH
    if _BUNDLED_DEX_PATH.exists()
    else (_REPO_DEX_PATH if _REPO_DEX_PATH.exists() else _REF_DEX_PATH)
)

_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_HP_TYPES = (
    "Fighting",
    "Flying",
    "Poison",
    "Ground",
    "Rock",
    "Bug",
    "Ghost",
    "Steel",
    "Fire",
    "Water",
    "Grass",
    "Electric",
    "Psychic",
    "Ice",
    "Dragon",
    "Dark",
)
_COMPACT_TYPE_BOOST_ITEMS: dict[str, str] = {
    "blackbelt": "Fighting",
    "blackglasses": "Dark",
    "charcoal": "Fire",
    "dragonfang": "Dragon",
    "hardstone": "Rock",
    "magnet": "Electric",
    "metalcoat": "Steel",
    "miracleseed": "Grass",
    "mysticwater": "Water",
    "nevermeltice": "Ice",
    "pinkbow": "Normal",
    "poisonbarb": "Poison",
    "polkadotbow": "Normal",
    "sharpbeak": "Flying",
    "silverpowder": "Bug",
    "softsand": "Ground",
    "spelltag": "Ghost",
    "twistedspoon": "Psychic",
}


def toid(text: str) -> str:
    """Mirror Showdown/nc2000-engine `toid`: lowercase alphanumeric only."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _tr(num: float) -> float:
    """JS `num >>> 0` (ToUint32 truncation) matching `battle::tr`."""
    if not math.isfinite(num):
        return 0.0
    n = math.trunc(num)
    return float(n % 4294967296)


def _parse_status_code(s: str | None) -> int:
    return {
        "slp": Status.SLP,
        "psn": Status.PSN,
        "brn": Status.BRN,
        "frz": Status.FRZ,
        "par": Status.PAR,
        "tox": Status.TOX,
    }.get(s or "", Status.NONE)


def _parse_vol_code(s: str | None) -> int:
    return {
        "confusion": VolBit.CONFUSION,
        "flinch": VolBit.FLINCH,
        "leechseed": VolBit.LEECH_SEED,
        "nightmare": VolBit.NIGHTMARE,
        "curse": VolBit.CURSE,
        "foresight": VolBit.FORESIGHT,
        "encore": VolBit.ENCORE,
        "disable": VolBit.DISABLE,
        "perishsong": VolBit.PERISH_SONG,
        "destinybond": VolBit.DESTINY_BOND,
    }.get(s or "", 0)


@dataclass(frozen=True)
class DexData:
    """Parsed and interned Gen 2 Stadium 2 Dex matching `nc2000_engine::dex::Dex`."""

    raw: dict[str, Any]
    type_names: list[str]
    type_index: dict[str, int]
    type_chart: np.ndarray  # int32[20, 20]
    special_type_mask: int
    normal_type: int
    fighting_type: int
    flying_type: int
    poison_type: int
    ground_type: int
    rock_type: int
    ghost_type: int
    steel_type: int
    fire_type: int
    water_type: int
    grass_type: int
    electric_type: int
    ice_type: int
    sandstorm_immune_mask: int
    poison_immune_mask: int
    species_keys: list[str]
    species_map: dict[str, int]
    item_keys: list[str]
    item_map: dict[str, int]
    move_keys: list[str]
    move_map: dict[str, int]


@lru_cache(maxsize=2)
def load_dex(dex_path: str | Path = DEFAULT_DEX_PATH) -> DexData:
    """Load `gen2stadium2.json` and build deterministic BTreeMap-sorted indices."""
    path = Path(dex_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Ensure synthetic 'recharge' move exists just like Dex::from_json
    moves_dict = dict(raw["moves"])
    if "recharge" not in moves_dict:
        moves_dict["recharge"] = {
            "num": 0,
            "name": "Recharge",
            "basePower": 0,
            "accuracy": True,
            "pp": 1,
            "priority": 0,
            "category": "Physical",
            "type": "???",
            "target": "normal",
            "flags": {},
            "callbacks": [],
        }

    # Type interning in BTreeMap key order of `typechart`, then "???"
    typechart = raw["typechart"]
    sorted_type_keys = sorted(typechart.keys())
    type_names: list[str] = []
    type_index: dict[str, int] = {}
    for tk in sorted_type_keys:
        td = typechart[tk]
        idx = len(type_names)
        type_names.append(td["name"])
        type_index[td["name"]] = idx
        type_index[tk] = idx
    unk_idx = len(type_names)
    type_names.append("???")
    type_index["???"] = unk_idx

    n_types = min(len(type_names), 19)
    type_chart = np.zeros((20, 20), dtype=np.int32)
    for def_key in sorted_type_keys:
        def_id = type_index[def_key]
        if def_id >= n_types:
            continue
        dmg_taken = typechart[def_key].get("damageTaken", {})
        for att_key, code in dmg_taken.items():
            if att_key in type_index:
                att_id = type_index[att_key]
                if att_id < n_types:
                    if code == 3:
                        type_chart[att_id, def_id] = -9
                    elif code == 1:
                        type_chart[att_id, def_id] = 1
                    elif code == 2:
                        type_chart[att_id, def_id] = -1
                    else:
                        type_chart[att_id, def_id] = 0

    special_type_mask = 0
    for t_name in (
        "Fire",
        "Water",
        "Grass",
        "Electric",
        "Psychic",
        "Ice",
        "Dragon",
        "Dark",
    ):
        if t_name in type_index:
            special_type_mask |= 1 << (type_index[t_name] & 31)

    normal_type = type_index["Normal"]
    fighting_type = type_index["Fighting"]
    flying_type = type_index["Flying"]
    poison_type = type_index.get("Poison", 0xFF)
    ground_type = type_index["Ground"]
    rock_type = type_index.get("Rock", 0xFF)
    ghost_type = type_index["Ghost"]
    steel_type = type_index.get("Steel", 0xFF)
    fire_type = type_index["Fire"]
    water_type = type_index["Water"]
    grass_type = type_index["Grass"]
    electric_type = type_index.get("Electric", 0xFF)
    ice_type = type_index.get("Ice", 0xFF)

    sandstorm_immune_mask = 0
    for t in (rock_type, ground_type, steel_type):
        if t < 32:
            sandstorm_immune_mask |= 1 << t

    poison_immune_mask = 0
    for t in (poison_type, steel_type):
        if t < 32:
            poison_immune_mask |= 1 << t

    species_keys = sorted(raw["species"].keys())
    species_map = {k: i for i, k in enumerate(species_keys)}

    item_keys = sorted(raw["items"].keys())
    item_map = {k: i for i, k in enumerate(item_keys)}

    move_keys = sorted(moves_dict.keys())
    move_map = {k: i for i, k in enumerate(move_keys)}

    raw_copy = dict(raw)
    raw_copy["moves"] = moves_dict

    return DexData(
        raw=raw_copy,
        type_names=type_names,
        type_index=type_index,
        type_chart=type_chart,
        special_type_mask=special_type_mask,
        normal_type=normal_type,
        fighting_type=fighting_type,
        flying_type=flying_type,
        poison_type=poison_type,
        ground_type=ground_type,
        rock_type=rock_type,
        ghost_type=ghost_type,
        steel_type=steel_type,
        fire_type=fire_type,
        water_type=water_type,
        grass_type=grass_type,
        electric_type=electric_type,
        ice_type=ice_type,
        sandstorm_immune_mask=sandstorm_immune_mask,
        poison_immune_mask=poison_immune_mask,
        species_keys=species_keys,
        species_map=species_map,
        item_keys=item_keys,
        item_map=item_map,
        move_keys=move_keys,
        move_map=move_map,
    )


def _compile_move_spec(dex: DexData, mkey: str) -> dict[str, Any]:
    """Compile a move entry from `gen2stadium2.json` matching `build_compact_dex_tables`."""
    md = dex.raw["moves"][mkey]
    var_bp_kind = {
        "return": 1,
        "frustration": 2,
        "hiddenpower": 3,
        "flail": 4,
        "reversal": 4,
        "magnitude": 5,
        "present": 6,
    }.get(mkey, 0)

    if var_bp_kind == 5:
        bp = 71
    elif var_bp_kind == 6:
        bp = 40
    else:
        bp = int(md.get("basePower", 0))

    acc_raw = md.get("accuracy", 100)
    always_hit_extra = bool(md.get("alwaysHit", False))
    if acc_raw is True:
        base_acc_pct = 100
        always_hit = True
    elif isinstance(acc_raw, (int, float)):
        base_acc_pct = max(0, min(100, int(acc_raw)))
        always_hit = always_hit_extra
    else:
        base_acc_pct = 100
        always_hit = always_hit_extra

    primary_status = _parse_status_code(md.get("status"))
    primary_vol = _parse_vol_code(md.get("volatileStatus"))

    sec_chance_256 = 0
    sec_status = 0
    sec_vol = 0
    sec_boost_stat = 0xFF
    sec_boost_delta = 0

    secondaries = md.get("secondaries") or []
    if secondaries:
        sec = secondaries[0]
        ch = max(0, min(100, int(sec.get("chance", 100) or 100)))
        sec_chance_256 = max(1, min(255, (ch * 255) // 100))
        sec_status = _parse_status_code(sec.get("status"))
        sec_vol = _parse_vol_code(sec.get("volatileStatus"))
        sec_boosts = sec.get("boosts") or {}
        found_sec_boost = False
        for k, v in sec_boosts.items():
            if k in _BOOST_KEYS:
                sec_boost_stat = _BOOST_KEYS.index(k)
                sec_boost_delta = int(v)
                found_sec_boost = True
                break
        if not found_sec_boost and "self" in sec and isinstance(sec["self"], dict):
            self_b = sec["self"].get("boosts") or {}
            for k, v in self_b.items():
                if k in _BOOST_KEYS:
                    sec_boost_stat = _BOOST_KEYS.index(k) | 0x80
                    sec_boost_delta = int(v)
                    break

    target = md.get("target", "normal")
    boosts_self = target in ("self", "allySide", "adjacentAllyOrSelf")
    raw_boosts = md.get("boosts") or {}
    if not raw_boosts and isinstance(md.get("self"), dict):
        self_b = md["self"].get("boosts") or {}
        if self_b:
            boosts_self = True
            raw_boosts = self_b

    boost_stats = [0xFF, 0xFF, 0xFF]
    boost_deltas = [0, 0, 0]
    b_idx = 0
    for k, v in raw_boosts.items():
        if k in _BOOST_KEYS and b_idx < 3:
            boost_stats[b_idx] = _BOOST_KEYS.index(k)
            boost_deltas[b_idx] = int(v)
            b_idx += 1

    special_kind = {
        "reflect": 1,
        "lightscreen": 2,
        "safeguard": 3,
        "spikes": 4,
        "curse": 5,
        "bellydrum": 6,
        "roar": 7,
        "whirlwind": 7,
        "encore": 8,
        "disable": 9,
        "sleeptalk": 10,
        "counter": 11,
        "mirrorcoat": 12,
        "flail": 13,
        "reversal": 13,
        "seismictoss": 14,
        "nightshade": 14,
        "superfang": 15,
        "thief": 16,
        "painsplit": 17,
        "batonpass": 18,
        "perishsong": 19,
        "destinybond": 20,
        "healbell": 21,
        "psychup": 22,
        "raindance": 23,
        "sunnyday": 24,
        "sandstorm": 25,
        "meanlook": 26,
        "spiderweb": 26,
        "haze": 27,
    }.get(mkey, 0)

    dmg_field = md.get("damage")
    if dmg_field == "level":
        fixed_damage = -1
    elif isinstance(dmg_field, (int, float)):
        fixed_damage = max(1, min(32767, int(dmg_field)))
    else:
        fixed_damage = 0

    mh = md.get("multihit")
    if isinstance(mh, int):
        multihit_mean_x2 = max(2, min(20, mh * 2))
    elif isinstance(mh, list) and len(mh) == 2:
        if mh[0] == 2 and mh[1] == 5:
            multihit_mean_x2 = 6
        else:
            multihit_mean_x2 = max(2, min(20, int(mh[0]) + int(mh[1])))
    else:
        multihit_mean_x2 = 2

    move_type = dex.type_index[md.get("type", "???")]
    cat = md.get("category", "Status")
    flags_dict = md.get("flags") or {}

    mflags = 0
    if cat == "Physical":
        mflags |= MoveFlag.IS_PHYSICAL
    if cat == "Status":
        mflags |= MoveFlag.IS_STATUS
    if always_hit:
        mflags |= MoveFlag.ALWAYS_HIT
    if md.get("ignoreAccuracy"):
        mflags |= MoveFlag.IGNORE_ACCURACY
    if md.get("ignoreEvasion"):
        mflags |= MoveFlag.IGNORE_EVASION
    if md.get("ignorePositiveEvasion"):
        mflags |= MoveFlag.IGNORE_POS_EVASION
    if md.get("selfdestruct"):
        mflags |= MoveFlag.IS_SELFDESTRUCT
    if move_type == dex.water_type:
        mflags |= MoveFlag.IS_WATER
    if move_type == dex.fire_type:
        mflags |= MoveFlag.IS_FIRE
    if mkey == "solarbeam":
        mflags |= MoveFlag.IS_SOLARBEAM
    if "authentic" in flags_dict:
        mflags |= MoveFlag.BYPASSES_SUB
    if mkey == "rest":
        mflags |= MoveFlag.IS_REST
    if mkey in (
        "recover",
        "softboiled",
        "milkdrink",
        "moonlight",
        "morningsun",
        "synthesis",
    ):
        mflags |= MoveFlag.IS_HALF_HEAL
    if mkey in ("rest", "recover", "softboiled", "milkdrink"):
        mflags |= MoveFlag.IS_HEAL_CYCLE
    if md.get("sleepUsable"):
        mflags |= MoveFlag.IS_SLEEP_USABLE
    if "nosleeptalk" not in flags_dict and "charge" not in flags_dict:
        mflags |= MoveFlag.CAN_SLEEP_TALK
    if mkey in ("protect", "detect"):
        mflags |= MoveFlag.IS_PROTECT
    if mkey == "substitute":
        mflags |= MoveFlag.IS_SUBSTITUTE
    if md.get("drain") is not None:
        mflags |= MoveFlag.DRAIN_HALF
    if md.get("recoil") is not None:
        mflags |= MoveFlag.RECOIL_QUARTER
    if boosts_self:
        mflags |= MoveFlag.BOOSTS_SELF

    return {
        "base_power": bp,
        "fixed_damage": fixed_damage,
        "primary_vol": primary_vol,
        "sec_vol": sec_vol,
        "move_type": move_type,
        "move_flags": mflags,
        "priority": int(md.get("priority", 0)),
        "base_acc_pct": base_acc_pct,
        "crit_ratio": max(0, min(5, int(md.get("critRatio", 1) or 1))),
        "primary_status": primary_status,
        "sec_chance_256": sec_chance_256,
        "sec_status": sec_status,
        "sec_boost_stat": sec_boost_stat,
        "sec_boost_delta": sec_boost_delta,
        "boost_stats": boost_stats,
        "boost_deltas": boost_deltas,
        "special_kind": special_kind,
        "var_bp_kind": var_bp_kind,
        "multihit_mean_x2": multihit_mean_x2,
        "pp": int(md.get("pp", 1)),
        "no_pp_boosts": bool(md.get("noPPBoosts", False)),
    }


def build_compact_battle(
    seed: str,
    p1_sets: list[dict[str, Any]],
    p2_sets: list[dict[str, Any]],
    p1_pick: tuple[int, int, int] = (0, 1, 2),
    p2_pick: tuple[int, int, int] = (0, 1, 2),
    dex_path: str | Path = DEFAULT_DEX_PATH,
) -> tuple[MatchTable, CompactBattle]:
    """Construct `(MatchTable, CompactBattle)` matching `Battle::from_fixture` + `team 1,2,3`."""
    dex = load_dex(dex_path)
    rng_seed = seed_from_str(seed)

    # First, simulate `Battle::from_fixture` gender draws across full rosters (p1 then p2)
    for side_sets in (p1_sets, p2_sets):
        for s in side_sets:
            sp_key = toid(s["species"])
            sp_data = dex.raw["species"][sp_key]
            g = s.get("gender") or sp_data.get("gender")
            if g not in ("M", "F", "N"):
                rng_seed, _ = prng_random(rng_seed, 2)

    # Prepare MatchTable arrays for the 6 picked Pokemon (`0..3` = P1, `3..6` = P2)
    type_mask_arr = np.zeros((6,), dtype=np.uint32)
    maxhp_arr = np.zeros((6,), dtype=np.int32)
    q1_8_arr = np.zeros((6,), dtype=np.int32)
    q1_16_arr = np.zeros((6,), dtype=np.int32)
    stored_stats_arr = np.zeros((6, 5), dtype=np.int32)
    max_pp_arr = np.zeros((6, 4), dtype=np.int32)
    types_arr = np.zeros((6, 2), dtype=np.int32)
    num_types_arr = np.zeros((6,), dtype=np.int32)
    level_arr = np.zeros((6,), dtype=np.int32)
    num_moves_arr = np.zeros((6,), dtype=np.int32)
    item_stat_kind_arr = np.zeros((6,), dtype=np.int32)
    item_boost_type_arr = np.full((6,), 0xFF, dtype=np.int32)
    item_flags_arr = np.zeros((6,), dtype=np.int32)

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

    hp_pool_arr = np.zeros((6, 2), dtype=np.int32)
    pp_arr = np.zeros((6, 4), dtype=np.int32)
    status_packed_arr = np.zeros((6,), dtype=np.int32)
    item_arr = np.zeros((6,), dtype=np.int32)
    vol_flags_arr = np.zeros((6,), dtype=np.int32)
    timers_arr = np.zeros((6, 4), dtype=np.int32)
    boosts_arr = np.zeros((6, 7), dtype=np.int32)
    slot_refs_arr = np.zeros((6,), dtype=np.int32)

    party_roster_arr = np.zeros((2, 3), dtype=np.int32)
    alive_mask_arr = np.zeros((2,), dtype=np.int32)

    picks = [p1_pick, p2_pick]
    rosters = [p1_sets, p2_sets]

    for side_n in range(2):
        for p_idx in range(3):
            c_idx = side_n * 3 + p_idx
            roster_slot = picks[side_n][p_idx]
            party_roster_arr[side_n, p_idx] = roster_slot
            alive_mask_arr[side_n] |= 1 << p_idx

            pset = rosters[side_n][roster_slot]
            sp_key = toid(pset["species"])
            sp_data = dex.raw["species"][sp_key]
            level = int(pset.get("level", 50))
            happiness = int(pset.get("happiness", 255))

            # IVs and EVs (Gen 2 DVs = even IVs)
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
            hp_pool_arr[c_idx, 0] = maxhp
            hp_pool_arr[c_idx, 1] = 0
            level_arr[c_idx] = level

            # Types
            sp_types = sp_data["types"]
            t0 = dex.type_index[sp_types[0]]
            t1 = dex.type_index[sp_types[1]] if len(sp_types) > 1 else t0
            types_arr[c_idx, 0] = t0
            types_arr[c_idx, 1] = t1
            num_types_arr[c_idx] = len(sp_types)
            tmask = (1 << (t0 & 31)) | (1 << (t1 & 31))
            type_mask_arr[c_idx] = np.uint32(tmask)

            # Hidden Power from DVs
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

            # Held item
            ikey = toid(pset.get("item", ""))
            if ikey and ikey in dex.item_map:
                iid = dex.item_map[ikey]
                item_arr[c_idx] = (iid & 0x7F) + 1
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

            # Moves
            raw_moves = pset.get("moves") or []
            nm = min(4, len(raw_moves))
            num_moves_arr[c_idx] = nm
            return_bp = (happiness * 10) // 25
            frustration_bp = ((255 - happiness) * 10) // 25

            for m_i in range(nm):
                mkey = toid(raw_moves[m_i])
                # Canonicalize hiddenpowerxxx -> hiddenpower
                if mkey.startswith("hiddenpower") and mkey != "hiddenpower":
                    mkey = "hiddenpower"
                cm = _compile_move_spec(dex, mkey)
                pp_ups = 0 if cm["no_pp_boosts"] else 3
                pp_val = (cm["pp"] * (5 + pp_ups)) // 5
                if cm["pp"] == 40:
                    pp_val -= pp_ups
                pp_val = max(0, min(64, pp_val))
                max_pp_arr[c_idx, m_i] = pp_val
                pp_arr[c_idx, m_i] = pp_val

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

    # Simulate TeamPreview `commit_choices` -> `speed_sort` tie-breakers for index 0, 1, 2
    for p_idx in range(3):
        if int(stored_stats_arr[p_idx, 4]) == int(stored_stats_arr[3 + p_idx, 4]):
            rng_seed, _ = prng_random(rng_seed, 2)

    # Simulate `ActionKind::Start` -> `switch_in` -> `insert_choice_run_switch` tie-breaker on leads
    if int(stored_stats_arr[0, 4]) == int(stored_stats_arr[3, 4]):
        rng_seed, _ = prng_random(rng_seed, 2)

    # End of turn 0 (`Start` -> `end_turn`): turn becomes 1 and Quick Claw roll is drawn
    rng_seed, qc_bool = prng_random_chance(rng_seed, 60, 256)
    quick_claw_roll = int(qc_bool)

    table = MatchTable(
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

    battle = CompactBattle(
        hp_pool=jnp.asarray(hp_pool_arr, dtype=jnp.int32),
        pp=jnp.asarray(pp_arr, dtype=jnp.int32),
        status_packed=jnp.asarray(status_packed_arr, dtype=jnp.int32),
        item=jnp.asarray(item_arr, dtype=jnp.int32),
        vol_flags=jnp.asarray(vol_flags_arr, dtype=jnp.int32),
        timers=jnp.asarray(timers_arr, dtype=jnp.int32),
        boosts=jnp.asarray(boosts_arr, dtype=jnp.int32),
        slot_refs=jnp.asarray(slot_refs_arr, dtype=jnp.int32),
        active_idx=jnp.array([0, 0], dtype=jnp.int32),
        party_roster=jnp.asarray(party_roster_arr, dtype=jnp.int32),
        alive_mask=jnp.asarray(alive_mask_arr, dtype=jnp.int32),
        reflect_turns=jnp.array([0, 0], dtype=jnp.int32),
        lightscreen_turns=jnp.array([0, 0], dtype=jnp.int32),
        safeguard_turns=jnp.array([0, 0], dtype=jnp.int32),
        side_flags=jnp.array([0, 0], dtype=jnp.int32),
        last_move_selfko=jnp.array([255, 255], dtype=jnp.int32),
        request=jnp.array([1, 1], dtype=jnp.int32),
        turn=jnp.int32(1),
        weather_kind=jnp.int32(CompactWeather.NONE),
        weather_turns=jnp.int32(0),
        quick_claw_roll=jnp.int32(quick_claw_roll),
        outcome=jnp.int32(0),
        speed_order=jnp.int32(0),
        clause_flags=jnp.int32(5),  # Sleep Clause (1) | Freeze Clause (4)
        hurt_damage_this_turn=jnp.array([0, 0], dtype=jnp.int32),
        hurt_cat_this_turn=jnp.array([0, 0], dtype=jnp.int32),
        prng_seed=rng_seed,
    )
    return table, battle
