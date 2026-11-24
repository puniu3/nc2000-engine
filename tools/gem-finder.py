#!/usr/bin/env python3
"""Minor-Pokemon Gem-Finder for NC2000 (Big-16 Anchor 1v1 + Adversarial Switch-In + Pareto Breakout + Preview Gate).

Pipeline:
  1. `generate`: Build Big-16 meta anchors, complete 44-Major reference sets, and
     tailored best-foot-forward sets for all non-dominated Minor species across
     3 resource tiers (L50_NO_LEFT, L50_LEFT, L55_ACE). Canonicalize & validate
     every set via WASM Validator + training-v3 normalizer.
  2. `run`: Stage 1 fast root-search screen (`gem_1v1 --mode root`), then Stage 2
     full played-out 1v1 + switch-in verification (`gem_1v1 --mode play`) on all
     Pareto-promising Minors + top Majors per tier.
  3. `analyze`: Extract Pareto envelope breakouts (1-anchor, 2-anchor, 3-anchor
     role compression vs ALL 44 Majors of the same tier) and Drop-In upgrades for
     flagship/pillar shells (`s0053-lefties`, `pb-84`, `thdrcomb`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "target/gem-finder"
GEM_BIN = ROOT / "target/release/examples/gem_1v1"

# Gen 2 physical types vs special types
PHYSICAL_TYPES = {"Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel"}
SPECIAL_TYPES = {"Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"}

# Pure evasion / swagger cheese excluded from auto-generated candidates
BANNED_AUTO_MOVES = {
    "doubleteam", "minimize", "swagger", "flash", "sandattack", "smokescreen",
    "kinesis", "attract", "sweetkiss", "supersonic", "confuseray",
    # OHKO (already absent) & unusable/1v1-broken
    "fissure", "horndrill", "guillotine", "splash", "teleport", "whirlwind", "roar",
    "batonpass", "spikes", "safeguard", "perishsong", "destinybond", "meanlook", "spiderweb",
    "explosion", "selfdestruct", "metronome", "mimic", "mirrormove", "bide", "focusenergy",
    "rage", "thrash", "outrage", "petaldance", "hyperbeam", "solarbeam", "skyattack",
    "skullbash", "razorwind", "fly", "dig", "rollout", "furycutter", "snore",
    "nightmare", "dreameater", "curse_ghost", "spite", "present", "magnitude",
}

# Known USA NYPC event moves / JP cart illegalities
NYPC_ILLEGAL: dict[str, set[str]] = {
    "snorlax": {"lovelykiss", "splash", "sweetkiss"},
    "nidoking": {"lovelykiss", "morningsun"},
    "nidoqueen": {"lovelykiss", "moonlight"},
    "politoed": {"lovelykiss", "growth"},
    "poliwrath": {"lovelykiss", "growth"},
    "poliwhirl": {"lovelykiss", "growth"},
    "granbull": {"lovelykiss"},
    "victreebel": {"lovelykiss"},
    "sneasel": {"moonlight"},
    "vaporeon": {"growth"},
    "jolteon": {"growth"},
    "flareon": {"growth"},
    "espeon": {"growth"},
    "umbreon": {"growth"},
    "eevee": {"growth"},
    "zapdos": {"batonpass", "lightscreen"},
    "pikachu": {"surf", "fly", "sing", "petaldance"},
    "raichu": {"surf", "fly", "sing", "petaldance"},
    "chansey": {"sweetkiss"},
    "blissey": {"sweetkiss"},
    "clefable": {"petaldance", "scaryface", "swift"},
    "wigglytuff": {"petaldance", "scaryface"},
    "miltank": {"sing"},
    "lapras": {"bite", "futuresight"},
    "tauros": {"quickattack"},
    "kangaskhan": {"faintattack"},
    "tentacruel": {"confuseray"},
    "golem": {"rapidspin"},
    "rhydon": {"crunch"},
    "machamp": {"thrash", "falseswipe"},
    "heracross": {"seismictoss"},
    "scizor": {"sonicboom"},
    "scyther": {"sonicboom"},
    "pinsir": {"rockthrow"},
    "qwilfish": {"doubleedge"},
    "shuckle": {"sweetscent"},
    "delibird": {"spikes", "payday"},
    "octillery": {"amnesia"},
    "kingdra": {"haze"},
    "starmie": {"twister"},
    "dewgong": {"flail"},
    "slowbro": {"bellydrum"},
    "slowking": {"bellydrum"},
    "marowak": {"furycutter"},
    "cubone": {"furycutter"},
    "dodrio": {"lowkick"},
    "farfetchd": {"furycutter"},
    "lickitung": {"doubleslap"},
    "tangela": {"synthesis"},
    "aerodactyl": {"rockthrow"},
    "kabutops": {"rockthrow"},
    "omastar": {"rockthrow"},
    "porygon2": {"barrier"},
    "xatu": {"safeguard"},
    "girafarig": {"futuresight"},
    "misdreavus": {"hypnosis"},
    "murkrow": {"beatup"},
    "wobbuffet": {"mimic"},
    "kingler": {"metalclaw"},
    "hypno": {"amnesia"},
    "exeggutor": {"sweetscent"},
    "alakazam": {"foresight"},
    "gengar": {"spite"},
    "tyranitar": {"rage"},
    "dragonite": {"hydropump"},
    "charizard": {"crunch"},
    "blastoise": {"zapcannon"},
    "venusaur": {"ancientpower"},
}

INCOMPATIBLE_COMBOS: list[tuple[str, set[str]]] = [
    ("marowak", {"bellydrum", "rockslide"}),
    ("marowak", {"bellydrum", "swordsdance"}),
    ("blissey", {"healbell", "counter"}),
    ("chansey", {"healbell", "counter"}),
    ("charizard", {"bellydrum", "crunch"}),
    ("snorlax", {"bellydrum", "lovelykiss"}),
    ("slowbro", {"bellydrum", "amnesia"}),
    ("slowking", {"bellydrum", "amnesia"}),
    ("quagsire", {"bellydrum", "ancientpower"}),
    ("azumarill", {"bellydrum", "perishsong"}),
    ("nidoking", {"counter", "morningsun"}),
    ("nidoqueen", {"counter", "moonlight"}),
]

SIGNATURE_ITEMS: dict[str, str] = {
    "pikachu": "Light Ball",
    "cubone": "Thick Club",
    "marowak": "Thick Club",
    "farfetchd": "Stick",
    "chansey": "Lucky Punch",
    "ditto": "Metal Powder",
}


def toid(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def load_dex_and_learnsets() -> tuple[dict[str, Any], dict[str, Any]]:
    dex = json.loads((ROOT / "data/gen2stadium2.json").read_text())
    ls = json.loads((ROOT / "data/learnsets-gen2.json").read_text())
    return dex, ls


def type_mult(dex: dict[str, Any], atk_type: str, def_types: list[str]) -> float:
    mult = 1.0
    tc = dex["typechart"]
    for dt in def_types:
        code = tc[dt.lower()]["damageTaken"].get(atk_type, 0)
        if code == 1:
            mult *= 2.0
        elif code == 2:
            mult *= 0.5
        elif code == 3:
            mult *= 0.0
    return mult


def gen2_stat(base: int, level: int, is_hp: bool, dv: int = 15, stat_exp: int = 65535) -> int:
    ev_term = int(math.ceil(math.sqrt(stat_exp))) // 4
    val = ((base + dv) * 2 + ev_term) * level // 100
    return val + level + 10 if is_hp else val + 5


def is_move_jp_legal(sp_id: str, move_id: str, level: int, ls_entry: dict[str, Any]) -> bool:
    if move_id not in ls_entry["moves"]:
        return False
    min_lv = ls_entry.get("moveMinLevel", {}).get(move_id, 50)
    if level < min_lv:
        return False
    if move_id in NYPC_ILLEGAL.get(sp_id, set()):
        return False
    return True


def set_is_jp_legal(sp_id: str, moves: list[str], level: int, ls_entry: dict[str, Any]) -> bool:
    mids = set()
    for m in moves:
        mid = toid(m)
        base = "hiddenpower" if mid.startswith("hiddenpower") else mid
        if not is_move_jp_legal(sp_id, base, level, ls_entry):
            return False
        mids.add(mid)
    for sp_chk, bad in INCOMPATIBLE_COMBOS:
        if sp_id == sp_chk and bad.issubset(mids):
            return False
    return True


def build_big16_anchors() -> list[dict[str, Any]]:
    """Canonical Big-16 meta anchors from team-pool-v3 & flagship/breaker pillars."""
    raw = [
        # --- 10 L55 Aces ---
        ("Lax55_Talk", "Snorlax", 55, "Miracle Berry", ["Return", "Curse", "Rest", "Sleep Talk"]),
        ("Lax55_Left", "Snorlax", 55, "Leftovers", ["Body Slam", "Earthquake", "Curse", "Rest"]),
        ("Zap55_HPI", "Zapdos", 55, "Leftovers", ["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
        ("Zap55_DP", "Zapdos", 55, "Leftovers", ["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        ("Raik55_Mag", "Raikou", 55, "Magnet", ["Thunder", "Hidden Power Ice", "Rest", "Sleep Talk"]),
        ("Wak55_HPB", "Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Hidden Power Bug", "Swords Dance"]),
        ("Ttar55_Mix", "Tyranitar", 55, "Leftovers", ["Rock Slide", "Earthquake", "Ice Beam", "Dynamic Punch"]),
        ("Cune55_Talk", "Suicune", 55, "Mint Berry", ["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
        ("P2_55_DT", "Porygon2", 55, "Bright Powder", ["Double-Edge", "Curse", "Double Team", "Recover"]),
        ("P2_55_Ice", "Porygon2", 55, "Leftovers", ["Return", "Ice Beam", "Curse", "Recover"]),
        # --- 6 L50 Meta Cores ---
        ("Lax50_Curse", "Snorlax", 50, "Leftovers", ["Body Slam", "Earthquake", "Curse", "Self-Destruct"]),
        ("Champ50_Talk", "Machamp", 50, "Scope Lens", ["Cross Chop", "Rock Slide", "Rest", "Sleep Talk"]),
        ("Venu50_Seed", "Venusaur", 50, "PRZ Cure Berry", ["Razor Leaf", "Leech Seed", "Sleep Powder", "Synthesis"]),
        ("Eggy50_Seed", "Exeggutor", 50, "Gold Berry", ["Psychic", "Leech Seed", "Sleep Powder", "Explosion"]),
        ("Skarm50_Curse", "Skarmory", 50, "Mint Berry", ["Drill Peck", "Curse", "Toxic", "Rest"]),
        ("Star50_Rec", "Starmie", 50, "Miracle Berry", ["Hydro Pump", "Psychic", "Thunder Wave", "Recover"]),
    ]
    out = []
    for aid, sp, lv, item, moves in raw:
        out.append({
            "id": aid,
            "set": {
                "name": sp,
                "species": sp,
                "item": item,
                "ability": "No Ability",
                "moves": moves,
                "nature": "Serious",
                "level": lv,
                "happiness": 255,
            },
        })
    return out


def approx_move_dmg_frac(
    dex: dict[str, Any],
    att_sp: dict[str, Any],
    att_lv: int,
    att_item: str,
    move_name: str,
    def_sp: dict[str, Any],
    def_lv: int,
) -> float:
    mid = toid(move_name)
    if mid.startswith("hiddenpower"):
        tname = move_name.replace("Hidden Power", "").strip()
        if not tname:
            return 0.0
        mtype = tname
        bp = 70
        acc = 1.0
        is_phys = mtype in PHYSICAL_TYPES
    else:
        mv = dex["moves"].get(mid)
        if not mv:
            return 0.0
        bp = mv.get("basePower", 0)
        if mid in ("return", "frustration"):
            bp = 102
        elif mid in ("lowkick",):
            bp = 50
        elif mid in ("seismicoss", "seismictoss", "nightshade"):
            def_hp = gen2_stat(def_sp["baseStats"]["hp"], def_lv, True)
            imm = type_mult(dex, mv["type"], def_sp["types"]) == 0.0
            return 0.0 if imm else att_lv / def_hp
        if bp <= 0:
            return 0.0
        mtype = mv["type"]
        raw_acc = mv.get("accuracy", 100)
        acc = 1.0 if raw_acc is True else float(raw_acc) / 100.0
        is_phys = mtype in PHYSICAL_TYPES

    eff = type_mult(dex, mtype, def_sp["types"])
    if eff == 0.0:
        return 0.0
    stab = 1.5 if mtype in att_sp["types"] else 1.0

    atk = gen2_stat(att_sp["baseStats"]["atk" if is_phys else "spa"], att_lv, False)
    if is_phys and att_item == "Thick Club" and att_sp["name"] in ("Marowak", "Cubone"):
        atk *= 2
    if not is_phys and att_item == "Light Ball" and att_sp["name"] == "Pikachu":
        atk *= 2
    dfn = gen2_stat(def_sp["baseStats"]["def" if is_phys else "spd"], def_lv, False)
    def_hp = gen2_stat(def_sp["baseStats"]["hp"], def_lv, True)

    dmg = (((2 * att_lv // 5 + 2) * bp * atk // max(1, dfn)) // 50 + 2) * stab * eff * 0.925
    # Slight penalty for low-accuracy moves in ranking
    return (dmg / def_hp) * (0.65 + 0.35 * acc)


def rank_species_attacks(
    dex: dict[str, Any],
    sp_id: str,
    level: int,
    item: str,
    ls_entry: dict[str, Any],
    anchors: list[dict[str, Any]],
) -> list[tuple[str, list[float]]]:
    sp_data = dex["species"][sp_id]
    cand_moves: list[str] = []
    for mid in ls_entry["moves"]:
        if mid in BANNED_AUTO_MOVES or mid == "hiddenpower":
            continue
        if not is_move_jp_legal(sp_id, mid, level, ls_entry):
            continue
        mv = dex["moves"].get(mid)
        if not mv:
            continue
        bp = mv.get("basePower", 0)
        if mid in ("return", "frustration"):
            bp = 102
        if bp < 50 and mid not in ("seismictoss", "nightshade", "quickattack", "machpunch", "extremespeed", "gigadrain", "feintattack", "faintattack", "pursuit", "bonemerang", "doublekick", "pinmissile"):
            continue
        # Skip severely inaccurate moves (<70% acc) except Dynamic Punch/Zap Cannon/Hypnosis
        raw_acc = mv.get("accuracy", 100)
        if raw_acc is not True and raw_acc < 70 and mid not in ("dynamicpunch", "zapcannon"):
            continue
        # Skip Frustration if Return is available
        if mid == "frustration" and is_move_jp_legal(sp_id, "return", level, ls_entry):
            continue
        cand_moves.append(mv["name"])

    # Also evaluate useful Hidden Power types
    if is_move_jp_legal(sp_id, "hiddenpower", level, ls_entry):
        for hpt in ["Ice", "Electric", "Ground", "Fighting", "Bug", "Fire", "Rock", "Grass", "Water", "Ghost", "Flying"]:
            cand_moves.append(f"Hidden Power {hpt}")

    scored: list[tuple[float, str, list[float]]] = []
    for mname in cand_moves:
        vec = []
        for a in anchors:
            asp = dex["species"][toid(a["set"]["species"])]
            vec.append(approx_move_dmg_frac(dex, sp_data, level, item, mname, asp, a["set"]["level"]))
        # Score = sum of capped per-anchor utility (value 2HKO/3HKO thresholds!)
        tot = sum(min(v, 0.75) for v in vec)
        if tot > 0.35:
            scored.append((tot, mname, vec))

    # Deduplicate same-type strictly weaker moves (keep highest scoring per type, plus special utility like Body Slam/Thunder/Ice Beam)
    scored.sort(key=lambda x: -x[0])
    out: list[tuple[str, list[float]]] = []
    seen_name = set()
    for _, mname, vec in scored:
        if mname in seen_name:
            continue
        seen_name.add(mname)
        out.append((mname, vec))
    return out


def pick_best_attack_pair(ranked: list[tuple[str, list[float]]], prefer_non_hp_first: bool = True) -> list[str]:
    if not ranked:
        return []
    # Pick best primary attack (preferring non-HP if close)
    best0 = ranked[0]
    if prefer_non_hp_first and best0[0].startswith("Hidden Power"):
        for cand in ranked[1:8]:
            if not cand[0].startswith("Hidden Power") and sum(min(v, 0.7) for v in cand[1]) >= 0.88 * sum(min(v, 0.7) for v in best0[1]):
                best0 = cand
                break
    # Pick 2nd attack maximizing marginal coverage over Big-16 (and never 2 Hidden Powers!)
    best1 = None
    best1_val = -1.0
    for mname, vec in ranked:
        if mname == best0[0]:
            continue
        if best0[0].startswith("Hidden Power") and mname.startswith("Hidden Power"):
            continue
        val = sum(min(max(a, b), 0.75) for a, b in zip(best0[1], vec))
        if val > best1_val:
            best1_val = val
            best1 = (mname, vec)
    res = [best0[0]]
    if best1:
        res.append(best1[0])
    return res


def pick_best_attack_trio(ranked: list[tuple[str, list[float]]]) -> list[str]:
    pair = pick_best_attack_pair(ranked)
    if len(pair) < 2:
        return pair
    vmap = {m: v for m, v in ranked}
    cur = [max(a, b) for a, b in zip(vmap[pair[0]], vmap[pair[1]])]
    has_hp = any(m.startswith("Hidden Power") for m in pair)
    best2 = None
    best2_val = -1.0
    for mname, vec in ranked:
        if mname in pair:
            continue
        if has_hp and mname.startswith("Hidden Power"):
            continue
        val = sum(min(max(a, b), 0.75) for a, b in zip(cur, vec))
        if val > best2_val:
            best2_val = val
            best2 = mname
    if best2:
        pair.append(best2)
    return pair


def generate_minor_sets_for_species(
    dex: dict[str, Any],
    sp_id: str,
    ls_entry: dict[str, Any],
    anchors: list[dict[str, Any]],
    bp_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sp_data = dex["species"][sp_id]
    sp_name = sp_data["name"]
    min_lv = ls_entry.get("minLevel", 50)
    bs = sp_data["baseStats"]
    sig_item = SIGNATURE_ITEMS.get(sp_id)

    def has(mid: str, lv: int) -> bool:
        return is_move_jp_legal(sp_id, mid, lv, ls_entry)

    def heal50(lv: int) -> str | None:
        for mid, nm in [
            ("recover", "Recover"),
            ("softboiled", "Soft-Boiled"),
            ("milkdrink", "Milk Drink"),
            ("synthesis", "Synthesis"),
            ("moonlight", "Moonlight"),
            ("morningsun", "Morning Sun"),
        ]:
            if has(mid, lv):
                return nm
        return None

    def sleep_mv(lv: int) -> str | None:
        for mid, nm in [
            ("spore", "Spore"),
            ("sleeppowder", "Sleep Powder"),
            ("lovelykiss", "Lovely Kiss"),
            ("hypnosis", "Hypnosis"),
            ("sing", "Sing"),
        ]:
            if has(mid, lv):
                return nm
        return None

    def setup_mvs(lv: int) -> list[str]:
        out = []
        # Check physical vs special bias
        if bs["atk"] >= 65 and has("swordsdance", lv):
            out.append("Swords Dance")
        if bs["atk"] >= 65 and has("bellydrum", lv):
            out.append("Belly Drum")
        if (bs["atk"] >= 65 or bs["def"] >= 75) and has("curse", lv):
            out.append("Curse")
        if bs["spa"] >= 75 and has("growth", lv):
            out.append("Growth")
        if bs["spd"] >= 75 and has("amnesia", lv):
            out.append("Amnesia")
        if bs["def"] >= 75 and has("barrier", lv):
            out.append("Barrier")
        if bs["def"] >= 75 and has("acidarmor", lv):
            out.append("Acid Armor")
        return out

    def tech_mvs(lv: int) -> list[str]:
        out = []
        for mid, nm in [
            ("leechseed", "Leech Seed"),
            ("encore", "Encore"),
            ("thunderwave", "Thunder Wave"),
            ("stunspore", "Stun Spore"),
            ("glare", "Glare"),
            ("counter", "Counter"),
            ("mirrorcoat", "Mirror Coat"),
            ("haze", "Haze"),
            ("charm", "Charm"),
            ("toxic", "Toxic"),
            ("reflect", "Reflect"),
            ("lightscreen", "Light Screen"),
            ("substitute", "Substitute"),
        ]:
            if has(mid, lv):
                out.append(nm)
        return out

    # Special case: Wobbuffet & Ditto & Shuckle
    generated: list[tuple[str, str, int, str, list[str]]] = []  # (tag, tier, lv, item, moves)

    if sp_id == "wobbuffet":
        for tier, lv, it in [("L50_NO_LEFT", 50, "Miracle Berry"), ("L50_LEFT", 50, "Leftovers"), ("L55_ACE", 55, "Leftovers")]:
            generated.append((f"wobb_{tier.lower()}", tier, lv, it, ["Counter", "Mirror Coat", "Encore", "Destiny Bond"]))
    elif sp_id == "shuckle":
        for tier, lv, it in [("L50_NO_LEFT", 50, "Mint Berry"), ("L50_LEFT", 50, "Leftovers")]:
            generated.append((f"shuck_tox_{tier.lower()}", tier, lv, it, ["Toxic", "Encore", "Rest", "Sleep Talk"]))
            generated.append((f"shuck_wrap_{tier.lower()}", tier, lv, it, ["Toxic", "Wrap", "Rest", "Encore"]))
    else:
        tier_specs = []
        if min_lv <= 50:
            tier_specs.append(("L50_NO_LEFT", 50, sig_item or "Miracle Berry", sig_item or "Mint Berry", sig_item or "Gold Berry"))
            tier_specs.append(("L50_LEFT", 50, sig_item or "Leftovers", sig_item or "Leftovers", sig_item or "Leftovers"))
        tier_specs.append(("L55_ACE", 55, sig_item or "Leftovers", sig_item or "Leftovers", sig_item or "Miracle Berry"))

        for tier, lv, it_std, it_rest, it_atk in tier_specs:
            # Skip redundant L50_LEFT when signature item overrides Leftovers
            if tier == "L50_LEFT" and sig_item:
                continue
            ranked = rank_species_attacks(dex, sp_id, lv, it_atk, ls_entry, anchors)
            pair = pick_best_attack_pair(ranked)
            trio = pick_best_attack_trio(ranked)
            h50 = heal50(lv)
            slp = sleep_mv(lv)
            setups = setup_mvs(lv)
            techs = tech_mvs(lv)

            # Shell 1: RestTalk (2 best attacks + Rest + Sleep Talk)
            if has("rest", lv) and has("sleeptalk", lv) and len(pair) >= 2:
                it_rt = sig_item or ("Leftovers" if tier != "L50_NO_LEFT" else "Mint Berry")
                generated.append((f"rt_{tier.lower()}", tier, lv, it_rt, [pair[0], pair[1], "Rest", "Sleep Talk"]))

            # Shell 2: Setup + Recovery (Curse / SD / Growth / Amnesia + best attack(s) + Heal/Rest)
            if setups and pair:
                su = setups[0]
                if h50 and len(pair) >= 2:
                    generated.append((f"su50_{tier.lower()}", tier, lv, it_std, [pair[0], pair[1], su, h50]))
                elif has("rest", lv) and has("sleeptalk", lv):
                    # Curse/Growth RestTalk (1 STAB + Setup + Rest + Sleep Talk)
                    it_rt = sig_item or ("Leftovers" if tier != "L50_NO_LEFT" else "Mint Berry")
                    generated.append((f"surt_{tier.lower()}", tier, lv, it_rt, [pair[0], su, "Rest", "Sleep Talk"]))
                elif has("rest", lv) and len(pair) >= 2:
                    it_r = sig_item or ("Leftovers" if tier != "L50_LEFT" and tier != "L55_ACE" else it_rest)
                    if tier == "L50_NO_LEFT":
                        it_r = sig_item or "Mint Berry"
                    generated.append((f"sur_{tier.lower()}", tier, lv, it_r, [pair[0], pair[1], su, "Rest"]))

            # Shell 3: 50% Heal + Bulky Utility (or Sleep + Heal)
            if h50 and len(pair) >= 1:
                if slp and len(pair) >= 2:
                    generated.append((f"slph_{tier.lower()}", tier, lv, it_std, [pair[0], pair[1], slp, h50]))
                elif techs and len(pair) >= 2:
                    generated.append((f"utlh_{tier.lower()}", tier, lv, it_std, [pair[0], pair[1], techs[0], h50]))
                elif len(trio) >= 3:
                    generated.append((f"atk3h_{tier.lower()}", tier, lv, it_std, [trio[0], trio[1], trio[2], h50]))

            # Shell 4: Sleep / Tech + Attacks (for mons without 50% heal, or offensive sleepers/techs)
            if slp and len(pair) >= 2 and not h50:
                m4 = setups[0] if setups else (trio[2] if len(trio) >= 3 else (techs[0] if techs else ("Rest" if has("rest", lv) else None)))
                if m4 and m4 not in pair and m4 != slp:
                    generated.append((f"slp_{tier.lower()}", tier, lv, it_atk, [pair[0], pair[1], slp, m4]))
            elif techs and len(trio) >= 3:
                # Pick first tech that makes sense in 1v1 (Encore/TWave/Counter/MirrorCoat/LeechSeed/Toxic)
                for tc in techs:
                    if tc not in trio:
                        generated.append((f"tech_{tier.lower()}", tier, lv, it_atk if tier == "L50_NO_LEFT" else it_std, [trio[0], trio[1], trio[2], tc]))
                        break

            # Shell 5: Special anti-meta Curse + 2 attacks + Rest (if setup[0] wasn't Curse and species has Curse + Rest)
            if "Curse" in setups and setups[0] != "Curse" and has("rest", lv) and len(pair) >= 2:
                it_r = sig_item or ("Mint Berry" if tier == "L50_NO_LEFT" else "Leftovers")
                generated.append((f"curse_{tier.lower()}", tier, lv, it_r, [pair[0], pair[1], "Curse", "Rest"]))

    # Also include existing belief-pool-v3 sets for this minor species (filtered for JP legality & no evasion cheese!)
    for idx, s in enumerate(bp_sets):
        lv = int(s["level"])
        it = s.get("item", "")
        moves = list(s["moves"])
        mids = {toid(m) for m in moves}
        if mids & {"doubleteam", "minimize", "swagger", "lovelykiss"}:
            continue
        if not set_is_jp_legal(sp_id, moves, lv, ls_entry):
            continue
        if lv == 55:
            tier = "L55_ACE"
        elif lv == 50 and it == "Leftovers":
            tier = "L50_LEFT"
        elif lv == 50:
            tier = "L50_NO_LEFT"
        else:
            continue
        generated.append((f"bp{idx}_{tier.lower()}", tier, lv, it, moves))

    # Deduplicate by (tier, lv, item, sorted_moves) and enforce JP legality + <= 4 unique moves
    out = []
    seen = set()
    # Cap per-tier variants per minor species so total stays compact (~4-6 per species)
    per_tier_cap = {"L50_NO_LEFT": 4, "L50_LEFT": 3, "L55_ACE": 3}
    per_tier_cnt: Counter[str] = Counter()

    for tag, tier, lv, it, moves in generated:
        if per_tier_cnt[tier] >= per_tier_cap[tier]:
            continue
        # Remove duplicates preserving order
        uniq_m = []
        for m in moves:
            if m not in uniq_m:
                uniq_m.append(m)
        if not uniq_m or len(uniq_m) > 4:
            continue
        if not set_is_jp_legal(sp_id, uniq_m, lv, ls_entry):
            continue
        key = (tier, lv, it, tuple(sorted(uniq_m)))
        if key in seen:
            continue
        seen.add(key)
        per_tier_cnt[tier] += 1
        out.append({
            "id": f"min:{sp_id}:{tag}",
            "species": sp_name,
            "tier": tier,
            "is_major": False,
            "set": {
                "name": sp_name,
                "species": sp_name,
                "item": it,
                "ability": "No Ability",
                "moves": uniq_m,
                "nature": "Serious",
                "level": lv,
                "happiness": 255,
            },
        })
    return out


def build_major_reference_pool(
    dex: dict[str, Any],
    ls: dict[str, Any],
    major_ids: set[str],
    tp3_teams: list[dict[str, Any]],
    bp3_teams: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect the real strongest sets across ALL 44 Major species, partitioned by tier."""
    freq: dict[tuple[str, str, int, str, tuple[str, ...]], tuple[int, list[str]]] = {}
    for src_w, teams in [(5, tp3_teams), (1, bp3_teams)]:
        for t in teams:
            for s in t["sets"]:
                sp_id = toid(s["species"])
                if sp_id not in major_ids:
                    continue
                lv = int(s["level"])
                it = s.get("item", "")
                moves = list(s["moves"])
                mids = {toid(m) for m in moves}
                # Exclude evasion/swagger cheese & NYPC illegal from reference baseline too
                if mids & {"doubleteam", "minimize", "swagger"}:
                    continue
                if not set_is_jp_legal(sp_id, moves, lv, ls["species"][sp_id]):
                    continue
                if lv == 55:
                    tier = "L55_ACE"
                elif lv == 50 and it == "Leftovers":
                    tier = "L50_LEFT"
                elif lv == 50:
                    tier = "L50_NO_LEFT"
                else:
                    continue
                key = (sp_id, tier, lv, it, tuple(sorted(moves)))
                prev_w, prev_m = freq.get(key, (0, moves))
                freq[key] = (prev_w + src_w, prev_m)

    # Keep top 3 sets per (major_species, tier)
    by_sp_tier: dict[tuple[str, str], list[tuple[int, tuple, list[str]]]] = defaultdict(list)
    for key, (w, orig_moves) in freq.items():
        sp_id, tier = key[0], key[1]
        by_sp_tier[(sp_id, tier)].append((w, key, orig_moves))

    out = []
    seen_key: set[tuple[str, str, int, str, tuple[str, ...]]] = set()
    by_sp_tier_out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for (sp_id, tier), items in sorted(by_sp_tier.items()):
        items.sort(key=lambda x: -x[0])
        sp_name = dex["species"][sp_id]["name"]
        for rank, (_, key, orig_moves) in enumerate(items[:3]):
            _, _, lv, it, _ = key
            seen_key.add(key)
            rec = {
                "id": f"maj:{sp_id}:{tier.lower()}:{rank}",
                "species": sp_name,
                "tier": tier,
                "is_major": True,
                "set": {
                    "name": sp_name,
                    "species": sp_name,
                    "item": it,
                    "ability": "No Ability",
                    "moves": orig_moves,
                    "nature": "Serious",
                    "level": lv,
                    "happiness": 255,
                },
            }
            out.append(rec)
            by_sp_tier_out[(sp_id, tier)].append(rec)

    # Ensure every Major species has a fair, complete baseline in ALL 3 tiers
    # (so Minors never "cheat" by winning a tier where a Major wasn't represented in bp3).
    for sp_id in sorted(major_ids):
        ls_entry = ls["species"][sp_id]
        min_lv = ls_entry.get("minLevel", 50)
        sp_name = dex["species"][sp_id]["name"]
        sig_item = SIGNATURE_ITEMS.get(sp_id)

        def add_synth(tier: str, lv: int, it: str, moves: list[str], tag: str) -> None:
            if not set_is_jp_legal(sp_id, moves, lv, ls_entry):
                return
            k = (sp_id, tier, lv, it, tuple(sorted(moves)))
            if k in seen_key:
                return
            seen_key.add(k)
            rec = {
                "id": f"maj:{sp_id}:{tier.lower()}:{tag}",
                "species": sp_name,
                "tier": tier,
                "is_major": True,
                "set": {
                    "name": sp_name,
                    "species": sp_name,
                    "item": it,
                    "ability": "No Ability",
                    "moves": list(moves),
                    "nature": "Serious",
                    "level": lv,
                    "happiness": 255,
                },
            }
            out.append(rec)
            by_sp_tier_out[(sp_id, tier)].append(rec)

        # Backfill L50_LEFT from L55_ACE & L50_NO_LEFT
        if min_lv <= 50 and not sig_item and len(by_sp_tier_out[(sp_id, "L50_LEFT")]) < 2:
            for src_tier in ("L55_ACE", "L50_NO_LEFT"):
                for idx_s, src in enumerate(by_sp_tier_out[(sp_id, src_tier)][:2]):
                    add_synth("L50_LEFT", 50, "Leftovers", src["set"]["moves"], f"syn_{src_tier.lower()}_{idx_s}")
                    if len(by_sp_tier_out[(sp_id, "L50_LEFT")]) >= 2:
                        break
                if len(by_sp_tier_out[(sp_id, "L50_LEFT")]) >= 2:
                    break

        # Backfill L50_NO_LEFT from L55_ACE & L50_LEFT
        if min_lv <= 50 and len(by_sp_tier_out[(sp_id, "L50_NO_LEFT")]) < 2:
            for src_tier in ("L55_ACE", "L50_LEFT"):
                for idx_s, src in enumerate(by_sp_tier_out[(sp_id, src_tier)][:2]):
                    mvs = src["set"]["moves"]
                    it = sig_item or ("Mint Berry" if any(toid(m) == "rest" for m in mvs) else "Miracle Berry")
                    add_synth("L50_NO_LEFT", 50, it, mvs, f"syn_{src_tier.lower()}_{idx_s}")
                    if len(by_sp_tier_out[(sp_id, "L50_NO_LEFT")]) >= 2:
                        break
                if len(by_sp_tier_out[(sp_id, "L50_NO_LEFT")]) >= 2:
                    break

        # Backfill L55_ACE from L50_LEFT & L50_NO_LEFT
        if len(by_sp_tier_out[(sp_id, "L55_ACE")]) < 2:
            for src_tier in ("L50_LEFT", "L50_NO_LEFT"):
                for idx_s, src in enumerate(by_sp_tier_out[(sp_id, src_tier)][:2]):
                    it = sig_item or "Leftovers"
                    add_synth("L55_ACE", 55, it, src["set"]["moves"], f"syn_{src_tier.lower()}_{idx_s}")
                    if len(by_sp_tier_out[(sp_id, "L55_ACE")]) >= 2:
                        break
                if len(by_sp_tier_out[(sp_id, "L55_ACE")]) >= 2:
                    break

    return out


def canonicalize_sets_via_wasm(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run every single PokemonSet through the real WASM Validator + training-v3 normalizer."""
    js = r"""
const fs = require('fs');
const wasm = require('./crates/wasm/pkg-node/nc2000_wasm.js');
const { normalizeCatalog } = require('./tools/normalize-training.js');
const dexJson = JSON.parse(fs.readFileSync('data/gen2stadium2.json', 'utf8'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

const dex = new wasm.Dex();
const validator = new wasm.Validator(dex);

const validEntries = [];
const rejected = [];

// Wrap each 1-mon set inside a dummy legal 6-mon shell for Validator::canonicalizeTeam + training-v3 normalization,
// or validate individual fields + canonicalizeTeam by padding with 5 harmless L50 mons!
const padSpecies = [
  ['Caterpie', 'Tackle'],
  ['Weedle', 'Poison Sting'],
  ['Pidgey', 'Tackle'],
  ['Rattata', 'Tackle'],
  ['Spearow', 'Peck'],
  ['Ekans', 'Wrap'],
];
function makePad(excludeSp) {
  const out = [];
  for (const [sp, mv] of padSpecies) {
    if (sp.toLowerCase() === excludeSp.toLowerCase()) continue;
    out.push({
      name: sp, species: sp, item: '', ability: 'No Ability',
      moves: [mv], nature: 'Serious', level: 50, happiness: 255
    });
    if (out.length === 5) break;
  }
  return out;
}

const catalogTeams = [];
const idxMap = [];

for (let i = 0; i < input.length; i++) {
  const e = input[i];
  const s = e.set;
  const team6 = [s, ...makePad(s.species)];
  const canon = JSON.parse(validator.canonicalizeTeam(JSON.stringify(team6)));
  if (!canon.ok) {
    rejected.push({ id: e.id, errors: canon.errors });
    continue;
  }
  const val = JSON.parse(validator.validateTeam(JSON.stringify(canon.team)));
  if (!val.ok) {
    rejected.push({ id: e.id, errors: val.errors });
    continue;
  }
  catalogTeams.push({ id: e.id, variantCluster: e.id, sets: canon.team });
  idxMap.push(i);
}

const norm = normalizeCatalog(catalogTeams, dexJson, false);
for (let k = 0; k < catalogTeams.length; k++) {
  const orig = input[idxMap[k]];
  const canonSet = norm.get(orig.id).sets[0];
  validEntries.push({ ...orig, set: canonSet });
}

validator.free();
dex.free();
console.log(JSON.stringify({ valid: validEntries, rejected }));
"""
    proc = subprocess.run(
        ["node", "-e", js],
        input=json.dumps(entries),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        check=True,
    )
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    if res["rejected"]:
        # Log rejected count to file inside target/gem-finder
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "rejected.json").write_text(json.dumps(res["rejected"], indent=2))
    return res["valid"]


def cmd_generate(args: argparse.Namespace) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    dex, ls = load_dex_and_learnsets()
    tp3 = json.loads((ROOT / "data/team-pool-v3/team-pool.json").read_text())["teams"]
    bp3 = json.loads((ROOT / "data/belief-pool-v3/belief-pool.json").read_text())["teams"]

    # Partition non-dominated species into Major (in tp3 or >=15 in bp3) vs Minor
    c_tp = Counter(s["species"] for t in tp3 for s in t["sets"])
    c_bp = Counter(s["species"] for t in bp3 for s in t["sets"])
    sp_dict = dex["species"]
    ls_sp = ls["species"]
    special_item_species = set(SIGNATURE_ITEMS.keys())

    legal = sorted(ls_sp.keys())
    dom_strict: dict[str, str] = {}
    for b in legal:
        if b in special_item_species:
            continue
        sb = sp_dict[b]
        tb = tuple(sorted(sb["types"]))
        mb = set(ls_sp[b]["moves"])
        minb = ls_sp[b].get("minLevel", 50)
        bsb = sb["baseStats"]
        for a in legal:
            if a == b:
                continue
            sa = sp_dict[a]
            if tuple(sorted(sa["types"])) != tb:
                continue
            if ls_sp[a].get("minLevel", 50) > minb:
                continue
            bsa = sa["baseStats"]
            if all(bsa[k] >= bsb[k] for k in bsb) and mb.issubset(set(ls_sp[a]["moves"])):
                if any(bsa[k] > bsb[k] for k in bsb) or len(ls_sp[a]["moves"]) > len(mb):
                    dom_strict[b] = a
                    break

    non_dom = [k for k in legal if k not in dom_strict]
    major_ids = {k for k in non_dom if c_tp[sp_dict[k]["name"]] >= 1 or c_bp[sp_dict[k]["name"]] >= 15}
    minor_ids = [
        k for k in non_dom
        if k not in major_ids
        and (sum(sp_dict[k]["baseStats"].values()) >= 360 or k in special_item_species or k in ("wobbuffet", "shuckle"))
    ]

    anchors = canonicalize_sets_via_wasm(build_big16_anchors())

    # Group belief-pool-v3 sets by minor species
    bp_by_sp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in bp3:
        for s in t["sets"]:
            bp_by_sp[toid(s["species"])].append(s)

    majors_raw = build_major_reference_pool(dex, ls, major_ids, tp3, bp3)
    minors_raw: list[dict[str, Any]] = []
    for sp_id in minor_ids:
        minors_raw.extend(generate_minor_sets_for_species(dex, sp_id, ls_sp[sp_id], anchors, bp_by_sp.get(sp_id, [])))

    all_cands = canonicalize_sets_via_wasm(majors_raw + minors_raw)
    n_maj = sum(1 for c in all_cands if c["is_major"])
    n_min = sum(1 for c in all_cands if not c["is_major"])
    tier_counts = Counter((c["is_major"], c["tier"]) for c in all_cands)

    doc = {
        "anchors": anchors,
        "candidates": all_cands,
        "meta": {
            "major_species": len(major_ids),
            "minor_species": len(minor_ids),
            "major_sets": n_maj,
            "minor_sets": n_min,
            "tier_counts": {f"{'maj' if k[0] else 'min'}:{k[1]}": v for k, v in sorted(tier_counts.items())},
        },
    }
    out_path = WORK / "input.json"
    out_path.write_text(json.dumps(doc))
    print(json.dumps({
        "stage": "generate",
        "anchors": len(anchors),
        "major_species": len(major_ids),
        "minor_species": len(minor_ids),
        "major_sets": n_maj,
        "minor_sets": n_min,
        "total_cells": len(all_cands) * len(anchors),
        "out": str(out_path),
    }))


def build_matrix(inp: dict[str, Any], res: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], dict[str, list[dict[str, Any]]]]:
    na = len(inp["anchors"])
    lead: dict[str, list[float]] = {}
    sw_ns: dict[str, list[float]] = {}
    sw_all: dict[str, list[float]] = {}
    meta: dict[str, list[dict[str, Any]]] = {}
    for cell in res["cells"]:
        cid = cell["cand_id"]
        ai = cell["anchor_idx"]
        if cid not in lead:
            lead[cid] = [0.0] * na
            sw_ns[cid] = [0.0] * na
            sw_all[cid] = [0.0] * na
            meta[cid] = [{} for _ in range(na)]
        lead[cid][ai] = cell["lead"]
        sw_ns[cid][ai] = cell["sw_no_sleep"]
        sw_all[cid][ai] = cell["sw_all"]
        meta[cid][ai] = cell
    return lead, sw_ns, sw_all, meta


def select_stage2_candidates(inp: dict[str, Any], s1: dict[str, Any], top_majors_per_tier: int = 999, top_minors_per_tier: int = 75) -> list[dict[str, Any]]:
    """Select all Majors + all Stage-1 Pareto-promising Minors for full played-out verification."""
    lead, sw_ns, _, _ = build_matrix(inp, s1)
    by_id = {c["id"]: c for c in inp["candidates"]}
    na = len(inp["anchors"])

    keep_ids: set[str] = set()
    for tier in ("L50_NO_LEFT", "L50_LEFT", "L55_ACE"):
        majs = [c["id"] for c in inp["candidates"] if c["is_major"] and c["tier"] == tier and c["id"] in lead]
        mins = [c["id"] for c in inp["candidates"] if not c["is_major"] and c["tier"] == tier and c["id"] in lead]

        # Keep ALL majors (there are only ~70-100 per tier, or keep top Pareto majors)
        maj_keep: set[str] = set()
        if len(majs) <= top_majors_per_tier:
            maj_keep.update(majs)
        else:
            # Keep every major that is top-3 on any anchor (lead or sw) or top-30 mean
            for ai in range(na):
                for mat in (lead, sw_ns):
                    for cid in sorted(majs, key=lambda x: -mat[x][ai])[:3]:
                        maj_keep.add(cid)
            for cid in sorted(majs, key=lambda x: -sum(lead[x]))[:25]:
                maj_keep.add(cid)
        keep_ids.update(maj_keep)

        # Score each minor by: (a) best margin vs major hull on any single anchor or pair (lead/sw_ns), (b) strong mean
        # Precompute major max per single anchor and pair
        maj_single_lead = [max((lead[m][i] for m in majs), default=0.0) for i in range(na)]
        maj_single_sw = [max((sw_ns[m][i] for m in majs), default=0.0) for i in range(na)]

        maj_pair_lead: dict[tuple[int, int], float] = {}
        maj_pair_sw: dict[tuple[int, int], float] = {}
        for i in range(na):
            for j in range(i + 1, na):
                maj_pair_lead[(i, j)] = max((min(lead[m][i], lead[m][j]) for m in majs), default=0.0)
                maj_pair_sw[(i, j)] = max((min(sw_ns[m][i], sw_ns[m][j]) for m in majs), default=0.0)

        scored_min: list[tuple[float, str]] = []
        for cid in mins:
            best_gap = -1.0
            for i in range(na):
                if lead[cid][i] >= 0.55:
                    best_gap = max(best_gap, lead[cid][i] - maj_single_lead[i])
                if sw_ns[cid][i] >= 0.50:
                    best_gap = max(best_gap, sw_ns[cid][i] - maj_single_sw[i])
            for (i, j), mb in maj_pair_lead.items():
                v = min(lead[cid][i], lead[cid][j])
                if v >= 0.55:
                    best_gap = max(best_gap, v - mb)
            for (i, j), mb in maj_pair_sw.items():
                v = min(sw_ns[cid][i], sw_ns[cid][j])
                if v >= 0.50:
                    best_gap = max(best_gap, v - mb)
            mean_bonus = 0.25 * (sum(lead[cid]) + sum(sw_ns[cid])) / (2 * na)
            scored_min.append((best_gap + mean_bonus, cid))

        scored_min.sort(key=lambda x: -x[0])
        # Ensure species diversity (at most 2 sets per minor species unless top-20)
        sp_cnt: Counter[str] = Counter()
        added = 0
        for rank, (_, cid) in enumerate(scored_min):
            sp = by_id[cid]["species"]
            if rank >= 20 and sp_cnt[sp] >= 2:
                continue
            keep_ids.add(cid)
            sp_cnt[sp] += 1
            added += 1
            if added >= top_minors_per_tier:
                break

    return [c for c in inp["candidates"] if c["id"] in keep_ids]


def cmd_run(args: argparse.Namespace) -> None:
    inp_path = WORK / "input.json"
    inp = json.loads(inp_path.read_text())

    # Stage 1: Fast root evaluation across ALL candidates × Big-16
    s1_out = WORK / "stage1-root.json"
    if not s1_out.exists() or args.force:
        cmd1 = [
            str(GEM_BIN),
            "--input", str(inp_path),
            "--out", str(s1_out),
            "--mode", "root",
            "--iters", str(args.root_iters),
            "--sw-iters", str(args.root_sw_iters),
            "--sw-samples", str(args.sw_samples),
            "--threads", str(args.threads),
        ]
        p1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
        s1_tel = json.loads(p1.stdout.strip().splitlines()[-1])
    else:
        s1_tel = {"cached": True}

    s1 = json.loads(s1_out.read_text())

    # Stage 2: Full played-out 1v1 + switch-in games on top Majors + Pareto-promising Minors
    s2_cands = select_stage2_candidates(
        inp, s1,
        top_majors_per_tier=args.stage2_majors_per_tier,
        top_minors_per_tier=args.stage2_minors_per_tier,
    )
    s2_in_path = WORK / "stage2-input.json"
    s2_in = {"anchors": inp["anchors"], "candidates": s2_cands}
    s2_in_path.write_text(json.dumps(s2_in))

    s2_out = WORK / "stage2-play.json"
    if not s2_out.exists() or args.force:
        cmd2 = [
            str(GEM_BIN),
            "--input", str(s2_in_path),
            "--out", str(s2_out),
            "--mode", "play",
            "--iters", str(args.play_iters),
            "--sw-iters", str(args.play_sw_iters),
            "--games", str(args.play_games),
            "--max-turns", str(args.max_turns),
            "--threads", str(args.threads),
        ]
        p2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
        s2_tel = json.loads(p2.stdout.strip().splitlines()[-1])
    else:
        s2_tel = {"cached": True}

    print(json.dumps({
        "stage": "run",
        "stage1": s1_tel,
        "stage2_cands": len(s2_cands),
        "stage2_majors": sum(1 for c in s2_cands if c["is_major"]),
        "stage2_minors": sum(1 for c in s2_cands if not c["is_major"]),
        "stage2": s2_tel,
    }))


def cmd_analyze(args: argparse.Namespace) -> None:
    s2_in = json.loads((WORK / "stage2-input.json").read_text())
    s2 = json.loads((WORK / "stage2-play.json").read_text())
    lead, sw_ns, sw_all, meta = build_matrix(s2_in, s2)
    anchors = s2_in["anchors"]
    a_ids = [a["id"] for a in anchors]
    na = len(anchors)
    by_id = {c["id"]: c for c in s2_in["candidates"]}

    # Also check Stage-1 vs Stage-2 correlation on overlapping cells
    s1 = json.loads((WORK / "stage1-root.json").read_text())
    s1_map = {(c["cand_id"], c["anchor_id"]): c for c in s1["cells"]}
    xs_l, ys_l, xs_s, ys_s = [], [], [], []
    for c in s2["cells"]:
        k = (c["cand_id"], c["anchor_id"])
        if k in s1_map:
            xs_l.append(s1_map[k]["lead"]); ys_l.append(c["lead"])
            xs_s.append(s1_map[k]["sw_no_sleep"]); ys_s.append(c["sw_no_sleep"])

    def corr(xs: list[float], ys: list[float]) -> float:
        n = len(xs)
        if n < 2: return 0.0
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        return num / max(1e-12, dx * dy)

    report: dict[str, Any] = {
        "stage1_vs_stage2_corr": {
            "lead_r": round(corr(xs_l, ys_l), 4),
            "sw_no_sleep_r": round(corr(xs_s, ys_s), 4),
            "cells": len(xs_l),
        },
        "anchors": a_ids,
        "tiers": {},
    }

    for tier in ("L50_NO_LEFT", "L50_LEFT", "L55_ACE"):
        majs = [c["id"] for c in s2_in["candidates"] if c["is_major"] and c["tier"] == tier and c["id"] in lead]
        mins = [c["id"] for c in s2_in["candidates"] if not c["is_major"] and c["tier"] == tier and c["id"] in lead]

        def analyze_metric(mat: dict[str, list[float]], min_floor: float, min_margin: float) -> dict[str, Any]:
            # Single-anchor breakouts
            single_hits = []
            for i in range(na):
                maj_best_id = max(majs, key=lambda m: mat[m][i])
                maj_best_v = mat[maj_best_id][i]
                for cid in mins:
                    v = mat[cid][i]
                    if v >= min_floor and v - maj_best_v >= min_margin:
                        single_hits.append({
                            "cand_id": cid,
                            "species": by_id[cid]["species"],
                            "set": by_id[cid]["set"],
                            "anchor": a_ids[i],
                            "val": round(v, 3),
                            "best_major": maj_best_id,
                            "best_major_species": by_id[maj_best_id]["species"],
                            "best_major_val": round(maj_best_v, 3),
                            "margin": round(v - maj_best_v, 3),
                        })
            single_hits.sort(key=lambda x: (-x["margin"], -x["val"]))

            # Pair role-compression breakouts: min(W[c,i], W[c,j]) > max_m min(W[m,i], W[m,j])
            pair_by_minor: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for i in range(na):
                for j in range(i + 1, na):
                    maj_best_id = max(majs, key=lambda m: min(mat[m][i], mat[m][j]))
                    maj_best_v = min(mat[maj_best_id][i], mat[maj_best_id][j])
                    for cid in mins:
                        v = min(mat[cid][i], mat[cid][j])
                        if v >= min_floor and v - maj_best_v >= min_margin:
                            pair_by_minor[cid].append({
                                "pair": [a_ids[i], a_ids[j]],
                                "min_val": round(v, 3),
                                "vals": [round(mat[cid][i], 3), round(mat[cid][j], 3)],
                                "best_major": by_id[maj_best_id]["species"],
                                "best_major_id": maj_best_id,
                                "best_major_val": round(maj_best_v, 3),
                                "margin": round(v - maj_best_v, 3),
                            })

            # Trio role-compression breakouts (focus on L55 aces indices 0..10 + Champ50/Venu50/Eggy50/Skarm50/Star50)
            trio_by_minor: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for i, j, k in itertools.combinations(range(na), 3):
                maj_best_id = max(majs, key=lambda m: min(mat[m][i], mat[m][j], mat[m][k]))
                maj_best_v = min(mat[maj_best_id][i], mat[maj_best_id][j], mat[maj_best_id][k])
                for cid in mins:
                    v = min(mat[cid][i], mat[cid][j], mat[cid][k])
                    if v >= min_floor and v - maj_best_v >= min_margin:
                        trio_by_minor[cid].append({
                            "trio": [a_ids[i], a_ids[j], a_ids[k]],
                            "min_val": round(v, 3),
                            "best_major": by_id[maj_best_id]["species"],
                            "best_major_val": round(maj_best_v, 3),
                            "margin": round(v - maj_best_v, 3),
                        })

            # Rank minors by strongest breakout + breadth of broken pairs/trios
            gem_summary = []
            all_gem_cids = set(pair_by_minor.keys()) | set(trio_by_minor.keys())
            for cid in all_gem_cids:
                ps = sorted(pair_by_minor.get(cid, []), key=lambda x: (-x["margin"], -x["min_val"]))
                ts = sorted(trio_by_minor.get(cid, []), key=lambda x: (-x["margin"], -x["min_val"]))
                best_p_margin = ps[0]["margin"] if ps else 0.0
                best_t_margin = ts[0]["margin"] if ts else 0.0
                wins_ge60 = [a_ids[i] for i in range(na) if mat[cid][i] >= 0.60]
                gem_summary.append({
                    "cand_id": cid,
                    "species": by_id[cid]["species"],
                    "level": by_id[cid]["set"]["level"],
                    "item": by_id[cid]["set"]["item"],
                    "moves": by_id[cid]["set"]["moves"],
                    "mean": round(sum(mat[cid]) / na, 3),
                    "n_ge60": len(wins_ge60),
                    "wins_ge60": wins_ge60,
                    "n_pair_breakouts": len(ps),
                    "best_pair_margin": best_p_margin,
                    "top_pairs": ps[:4],
                    "n_trio_breakouts": len(ts),
                    "best_trio_margin": best_t_margin,
                    "top_trios": ts[:3],
                    "cap_rate_avg": round(sum(meta[cid][i]["cap_rate"] for i in range(na)) / na, 3),
                })
            gem_summary.sort(key=lambda x: (-(x["best_trio_margin"] * 1.2 + x["best_pair_margin"]), -x["n_trio_breakouts"], -x["n_pair_breakouts"]))
            return {
                "single_breakouts": single_hits[:20],
                "gem_breakouts": gem_summary[:25],
            }

        # Top overall minor performers by mean & count>=60% (even if inside hull)
        top_by_sw = sorted(mins, key=lambda c: (-sum(1 for v in sw_ns[c] if v >= 0.60), -sum(sw_ns[c])))[:15]
        top_by_lead = sorted(mins, key=lambda c: (-sum(1 for v in lead[c] if v >= 0.60), -sum(lead[c])))[:15]

        report["tiers"][tier] = {
            "n_majors": len(majs),
            "n_minors": len(mins),
            "switchin_breakouts": analyze_metric(sw_ns, min_floor=0.58, min_margin=0.05),
            "lead_breakouts": analyze_metric(lead, min_floor=0.62, min_margin=0.05),
            "top_minors_by_sw_coverage": [
                {
                    "cand_id": c,
                    "species": by_id[c]["species"],
                    "item": by_id[c]["set"]["item"],
                    "moves": by_id[c]["set"]["moves"],
                    "sw_mean": round(sum(sw_ns[c]) / na, 3),
                    "lead_mean": round(sum(lead[c]) / na, 3),
                    "sw_ge60": [a_ids[i] for i in range(na) if sw_ns[c][i] >= 0.60],
                }
                for c in top_by_sw
            ],
            "top_minors_by_lead_coverage": [
                {
                    "cand_id": c,
                    "species": by_id[c]["species"],
                    "item": by_id[c]["set"]["item"],
                    "moves": by_id[c]["set"]["moves"],
                    "lead_mean": round(sum(lead[c]) / na, 3),
                    "sw_mean": round(sum(sw_ns[c]) / na, 3),
                    "lead_ge60": [a_ids[i] for i in range(na) if lead[c][i] >= 0.60],
                }
                for c in top_by_lead
            ],
        }

    # Drop-in Hole-Filler query:
    # Target breakers of s0053-lefties:
    #   pb-84 core: [Lax55_Talk, Raik55_Mag, Champ50_Talk]
    #   offantom core: [P2_55_DT, Lax50_Curse, Lax55_Left]
    #   thdrcomb core: [Ttar55_Mix, Zap55_HPI, Venu50_Seed]
    target_sets = {
        "anti_pb84_core": ["Raik55_Mag", "Champ50_Talk", "Lax55_Talk"],
        "anti_offantom_core": ["P2_55_DT", "P2_55_Ice", "Lax55_Left"],
        "anti_pb84_plus_offantom": ["Raik55_Mag", "P2_55_DT", "Champ50_Talk"],
        "anti_electric_duo": ["Zap55_HPI", "Zap55_DP", "Raik55_Mag"],
        "anti_physical_breakers": ["Wak55_HPB", "Ttar55_Mix", "Lax55_Left"],
        "anti_s0053_aces": ["Zap55_HPI", "Wak55_HPB", "Cune55_Talk"],
    }
    a_idx = {aid: i for i, aid in enumerate(a_ids)}
    report["dropin_queries"] = {}
    for qname, targets in target_sets.items():
        idxs = [a_idx[t] for t in targets]
        tier_res = {}
        for tier in ("L50_NO_LEFT", "L50_LEFT", "L55_ACE"):
            cands = [c["id"] for c in s2_in["candidates"] if c["tier"] == tier and c["id"] in sw_ns]
            scored = []
            for cid in cands:
                sw_vals = [sw_ns[cid][i] for i in idxs]
                ld_vals = [lead[cid][i] for i in idxs]
                # Combined score: prioritize switch-in worst-case + mean, tie-break with lead
                sc = 0.55 * min(sw_vals) + 0.25 * (sum(sw_vals) / len(idxs)) + 0.20 * (sum(ld_vals) / len(idxs))
                scored.append((sc, cid, sw_vals, ld_vals))
            scored.sort(key=lambda x: -x[0])
            best_maj = next((x for x in scored if by_id[x[1]]["is_major"]), None)
            top_mins = [x for x in scored if not by_id[x[1]]["is_major"]][:8]
            tier_res[tier] = {
                "best_major": {
                    "species": by_id[best_maj[1]]["species"],
                    "item": by_id[best_maj[1]]["set"]["item"],
                    "moves": by_id[best_maj[1]]["set"]["moves"],
                    "score": round(best_maj[0], 3),
                    "sw": [round(v, 3) for v in best_maj[2]],
                    "lead": [round(v, 3) for v in best_maj[3]],
                } if best_maj else None,
                "top_minors": [
                    {
                        "cand_id": cid,
                        "species": by_id[cid]["species"],
                        "item": by_id[cid]["set"]["item"],
                        "moves": by_id[cid]["set"]["moves"],
                        "score": round(sc, 3),
                        "sw": [round(v, 3) for v in swv],
                        "lead": [round(v, 3) for v in ldv],
                    }
                    for sc, cid, swv, ldv in top_mins
                ],
            }
        report["dropin_queries"][qname] = {"targets": targets, "by_tier": tier_res}

    out_path = WORK / "gem-report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "stage": "analyze",
        "lead_r": report["stage1_vs_stage2_corr"]["lead_r"],
        "sw_r": report["stage1_vs_stage2_corr"]["sw_no_sleep_r"],
        "out": str(out_path),
    }))


def canonicalize_full_teams_via_wasm(teams_map: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Validate & training-v3 normalize full 6-mon teams via real WASM Validator."""
    js = r"""
const fs = require('fs');
const wasm = require('./crates/wasm/pkg-node/nc2000_wasm.js');
const { normalizeCatalog } = require('./tools/normalize-training.js');
const dexJson = JSON.parse(fs.readFileSync('data/gen2stadium2.json', 'utf8'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

const dex = new wasm.Dex();
const validator = new wasm.Validator(dex);
try {
  const entries = [];
  for (const [id, sets] of Object.entries(input)) {
    const canon = JSON.parse(validator.canonicalizeTeam(JSON.stringify(sets)));
    if (!canon.ok) throw new Error(`canonicalize failed ${id}: ${JSON.stringify(canon.errors)}`);
    entries.push({ id, variantCluster: id, sets: canon.team });
  }
  const opt = normalizeCatalog(entries, dexJson, false);
  const out = {};
  for (const e of entries) {
    const sets = opt.get(e.id).sets;
    const vr = JSON.parse(validator.validateTeam(JSON.stringify(sets)));
    if (!vr.ok) throw new Error(`validateTeam failed ${e.id}: ${JSON.stringify(vr.errors)}`);
    out[e.id] = sets;
  }
  console.log(JSON.stringify({ ok: true, teams: out }));
} finally {
  validator.free();
  dex.free();
}
"""
    proc = subprocess.run(
        ["node", "-e", js],
        input=json.dumps(teams_map),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Full team validation failed:\n{proc.stderr or proc.stdout}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    return res["teams"]


def cmd_preview_gate(args: argparse.Namespace) -> None:
    """Step 3 Exit Gate: Splice top minor gems into flagship/pillar shells and measure real Team Preview pick rate."""
    import copy, hashlib, os, shutil

    def u32(tag: str) -> int:
        return int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=4).digest(), "little")

    tp3 = json.loads((ROOT / "data/team-pool-v3/team-pool.json").read_text())["teams"]
    by_id = {t["id"]: t for t in tp3}

    pb84_base = copy.deepcopy(by_id["pb-84"]["sets"])
    s0053_base = copy.deepcopy(by_id["s0053-lefties-hpi"]["sets"])

    def mk(sp: str, lv: int, item: str, moves: list[str]) -> dict[str, Any]:
        return {
            "name": sp, "species": sp, "item": item, "ability": "No Ability",
            "moves": moves, "nature": "Serious", "level": lv, "happiness": 255,
        }

    def splice(base: list[dict[str, Any]], slot_idx: int, mon: dict[str, Any], extra_item_fix: tuple[int, str] | None = None) -> list[dict[str, Any]]:
        t = copy.deepcopy(base)
        if extra_item_fix is not None:
            t[extra_item_fix[0]]["item"] = extra_item_fix[1]
        t[slot_idx] = mon
        return t

    raw_cands: dict[str, list[dict[str, Any]]] = {
        # pb-84 Slot 6 (Umbreon50@Leftovers -> minor L50@Leftovers)
        "gate:pb84_base": pb84_base,
        "gate:pb84_shuckle50": splice(pb84_base, 5, mk("Shuckle", 50, "Leftovers", ["Toxic", "Encore", "Rest", "Sleep Talk"])),
        "gate:pb84_piloswine50": splice(pb84_base, 5, mk("Piloswine", 50, "Leftovers", ["Earthquake", "Ice Beam", "Rest", "Sleep Talk"])),
        "gate:pb84_nidoqueen50": splice(pb84_base, 5, mk("Nidoqueen", 50, "Leftovers", ["Earthquake", "Curse", "Rest", "Sleep Talk"])),
        "gate:pb84_slowking50": splice(pb84_base, 5, mk("Slowking", 50, "Leftovers", ["Surf", "Fire Blast", "Rest", "Sleep Talk"])),
        "gate:pb84_ampharos50": splice(pb84_base, 5, mk("Ampharos", 50, "Leftovers", ["Thunder", "Hidden Power Ice", "Rest", "Sleep Talk"])),
        "gate:pb84_lanturn50": splice(pb84_base, 5, mk("Lanturn", 50, "Leftovers", ["Thunder", "Hydro Pump", "Rest", "Sleep Talk"])),
        # s0053-lefties-hpi Drop-Ins (Slot 1 Electrode50 or Slot 2 Suicune55)
        "gate:s0053_base": s0053_base,
        "gate:s0053_pilo50_s1": splice(s0053_base, 0, mk("Piloswine", 50, "Miracle Berry", ["Earthquake", "Ice Beam", "Rest", "Sleep Talk"])),
        "gate:s0053_queen50_s1": splice(s0053_base, 0, mk("Nidoqueen", 50, "Miracle Berry", ["Earthquake", "Curse", "Rest", "Sleep Talk"])),
        "gate:s0053_shuck50_s1": splice(s0053_base, 0, mk("Shuckle", 50, "Miracle Berry", ["Toxic", "Encore", "Rest", "Sleep Talk"])),
        "gate:s0053_mag55_s2": splice(s0053_base, 1, mk("Magneton", 55, "Mint Berry", ["Thunder", "Hidden Power Ice", "Rest", "Sleep Talk"])),
        "gate:s0053_muk55_s2": splice(s0053_base, 1, mk("Muk", 55, "Mint Berry", ["Sludge Bomb", "Curse", "Rest", "Sleep Talk"])),
        "gate:s0053_espeon55_s2": splice(s0053_base, 1, mk("Espeon", 55, "Miracle Berry", ["Psychic", "Zap Cannon", "Charm", "Morning Sun"]), extra_item_fix=(0, "PRZ Cure Berry")),
        "gate:s0053_vap55_s2": splice(s0053_base, 1, mk("Vaporeon", 55, "Mint Berry", ["Hydro Pump", "Curse", "Rest", "Sleep Talk"])),
    }
    validated = canonicalize_full_teams_via_wasm(raw_cands)

    panel_ids = [
        "s0053-lefties-hpi",
        "pb-84",
        "mjj-2014-offantom",
        "mjj-2014-thdrcomb",
        "pb-6",
        "pb-122",
        "pb-131",
        "rental-cban-8",
    ]
    pools_teams = [{"id": f"v3:{t['id']}", "sets": t["sets"]} for t in tp3]
    mixtures: dict[str, Any] = {}
    for t in tp3:
        tid = t["id"]
        if tid in ("s0053-lefties-hpi", "s0053-lefties-dp"):
            mixtures["shell:s0053-lefties"] = [
                {"id": "v3:s0053-lefties-hpi", "weight": 0.5},
                {"id": "v3:s0053-lefties-dp", "weight": 0.5},
            ]
        else:
            mixtures[f"shell:{tid}"] = [{"id": f"v3:{tid}", "weight": 1.0}]

    for cid, sets in validated.items():
        pools_teams.append({"id": cid, "sets": sets})
        mixtures[f"mix:{cid}"] = [{"id": cid, "weight": 1.0}]

    gate_dir = (Path(args.work_dir) if getattr(args, "work_dir", None) else WORK) / "preview-gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    pools_path = gate_dir / "pools.json"
    sched_path = gate_dir / "schedule.jsonl"
    out_dir = gate_dir / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pools_path.write_text(json.dumps({"teams": pools_teams, "mixtures": mixtures}, separators=(",", ":")))

    rows = []
    idx = 0
    for cid in validated:
        for pid in panel_ids:
            opp_mix = "shell:s0053-lefties" if pid == "s0053-lefties-hpi" else f"shell:{pid}"
            rows.append({
                "block": idx,
                "cell": {"kind": "gem_preview_gate", "cand": cid, "opp": pid},
                "rowSeat": 0,
                "ids": [cid, f"v3:{pid}"],
                "seeds": [u32(f"gg-s0:{cid}:{pid}"), u32(f"gg-s1:{cid}:{pid}")],
                "battleSeed": u32(f"gg-b:{cid}:{pid}"),
                "mixtures": [f"mix:{cid}", opp_mix],
            })
            idx += 1
    sched_path.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")

    nash_bin = ROOT / "target/native-pgo/release/examples/nash_refresh"
    env = dict(os.environ)
    env["NASH_POOLS"] = str(pools_path)
    env["NASH_PREVIEW_ITERS"] = str(args.preview_iters)
    env["NASH_PREVIEW_DIST_K"] = str(args.k)
    env["NASH_PREVIEW_DIST_ONLY"] = "1"
    env["NASH_PREVIEW_CACHE"] = "0"

    t0 = time.monotonic()
    subprocess.run(
        [str(nash_bin), str(sched_path), str(out_dir), str(args.threads)],
        env=env,
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0

    dist_doc = json.loads((out_dir / "preview-dist.json").read_text())
    entries = dist_doc.get("entries", {})
    by_own_opp: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for ent in entries.values():
        own = ent.get("own")
        opp = ent.get("opp_mix")
        if own in validated and opp:
            for act, cnt in ent.get("counts", {}).items():
                by_own_opp[own][opp][act] += int(cnt)

    summary = []
    for cid, sets in validated.items():
        slot_idx = 5 if cid.startswith("gate:pb84_") else (0 if "_s1" in cid else 1)
        if cid == "gate:s0053_base":
            slot_idx = 1
        opps = by_own_opp.get(cid, {})
        n_opp = max(1, len(opps))
        slot_rates = [0.0] * 6
        per_opp_target: dict[str, float] = {}
        for opp_mix, cnts in sorted(opps.items()):
            tot = sum(cnts.values())
            if tot <= 0:
                continue
            opp_s = [0.0] * 6
            for act, c in cnts.items():
                w = c / tot
                for snum in [int(x.strip()) for x in act.replace("team", "").strip().split(",")]:
                    if 1 <= snum <= 6:
                        opp_s[snum - 1] += w
            for i in range(6):
                slot_rates[i] += opp_s[i] / n_opp
            per_opp_target[opp_mix.replace("shell:", "")] = round(opp_s[slot_idx] * 100.0, 1)
        tpk = round(slot_rates[slot_idx] * 100.0, 1)
        # Empirically calibrated policy-v2 (SteppedCfPreview 240k + CfValuePuct) detection zones:
        # <10% = dead/ignored, 10-42% = sweet spot, >42% = OOD CF-V_net over-pick risk (0/12 passed naive InWR-OutWR>=+5pp screen; mean CRN InΔ = -13.7pp).
        v2_zone = "DEAD(<10%)" if tpk < 10.0 else ("SWEET[10-42%]" if tpk <= 42.0 else "OVERPICK_RISK(>42%)")
        summary.append({
            "id": cid,
            "target_slot": slot_idx + 1,
            "target_species": sets[slot_idx]["species"],
            "target_pick_pct": tpk,
            "policy_v2_zone": v2_zone,
            "min_slot_pct": round(min(slot_rates) * 100.0, 1),
            "all_slot_pcts": [round(r * 100.0, 1) for r in slot_rates],
            "per_opp_target_pct": per_opp_target,
        })

    manifest_path = out_dir / "manifest.json"
    policy_epoch = (
        json.loads(manifest_path.read_text()).get("policyEpoch", "unknown")
        if manifest_path.exists()
        else "unknown"
    )
    rep_path = gate_dir / "gate-report.json"
    rep_path.write_text(json.dumps({
        "policy_epoch": policy_epoch,
        "preview_iters": args.preview_iters,
        "k": args.k,
        "v2_thresholds": {
            "sweet_pick_pct": [10.0, 42.0],
            "overpick_risk_gt_pct": 42.0,
            "live_crn_in_delta_min_pp": 0.0,
            "live_in_minus_out_wr_screen_pp": 5.0,
        },
        "elapsed_s": round(elapsed, 1),
        "results": summary,
    }, indent=2))
    print(json.dumps({"stage": "preview-gate", "policy_epoch": policy_epoch, "cands": len(summary), "preview_iters": args.preview_iters, "k": args.k, "elapsed_s": round(elapsed, 1), "out": str(rep_path)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("generate")

    pr = sub.add_parser("run")
    pr.add_argument("--root-iters", type=int, default=400)
    pr.add_argument("--root-sw-iters", type=int, default=140)
    pr.add_argument("--sw-samples", type=int, default=5)
    pr.add_argument("--play-iters", type=int, default=120)
    pr.add_argument("--play-sw-iters", type=int, default=100)
    pr.add_argument("--play-games", type=int, default=20)
    pr.add_argument("--max-turns", type=int, default=150)
    pr.add_argument("--stage2-majors-per-tier", type=int, default=120)
    pr.add_argument("--stage2-minors-per-tier", type=int, default=60)
    pr.add_argument("--threads", type=int, default=10)
    pr.add_argument("--force", action="store_true")

    sub.add_parser("analyze")

    live_preview_default = int(json.loads((ROOT / "data/live-search.json").read_text()).get("unknownPreview", 240000))
    pg = sub.add_parser("preview-gate")
    pg.add_argument("--preview-iters", type=int, default=live_preview_default)
    pg.add_argument("--k", type=int, default=4)
    pg.add_argument("--threads", type=int, default=10)
    pg.add_argument("--work-dir", default=None, help="Override work dir (default: target/gem-finder)")

    args = ap.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "preview-gate":
        cmd_preview_gate(args)


if __name__ == "__main__":
    main()
