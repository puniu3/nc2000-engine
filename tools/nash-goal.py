#!/usr/bin/env python3
"""Autonomous multi-phase Nash equilibrium verification & certification driver for NC2000.

Supports:
  - Exact pair-level pooled production aggregation across target/nash-ledger.jsonl and prod-*/results.jsonl
  - Core Nash solve, bootstrap, +-1sigma sensitivity, and per-variant internal diagnostics
  - General matchup schedule builder (with preview cache buckets OR strictly independent no-cache holdout)
  - Generalized external challenger screen (9k/81k & 243k/729k) against multi-shell cores
  - Strictly independent holdout certification (previewSeeds omitted, one-sided 95% UCB <= 55%)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK = ROOT / "target/nash-work/elim-v1"
DEFAULT_BIN = ROOT / "target/native-pgo/release/examples/nash_refresh"


def _load_mod(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ELIM = _load_mod("nash_elim", ROOT / "tools/nash-elim.py")
STATS = ELIM.STATS
LEDGER = ELIM.LEDGER


def u32_hash(tag: str) -> int:
    return int.from_bytes(hashlib.blake2b(tag.encode("utf-8"), digest_size=4).digest(), "little")


def seat_score(outcome: Any, seat: int) -> float:
    s = str(outcome).lower()
    if s in ("p1", "0", "seat0", "win", "p0"):
        return 1.0 if seat == 0 else 0.0
    if s in ("p2", "1", "seat1", "loss"):
        return 0.0 if seat == 0 else 1.0
    return 0.5


def p0_score(outcome: Any) -> float:
    return seat_score(outcome, 0)


def pair_stats(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.5, 0.5
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.5
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0.0:
        se = 0.5 / (n + 1)
    return mu, se


def validate_teams_wasm(teams_map: dict[str, list[dict[str, Any]]]) -> None:
    if not teams_map:
        return
    js = (
        "const wasm = require('./crates/wasm/pkg-node/nc2000_wasm.js');"
        "const fs = require('fs');"
        "const data = JSON.parse(fs.readFileSync(0, 'utf8'));"
        "const dex = new wasm.Dex();"
        "const v = new wasm.Validator(dex);"
        "const errs = {};"
        "for (const [id, sets] of Object.entries(data)) {"
        "  const r = JSON.parse(v.validateTeam(JSON.stringify(sets)));"
        "  if (!r.ok) errs[id] = r.errors;"
        "}"
        "console.log(JSON.stringify({ ok: Object.keys(errs).length === 0, errs }));"
    )
    proc = subprocess.run(
        ["node", "-e", js],
        input=json.dumps(teams_map),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        check=True,
    )
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["ok"], f"WASM Validator rejected teams: {verdict['errs']}"


def build_all_sets_and_profiles() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    _, raw_sets = ELIM.load_full_catalog()
    p0_doc = json.loads((ROOT / "target/nash-work/psro-v1/pools-p0.json").read_text())
    for t in p0_doc.get("teams", []):
        raw_sets[t["id"]] = t["sets"]
    base_teams, _ = LEDGER.load_catalog_and_mixtures(None)

    all_sets = dict(raw_sets)
    bare3_teams, _ = ELIM.build_bare3_camo_variants(raw_sets)
    all_sets.update(bare3_teams)
    s08_teams = ELIM.build_s08_upgrade_candidates(raw_sets)
    for k, v in s08_teams.items():
        if k not in all_sets:
            all_sets[k] = v

    custom_path = DEFAULT_WORK / "custom-teams.json"
    if custom_path.exists():
        cdoc = json.loads(custom_path.read_text())
        for k, v in cdoc.get("teams", {}).items():
            all_sets[k] = v

    profiles: dict[str, list[dict[str, Any]]] = {
        "S07_0_100": [{"id": "sample-07", "weight": 1.00}],
        "S07_30_70": [
            {"id": "sample-07-beq-g2-c4", "weight": 0.30},
            {"id": "sample-07", "weight": 0.70},
        ],
        "S07_50_50": [
            {"id": "sample-07-beq-g2-c4", "weight": 0.50},
            {"id": "sample-07", "weight": 0.50},
        ],
        "S07_70_30": [
            {"id": "sample-07-beq-g2-c4", "weight": 0.70},
            {"id": "sample-07", "weight": 0.30},
        ],
        "APEX4": copy.deepcopy(ELIM.STEP_A_PROFILES["APEX4"]),
        "APEX2": [
            {"id": "apex-c04-bug-3atkstar-boomgen", "weight": 0.50},
            {"id": "apex-c08-surgical-142", "weight": 0.50},
        ],
        "APEX3": [
            {"id": "apex-c04-bug-3atkstar-boomgen", "weight": 0.40},
            {"id": "apex-c08-surgical-142", "weight": 0.40},
            {"id": "red-v10-s1105-r2-s0142", "weight": 0.20},
        ],
        "D5": copy.deepcopy(ELIM.STEP_A_PROFILES["D5"]),
        "D2": [
            {"id": "deme-r1-00-s0030", "weight": 0.50},
            {"id": "deme-r1-00-s0030-beq-g2-c3-beq-g3-c5-legal", "weight": 0.50},
        ],
        "D3": [
            {"id": "deme-r1-00-s0030", "weight": 0.40},
            {"id": "deme-r1-00-s0030-beq-g2-c3-legal", "weight": 0.30},
            {"id": "deme-r1-00-s0030-beq-g2-c3-beq-g3-c5-legal", "weight": 0.30},
        ],
        "PB132_50_50": copy.deepcopy(ELIM.STEP_A_PROFILES["PB132_50_50"]),
        "thdr:zap-talk": [
            {"id": "mjj-2014-thdrcomb__56b67d", "weight": 0.50},
            {"id": "thdr-v-zap-talk", "weight": 0.50},
        ],
        "thdr:zap-talk-30-70": [
            {"id": "mjj-2014-thdrcomb__56b67d", "weight": 0.30},
            {"id": "thdr-v-zap-talk", "weight": 0.70},
        ],
        "thdr:zap-talk-20-80": [
            {"id": "mjj-2014-thdrcomb__56b67d", "weight": 0.20},
            {"id": "thdr-v-zap-talk", "weight": 0.80},
        ],
        "thdr:zap-talk-40-60": [
            {"id": "mjj-2014-thdrcomb__56b67d", "weight": 0.40},
            {"id": "thdr-v-zap-talk", "weight": 0.60},
        ],
        "thdr:lax-curse": [
            {"id": "mjj-2014-thdrcomb__56b67d", "weight": 0.50},
            {"id": "thdr-v-lax-curse", "weight": 0.50},
        ],
        "thdr:tri-20-50-30": [
            {"id": "mjj-2014-thdrcomb__56b67d", "weight": 0.20},
            {"id": "thdr-v-zap-talk", "weight": 0.50},
            {"id": "thdr-v-lax-curse", "weight": 0.30},
        ],
        "thdr:tri-25-50-25": [
            {"id": "mjj-2014-thdrcomb__56b67d", "weight": 0.25},
            {"id": "thdr-v-zap-talk", "weight": 0.50},
            {"id": "thdr-v-lax-curse", "weight": 0.25},
        ],
    }
    if custom_path.exists():
        cdoc = json.loads(custom_path.read_text())
        for pname, mems in cdoc.get("profiles", {}).items():
            profiles[pname] = mems

    for tid in all_sets:
        if tid not in profiles:
            profiles[tid] = [{"id": tid, "weight": 1.0}]
    profiles["sample-07"] = profiles["S07_0_100"]

    return all_sets, profiles, base_teams


EXPLICIT_ALIAS: dict[str, str] = {
    "s:sample-07": "S07_0_100",
    "sample-07:1.0": "S07_0_100",
    "sample-07": "S07_0_100",
    "S07_0_100": "S07_0_100",
    "S07_30_70": "S07_30_70",
    "S07_50_50": "S07_50_50",
    "S07_70_30": "S07_70_30",
    "sample-07-beq-g2-c4:0.7+sample-07:0.3": "S07_70_30",
    "APEX4": "APEX4",
    "apex-c04-bug-3atkstar-boomgen:0.4+red-v10-s1105-r2-s0142:0.3+apex-c08-surgical-142:0.15+red-v10-s1105-r1-s0148:0.15": "APEX4",
    "D5": "D5",
    "deme-r1-00-s0030:0.2+deme-r1-00-s0030-beq-g2-c3-legal:0.2+deme-r1-00-s0030-beq-g2-c3-beq-g3-c0-legal:0.2+deme-r1-00-s0030-beq-g2-c3-beq-g3-c4-legal:0.2+deme-r1-00-s0030-beq-g2-c3-beq-g3-c5-legal:0.2": "D5",
    "PB132_50_50": "PB132_50_50",
    "pb-132__74c6b5:0.5+pb-131__65e4fe:0.5": "PB132_50_50",
    "s0155": "red-v10-s1105-r2-s0155",
}


def canon_label(lbl: str) -> str:
    if lbl in EXPLICIT_ALIAS:
        return EXPLICIT_ALIAS[lbl]
    if lbl.startswith("s:"):
        base = lbl[2:]
        return EXPLICIT_ALIAS.get(base, base)
    if lbl.endswith(":1.0") and "+" not in lbl:
        base = lbl[:-4]
        return EXPLICIT_ALIAS.get(base, base)
    return lbl


def load_all_prod_pairs(
    work_dir: Path,
    exclude_prefixes: tuple[str, ...] = ("prod-holdout",),
) -> dict[str, Any]:
    """Load and deduplicate all production 243k/729k games into exact pair-level buckets."""
    sources: list[tuple[Path, bool]] = []
    ledger_p = ROOT / "target/nash-ledger.jsonl"
    if ledger_p.exists():
        sources.append((ledger_p, True))
    for d in sorted(work_dir.glob("prod-*")):
        if not d.is_dir():
            continue
        if any(d.name.startswith(pref) for pref in exclude_prefixes):
            continue
        rp = d / "results.jsonl"
        if rp.exists():
            sources.append((rp, False))

    pair_buckets: dict[tuple[str, str, str, str, str], list[tuple[float, int]]] = defaultdict(list)
    seen_games: set[tuple[Any, ...]] = set()
    prev_seeds_seen: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    seat_stats: dict[tuple[str, str], list[float]] = defaultdict(list)
    slot_picks: dict[str, Counter[int]] = defaultdict(Counter)
    slot_games: Counter[str] = Counter()
    total_games = 0

    for src_path, is_ledger in sources:
        src_tag = "ledger" if is_ledger else src_path.parent.name
        for line in src_path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if is_ledger:
                lbls = r.get("profile_labels") or ["", ""]
                pa, pb = canon_label(lbls[0]), canon_label(lbls[1])
            else:
                mix = r.get("mixtures")
                if mix:
                    pa, pb = canon_label(mix[0]), canon_label(mix[1])
                else:
                    pa, pb = canon_label(f"s:{r['ids'][0]}"), canon_label(f"s:{r['ids'][1]}")
            if not pa or not pb or pa == pb:
                continue
            ia, ib = r["ids"][0], r["ids"][1]
            bs = r.get("battleSeed")
            sds = tuple(r.get("seeds", []))
            dedup_key = (pa, pb, ia, ib, bs, sds)
            if dedup_key in seen_games:
                continue
            seen_games.add(dedup_key)
            total_games += 1
            sc0 = p0_score(r["outcome"])

            pr = r.get("preview")
            if isinstance(pr, list) and len(pr) == 2:
                for side, prof in ((0, pa), (1, pb)):
                    if isinstance(pr[side], dict):
                        inp = pr[side].get("input", "")
                        if inp.startswith("team "):
                            slot_games[prof] += 1
                            for s in inp[5:].split(","):
                                slot_picks[prof][int(s.strip())] += 1

            s_sorted = tuple(sorted(sds))
            pkey = f"{src_tag}/{bs}/{s_sorted}"
            psds = r.get("previewSeeds")
            if pa < pb:
                lo, hi, ilo, ihi, scl, seat_lo = pa, pb, ia, ib, sc0, 0
                if psds and len(psds) == 2:
                    prev_seeds_seen[(lo, hi)].add((int(psds[0]), int(psds[1])))
                else:
                    prev_seeds_seen[(lo, hi)].add((int(sds[0]), int(sds[1])))
            else:
                lo, hi, ilo, ihi, scl, seat_lo = pb, pa, ib, ia, 1.0 - sc0, 1
                if psds and len(psds) == 2:
                    prev_seeds_seen[(lo, hi)].add((int(psds[1]), int(psds[0])))
                else:
                    prev_seeds_seen[(lo, hi)].add((int(sds[1]), int(sds[0])))

            pair_buckets[(lo, hi, ilo, ihi, pkey)].append((scl, seat_lo))
            st = seat_stats[(lo, hi)]
            if not st:
                st.extend([0.0, 0.0, 0.0, 0.0])
            if seat_lo == 0:
                st[0] += scl
                st[1] += 1.0
            else:
                st[2] += scl
                st[3] += 1.0

    sub_pairs: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    sub_singles: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (lo, hi, ilo, ihi, _pkey), entries in pair_buckets.items():
        if len(entries) == 2:
            sub_pairs[(lo, hi, ilo, ihi)].append(0.5 * (entries[0][0] + entries[1][0]))
        else:
            for sc, _ in entries:
                sub_singles[(lo, hi, ilo, ihi)].append(sc)

    return {
        "sources": [str(p) for p, _ in sources],
        "total_games": total_games,
        "sub_pairs": sub_pairs,
        "sub_singles": sub_singles,
        "prev_seeds_seen": prev_seeds_seen,
        "seat_stats": seat_stats,
        "slot_picks": slot_picks,
        "slot_games": slot_games,
    }


def eval_edge(
    data: dict[str, Any],
    profiles: dict[str, list[dict[str, Any]]],
    pa: str,
    pb: str,
) -> dict[str, Any]:
    if pa == pb:
        return {"mean": 0.5, "se": 0.0, "games": 0, "ok": True, "by_a": {}, "by_b": {}, "prev_seeds": 0}
    flip = pa > pb
    lo, hi = (pb, pa) if flip else (pa, pb)
    sub_pairs = data["sub_pairs"]
    sub_singles = data["sub_singles"]

    ev = 0.0
    var = 0.0
    ng = 0
    ok = True
    by_lo: dict[str, dict[str, Any]] = {}
    by_hi: dict[str, dict[str, Any]] = {}

    lo_mems = profiles[lo]
    hi_mems = profiles[hi]

    sub_m: dict[tuple[str, str], tuple[float, float, int]] = {}
    for ma in lo_mems:
        ia = ma["id"]
        for mb in hi_mems:
            ib = mb["id"]
            ps = list(sub_pairs[(lo, hi, ia, ib)])
            sg = list(sub_singles[(lo, hi, ia, ib)])
            for k in range(0, len(sg) - 1, 2):
                ps.append(0.5 * (sg[k] + sg[k + 1]))
            if len(sg) % 2 == 1:
                ps.append(sg[-1])
            g_sub = len(sub_pairs[(lo, hi, ia, ib)]) * 2 + len(sg)
            if g_sub == 0:
                ok = False
                sub_m[(ia, ib)] = (0.5, 0.5, 0)
            else:
                mu, se = pair_stats(ps)
                sub_m[(ia, ib)] = (mu, se, g_sub)

    for ma in lo_mems:
        ia, wa = ma["id"], float(ma["weight"])
        rev = 0.0
        rvar = 0.0
        rg = 0
        for mb in hi_mems:
            ib, wb = mb["id"], float(mb["weight"])
            mu, se, gs = sub_m[(ia, ib)]
            ev += (wa * wb) * mu
            var += ((wa * wb) ** 2) * (se ** 2)
            rev += wb * mu
            rvar += (wb ** 2) * (se ** 2)
            rg += gs
            ng += gs
        by_lo[ia] = {"mean": round(rev, 4), "se": round(math.sqrt(rvar), 4), "games": rg}

    for mb in hi_mems:
        ib, wb = mb["id"], float(mb["weight"])
        cev = 0.0
        cvar = 0.0
        cg = 0
        for ma in lo_mems:
            ia, wa = ma["id"], float(ma["weight"])
            mu, se, gs = sub_m[(ia, ib)]
            cev += wa * (1.0 - mu)
            cvar += (wa ** 2) * (se ** 2)
            cg += gs
        by_hi[ib] = {"mean": round(cev, 4), "se": round(math.sqrt(cvar), 4), "games": cg}

    st = data["seat_stats"].get((lo, hi))
    seat_bal = None
    if st and st[1] > 0 and st[3] > 0:
        s0 = st[0] / st[1]
        s1 = st[2] / st[3]
        seat_bal = [round(1.0 - s1, 4), round(1.0 - s0, 4)] if flip else [round(s0, 4), round(s1, 4)]

    return {
        "mean": round(1.0 - ev if flip else ev, 4),
        "se": round(math.sqrt(var), 4),
        "games": ng,
        "ok": ok,
        "prev_seeds": len(data["prev_seeds_seen"].get((lo, hi), set())),
        "seat_p0_p1": seat_bal,
        "by_a": by_hi if flip else by_lo,
        "by_b": by_lo if flip else by_hi,
    }


def solve_core(
    data: dict[str, Any],
    profiles: dict[str, list[dict[str, Any]]],
    shells: list[str],
    n_boot: int = 2000,
    seed: int = 20261121,
) -> dict[str, Any]:
    k = len(shells)
    M = [[0.5] * k for _ in range(k)]
    S = [[0.0] * k for _ in range(k)]
    G = [[0] * k for _ in range(k)]
    P = [[0] * k for _ in range(k)]
    edge_details: dict[str, Any] = {}
    for i in range(k):
        for j in range(i + 1, k):
            e = eval_edge(data, profiles, shells[i], shells[j])
            assert e["ok"], f"Missing subcells for {shells[i]} vs {shells[j]}: {e}"
            M[i][j], M[j][i] = e["mean"], round(1.0 - e["mean"], 4)
            S[i][j] = S[j][i] = e["se"]
            G[i][j] = G[j][i] = e["games"]
            P[i][j] = P[j][i] = e["prev_seeds"]
            edge_details[f"{shells[i]} vs {shells[j]}"] = e

    pt = STATS.solve_symmetric_nash(M, iters=100_000)
    rng = random.Random(seed)
    bws: list[list[float]] = []
    for _ in range(n_boot):
        mb = [[0.5] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                x = max(0.0, min(1.0, M[i][j] + rng.gauss(0.0, S[i][j])))
                mb[i][j], mb[j][i] = x, 1.0 - x
        bws.append(STATS.solve_symmetric_nash(mb, iters=8_000))
    bm = [sum(b[i] for b in bws) / n_boot for i in range(k)]
    bm_sum = sum(bm)
    bm_norm = [x / bm_sum for x in bm]

    summary = []
    for i, nm in enumerate(shells):
        col = sorted(b[i] for b in bws)
        ev_pt = sum(pt[j] * M[i][j] for j in range(k))
        se_pt = math.sqrt(sum((pt[j] * S[i][j]) ** 2 for j in range(k) if j != i))
        ev_bt = sum(bm_norm[j] * M[i][j] for j in range(k))
        se_bt = math.sqrt(sum((bm_norm[j] * S[i][j]) ** 2 for j in range(k) if j != i))
        z_pt = (ev_pt - 0.5) / se_pt if se_pt > 1e-9 else 0.0
        summary.append({
            "name": nm,
            "point": round(pt[i], 4),
            "boot_mean": round(bm_norm[i], 4),
            "boot_p_gt_5pct": round(sum(1 for x in col if x > 0.05) / n_boot, 3),
            "boot_iqr": [round(col[int(0.25 * n_boot)], 4), round(col[int(0.75 * n_boot)], 4)],
            "ev_vs_point": round(ev_pt, 4),
            "se_vs_point": round(se_pt, 4),
            "z_vs_point": round(z_pt, 2),
            "ev_vs_boot": round(ev_bt, 4),
            "se_vs_boot": round(se_bt, 4),
            "mean_vs_core": round(sum(M[i][j] for j in range(k) if j != i) / (k - 1), 4),
        })

    sens = []
    for i in range(k):
        for j in range(i + 1, k):
            max_tv = 0.0
            w_m: list[float] = []
            w_p: list[float] = []
            for sign in (-1, 1):
                Mp = [r[:] for r in M]
                v = max(0.0, min(1.0, M[i][j] + sign * S[i][j]))
                Mp[i][j], Mp[j][i] = v, 1.0 - v
                wp = STATS.solve_symmetric_nash(Mp, iters=40_000)
                tv = 0.5 * sum(abs(wp[idx] - pt[idx]) for idx in range(k))
                if tv > max_tv:
                    max_tv = tv
                if sign == -1:
                    w_m = wp
                else:
                    w_p = wp
            sens.append({
                "edge": f"{shells[i]} vs {shells[j]}",
                "mean": M[i][j],
                "se": S[i][j],
                "games": G[i][j],
                "prev_seeds": P[i][j],
                "max_tv_1sig": round(max_tv, 3),
                "minus1sig": {shells[idx]: round(w_m[idx], 3) for idx in range(k) if w_m[idx] >= 0.01},
                "plus1sig": {shells[idx]: round(w_p[idx], 3) for idx in range(k) if w_p[idx] >= 0.01},
            })
    sens.sort(key=lambda x: -x["max_tv_1sig"])

    internal_var_audit: dict[str, Any] = {}
    for i, nm in enumerate(shells):
        mems = profiles[nm]
        if len(mems) <= 1:
            continue
        per_v: dict[str, Any] = {}
        for m in mems:
            vid = m["id"]
            ev_p_non = var_p_non = 0.0
            ev_b_non = var_b_non = 0.0
            ev_eq = var_eq = 0.0
            per_opp: dict[str, Any] = {}
            for j, op in enumerate(shells):
                if j == i:
                    continue
                ek = f"{shells[i]} vs {shells[j]}" if i < j else f"{shells[j]} vs {shells[i]}"
                ed = edge_details[ek]
                bv = ed["by_a"][vid] if i < j else ed["by_b"][vid]
                mu_v, se_v, g_v = bv["mean"], bv["se"], bv["games"]
                per_opp[op] = {"mean": mu_v, "se": se_v, "games": g_v}
                ev_p_non += pt[j] * mu_v
                var_p_non += (pt[j] * se_v) ** 2
                ev_b_non += bm_norm[j] * mu_v
                var_b_non += (bm_norm[j] * se_v) ** 2
                ev_eq += mu_v / (k - 1)
                var_eq += (se_v / (k - 1)) ** 2
            ev_p_full = ev_p_non + pt[i] * 0.5
            se_p_full = math.sqrt(var_p_non)
            ev_b_full = ev_b_non + bm_norm[i] * 0.5
            se_b_full = math.sqrt(var_b_non)
            per_v[vid] = {
                "weight": m["weight"],
                "ev_vs_eq": round(ev_eq, 4),
                "se_vs_eq": round(math.sqrt(var_eq), 4),
                "ev_vs_pt": round(ev_p_full, 4),
                "se_vs_pt": round(se_p_full, 4),
                "ucb95_pt": round(ev_p_full + 1.645 * se_p_full, 4),
                "ev_vs_boot": round(ev_b_full, 4),
                "se_vs_boot": round(se_b_full, 4),
                "ucb95_boot": round(ev_b_full + 1.645 * se_b_full, 4),
                "per_opp": per_opp,
            }
        internal_var_audit[nm] = per_v

    return {
        "shells": shells,
        "games_min": min(G[i][j] for i in range(k) for j in range(i + 1, k)),
        "games_total_in_matrix": sum(G[i][j] for i in range(k) for j in range(i + 1, k)),
        "summary": summary,
        "sensitivity": sens,
        "internal_variant_audit": internal_var_audit,
        "matrix": M,
        "se_matrix": S,
        "games_matrix": G,
        "prev_seeds_matrix": P,
    }


def eval_challengers_against_pool(
    data: dict[str, Any],
    profiles: dict[str, list[dict[str, Any]]],
    pool_shells: list[str],
    pool_weights: list[float],
    challengers: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ch in challengers:
        ev = 0.0
        var = 0.0
        missing: list[str] = []
        assumed_mirror = False
        per_opp: dict[str, Any] = {}
        for s, w in zip(pool_shells, pool_weights):
            if w <= 1e-9:
                continue
            if ch == s:
                ev += w * 0.5
                per_opp[s] = {"mean": 0.5, "se": 0.0, "games": 0}
                continue
            e = eval_edge(data, profiles, ch, s)
            if e["ok"]:
                ev += w * e["mean"]
                var += (w * e["se"]) ** 2
                per_opp[s] = {"mean": e["mean"], "se": e["se"], "games": e["games"]}
            else:
                missing.append(s)
                ev += w * 0.5
                assumed_mirror = True
                per_opp[s] = {"mean": None, "games": e["games"]}
        se = math.sqrt(var)
        out[ch] = {
            "ev": round(ev, 4),
            "se": round(se, 4),
            "ucb95": round(ev + 1.645 * se, 4),
            "missing_edges": missing,
            "assumed_mirror": assumed_mirror,
            "per_opp": per_opp,
        }
    return out


# ---------------------------------------------------------------------------
# Subcommand: audit-core
# ---------------------------------------------------------------------------

def cmd_audit_core(args: argparse.Namespace) -> None:
    _, profiles, _ = build_all_sets_and_profiles()
    shells = [s.strip() for s in args.shells.split(",") if s.strip()]
    data = load_all_prod_pairs(args.work_dir)
    res = solve_core(data, profiles, shells, n_boot=args.boot)

    devs = [
        "mjj-2014-thdrcomb__56b67d",
        "thdr:zap-talk",
        "thdr-v-zap-talk",
        "S07_50_50",
        "pb-58",
    ]
    if args.extra_devs:
        for d in args.extra_devs.split(","):
            if d.strip() and d.strip() not in devs:
                devs.append(d.strip())
    pt_w = [s["point"] for s in res["summary"]]
    bt_w = [s["boot_mean"] for s in res["summary"]]
    res["dev_vs_point"] = eval_challengers_against_pool(data, profiles, shells, pt_w, devs)
    res["dev_vs_boot"] = eval_challengers_against_pool(data, profiles, shells, bt_w, devs)
    res["sources_count"] = len(data["sources"])
    res["total_prod_games_loaded"] = data["total_games"]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n")

    print(json.dumps({
        "cmd": "audit-core",
        "out": str(args.out) if args.out else None,
        "games_min": res["games_min"],
        "games_matrix_sum": res["games_total_in_matrix"],
        "point": {s["name"]: s["point"] for s in res["summary"]},
        "boot": {s["name"]: s["boot_mean"] for s in res["summary"]},
        "z_vs_pt": {s["name"]: s["z_vs_point"] for s in res["summary"]},
        "top_sens": [[e["edge"], e["mean"], e["se"], e["games"], e["max_tv_1sig"]] for e in res["sensitivity"][:4]],
        "dev_pt": {k: [v["ev"], v["ucb95"]] for k, v in res["dev_vs_point"].items()},
        "dev_boot": {k: [v["ev"], v["ucb95"]] for k, v in res["dev_vs_boot"].items()},
    }))


# ---------------------------------------------------------------------------
# Subcommand: prepare-matchups (General chunked schedule builder w/ optional no-cache)
# ---------------------------------------------------------------------------

PHASE1_CORE5_HP: list[tuple[str, str, Any, str]] = [
    ("S07_0_100", "D5", 5, "a"),                                                                   # 50g
    ("S07_0_100", "D5", 5, "b"),                                                                   # 50g -> 150g
    ("S07_0_100", "PB132_50_50", 12, "a"),                                                         # 48g -> 96g
    ("APEX4", "PB132_50_50", 3, "a"),                                                              # 48g -> 96g
    ("thdr:zap-talk-30-70", "PB132_50_50", {"mjj-2014-thdrcomb__56b67d": 4, "thdr-v-zap-talk": 8}, "a"),  # 48g -> 96g
    ("thdr:zap-talk-30-70", "APEX4", {"mjj-2014-thdrcomb__56b67d": 2, "thdr-v-zap-talk": 4}, "a"),        # 48g -> 112g
    ("APEX4", "D5", 1, "a"),                                                                       # 40g -> 120g
    ("D5", "PB132_50_50", 2, "a"),                                                                 # 40g -> 100g
    ("thdr:zap-talk-30-70", "S07_0_100", {"mjj-2014-thdrcomb__56b67d": 6, "thdr-v-zap-talk": 12}, "a"),   # 36g
    ("thdr:zap-talk-30-70", "S07_0_100", {"mjj-2014-thdrcomb__56b67d": 6, "thdr-v-zap-talk": 12}, "b"),   # 36g -> 120g
    ("S07_0_100", "APEX4", 4, "a"),                                                                # 32g
    ("S07_0_100", "APEX4", 4, "b"),                                                                # 32g -> 114g
    ("thdr:zap-talk-30-70", "D5", {"mjj-2014-thdrcomb__56b67d": 1, "thdr-v-zap-talk": 2}, "a"),           # 30g
    ("thdr:zap-talk-30-70", "D5", {"mjj-2014-thdrcomb__56b67d": 1, "thdr-v-zap-talk": 2}, "b"),           # 30g -> 120g
]


def _write_pools(
    out_dir: Path,
    used_profiles: set[str],
    all_sets: dict[str, list[dict[str, Any]]],
    profiles: dict[str, list[dict[str, Any]]],
    base_teams: dict[str, Any],
) -> Path:
    needed_ids: set[str] = set()
    mixtures: dict[str, list[dict[str, Any]]] = {}
    for pname in sorted(used_profiles):
        mems = profiles[pname]
        if len(mems) > 1:
            sigs = {
                tuple((m["species"], m["level"], m.get("gender")) for m in all_sets[mem["id"]])
                for mem in mems
            }
            assert len(sigs) == 1, f"Profile {pname} has mismatched preview signatures: {sigs}"
        mname = pname if len(mems) > 1 else f"s:{mems[0]['id']}"
        mixtures[mname] = [{"id": m["id"], "weight": float(m["weight"])} for m in mems]
        for m in mems:
            needed_ids.add(m["id"])

    extra_teams = [{"id": tid, "sets": all_sets[tid]} for tid in sorted(needed_ids) if tid not in base_teams]
    pools_path = out_dir / "pools.json"
    pools_path.write_text(json.dumps({"teams": extra_teams, "mixtures": mixtures}, separators=(",", ":")) + "\n")
    return pools_path


def cmd_prepare_matchups(args: argparse.Namespace) -> None:
    all_sets, profiles, base_teams = build_all_sets_and_profiles()
    if args.preset == "phase1-core5-hp":
        spec = PHASE1_CORE5_HP
    elif args.spec_file:
        raw_spec = json.loads(args.spec_file.read_text())
        spec = [(x[0], x[1], x[2], x[3] if len(x) > 3 else f"c{idx}") for idx, x in enumerate(raw_spec)]
    else:
        raise ValueError("Provide --preset or --spec-file")

    used_profiles: set[str] = set()
    for rname, cname, _, _ in spec:
        used_profiles.add(rname)
        used_profiles.add(cname)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pools_path = _write_pools(out_dir, used_profiles, all_sets, profiles, base_teams)

    rows: list[dict[str, Any]] = []
    chunk_breaks: list[int] = [0]
    blk = 0
    seed_prefix = args.seed_prefix
    dist_cache = str(args.preview_dist_cache.resolve()) if args.preview_dist_cache else None
    dist_k = int(args.preview_dist_k) if args.preview_dist_k is not None else None
    use_dist = bool(dist_cache or (dist_k is not None and dist_k >= 2))
    no_cache = bool(args.no_preview_cache) or use_dist

    for rname, cname, pspec, subtag in spec:
        rmems = profiles[rname]
        cmems = profiles[cname]
        rmix = rname if len(rmems) > 1 else f"s:{rmems[0]['id']}"
        cmix = cname if len(cmems) > 1 else f"s:{cmems[0]['id']}"
        tot_c_pairs = sum(int(pspec[rm["id"]]) if isinstance(pspec, dict) else int(pspec) for rm in rmems)
        c_bkts = 3 if tot_c_pairs >= 8 else 2
        r_rep: dict[str, int] = defaultdict(int)
        c_rep: dict[str, int] = defaultdict(int)
        for rm in rmems:
            rid = rm["id"]
            n_p = int(pspec[rid]) if isinstance(pspec, dict) else int(pspec)
            if n_p <= 0:
                continue
            r_bkts = 3 if (n_p * len(cmems)) >= 8 else 2
            for cm in cmems:
                cid = cm["id"]
                for vrep in range(n_p):
                    tag = f"{seed_prefix}/{subtag}/{rname}/{rid}/vs/{cname}/{cid}/r{vrep}"
                    sr = u32_hash(f"{tag}/r")
                    sc = u32_hash(f"{tag}/c")
                    bs = u32_hash(f"{tag}/battle")
                    pr = u32_hash(f"{seed_prefix}/{subtag}/{rname}/{rid}/vs/{cname}/prev-r-k{r_rep[rid] % r_bkts}")
                    pc = u32_hash(f"{seed_prefix}/{subtag}/{rname}/vs/{cname}/{cid}/prev-c-k{c_rep[cid] % c_bkts}")
                    for row_seat in (0, 1):
                        if row_seat == 0:
                            ids, seeds, pseeds, mix = [rid, cid], [sr, sc], [pr, pc], [rmix, cmix]
                        else:
                            ids, seeds, pseeds, mix = [cid, rid], [sc, sr], [pc, pr], [cmix, rmix]
                        row_obj: dict[str, Any] = {
                            "block": blk,
                            "cell": {
                                "kind": "goal_matchup",
                                "row": rname,
                                "col": cname,
                                "row_id": rid,
                                "col_id": cid,
                                "rep": vrep,
                                "subtag": subtag,
                            },
                            "rowSeat": row_seat,
                            "ids": ids,
                            "seeds": seeds,
                            "battleSeed": bs,
                            "mixtures": mix,
                        }
                        if not no_cache:
                            row_obj["previewSeeds"] = pseeds
                        rows.append(row_obj)
                    blk += 1
                    r_rep[rid] += 1
                    c_rep[cid] += 1
        chunk_breaks.append(len(rows))

    (out_dir / "schedule.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")
    meta = {
        "kind": "goal-matchups",
        "pools": str(pools_path),
        "no_preview_cache": bool(args.no_preview_cache) and not use_dist,
        "preview_dist_cache": dist_cache,
        "preview_dist_k": dist_k,
        "chunk_breaks": chunk_breaks,
        "total_pairs": blk,
        "total_games": len(rows),
        "preview_iters": args.preview_iters,
        "battle_iters": args.battle_iters,
        "battle_mode": args.battle_mode,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({
        "cmd": "prepare-matchups",
        "dir": str(out_dir),
        "chunks": len(chunk_breaks) - 1,
        "pairs": blk,
        "games": len(rows),
        "no_preview_cache": no_cache,
    }))


# ---------------------------------------------------------------------------
# Subcommand: prepare-ext-screen & analyze-ext-screen
# ---------------------------------------------------------------------------

def cmd_prepare_ext_screen(args: argparse.Namespace) -> None:
    all_sets, profiles, base_teams = build_all_sets_and_profiles()
    challengers = json.loads(args.challengers_file.read_text())
    opponents_spec: list[list[Any]] = json.loads(args.opponents_json)
    used_profiles: set[str] = set(challengers)
    for op, _ in opponents_spec:
        used_profiles.add(op)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pools_path = _write_pools(out_dir, used_profiles, all_sets, profiles, base_teams)

    rows: list[dict[str, Any]] = []
    chunk_breaks: list[int] = [0]
    blk = 0
    seed_prefix = args.seed_prefix
    dist_cache = str(args.preview_dist_cache.resolve()) if args.preview_dist_cache else None
    dist_k = int(args.preview_dist_k) if args.preview_dist_k is not None else None
    use_dist = bool(dist_cache or (dist_k is not None and dist_k >= 2))
    no_cache = bool(args.no_preview_cache) or use_dist
    per_chunk_cands = max(1, int(args.cands_per_chunk))

    for ci, ch in enumerate(challengers):
        cmems = profiles[ch]
        cmix = ch if len(cmems) > 1 else f"s:{cmems[0]['id']}"
        for op, pspec in opponents_spec:
            omems = profiles[op]
            omix = op if len(omems) > 1 else f"s:{omems[0]['id']}"
            c_rep: dict[str, int] = defaultdict(int)
            o_rep: dict[str, int] = defaultdict(int)
            tot_c = sum(int(pspec[om["id"]]) if isinstance(pspec, dict) else int(pspec) for om in omems)
            c_bkts = int(args.preview_buckets) if tot_c >= 6 else min(2, int(args.preview_buckets))
            for cm in cmems:
                cid = cm["id"]
                for om in omems:
                    oid = om["id"]
                    n_p = int(pspec[oid]) if isinstance(pspec, dict) else int(pspec)
                    if n_p <= 0:
                        continue
                    o_bkts = min(int(args.preview_buckets), max(1, n_p)) if n_p >= 4 else min(2, max(1, n_p))
                    for vrep in range(n_p):
                        tag = f"{seed_prefix}/crn/{op}/{oid}/r{vrep}"
                        sc = u32_hash(f"{tag}/c")
                        so = u32_hash(f"{tag}/o")
                        bs = u32_hash(f"{tag}/battle")
                        pc = u32_hash(f"{seed_prefix}/crn/{op}/prev-c-k{c_rep[cid] % c_bkts}")
                        po = u32_hash(f"{seed_prefix}/crn/{op}/{oid}/prev-o-k{o_rep[oid] % o_bkts}")
                        for row_seat in (0, 1):
                            if row_seat == 0:
                                ids, seeds, pseeds, mix = [cid, oid], [sc, so], [pc, po], [cmix, omix]
                            else:
                                ids, seeds, pseeds, mix = [oid, cid], [so, sc], [po, pc], [omix, cmix]
                            row_obj: dict[str, Any] = {
                                "block": blk,
                                "cell": {
                                    "kind": "ext_screen",
                                    "challenger": ch,
                                    "opponent": op,
                                    "ch_id": cid,
                                    "opp_id": oid,
                                    "rep": vrep,
                                },
                                "rowSeat": row_seat,
                                "ids": ids,
                                "seeds": seeds,
                                "battleSeed": bs,
                                "mixtures": mix,
                            }
                            if not no_cache:
                                row_obj["previewSeeds"] = pseeds
                            rows.append(row_obj)
                        blk += 1
                        c_rep[cid] += 1
                        o_rep[oid] += 1
        if (ci + 1) % per_chunk_cands == 0 or (ci + 1) == len(challengers):
            chunk_breaks.append(len(rows))

    (out_dir / "schedule.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")
    meta = {
        "kind": "ext-screen",
        "pools": str(pools_path),
        "challengers": challengers,
        "opponents": opponents_spec,
        "no_preview_cache": bool(args.no_preview_cache) and not use_dist,
        "preview_dist_cache": dist_cache,
        "preview_dist_k": dist_k,
        "chunk_breaks": chunk_breaks,
        "total_pairs": blk,
        "total_games": len(rows),
        "preview_iters": args.preview_iters,
        "battle_iters": args.battle_iters,
        "battle_mode": args.battle_mode,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({
        "cmd": "prepare-ext-screen",
        "dir": str(out_dir),
        "challengers": len(challengers),
        "chunks": len(chunk_breaks) - 1,
        "pairs": blk,
        "games": len(rows),
    }))


def cmd_analyze_ext_screen(args: argparse.Namespace) -> None:
    _, profiles, _ = build_all_sets_and_profiles()
    run_dir: Path = args.dir
    meta = json.loads((run_dir / "meta.json").read_text())
    challengers: list[str] = meta["challengers"]
    opponents: list[str] = [op for op, _ in meta["opponents"]]

    scenarios: dict[str, dict[str, float]] = json.loads(args.scenarios_json)
    if "equal" not in scenarios:
        scenarios["equal"] = {op: 1.0 / len(opponents) for op in opponents}

    raw: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    slot_cnt: dict[str, Counter[int]] = defaultdict(Counter)
    g_cnt: Counter[str] = Counter()
    for line in (run_dir / "results.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r["cell"]
        rs = int(r["rowSeat"])
        sc = seat_score(r["outcome"], rs)
        raw[(c["challenger"], c["opponent"], c["ch_id"], c["opp_id"], int(c["rep"]))].append(sc)
        pr = r.get("preview")
        if isinstance(pr, list) and len(pr) == 2 and isinstance(pr[rs], dict):
            inp = pr[rs].get("input", "")
            if inp.startswith("team "):
                g_cnt[c["challenger"]] += 1
                for s in inp[5:].split(","):
                    slot_cnt[c["challenger"]][int(s.strip())] += 1

    sub_p: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (ch, op, cid, oid, _rep), scs in raw.items():
        if len(scs) == 2:
            sub_p[(ch, op, cid, oid)].append(0.5 * (scs[0] + scs[1]))

    rows_out: list[dict[str, Any]] = []
    for ch in challengers:
        per_opp: dict[str, dict[str, Any]] = {}
        for op in opponents:
            ev = 0.0
            var = 0.0
            ng = 0
            for cm in profiles[ch]:
                cid, cw = cm["id"], float(cm["weight"])
                for om in profiles[op]:
                    oid, ow = om["id"], float(om["weight"])
                    ps = sub_p[(ch, op, cid, oid)]
                    mu, se = pair_stats(ps)
                    ev += (cw * ow) * mu
                    var += ((cw * ow) ** 2) * (se ** 2)
                    ng += len(ps) * 2
            per_opp[op] = {"mean": round(ev, 4), "se": round(math.sqrt(var), 4), "games": ng}

        scen_out: dict[str, dict[str, float]] = {}
        max_sc_ev = 0.0
        for sname, wmap in scenarios.items():
            ev = sum(float(wmap.get(op, 0.0)) * per_opp[op]["mean"] for op in opponents)
            se = math.sqrt(sum((float(wmap.get(op, 0.0)) * per_opp[op]["se"]) ** 2 for op in opponents))
            scen_out[sname] = {
                "ev": round(ev, 4),
                "se": round(se, 4),
                "ucb95": round(ev + 1.645 * se, 4),
                "lcb95": round(ev - 1.645 * se, 4),
            }
            if ev > max_sc_ev:
                max_sc_ev = ev

        ng_slot = max(1, g_cnt[ch])
        slots = [round(slot_cnt[ch][i] / ng_slot, 3) for i in range(1, 7)]
        rows_out.append({
            "challenger": ch,
            "max_scenario_ev": round(max_sc_ev, 4),
            "scenarios": scen_out,
            "max_opp": round(max(v["mean"] for v in per_opp.values()), 4),
            "min_opp": round(min(v["mean"] for v in per_opp.values()), 4),
            "per_opp": per_opp,
            "slot_picks": slots,
            "min_pick": min(slots),
            "zero_slots": sum(1 for x in slots if x == 0.0),
        })

    rows_out.sort(key=lambda x: -x["max_scenario_ev"])
    report = {
        "dir": str(run_dir),
        "games": sum(len(v) * 2 for v in sub_p.values()),
        "scenarios": scenarios,
        "ranking": rows_out,
    }
    out_path = args.out or (run_dir / "ext-screen-report.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "cmd": "analyze-ext-screen",
        "out": str(out_path),
        "games": report["games"],
        "top15": [
            {
                "ch": r["challenger"],
                "max_ev": r["max_scenario_ev"],
                "scen": {k: v["ev"] for k, v in r["scenarios"].items()},
                "min_pick": r["min_pick"],
            }
            for r in rows_out[:15]
        ],
    }))


# ---------------------------------------------------------------------------
# Subcommand: prepare-holdout & analyze-holdout (Strictly Independent NO-CACHE Certification)
# ---------------------------------------------------------------------------

def cmd_prepare_holdout(args: argparse.Namespace) -> None:
    all_sets, profiles, base_teams = build_all_sets_and_profiles()
    pool_doc = json.loads(args.pool_json.read_text())
    pool_w: dict[str, float] = {k: float(v) for k, v in pool_doc["pool"].items() if float(v) > 0.0}
    w_sum = sum(pool_w.values())
    pool_w = {k: v / w_sum for k, v in pool_w.items()}

    challengers = pool_doc["challengers"]
    used_profiles: set[str] = set(pool_w.keys()) | {c["name"] for c in challengers}

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pools_path = _write_pools(out_dir, used_profiles, all_sets, profiles, base_teams)

    rows: list[dict[str, Any]] = []
    chunk_breaks: list[int] = [0]
    blk = 0
    seed_prefix = args.seed_prefix
    chunk_pairs_limit = max(4, int(args.pairs_per_chunk))

    for ch_obj in challengers:
        ch = ch_obj["name"]
        tot_pairs = int(ch_obj["pairs"])
        cmems = profiles[ch]
        cmix = ch if len(cmems) > 1 else f"s:{cmems[0]['id']}"

        slots_weights: list[tuple[str, str, str, float]] = []
        for cm in cmems:
            for op, pw in pool_w.items():
                for om in profiles[op]:
                    slots_weights.append((cm["id"], op, om["id"], float(cm["weight"]) * pw * float(om["weight"])))
        raw_alloc = [tot_pairs * w for _, _, _, w in slots_weights]
        int_alloc = [int(math.floor(x)) for x in raw_alloc]
        rem = tot_pairs - sum(int_alloc)
        order = sorted(range(len(slots_weights)), key=lambda i: -(raw_alloc[i] - int_alloc[i]))
        for idx in order[:rem]:
            int_alloc[idx] += 1

        ch_rows: list[dict[str, Any]] = []
        for (cid, op, oid, _w), n_p in zip(slots_weights, int_alloc):
            if n_p <= 0:
                continue
            omems = profiles[op]
            omix = op if len(omems) > 1 else f"s:{omems[0]['id']}"
            for vrep in range(n_p):
                tag = f"{seed_prefix}/holdout/{ch}/{cid}/vs/{op}/{oid}/r{vrep}"
                sc0 = u32_hash(f"{tag}/s0/c")
                so0 = u32_hash(f"{tag}/s0/o")
                sc1 = u32_hash(f"{tag}/s1/c")
                so1 = u32_hash(f"{tag}/s1/o")
                bs = u32_hash(f"{tag}/battle")
                for row_seat in (0, 1):
                    if row_seat == 0:
                        ids, seeds, mix = [cid, oid], [sc0, so0], [cmix, omix]
                    else:
                        ids, seeds, mix = [oid, cid], [so1, sc1], [omix, cmix]
                    ch_rows.append({
                        "block": blk,
                        "cell": {
                            "kind": "holdout_cert",
                            "challenger": ch,
                            "opponent": op,
                            "ch_id": cid,
                            "opp_id": oid,
                            "rep": vrep,
                        },
                        "rowSeat": row_seat,
                        "ids": ids,
                        "seeds": seeds,
                        "battleSeed": bs,
                        "mixtures": mix,
                    })
                blk += 1

        g_lim = chunk_pairs_limit * 2
        for start in range(0, len(ch_rows), g_lim):
            rows.extend(ch_rows[start : start + g_lim])
            chunk_breaks.append(len(rows))

    (out_dir / "schedule.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")
    meta = {
        "kind": "holdout-cert",
        "pools": str(pools_path),
        "pool": pool_w,
        "challengers": challengers,
        "no_preview_cache": True,
        "chunk_breaks": chunk_breaks,
        "total_pairs": blk,
        "total_games": len(rows),
        "preview_iters": args.preview_iters,
        "battle_iters": args.battle_iters,
        "battle_mode": args.battle_mode,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({
        "cmd": "prepare-holdout",
        "dir": str(out_dir),
        "challengers": len(challengers),
        "chunks": len(chunk_breaks) - 1,
        "pairs": blk,
        "games": len(rows),
    }))


def cmd_analyze_holdout(args: argparse.Namespace) -> None:
    run_dir: Path = args.dir
    meta = json.loads((run_dir / "meta.json").read_text())
    pool_w: dict[str, float] = meta["pool"]
    challengers = [c["name"] for c in meta["challengers"]]

    raw: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for line in (run_dir / "results.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r["cell"]
        raw[(c["challenger"], c["opponent"], c["ch_id"], c["opp_id"], int(c["rep"]))].append(
            seat_score(r["outcome"], int(r["rowSeat"]))
        )

    by_ch_pairs: dict[str, list[float]] = defaultdict(list)
    by_ch_opp: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (ch, op, _cid, _oid, _rep), scs in raw.items():
        if len(scs) == 2:
            ps = 0.5 * (scs[0] + scs[1])
            by_ch_pairs[ch].append(ps)
            by_ch_opp[(ch, op)].append(ps)

    hits = misses = 0
    for df in sorted((run_dir / "chunks").glob("chunk-*.done.json")):
        d = json.loads(df.read_text())
        inner = json.loads(d["remote_stdout_last"]) if "remote_stdout_last" in d else d
        hits += int(inner.get("preview_cache_hits", 0))
        misses += int(inner.get("preview_cache_misses", 0))

    summary: list[dict[str, Any]] = []
    for ch in challengers:
        ps = by_ch_pairs[ch]
        mu, se = pair_stats(ps)
        st_ev = 0.0
        st_var = 0.0
        per_opp: dict[str, Any] = {}
        all_opps_present = True
        for op, w in pool_w.items():
            ops = by_ch_opp[(ch, op)]
            if len(ops) >= 2:
                omu, ose = pair_stats(ops)
                st_ev += w * omu
                st_var += (w * ose) ** 2
                per_opp[op] = {"mean": round(omu, 4), "se": round(ose, 4), "games": len(ops) * 2}
            elif len(ops) == 1:
                st_ev += w * ops[0]
                st_var += (w * 0.5) ** 2
                per_opp[op] = {"mean": round(ops[0], 4), "se": 0.5, "games": 2}
            else:
                all_opps_present = False
                per_opp[op] = {"mean": None, "games": 0}
        st_se = math.sqrt(st_var) if all_opps_present else se
        use_ev = st_ev if all_opps_present else mu
        use_se = max(se, st_se) if all_opps_present else se
        df = max(1, len(ps) - 1)
        t95 = 1.6449 + 1.53 / df + 1.2 / (df * df)
        ucb95_z = use_ev + 1.6449 * use_se
        ucb95_t = use_ev + t95 * use_se
        summary.append({
            "challenger": ch,
            "games": len(ps) * 2,
            "pairs": len(ps),
            "pooled_mean": round(mu, 4),
            "strat_mean": round(use_ev, 4),
            "se_pooled": round(se, 4),
            "se_strat": round(st_se, 4),
            "ucb95_z": round(ucb95_z, 4),
            "ucb95_t": round(ucb95_t, 4),
            "pass_55pct_t": ucb95_t <= 0.55,
            "per_opp": per_opp,
        })

    summary.sort(key=lambda x: -x["ucb95_t"])
    report = {
        "dir": str(run_dir),
        "pool": pool_w,
        "total_games": sum(r["games"] for r in summary),
        "preview_cache_hits": hits,
        "preview_cache_misses": misses,
        "all_pass_ucb95_t_le_55": all(r["pass_55pct_t"] for r in summary),
        "worst_ucb95_t": summary[0]["ucb95_t"] if summary else None,
        "worst_challenger": summary[0]["challenger"] if summary else None,
        "challengers": summary,
    }
    out_path = args.out or (run_dir / "holdout-cert-report.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "cmd": "analyze-holdout",
        "out": str(out_path),
        "games": report["total_games"],
        "cache_hits": hits,
        "all_pass_55": report["all_pass_ucb95_t_le_55"],
        "results": [
            {
                "ch": r["challenger"],
                "g": r["games"],
                "ev": r["strat_mean"],
                "se": round(max(r["se_pooled"], r["se_strat"]), 4),
                "ucb95_t": r["ucb95_t"],
                "pass": r["pass_55pct_t"],
            }
            for r in summary
        ],
    }))


# ---------------------------------------------------------------------------
# Subcommand: precompute-preview-dist (Dual-node K-seed Legacy Preview Distribution Cache)
# ---------------------------------------------------------------------------

def cmd_precompute_preview_dist(args: argparse.Namespace) -> None:
    import time
    t0 = time.monotonic()
    all_sets, profiles, base_teams = build_all_sets_and_profiles()

    edge_pairs: list[tuple[str, str]] = []
    if args.shells:
        sh = [s.strip() for s in args.shells.split(",") if s.strip()]
        for i in range(len(sh)):
            for j in range(i + 1, len(sh)):
                edge_pairs.append((sh[i], sh[j]))
    if args.challengers_file and args.opponents:
        chs = json.loads(args.challengers_file.read_text())
        ops = [s.strip() for s in args.opponents.split(",") if s.strip()]
        for ch in chs:
            for op in ops:
                if ch != op:
                    edge_pairs.append((ch, op))
    assert edge_pairs, "Provide --shells and/or (--challengers-file + --opponents)"

    used_profiles: set[str] = set()
    for a, b in edge_pairs:
        used_profiles.add(a)
        used_profiles.add(b)

    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    pools_path = _write_pools(work_dir, used_profiles, all_sets, profiles, base_teams)

    # Deduplicate minimal representative tasks covering every (own_var, opp_mix, side) pair.
    # For each ordered (A_profile, B_profile): every member of A only needs ONE representative
    # member of B (all members of B share exact preview signature + identical disclosed mixture B).
    seen_tasks: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    blk = 0
    for a, b in edge_pairs:
        amems = profiles[a]
        bmems = profiles[b]
        amix = a if len(amems) > 1 else f"s:{amems[0]['id']}"
        bmix = b if len(bmems) > 1 else f"s:{bmems[0]['id']}"
        b0 = bmems[0]["id"]
        a0 = amems[0]["id"]
        # Cover all members of A vs B (both seats) and all members of B vs A (both seats)
        pairs_to_cover = [(am["id"], b0, amix, bmix) for am in amems] + [(a0, bm["id"], amix, bmix) for bm in bmems]
        for aid, bid, mx_a, mx_b in pairs_to_cover:
            tkey = (aid, bid, mx_a, mx_b)
            if tkey in seen_tasks:
                continue
            seen_tasks.add(tkey)
            for seat in (0, 1):
                ids = [aid, bid] if seat == 0 else [bid, aid]
                mix = [mx_a, mx_b] if seat == 0 else [mx_b, mx_a]
                rows.append({
                    "block": blk,
                    "cell": {"kind": "precompute_dist", "a": a, "b": b, "aid": aid, "bid": bid},
                    "rowSeat": seat,
                    "ids": ids,
                    "seeds": [1001 + blk * 2 + seat, 2001 + blk * 2 + seat],
                    "battleSeed": 3001 + blk,
                    "mixtures": mix,
                })
            blk += 1

    (work_dir / "schedule.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    cache_path: Path = args.cache
    # Chunk by rows_per_chunk so mini1 + mini2 + Colab load-balance cleanly
    step = max(2, int(args.rows_per_chunk) + (int(args.rows_per_chunk) % 2))
    if not any(chunks_dir.glob("chunk-*.jsonl")):
        step = ELIM.MINI2.recommend_chunk_size(
            step,
            colab_nodes=args.colab or None,
            preview_iters=args.preview_iters,
            battle_iters=243000,
            battle_mode="adaptive",
            preview_cache=True,
            preview_dist_cache=cache_path if cache_path.exists() else None,
            preview_dist_k=args.k,
            preview_dist_only=True,
        )
    chunks: list[dict[str, Any]] = []
    for ci, start in enumerate(range(0, len(rows), step)):
        c_rows = rows[start : start + step]
        c_sched = chunks_dir / f"chunk-{ci:03d}.jsonl"
        c_out = chunks_dir / f"chunk-{ci:03d}-out"
        c_done = chunks_dir / f"chunk-{ci:03d}.done.json"
        c_sched.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in c_rows) + "\n")
        chunks.append({
            "ci": ci,
            "sched": c_sched,
            "out": c_out,
            "done": c_done,
            "n_rows": len(c_rows),
        })

    dispatch = ELIM.MINI2.run_chunks_dual_node(
        chunks,
        binary=args.binary,
        pools_path=pools_path,
        local_workers=args.local_workers,
        mini2_workers=args.mini2_workers,
        use_mini2=args.mini2,
        colab_nodes=args.colab or None,
        colab_binary=args.colab_binary,
        colab_slots=args.colab_slots,
        colab_threads=args.colab_threads,
        preview_iters=args.preview_iters,
        battle_iters=243000,
        battle_mode="adaptive",
        preview_cache=True,
        preview_dist_cache=cache_path if cache_path.exists() else None,
        preview_dist_k=args.k,
        preview_dist_only=True,
    )

    merged: dict[str, Any] = {}
    if cache_path.exists():
        prev = json.loads(cache_path.read_text())
        merged.update(prev.get("entries", {}))
    added = 0
    updated = 0
    for ch in chunks:
        p = ch["out"] / "preview-dist.json"
        doc = json.loads(p.read_text())
        for khex, entry in doc.get("entries", {}).items():
            old = merged.get(khex)
            if old is None:
                merged[khex] = entry
                added += 1
            elif int(entry["k"]) > int(old["k"]):
                merged[khex] = entry
                updated += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps({"format": "nc2000-preview-dist-v1", "entries": merged}, separators=(",", ":")) + "\n")
    tmp_path.replace(cache_path)

    # Diagnostics: entropy & support size across all merged entries
    supports = [len(e["counts"]) for e in merged.values()]
    non_pure = sum(1 for s in supports if s > 1)
    top1_shares = [max(e["counts"].values()) / max(1, e["k"]) for e in merged.values()]
    print(json.dumps({
        "cmd": "precompute-preview-dist",
        "cache": str(cache_path),
        "k": args.k,
        "total_entries": len(merged),
        "added": added,
        "updated": updated,
        "non_pure_entries": non_pure,
        "mean_support": round(sum(supports) / max(1, len(supports)), 2),
        "mean_top1_share": round(sum(top1_shares) / max(1, len(top1_shares)), 3),
        "wall_s": round(time.monotonic() - t0, 2),
        **dispatch,
    }))


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-phase NC2000 Nash verification & certification driver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ac = sub.add_parser("audit-core")
    p_ac.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    p_ac.add_argument("--shells", type=str, default="S07_0_100,APEX4,D5,PB132_50_50,thdr:zap-talk-30-70")
    p_ac.add_argument("--extra-devs", type=str, default="")
    p_ac.add_argument("--boot", type=int, default=2000)
    p_ac.add_argument("--out", type=Path, default=None)
    p_ac.set_defaults(func=cmd_audit_core)

    p_pm = sub.add_parser("prepare-matchups")
    p_pm.add_argument("--preset", type=str, default=None)
    p_pm.add_argument("--spec-file", type=Path, default=None)
    p_pm.add_argument("--out-dir", type=Path, required=True)
    p_pm.add_argument("--seed-prefix", type=str, required=True)
    p_pm.add_argument("--no-preview-cache", action="store_true", default=False)
    p_pm.add_argument("--preview-dist-cache", type=Path, default=None)
    p_pm.add_argument("--preview-dist-k", type=int, default=None)
    p_pm.add_argument("--preview-iters", type=int, default=729000)
    p_pm.add_argument("--battle-iters", type=int, default=243000)
    p_pm.add_argument("--battle-mode", choices=["fixed", "adaptive"], default="adaptive")
    p_pm.set_defaults(func=cmd_prepare_matchups)

    p_pe = sub.add_parser("prepare-ext-screen")
    p_pe.add_argument("--challengers-file", type=Path, required=True)
    p_pe.add_argument("--opponents-json", type=str, required=True)
    p_pe.add_argument("--out-dir", type=Path, required=True)
    p_pe.add_argument("--seed-prefix", type=str, required=True)
    p_pe.add_argument("--cands-per-chunk", type=int, default=4)
    p_pe.add_argument("--preview-buckets", type=int, default=2)
    p_pe.add_argument("--no-preview-cache", action="store_true", default=False)
    p_pe.add_argument("--preview-dist-cache", type=Path, default=None)
    p_pe.add_argument("--preview-dist-k", type=int, default=None)
    p_pe.add_argument("--preview-iters", type=int, default=81000)
    p_pe.add_argument("--battle-iters", type=int, default=9000)
    p_pe.add_argument("--battle-mode", choices=["fixed", "adaptive"], default="fixed")
    p_pe.set_defaults(func=cmd_prepare_ext_screen)

    p_pd = sub.add_parser("precompute-preview-dist")
    p_pd.add_argument("--shells", type=str, default="")
    p_pd.add_argument("--challengers-file", type=Path, default=None)
    p_pd.add_argument("--opponents", type=str, default="")
    p_pd.add_argument("--cache", type=Path, required=True)
    p_pd.add_argument("--work-dir", type=Path, required=True)
    p_pd.add_argument("--k", type=int, default=16)
    p_pd.add_argument("--preview-iters", type=int, default=729000)
    p_pd.add_argument("--rows-per-chunk", type=int, default=12)
    p_pd.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p_pd.add_argument("--local-workers", type=int, default=10)
    p_pd.add_argument("--mini2-workers", type=int, default=12)
    p_pd.add_argument("--mini2", action="store_true", default=False)
    p_pd.add_argument("--colab", type=str, default="", help="Colab node pool via colab-barn (e.g. G4, G4,G4, V5E1, auto)")
    p_pd.add_argument("--colab-slots", type=int, default=None)
    p_pd.add_argument("--colab-threads", type=int, default=None)
    p_pd.add_argument("--colab-binary", type=Path, default=None)
    p_pd.set_defaults(func=cmd_precompute_preview_dist)

    p_ae = sub.add_parser("analyze-ext-screen")
    p_ae.add_argument("--dir", type=Path, required=True)
    p_ae.add_argument("--scenarios-json", type=str, required=True)
    p_ae.add_argument("--out", type=Path, default=None)
    p_ae.set_defaults(func=cmd_analyze_ext_screen)

    p_ph = sub.add_parser("prepare-holdout")
    p_ph.add_argument("--pool-json", type=Path, required=True)
    p_ph.add_argument("--out-dir", type=Path, required=True)
    p_ph.add_argument("--seed-prefix", type=str, required=True)
    p_ph.add_argument("--pairs-per-chunk", type=int, default=16)
    p_ph.add_argument("--preview-iters", type=int, default=729000)
    p_ph.add_argument("--battle-iters", type=int, default=243000)
    p_ph.add_argument("--battle-mode", choices=["fixed", "adaptive"], default="adaptive")
    p_ph.set_defaults(func=cmd_prepare_holdout)

    p_ah = sub.add_parser("analyze-holdout")
    p_ah.add_argument("--dir", type=Path, required=True)
    p_ah.add_argument("--out", type=Path, default=None)
    p_ah.set_defaults(func=cmd_analyze_holdout)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
