#!/usr/bin/env python3
"""Okimono (dead-slot / hitchhiker) exposure, mutation screening, and strict NO-CACHE LIVE A/B
verification harness for NC2000 team catalogs (defaults to `data/live-search.json` budgets).

Subcommands:
  census-run       Exact preview census across all teams x shells of a catalog.
  census-analyze   Aggregate preview-dist into per-team, per-slot pick rates & okimono ranking.
  preview-screen   Exact preview activation screen for candidate mutations vs panel.
  live-ab          Paired CRN strictly NO-CACHE LIVE A/B evaluation + H2H.
  live-analyze     Analyze paired CRN LIVE A/B results (WR, paired delta, t-test, per-slot picks/InWR/InΔ).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "target/nash-work/okimono-v1"
DEFAULT_BIN = ROOT / "target/native-pgo/release/examples/nash_refresh"
TEAM_POOL_V3 = ROOT / "data/team-pool-v3/team-pool.json"
LIVE_SEARCH_CFG = json.loads((ROOT / "data/live-search.json").read_text())
DEFAULT_PREVIEW_ITERS = int(LIVE_SEARCH_CFG.get("unknownPreview", 240000))
DEFAULT_BATTLE_ITERS = int(LIVE_SEARCH_CFG.get("battle", {}).get("adaptive", 243000))


def _load_mod(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MINI2 = _load_mod("mini2_nash", ROOT / "tools/mini2_nash.py")


def u32_hash(tag: str) -> int:
    return int.from_bytes(hashlib.blake2b(tag.encode("utf-8"), digest_size=4).digest(), "little")


def load_v3_catalog(catalog_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    doc = json.loads((catalog_path or TEAM_POOL_V3).read_text())
    teams = doc["teams"]
    by_id = {t["id"]: t for t in teams}
    # Build shell mixtures using prefixed team IDs `v3:<id>` to avoid collisions with team-pool-v2.
    shells: dict[str, list[dict[str, Any]]] = {}
    for t in teams:
        tid = t["id"]
        if tid in ("s0053-lefties-hpi", "s0053-lefties-dp"):
            shells["shell:s0053-lefties"] = [
                {"id": "v3:s0053-lefties-hpi", "weight": 0.5},
                {"id": "v3:s0053-lefties-dp", "weight": 0.5},
            ]
        else:
            shells[f"shell:{tid}"] = [{"id": f"v3:{tid}", "weight": 1.0}]
    return teams, by_id, shells


def canonicalize_and_validate_teams(teams_map: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Canonicalize, training-v3 normalize, and strictly validate (WASM + JP NC2000 cart rules)."""
    js = r"""
const fs = require('fs');
const wasm = require('./crates/wasm/pkg-node/nc2000_wasm.js');
const { normalizeCatalog } = require('./tools/normalize-training.js');
const { attractGenderCensus } = require('./tools/attract-gender-census.js');
const dex = JSON.parse(fs.readFileSync('data/gen2stadium2.json', 'utf8'));
const bakePrior = JSON.parse(fs.readFileSync('data/belief-pool-v3/bake-prior.json', 'utf8'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

function checkJp(id, sets) {
  if (!Array.isArray(sets) || sets.length !== 6) throw new Error(`${id}: need 6 sets`);
  const items = new Set();
  const species = new Set();
  let l55 = 0;
  for (const s of sets) {
    if (species.has(s.species)) throw new Error(`${id}: duplicate species ${s.species}`);
    species.add(s.species);
    if (s.level > 55 || s.level < 50) throw new Error(`${id}: illegal level ${s.species} L${s.level}`);
    if (s.level === 55) l55 += 1;
    const mv = new Set(s.moves);
    if (s.species === 'Snorlax' && mv.has('Lovely Kiss')) throw new Error(`${id}: NYPC Lovely Kiss Snorlax`);
    if (s.species === 'Marowak' && mv.has('Belly Drum') && mv.has('Rock Slide')) throw new Error(`${id}: illegal Belly Drum + Rock Slide Marowak`);
    if (s.species === 'Zapdos' && s.level < 61 && mv.has('Light Screen')) throw new Error(`${id}: illegal Light Screen Zapdos L${s.level}`);
    if (s.species === 'Zapdos' && mv.has('Baton Pass')) throw new Error(`${id}: XD Baton Pass Zapdos`);
    if (s.species === 'Tyranitar' && s.level < 55) throw new Error(`${id}: Tyranitar illegal below L55`);
    if (s.species === 'Dragonite' && s.level < 55) throw new Error(`${id}: Dragonite illegal below L55`);
    if (s.species === 'Skarmory' && mv.has('Drill Peck') && mv.has('Whirlwind')) {
      // Legal via Fearow father in Gen 2! Keep allowed.
    }
    if (s.species === 'Charmander' || s.species === 'Charizard') {
      if (mv.has('Belly Drum') && mv.has('Crunch')) throw new Error(`${id}: incompatible egg combo`);
    }
    if (s.species === 'Blissey' && mv.has('Heal Bell') && mv.has('Counter')) {
      // Counter is Gen1 TM18, Heal Bell is Gen2 egg move — incompatible in Gen2!
      throw new Error(`${id}: incompatible Gen1 TM Counter + Gen2 Egg Heal Bell on Blissey`);
    }
    if (s.species === 'Snorlax' && mv.has('Curse') && mv.has('Self-Destruct')) {
      // Legal via Gen1 TM36 + Gen2 TM03
    }
    if (s.item) {
      if (items.has(s.item)) throw new Error(`${id}: duplicate item ${s.item}`);
      items.add(s.item);
    }
  }
}

const engineDex = new wasm.Dex();
const validator = new wasm.Validator(engineDex);
try {
  const entries = [];
  for (const [id, sets] of Object.entries(input)) {
    const canon = JSON.parse(validator.canonicalizeTeam(JSON.stringify(sets)));
    if (!canon.ok) throw new Error(`canonicalize failed ${id}: ${JSON.stringify(canon.errors)}`);
    entries.push({ id, variantCluster: id, sets: canon.team });
  }
  const attract = attractGenderCensus(bakePrior);
  const opt = normalizeCatalog(entries, dex, attract.preferFemale);
  const out = {};
  for (const e of entries) {
    const sets = opt.get(e.id).sets;
    checkJp(e.id, sets);
    const vr = JSON.parse(validator.validateTeam(JSON.stringify(sets)));
    if (!vr.ok) throw new Error(`validateTeam failed ${e.id}: ${JSON.stringify(vr.errors || vr.findings)}`);
    const b = new wasm.Battle(engineDex, JSON.stringify(sets), JSON.stringify(sets), '1,2,3,4');
    b.free();
    out[e.id] = sets;
  }
  console.log(JSON.stringify({ ok: true, teams: out }));
} finally {
  validator.free();
  engineDex.free();
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
        raise RuntimeError(f"Team validation failed:\n{proc.stderr or proc.stdout}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res.get("ok"), f"Team validation not ok: {res}"
    return res["teams"]


def run_preview_dist_chunks(
    run_dir: Path,
    pools: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    k: int = 4,
    rows_per_chunk: int = 10,
    preview_iters: int = DEFAULT_PREVIEW_ITERS,
    preloaded_cache: Path | None = None,
    local_only: bool = False,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    pools_path = run_dir / "pools.json"
    sched_path = run_dir / "schedule.jsonl"
    pools_path.write_text(json.dumps(pools, separators=(",", ":")))
    sched_path.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")

    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for ci, start in enumerate(range(0, len(rows), rows_per_chunk)):
        batch = rows[start : start + rows_per_chunk]
        c_sched = chunks_dir / f"chunk-{ci:03d}.jsonl"
        c_out = chunks_dir / f"chunk-{ci:03d}-out"
        c_done = chunks_dir / f"chunk-{ci:03d}.done.json"
        if not c_sched.exists():
            c_sched.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in batch) + "\n")
        chunks.append({
            "ci": ci,
            "sched": c_sched,
            "out": c_out,
            "done": c_done,
            "n_rows": len(batch),
        })

    t0 = time.monotonic()
    tel = MINI2.run_chunks_dual_node(
        chunks,
        binary=DEFAULT_BIN,
        pools_path=pools_path,
        local_workers=10,
        mini2_workers=12,
        use_mini2=True,
        colab_nodes="off" if local_only else None,
        preview_iters=preview_iters,
        preview_cache=False,
        preview_dist_cache=preloaded_cache,
        preview_dist_k=k,
        preview_dist_only=True,
    )

    # Merge all chunk preview-dist.json outputs
    merged: dict[str, Any] = {}
    if preloaded_cache and preloaded_cache.exists():
        merged.update(json.loads(preloaded_cache.read_text()).get("entries", {}))
    for ch in chunks:
        p = ch["out"] / "preview-dist.json"
        if not p.exists():
            raise RuntimeError(f"Missing chunk preview-dist.json: {p}")
        data = json.loads(p.read_text())
        for key, ent in data.get("entries", {}).items():
            prev = merged.get(key)
            if prev is None or ent.get("k", 0) >= prev.get("k", 0):
                merged[key] = ent

    out_path = run_dir / "preview-dist.json"
    out_path.write_text(json.dumps({"format": "nc2000-preview-dist-v1", "entries": merged}, indent=1))
    print(json.dumps({
        "cmd": "preview-dist",
        "dir": str(run_dir),
        "rows": len(rows),
        "chunks": len(chunks),
        "entries": len(merged),
        "k": k,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "dispatch": tel,
    }))
    return out_path


def cmd_census_run(args: argparse.Namespace) -> None:
    teams, _, shells = load_v3_catalog(Path(args.catalog) if args.catalog else None)
    pools = {
        "teams": [{"id": f"v3:{t['id']}", "sets": t["sets"]} for t in teams],
        "mixtures": shells,
    }
    # Schedule all unordered pairs (i <= j) across all 33 teams so every team faces all 32 shells.
    # For each row (A, B), both side=0 (A vs shell(B)) and side=1 (B vs shell(A)) are computed.
    def mix_for(tid: str) -> str:
        return "shell:s0053-lefties" if tid in ("s0053-lefties-hpi", "s0053-lefties-dp") else f"shell:{tid}"

    rows: list[dict[str, Any]] = []
    idx = 0
    for i in range(len(teams)):
        for j in range(i, len(teams)):
            a = teams[i]["id"]
            b = teams[j]["id"]
            rows.append({
                "block": idx,
                "cell": {"kind": "census_v3", "a": a, "b": b},
                "rowSeat": 0,
                "ids": [f"v3:{a}", f"v3:{b}"],
                "seeds": [u32_hash(f"cen-s0:{a}:{b}"), u32_hash(f"cen-s1:{a}:{b}")],
                "battleSeed": u32_hash(f"cen-b:{a}:{b}"),
                "mixtures": [mix_for(a), mix_for(b)],
            })
            idx += 1

    run_dir = WORK / args.name
    run_preview_dist_chunks(
        run_dir,
        pools,
        rows,
        k=args.k,
        rows_per_chunk=args.rows_per_chunk,
        preview_iters=args.preview_iters,
        local_only=args.local_only,
    )


def summarize_preview_dist(dist_path: Path, teams_by_prefixed_id: dict[str, list[dict[str, Any]]], target_ids: list[str] | None = None) -> list[dict[str, Any]]:
    doc = json.loads(dist_path.read_text())
    entries = doc.get("entries", {})
    # Group by (own_id, opp_mix); if both sides exist for same (own, opp_mix), pool counts cleanly.
    by_own_opp: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for ent in entries.values():
        own = ent.get("own")
        opp = ent.get("opp_mix")
        if not own or not opp:
            continue
        if target_ids is not None and own not in target_ids:
            continue
        for act, cnt in ent.get("counts", {}).items():
            by_own_opp[own][opp][act] += int(cnt)

    report: list[dict[str, Any]] = []
    ordered_owns = target_ids if target_ids is not None else sorted(by_own_opp.keys())
    for own in ordered_owns:
        opps = by_own_opp.get(own, {})
        if not opps:
            continue
        # Macro-average across opponent shells so each opponent shell has equal weight!
        slot_sum = [0.0] * 6
        lead_sum = [0.0] * 6
        slot_max_opp = [0.0] * 6
        slot_nz_opps = [0] * 6
        combo_weights: Counter[str] = Counter()
        total_samples = 0
        for opp, cnts in opps.items():
            n_opp = sum(cnts.values())
            if n_opp <= 0:
                continue
            total_samples += n_opp
            opp_slot = [0.0] * 6
            for act, c in cnts.items():
                w = c / n_opp
                combo_weights[act] += w
                nums = [int(x.strip()) for x in act.replace("team", "").strip().split(",")]
                if nums:
                    lead_sum[nums[0] - 1] += w
                for slot in nums:
                    if 1 <= slot <= 6:
                        opp_slot[slot - 1] += w
            for si in range(6):
                slot_sum[si] += opp_slot[si]
                if opp_slot[si] > slot_max_opp[si]:
                    slot_max_opp[si] = opp_slot[si]
                if opp_slot[si] > 0.0:
                    slot_nz_opps[si] += 1

        n_opps = len(opps)
        slot_rates = [slot_sum[i] / n_opps for i in range(6)]
        lead_rates = [lead_sum[i] / n_opps for i in range(6)]
        min_rate = min(slot_rates)
        min_idx = slot_rates.index(min_rate)
        sets = teams_by_prefixed_id.get(own, [])
        slot_details = []
        for si in range(6):
            sp = sets[si]["species"] if si < len(sets) else f"s{si+1}"
            lv = sets[si]["level"] if si < len(sets) else 0
            it = sets[si]["item"] if si < len(sets) else ""
            slot_details.append({
                "slot": si + 1,
                "species": sp,
                "level": lv,
                "item": it,
                "pick_pct": round(100.0 * slot_rates[si], 2),
                "lead_pct": round(100.0 * lead_rates[si], 2),
                "nz_opps": slot_nz_opps[si],
                "max_opp_pct": round(100.0 * slot_max_opp[si], 1),
            })
        # Effective number of combos (exp entropy)
        tot_w = sum(combo_weights.values())
        ent = -sum((w / tot_w) * math.log(w / tot_w) for w in combo_weights.values() if w > 0) if tot_w > 0 else 0.0
        report.append({
            "id": own.removeprefix("v3:"),
            "raw_id": own,
            "n_opps": n_opps,
            "total_samples": total_samples,
            "min_pick_pct": round(100.0 * min_rate, 2),
            "min_slot": min_idx + 1,
            "min_species": slot_details[min_idx]["species"],
            "n_lt5": sum(1 for r in slot_rates if r < 0.05),
            "n_lt10": sum(1 for r in slot_rates if r < 0.10),
            "n_lt15": sum(1 for r in slot_rates if r < 0.15),
            "eff_combos": round(math.exp(ent), 2),
            "slots": slot_details,
        })
    return report


def cmd_census_analyze(args: argparse.Namespace) -> None:
    teams, _, _ = load_v3_catalog(Path(args.catalog) if args.catalog else None)
    by_pref = {f"v3:{t['id']}": t["sets"] for t in teams}
    dist_path = WORK / args.name / "preview-dist.json"
    rep = summarize_preview_dist(dist_path, by_pref)
    rep.sort(key=lambda r: (r["min_pick_pct"], -r["n_lt10"], -r["n_lt15"]))
    out_json = WORK / args.name / "census-report.json"
    out_json.write_text(json.dumps(rep, indent=2))
    print(f"=== EXACT PREVIEW OKIMONO CENSUS ({len(rep)} teams, {WORK.name}/{args.name}) ===")
    print(f"{'Team':31s} | {'Opps':>4s} | {'s1':>5s} {'s2':>5s} {'s3':>5s} {'s4':>5s} {'s5':>5s} {'s6':>5s} | {'Min%':>5s} {'<5':>2s} {'<10':>3s} | Okimono Slot(s)")
    print("-" * 126)
    for r in rep:
        rates = " ".join(f"{s['pick_pct']:5.1f}" for s in r["slots"])
        bad = [f"s{s['slot']}:{s['species']}({s['pick_pct']:.1f}%,{s['nz_opps']}/{r['n_opps']})" for s in r["slots"] if s["pick_pct"] < 15.0]
        if not bad:
            ms = r["slots"][r["min_slot"] - 1]
            bad = [f"clean(min s{ms['slot']}:{ms['species']}={ms['pick_pct']:.1f}%)"]
        print(f"{r['id']:31s} | {r['n_opps']:4d} | {rates} | {r['min_pick_pct']:5.1f} {r['n_lt5']:2d} {r['n_lt10']:3d} | {', '.join(bad)}")


def cmd_preview_screen(args: argparse.Namespace) -> None:
    """Screen candidate mutated teams against a strong-pool opponent panel at exact 729k preview."""
    spec = json.loads(Path(args.spec).read_text())
    cat_path = Path(args.catalog) if args.catalog else (Path(spec["catalog"]) if spec.get("catalog") else None)
    teams, _, shells = load_v3_catalog(cat_path)
    panel_shells: list[str] = spec["panel_shells"]
    raw_cands: dict[str, list[dict[str, Any]]] = spec["candidates"]
    validated = canonicalize_and_validate_teams(raw_cands)

    # Save validated canonical sets back for downstream LIVE A/B reuse
    run_dir = WORK / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validated-candidates.json").write_text(json.dumps(validated, indent=1))

    pools_teams = [{"id": f"v3:{t['id']}", "sets": t["sets"]} for t in teams]
    mixtures = dict(shells)
    for cid, sets in validated.items():
        pools_teams.append({"id": cid, "sets": sets})
        mixtures[f"mut:{cid}"] = [{"id": cid, "weight": 1.0}]
    # Support optional multi-variant mixtures in spec
    for mname, mems in spec.get("mixtures", {}).items():
        mixtures[mname] = mems

    # Map panel shell -> representative team id
    def rep_team(sh: str) -> str:
        if sh == "shell:s0053-lefties":
            return "v3:s0053-lefties-hpi"
        return f"v3:{sh.removeprefix('shell:')}"

    rows: list[dict[str, Any]] = []
    idx = 0
    for cid in validated:
        for sh in panel_shells:
            rows.append({
                "block": idx,
                "cell": {"kind": "mut_preview", "cand": cid, "opp": sh},
                "rowSeat": 0,
                "ids": [cid, rep_team(sh)],
                "seeds": [u32_hash(f"mp-s0:{cid}:{sh}"), u32_hash(f"mp-s1:{cid}:{sh}")],
                "battleSeed": u32_hash(f"mp-b:{cid}:{sh}"),
                "mixtures": [f"mut:{cid}", sh],
            })
            idx += 1

    dist_path = run_preview_dist_chunks(
        run_dir,
        {"teams": pools_teams, "mixtures": mixtures},
        rows,
        k=args.k,
        rows_per_chunk=args.rows_per_chunk,
        preview_iters=args.preview_iters,
        local_only=args.local_only,
    )
    rep = summarize_preview_dist(dist_path, validated, target_ids=list(validated.keys()))
    (run_dir / "screen-report.json").write_text(json.dumps(rep, indent=2))
    print(f"=== {args.preview_iters // 1000}k MUTATION ACTIVATION SCREEN ({len(rep)} candidates vs {len(panel_shells)} shells) ===")
    for r in rep:
        rates = " ".join(f"{s['pick_pct']:5.1f}" for s in r["slots"])
        print(f"  {r['id']:34s} | min={r['min_pick_pct']:5.1f}% (<5:{r['n_lt5']},<10:{r['n_lt10']}) eff={r['eff_combos']:4.1f} | [{rates}]")


def cmd_live_ab(args: argparse.Namespace) -> None:
    """Run paired CRN strictly NO-CACHE LIVE (243k/729k) A/B evaluation + optional H2H."""
    spec = json.loads(Path(args.spec).read_text())
    cat_path = Path(args.catalog) if args.catalog else (Path(spec["catalog"]) if spec.get("catalog") else None)
    teams, _, shells = load_v3_catalog(cat_path)
    panel_shells: list[str] = spec["panel_shells"]
    pairs_per_opp: int = int(spec.get("pairs_per_opp", 2))
    h2h_pairs: int = int(spec.get("h2h_pairs", 0))
    seed_tag: str = spec.get("seed_tag", args.name)

    raw_cands: dict[str, list[dict[str, Any]]] = spec.get("candidates", {})
    validated = canonicalize_and_validate_teams(raw_cands) if raw_cands else {}

    pools_teams = [{"id": f"v3:{t['id']}", "sets": t["sets"]} for t in teams]
    mixtures = dict(shells)
    for cid, sets in validated.items():
        pools_teams.append({"id": cid, "sets": sets})
        mixtures[f"mut:{cid}"] = [{"id": cid, "weight": 1.0}]
    for mname, mems in spec.get("mixtures", {}).items():
        mixtures[mname] = mems

    def resolve_entry(name: str) -> tuple[list[tuple[str, float]], str]:
        if name in mixtures:
            mems = [(m["id"], float(m["weight"])) for m in mixtures[name]]
            return mems, name
        if f"shell:{name}" in mixtures:
            mems = [(m["id"], float(m["weight"])) for m in mixtures[f"shell:{name}"]]
            return mems, f"shell:{name}"
        if f"mut:{name}" in mixtures:
            return [(name, 1.0)], f"mut:{name}"
        raise KeyError(f"Unknown entry {name}")

    rows: list[dict[str, Any]] = []
    block = 0

    # Group matchups: each group = {"base": orig_name, "mutants": [m1, m2, ...]}
    # All members in a group share IDENTICAL (opp, opp_realized_id, pair_idx, battleSeed) for CRN pairing!
    for grp in spec.get("groups", []):
        gname = grp["name"]
        members = [grp["base"]] + list(grp["mutants"])
        for opp_idx, opp_sh in enumerate(panel_shells):
            opp_mems, opp_mix = resolve_entry(opp_sh)
            for pi in range(pairs_per_opp):
                opp_id = opp_mems[(opp_idx + pi) % len(opp_mems)][0]
                bseed = u32_hash(f"crn-battle:{seed_tag}:{gname}:{opp_sh}:p{pi}")
                for m in members:
                    m_mems, m_mix = resolve_entry(m)
                    m_id = m_mems[(opp_idx + pi) % len(m_mems)][0]
                    for row_seat in (0, 1):
                        if row_seat == 0:
                            ids = [m_id, opp_id]
                            mixs = [m_mix, opp_mix]
                        else:
                            ids = [opp_id, m_id]
                            mixs = [opp_mix, m_mix]
                        s0 = u32_hash(f"crn-s0:{seed_tag}:{gname}:{m}:{opp_sh}:p{pi}:r{row_seat}")
                        s1 = u32_hash(f"crn-s1:{seed_tag}:{gname}:{m}:{opp_sh}:p{pi}:r{row_seat}")
                        rows.append({
                            "block": block,
                            "cell": {
                                "kind": "live_ab",
                                "group": gname,
                                "subject": m,
                                "is_base": m == grp["base"],
                                "opp": opp_sh,
                                "pair": pi,
                            },
                            "rowSeat": row_seat,
                            "ids": ids,
                            "seeds": [s0, s1],
                            "battleSeed": bseed,
                            "mixtures": mixs,
                        })
                    block += 1

        # Optional direct H2H: each mutant vs base
        if h2h_pairs > 0:
            base_mems, base_mix = resolve_entry(grp["base"])
            for mut in grp["mutants"]:
                mut_mems, mut_mix = resolve_entry(mut)
                for pi in range(h2h_pairs):
                    b_id = base_mems[pi % len(base_mems)][0]
                    m_id = mut_mems[pi % len(mut_mems)][0]
                    bseed = u32_hash(f"h2h-battle:{seed_tag}:{gname}:{mut}:p{pi}")
                    for row_seat in (0, 1):
                        ids = [m_id, b_id] if row_seat == 0 else [b_id, m_id]
                        mixs = [mut_mix, base_mix] if row_seat == 0 else [base_mix, mut_mix]
                        s0 = u32_hash(f"h2h-s0:{seed_tag}:{gname}:{mut}:p{pi}:r{row_seat}")
                        s1 = u32_hash(f"h2h-s1:{seed_tag}:{gname}:{mut}:p{pi}:r{row_seat}")
                        rows.append({
                            "block": block,
                            "cell": {
                                "kind": "live_h2h",
                                "group": gname,
                                "subject": mut,
                                "opp": grp["base"],
                                "pair": pi,
                            },
                            "rowSeat": row_seat,
                            "ids": ids,
                            "seeds": [s0, s1],
                            "battleSeed": bseed,
                            "mixtures": mixs,
                        })
                    block += 1

    # Also support standalone extra pairs (e.g. holdout matchups)
    for ex in spec.get("extra_matchups", []):
        a_mems, a_mix = resolve_entry(ex["a"])
        b_mems, b_mix = resolve_entry(ex["b"])
        npairs = int(ex["pairs"])
        a_off = int(ex.get("a_offset", 0))
        b_off = int(ex.get("b_offset", 0))
        for pi in range(npairs):
            aid = a_mems[(pi + a_off) % len(a_mems)][0]
            bid = b_mems[(pi + b_off) % len(b_mems)][0]
            bseed = u32_hash(f"ex-battle:{seed_tag}:{ex['a']}:{ex['b']}:p{pi}")
            for row_seat in (0, 1):
                ids = [aid, bid] if row_seat == 0 else [bid, aid]
                mixs = [a_mix, b_mix] if row_seat == 0 else [b_mix, a_mix]
                s0 = u32_hash(f"ex-s0:{seed_tag}:{ex['a']}:{ex['b']}:p{pi}:r{row_seat}")
                s1 = u32_hash(f"ex-s1:{seed_tag}:{ex['a']}:{ex['b']}:p{pi}:r{row_seat}")
                rows.append({
                    "block": block,
                    "cell": {"kind": "extra", "group": ex.get("group", "extra"), "subject": ex["a"], "opp": ex["b"], "pair": pi},
                    "rowSeat": row_seat,
                    "ids": ids,
                    "seeds": [s0, s1],
                    "battleSeed": bseed,
                    "mixtures": mixs,
                })
            block += 1

    run_dir = WORK / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    pools_path = run_dir / "pools.json"
    sched_path = run_dir / "schedule.jsonl"
    pools_path.write_text(json.dumps({"teams": pools_teams, "mixtures": mixtures}, separators=(",", ":")))
    sched_path.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")
    (run_dir / "meta.json").write_text(json.dumps({
        "name": args.name,
        "spec": str(args.spec),
        "total_games": len(rows),
        "preview_iters": args.preview_iters,
        "battle_iters": args.battle_iters,
        "battle_mode": args.battle_mode,
        "strict_no_cache_live": True,
    }, indent=2))

    chunk_games = max(2, args.chunk_games + (args.chunk_games % 2))
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for ci, start in enumerate(range(0, len(rows), chunk_games)):
        batch = rows[start : start + chunk_games]
        c_sched = chunks_dir / f"chunk-{ci:03d}.jsonl"
        c_out = chunks_dir / f"chunk-{ci:03d}-out"
        c_done = chunks_dir / f"chunk-{ci:03d}.done.json"
        if not c_sched.exists():
            c_sched.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in batch) + "\n")
        chunks.append({
            "ci": ci,
            "sched": c_sched,
            "out": c_out,
            "done": c_done,
            "n_rows": len(batch),
        })

    t0 = time.monotonic()
    tel = MINI2.run_chunks_dual_node(
        chunks,
        binary=DEFAULT_BIN,
        pools_path=pools_path,
        local_workers=10,
        mini2_workers=12,
        use_mini2=True,
        colab_nodes="off" if args.local_only else None,
        preview_iters=args.preview_iters,
        battle_iters=args.battle_iters,
        battle_mode=args.battle_mode,
        preview_cache=False,
        preview_dist_cache=None,
        preview_dist_k=None,
        preview_dist_only=False,
    )

    # Collect results.jsonl and assert 100% live no-cache hygiene
    out_lines: list[str] = []
    src_counts: Counter[str] = Counter()
    for ch in chunks:
        files = sorted(
            [p for p in ch["out"].glob("*.json") if p.name not in ("manifest.json", "preview-dist-new.json")],
            key=lambda p: int(p.stem),
        )
        assert len(files) == ch["n_rows"], f"Chunk {ch['ci']} expected {ch['n_rows']} got {len(files)}"
        for fp in files:
            rec = json.loads(fp.read_text())
            for ps in rec.get("preview", []):
                src_counts[ps.get("source", "unknown")] += 1
            out_lines.append(json.dumps(rec, separators=(",", ":")))

    results_path = run_dir / "results.jsonl"
    results_path.write_text("\n".join(out_lines) + "\n")
    elapsed = round(time.monotonic() - t0, 1)
    print(json.dumps({
        "cmd": "live-ab",
        "dir": str(run_dir),
        "games": len(out_lines),
        "preview_sources": dict(src_counts),
        "elapsed_s": elapsed,
        "games_per_hr": round(len(out_lines) * 3600.0 / max(1.0, elapsed), 1),
        "dispatch": tel,
    }))


def outcome_score(outcome: str, seat: int) -> float:
    o = str(outcome).lower()
    if o in ("p1", "0", "seat0", "win", "p0"):
        return 1.0 if seat == 0 else 0.0
    if o in ("p2", "1", "seat1", "loss"):
        return 0.0 if seat == 0 else 1.0
    return 0.5


def mean_se(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return mu, math.sqrt(var / n)


def cmd_live_analyze(args: argparse.Namespace) -> None:
    run_dir = WORK / args.name
    results_path = run_dir / "results.jsonl"
    records = [json.loads(l) for l in results_path.open() if l.strip()]

    src_counts: Counter[str] = Counter()
    # Structure: ab[(group, subject)][(opp, pair)] -> [score_seat0, score_seat1]
    ab_pairs: dict[tuple[str, str], dict[tuple[str, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    # Per-game CRN key: (opp, pair, rowSeat) -> (score, slot_set) for Simpson-free InΔ/OutΔ calculation
    ab_game_map: dict[tuple[str, str], dict[tuple[str, int, int], tuple[float, set[int]]]] = defaultdict(dict)
    ab_picks: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * 6)
    ab_slot_pk_sum: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0] * 6)
    ab_slot_unpk_sum: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0] * 6)
    ab_slot_unpk_n: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * 6)
    ab_games: dict[tuple[str, str], int] = defaultdict(int)
    ab_opp_scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_base: dict[str, str] = {}
    group_muts: dict[str, list[str]] = defaultdict(list)

    h2h_pairs: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        for ps in r.get("preview", []):
            src_counts[ps.get("source", "?")] += 1
        cell = r.get("cell") or {}
        kind = cell.get("kind")
        rs = int(r["rowSeat"])
        sc = outcome_score(r["outcome"], rs)
        pstr = r["preview"][rs]["input"]
        slots = [int(x.strip()) for x in pstr.replace("team", "").strip().split(",")]

        if kind in ("live_ab", "extra"):
            g = cell["group"]
            subj = cell["subject"]
            opp = cell["opp"]
            pi = int(cell["pair"])
            if cell.get("is_base"):
                group_base[g] = subj
            elif subj not in group_muts[g] and kind == "live_ab":
                group_muts[g].append(subj)
            ab_pairs[(g, subj)][(opp, pi)].append(sc)
            ab_opp_scores[(g, subj)][opp].append(sc)
            ab_games[(g, subj)] += 1
            slot_set = {sl for sl in slots if 1 <= sl <= 6}
            ab_game_map[(g, subj)][(opp, pi, rs)] = (sc, slot_set)
            for idx_s in range(6):
                if (idx_s + 1) in slot_set:
                    ab_picks[(g, subj)][idx_s] += 1
                    ab_slot_pk_sum[(g, subj)][idx_s] += sc
                else:
                    ab_slot_unpk_n[(g, subj)][idx_s] += 1
                    ab_slot_unpk_sum[(g, subj)][idx_s] += sc
        elif kind == "live_h2h":
            g = cell["group"]
            subj = cell["subject"]
            pi = int(cell["pair"])
            h2h_pairs[(g, subj)][pi].append(sc)

    summary: dict[str, Any] = {
        "name": args.name,
        "total_games": len(records),
        "preview_sources": dict(src_counts),
        "groups": {},
    }
    print(f"=== STRICT NO-CACHE LIVE A/B REPORT: {args.name} ({len(records)}g, sources={dict(src_counts)}) ===")
    groups = sorted(set(g for g, _ in ab_pairs.keys()))
    for g in groups:
        base = group_base.get(g)
        subjs = ([base] if base else []) + [m for m in group_muts.get(g, []) if m != base]
        # Also catch extra-only subjects
        for gg, ss in sorted(ab_pairs.keys()):
            if gg == g and ss not in subjs:
                subjs.append(ss)
        print(f"\n[Group: {g}]")
        base_pm = {k: sum(v) / len(v) for k, v in ab_pairs[(g, base)].items()} if base else {}
        g_out = []
        for s in subjs:
            pm = {k: sum(v) / len(v) for k, v in ab_pairs[(g, s)].items()}
            vals = list(pm.values())
            mu, se = mean_se(vals)
            ng = ab_games[(g, s)]
            pk_rates = [100.0 * c / ng for c in ab_picks[(g, s)]] if ng > 0 else [0.0] * 6
            min_pk = min(pk_rates)
            opp_wrs = {o: sum(v) / len(v) for o, v in ab_opp_scores[(g, s)].items()}
            worst_opp = min(opp_wrs.items(), key=lambda kv: kv[1]) if opp_wrs else ("", 0.0)
            n_ge50 = sum(1 for v in opp_wrs.values() if v >= 0.5)
            n_lt35 = sum(1 for v in opp_wrs.values() if v < 0.35)

            delta_str = "BASE"
            d_mu = d_se = z_val = None
            slot_pk_delta: list[float | None] | None = None
            slot_unpk_delta: list[float | None] | None = None
            slot_pk_contrib: list[float | None] | None = None
            slot_unpk_contrib: list[float | None] | None = None
            if base and s != base:
                common = sorted(set(pm.keys()) & set(base_pm.keys()))
                diffs = [pm[k] - base_pm[k] for k in common]
                d_mu, d_se = mean_se(diffs)
                z_val = d_mu / d_se if d_se > 0 else 0.0
                delta_str = f"{100*d_mu:+5.1f}pp ±{100*d_se:4.1f} (z={z_val:+.2f})"
                cg_map = ab_game_map[(g, s)]
                bg_map = ab_game_map[(g, base)]
                common_g = sorted(set(cg_map.keys()) & set(bg_map.keys()))
                if common_g:
                    ng_c = len(common_g)
                    slot_pk_delta = []
                    slot_unpk_delta = []
                    slot_pk_contrib = []
                    slot_unpk_contrib = []
                    for idx_s in range(6):
                        in_d = [cg_map[k][0] - bg_map[k][0] for k in common_g if (idx_s + 1) in cg_map[k][1]]
                        out_d = [cg_map[k][0] - bg_map[k][0] for k in common_g if (idx_s + 1) not in cg_map[k][1]]
                        slot_pk_delta.append(round(sum(in_d) / len(in_d), 4) if in_d else None)
                        slot_unpk_delta.append(round(sum(out_d) / len(out_d), 4) if out_d else None)
                        slot_pk_contrib.append(round(sum(in_d) / ng_c, 4))
                        slot_unpk_contrib.append(round(sum(out_d) / ng_c, 4))

            h2h_str = ""
            h2h_mu = h2h_se = None
            if (g, s) in h2h_pairs:
                hvals = [sum(v) / len(v) for v in h2h_pairs[(g, s)].values()]
                h2h_mu, h2h_se = mean_se(hvals)
                h2h_str = f" | H2H={100*h2h_mu:5.1f}%±{100*h2h_se:4.1f}({len(hvals)*2}g)"

            pk_str = " ".join(f"{x:4.0f}" for x in pk_rates)
            print(
                f"  {s:32s} | {ng:3d}g WR={100*mu:5.1f}%±{100*se:4.1f} | {delta_str:25s}{h2h_str}"
                f" | >=50:{n_ge50:2d}/{len(opp_wrs)} <35:{n_lt35:2d} | minPk={min_pk:4.1f}% [{pk_str}]"
            )
            g_out.append({
                "subject": s,
                "is_base": s == base,
                "games": ng,
                "wr": round(mu, 4),
                "se": round(se, 4),
                "paired_delta": round(d_mu, 4) if d_mu is not None else None,
                "paired_se": round(d_se, 4) if d_se is not None else None,
                "paired_z": round(z_val, 3) if z_val is not None else None,
                "h2h_wr": round(h2h_mu, 4) if h2h_mu is not None else None,
                "h2h_se": round(h2h_se, 4) if h2h_se is not None else None,
                "min_pick_pct": round(min_pk, 2),
                "slot_picks_pct": [round(x, 2) for x in pk_rates],
                "slot_picked_n": list(ab_picks[(g, s)]),
                "slot_picked_wr": [
                    round(ab_slot_pk_sum[(g, s)][i] / ab_picks[(g, s)][i], 4) if ab_picks[(g, s)][i] > 0 else None
                    for i in range(6)
                ],
                "slot_unpicked_wr": [
                    round(ab_slot_unpk_sum[(g, s)][i] / ab_slot_unpk_n[(g, s)][i], 4) if ab_slot_unpk_n[(g, s)][i] > 0 else None
                    for i in range(6)
                ],
                "slot_picked_delta": slot_pk_delta,
                "slot_unpicked_delta": slot_unpk_delta,
                "slot_picked_contrib": slot_pk_contrib,
                "slot_unpicked_contrib": slot_unpk_contrib,
                "ge50": n_ge50,
                "lt35": n_lt35,
                "n_opps": len(opp_wrs),
                "worst_opp": {"opp": worst_opp[0], "wr": round(worst_opp[1], 4)},
                "per_opp": {o: round(v, 4) for o, v in sorted(opp_wrs.items())},
            })
        summary["groups"][g] = g_out

    (run_dir / "ab-report.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=None, help="Override work dir (default: target/nash-work/okimono-v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_wd(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--work-dir", default=argparse.SUPPRESS, help="Override work dir (default: target/nash-work/okimono-v1)")

    p = sub.add_parser("census-run")
    add_wd(p)
    p.add_argument("--name", default="census-v3-k4")
    p.add_argument("--catalog", default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--rows-per-chunk", type=int, default=10)
    p.add_argument("--preview-iters", type=int, default=DEFAULT_PREVIEW_ITERS)
    p.add_argument("--local-only", action="store_true")
    p.set_defaults(fn=cmd_census_run)

    p = sub.add_parser("census-analyze")
    add_wd(p)
    p.add_argument("--name", default="census-v3-k4")
    p.add_argument("--catalog", default=None)
    p.set_defaults(fn=cmd_census_analyze)

    p = sub.add_parser("preview-screen")
    add_wd(p)
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--catalog", default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--rows-per-chunk", type=int, default=8)
    p.add_argument("--preview-iters", type=int, default=DEFAULT_PREVIEW_ITERS)
    p.add_argument("--local-only", action="store_true")
    p.set_defaults(fn=cmd_preview_screen)

    p = sub.add_parser("live-ab")
    add_wd(p)
    p.add_argument("--name", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--catalog", default=None)
    p.add_argument("--chunk-games", type=int, default=16)
    p.add_argument("--preview-iters", type=int, default=DEFAULT_PREVIEW_ITERS)
    p.add_argument("--battle-iters", type=int, default=DEFAULT_BATTLE_ITERS)
    p.add_argument("--battle-mode", default="adaptive")
    p.add_argument("--local-only", action="store_true")
    p.set_defaults(fn=cmd_live_ab)

    p = sub.add_parser("live-analyze")
    add_wd(p)
    p.add_argument("--name", required=True)
    p.set_defaults(fn=cmd_live_analyze)

    args = ap.parse_args()
    if getattr(args, "work_dir", None):
        WORK = Path(args.work_dir)
    args.fn(args)


if __name__ == "__main__":
    main()
