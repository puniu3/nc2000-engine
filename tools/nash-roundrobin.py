#!/usr/bin/env python3
"""Shell-level multi-variant (S, q) Round-Robin scheduler, runner, and Nash analyzer.

Enforces the exact disclosed-pool game invariants:
  1. Single-shell rows: every mixture row represents one public Turn-0 shell S
     with conditional variant weights q (sum q = 1, all q_v > 0).
  2. 1 shell = 1 row: all rows in a matrix have pairwise-distinct public shells
     (verified by canonical shell signature AND byte-identical/distinct Turn-0
     PlayerChannel frames via Node WASM when available).
  3. Exact conditional-profile ledger reuse: archived production games from
     `target/nash-ledger.jsonl` are reused ONLY when canonical `(profile_hash_A,
     profile_hash_B)` match identically (exact shell + exact sets + exact q).
  4. Unbiased stratified conditional estimation:
       M_AB = sum_{a,b} q_A(a) q_B(b) * mean(X_{ab})
       Var(M_AB) = sum_{a,b} (q_A(a) q_B(b))^2 * s^2_{ab} / n_{ab}
     with conservative variance fallback when n_{ab} == 1 and explicit reporting
     of `covered_mass` and `min_cell_pairs`.
  5. Per-variant within-shell diagnostics (`by_variant`) against the solved
     Nash support P*, plus RM+ bootstrap uncertainty & active-learning next_cells.

Subcommands:
  verify     Validate legality, weights, single-shell mixtures, distinct shells,
             and Turn-0 PlayerChannel frames.
  prepare    Build a ledger-aware deficit schedule.jsonl + meta.json.
  run        Execute schedule.jsonl locally in resumable chunk checkpoints via
             native CS-PGO `nash_refresh`, optionally merging into a ledger.
  analyze    Compute exact conditional-weighted matrix, RM+ Nash equilibrium,
             bootstrap stability, active-learning priorities, and by_variant EV.
  self-test  Run offline unit tests on invariants, deficit allocation, and math.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATS = SourceFileLoader("nash_stats", str(ROOT / "tools/nash-stats.py")).load_module()
LEDGER = SourceFileLoader("nash_ledger", str(ROOT / "tools/nash-ledger.py")).load_module()
MINI2 = SourceFileLoader("mini2_nash", str(ROOT / "tools/mini2_nash.py")).load_module()

DEFAULT_BIN = ROOT / "target/native-pgo/release/examples/nash_refresh"
NODE_WASM_PKG = ROOT / "crates/wasm/pkg-node/nc2000_wasm.js"


def u32_hash(tag: str) -> int:
    return int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:4], "big")


def seat_score(outcome: str, seat: int) -> float:
    if outcome in ("tie", "cap", "draw"):
        return 0.5
    win_seat = 0 if outcome == "p1" else 1
    return 1.0 if win_seat == seat else 0.0


def canonical_mixture_profile(
    name: str,
    members: list[dict[str, Any]],
    teams: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Validate that `members` form a legal single-shell conditional profile (S, q)."""
    if not members:
        raise ValueError(f"mixture {name!r} is empty")
    total_w = 0.0
    sigs = set()
    by_set: dict[str, tuple[str, float]] = {}
    for m in members:
        tid = m["id"]
        w = float(m["weight"])
        if w <= 0.0 or not math.isfinite(w):
            raise ValueError(f"mixture {name!r} member {tid!r} has non-positive weight {w}")
        if tid not in teams:
            raise ValueError(f"mixture {name!r} references unknown team id {tid!r}")
        total_w += w
        sets = teams[tid]
        sigs.add(LEDGER.shell_signature(sets))
        sh = LEDGER.sha256_json(sets)
        if sh in by_set:
            prev_id, prev_w = by_set[sh]
            by_set[sh] = (min(prev_id, tid), prev_w + w)
        else:
            by_set[sh] = (tid, w)

    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"mixture {name!r} weights sum to {total_w:.9f}, expected 1.0")
    if len(sigs) != 1:
        raise ValueError(
            f"mixture {name!r} spans {len(sigs)} distinct public shells; "
            f"each Round-Robin row MUST be a single public shell (S, q)"
        )

    sig = next(iter(sigs))
    # Normalize weights cleanly to 1.0 (matching tools/nash-ledger.py canonicalization)
    variants = []
    for set_hash, (tid, w) in sorted(by_set.items(), key=lambda kv: kv[1][0]):
        variants.append({
            "id": tid,
            "set_hash": set_hash,
            "weight": w / total_w,
        })
    canon = sorted((v["set_hash"], round(v["weight"], 8)) for v in variants)
    profile_hash = LEDGER.sha256_json({"shell": sig, "cond": canon})
    shell_hash = LEDGER.sha256_json(sig)
    return {
        "name": name,
        "shell_hash": shell_hash,
        "profile_hash": profile_hash,
        "species": [s[0] for s in sig],
        "variants": variants,
    }


def validate_profiles(
    names: list[str],
    pools_path: Path,
    *,
    verify_frames: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Load pools, enforce single-shell rows + pairwise-distinct shells + optional Turn-0 frame check."""
    base_teams, _ = LEDGER.load_catalog_and_mixtures(None)
    extra_doc = json.loads(pools_path.read_text())
    dup_base = [t["id"] for t in extra_doc.get("teams", []) if t["id"] in base_teams]
    if dup_base:
        raise ValueError(
            f"pools file {pools_path} repeats base team IDs already loaded by nash_refresh: "
            f"{dup_base[:5]} (omit base teams from extra.teams)"
        )
    teams, mixtures = LEDGER.load_catalog_and_mixtures(pools_path)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate mixture names in {names}")

    profiles: list[dict[str, Any]] = []
    seen_shells: dict[str, str] = {}
    for name in names:
        if name not in mixtures:
            raise ValueError(f"mixture {name!r} not found in {pools_path}")
        prof = canonical_mixture_profile(name, mixtures[name], teams)
        sh = prof["shell_hash"]
        if sh in seen_shells:
            raise ValueError(
                f"1-shell=1-row violation: mixtures {seen_shells[sh]!r} and {name!r} "
                f"share identical public shell_hash {sh} ({prof['species']})"
            )
        seen_shells[sh] = name
        profiles.append(prof)

    if verify_frames:
        if not NODE_WASM_PKG.exists():
            raise RuntimeError(
                f"--verify-frames requested but {NODE_WASM_PKG} not found; "
                f"build with `crates/wasm/build.sh nodejs` or pass --no-verify-frames"
            )
        verify_turn0_frames_node(pools_path, profiles)

    return teams, profiles


def verify_turn0_frames_node(pools_path: Path, profiles: list[dict[str, Any]]) -> None:
    """Validate team legality, within-row byte-identical Turn-0 frames, and across-row frame distinctness."""
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = process.argv[1];
const poolsPath = process.argv[2];
const spec = JSON.parse(process.argv[3]);
const wasm = require(path.join(root, 'crates/wasm/pkg-node/nc2000_wasm.js'));

const catalog = new Map();
for (const rel of [
  'data/kodoku-beq-v1/novel_evolved_teams.json',
  'data/team-pool-v2/team-pool.json',
  'data/kodoku-r3/conclusion-equilibrium-deme.json',
]) {
  const doc = JSON.parse(fs.readFileSync(path.join(root, rel), 'utf8'));
  for (const t of doc.teams || []) catalog.set(t.id, t.sets);
}
const extra = JSON.parse(fs.readFileSync(poolsPath, 'utf8'));
for (const t of extra.teams || []) catalog.set(t.id, t.sets);

const dex = new wasm.Dex();
const validator = new wasm.Validator(dex);
const oppRef = JSON.stringify(catalog.get('sample-07'));
const rowFramesBySide = [new Map(), new Map()];

for (const row of spec) {
  for (const v of row.variants) {
    const sets = catalog.get(v.id);
    assert(sets, `missing team sets for ${v.id}`);
    const res = JSON.parse(validator.validateTeam(JSON.stringify(sets)));
    assert(res.ok, `illegal team ${v.id}: ${JSON.stringify(res)}`);
  }
  for (const side of [0, 1]) {
    const frames = new Set();
    for (const v of row.variants) {
      const own = JSON.stringify(catalog.get(v.id));
      const teams = side === 0 ? [oppRef, own] : [own, oppRef];
      const b = new wasm.Battle(dex, teams[0], teams[1], wasm.deriveBattleSeed(901));
      const c = new wasm.PlayerChannel(side);
      frames.add(c.frame(b));
      c.free();
      b.free();
    }
    assert.equal(frames.size, 1, `non-identical Turn-0 frames inside ${row.name} (side ${side})`);
    const frame = [...frames][0];
    if (rowFramesBySide[side].has(frame)) {
      throw new Error(
        `1-shell=1-row frame collision on side ${side}: ${rowFramesBySide[side].get(frame)} and ${row.name}`
      );
    }
    rowFramesBySide[side].set(frame, row.name);
  }
}
validator.free();
dex.free();
console.log(JSON.stringify({ok: true, rows: spec.length}));
"""
    spec_payload = json.dumps([
        {"name": p["name"], "variants": [{"id": v["id"]} for v in p["variants"]]}
        for p in profiles
    ])
    proc = subprocess.run(
        ["node", "-e", script, str(ROOT), str(pools_path), spec_payload],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Turn-0 frame/legality verification failed:\n{proc.stderr or proc.stdout}")


def load_ledger_pair_records(
    ledger_paths: list[Path],
    target_profile_hashes: set[str],
) -> dict[tuple[str, str, str, str, int, tuple[int, int]], dict[int, float]]:
    """Load game records matching pairs of profiles in `target_profile_hashes`.

    Returns map:
      canonical_pair_key = (prof_lo, hash_lo, prof_hi, hash_hi, battle_seed, sorted_seeds)
      -> {seat_of_lo: score_of_lo}
    where ordering is lexicographic on `(prof_a, hash_a) < (prof_b, hash_b)` (note
    prof_lo != prof_hi is guaranteed for off-diagonal distinct shells).
    """
    pairs: dict[tuple[str, str, str, str, int, tuple[int, int]], dict[int, float]] = defaultdict(dict)
    for lp in ledger_paths:
        if not lp.exists():
            continue
        with lp.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                p0, p1 = r["profiles"]
                if p0 == p1 or p0 not in target_profile_hashes or p1 not in target_profile_hashes:
                    continue
                h0, h1 = r["realized_hashes"]
                bs = int(r["battleSeed"])
                ss = tuple(sorted(int(x) for x in r["seeds"]))
                s0 = seat_score(r["outcome"], 0)
                if p0 < p1:
                    k = (p0, h0, p1, h1, bs, ss)
                    pairs[k][0] = s0
                else:
                    k = (p1, h1, p0, h0, bs, ss)
                    pairs[k][1] = 1.0 - s0
    return pairs


def load_result_pair_records(
    result_sources: list[Path],
    profiles_by_name: dict[str, dict[str, Any]],
    teams: dict[str, list[dict[str, Any]]],
    target_profile_hashes: set[str],
) -> tuple[dict[tuple[str, str, str, str, int, tuple[int, int]], dict[int, float]], int]:
    """Load result records from `*.json` directories or `.jsonl` files into canonical pair map."""
    pairs: dict[tuple[str, str, str, str, int, tuple[int, int]], dict[int, float]] = defaultdict(dict)
    raw_count = 0

    def ingest(r: dict[str, Any]) -> None:
        nonlocal raw_count
        mix = r.get("mixtures")
        if not mix or len(mix) != 2:
            return
        m0, m1 = mix[0], mix[1]
        if m0 not in profiles_by_name or m1 not in profiles_by_name:
            return
        p0 = profiles_by_name[m0]["profile_hash"]
        p1 = profiles_by_name[m1]["profile_hash"]
        if p0 == p1 or p0 not in target_profile_hashes or p1 not in target_profile_hashes:
            return
        id0, id1 = r["ids"]
        if id0 not in teams or id1 not in teams:
            return
        h0 = LEDGER.sha256_json(teams[id0])
        h1 = LEDGER.sha256_json(teams[id1])
        bs = int(r["battleSeed"])
        ss = tuple(sorted(int(x) for x in r["seeds"]))
        s0 = seat_score(r["outcome"], 0)
        raw_count += 1
        if p0 < p1:
            k = (p0, h0, p1, h1, bs, ss)
            pairs[k][0] = s0
        else:
            k = (p1, h1, p0, h0, bs, ss)
            pairs[k][1] = 1.0 - s0

    for src in result_sources:
        if src.is_dir():
            for p in sorted(src.glob("*.json")):
                if p.name == "manifest.json":
                    continue
                ingest(json.loads(p.read_text()))
        elif src.is_file():
            with src.open() as fh:
                for line in fh:
                    if line.strip():
                        ingest(json.loads(line))
        else:
            raise FileNotFoundError(f"result source not found: {src}")
    return pairs, raw_count


def stratified_target_counts(
    qa: list[dict[str, Any]],
    qb: list[dict[str, Any]],
    target_pairs: int,
    min_cell_pairs: int,
    auto_cover: bool,
) -> dict[tuple[str, str], int]:
    """Allocate `target_pairs` across `(va, vb)` with floor `min_cell_pairs` and proportional remainder."""
    cells = [(va["id"], vb["id"], va["weight"] * vb["weight"]) for va in qa for vb in qb]
    floor_total = min_cell_pairs * len(cells)
    if target_pairs < floor_total:
        if auto_cover:
            target_pairs = floor_total
        else:
            raise ValueError(
                f"target_pairs={target_pairs} < min_cell_pairs*n_cells={floor_total} "
                f"({len(qa)}x{len(qb)} cells); increase --pairs, pass --auto-cover, or use --mode mc"
            )
    rem = target_pairs - floor_total
    raw = [rem * w for _, _, w in cells]
    floors = [int(math.floor(x)) for x in raw]
    leftover = rem - sum(floors)
    order = sorted(range(len(cells)), key=lambda idx: (-(raw[idx] - floors[idx]), cells[idx][0], cells[idx][1]))
    for idx in order[:leftover]:
        floors[idx] += 1
    return {
        (cells[idx][0], cells[idx][1]): min_cell_pairs + floors[idx]
        for idx in range(len(cells))
    }


def parse_pair_overrides(raw: str | None, valid_names: set[str]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    if not raw:
        return out
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        lhs, rhs = tok.split("=", 1)
        a, b = lhs.split(":", 1)
        a, b = a.strip(), b.strip()
        if a not in valid_names or b not in valid_names or a == b:
            raise ValueError(f"invalid override {tok!r} for mixtures {sorted(valid_names)}")
        k = (a, b) if a < b else (b, a)
        out[k] = int(rhs.strip())
    return out


def cmd_verify(args: argparse.Namespace) -> None:
    _, profiles = validate_profiles(
        [s.strip() for s in args.mixtures.split(",") if s.strip()],
        args.pools,
        verify_frames=not args.no_verify_frames,
    )
    print(json.dumps({
        "ok": True,
        "verified_frames": not args.no_verify_frames,
        "n_shells": len(profiles),
        "shells": [
            {
                "name": p["name"],
                "shell_hash": p["shell_hash"],
                "profile_hash": p["profile_hash"],
                "n_variants": len(p["variants"]),
                "species": p["species"],
            }
            for p in profiles
        ],
    }))


def cmd_prepare(args: argparse.Namespace) -> None:
    names = [s.strip() for s in args.mixtures.split(",") if s.strip()]
    teams, profiles = validate_profiles(
        names,
        args.pools,
        verify_frames=not args.no_verify_frames,
    )
    prof_by_name = {p["name"]: p for p in profiles}
    target_hashes = {p["profile_hash"] for p in profiles}
    overrides = parse_pair_overrides(args.pair_overrides, set(names))

    # Load existing complete seat pairs from ledger(s)
    existing_raw = load_ledger_pair_records(args.ledger, target_hashes) if args.ledger else {}
    existing_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    existing_seed_keys: set[tuple[str, str, int, tuple[int, int]]] = set()
    for (p_lo, h_lo, p_hi, h_hi, bs, ss), seats in existing_raw.items():
        existing_seed_keys.add((p_lo, p_hi, bs, ss))
        if 0 in seats and 1 in seats:
            existing_counts[(p_lo, h_lo, p_hi, h_hi)] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sched: list[dict[str, Any]] = []
    matchup_plan: list[dict[str, Any]] = []
    blk = 0

    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            pa, pb = profiles[i], profiles[j]
            ma, mb = pa["name"], pb["name"]
            pha, phb = pa["profile_hash"], pb["profile_hash"]
            ov_key = (ma, mb) if ma < mb else (mb, ma)
            base_pairs = overrides.get(ov_key, args.pairs)

            # Count existing pairs per (va_id, vb_id)
            ex_by_cell: dict[tuple[str, str], int] = {}
            for va in pa["variants"]:
                for vb in pb["variants"]:
                    if pha < phb:
                        cnt = existing_counts.get((pha, va["set_hash"], phb, vb["set_hash"]), 0)
                    else:
                        cnt = existing_counts.get((phb, vb["set_hash"], pha, va["set_hash"]), 0)
                    ex_by_cell[(va["id"], vb["id"])] = cnt
            ex_total = sum(ex_by_cell.values())

            if args.mode == "stratified":
                if args.add_pairs is not None:
                    # Allocate `add_pairs` proportionally and add on top of existing
                    add_alloc = stratified_target_counts(
                        pa["variants"], pb["variants"], args.add_pairs, 0, True
                    )
                    deficits = {
                        k: max(add_alloc[k], max(0, args.min_cell_pairs - ex_by_cell[k]))
                        for k in ex_by_cell
                    }
                    target_alloc = {k: ex_by_cell[k] + deficits[k] for k in ex_by_cell}
                else:
                    target_alloc = stratified_target_counts(
                        pa["variants"],
                        pb["variants"],
                        base_pairs,
                        args.min_cell_pairs,
                        args.auto_cover,
                    )
                    deficits = {k: max(0, target_alloc[k] - ex_by_cell[k]) for k in ex_by_cell}

                new_pairs_here = 0
                for va in pa["variants"]:
                    for vb in pb["variants"]:
                        ida, idb = va["id"], vb["id"]
                        need = deficits[(ida, idb)]
                        ex_here = ex_by_cell[(ida, idb)]
                        for r in range(need):
                            rep_idx = ex_here + r
                            # Guarantee strictly fresh seeds not in ledger
                            salt = 0
                            while True:
                                suffix = "" if salt == 0 else f"/s{salt}"
                                tag = f"nc2000/rr/{args.seed_tag}/{pha}/{phb}/{ida}/{idb}/r{rep_idx}{suffix}"
                                sa = u32_hash(f"{tag}/a")
                                sb = u32_hash(f"{tag}/b")
                                bs = u32_hash(f"{tag}/battle")
                                sk = (min(pha, phb), max(pha, phb), bs, tuple(sorted((sa, sb))))
                                if sk not in existing_seed_keys:
                                    existing_seed_keys.add(sk)
                                    break
                                salt += 1
                            for row_seat in (0, 1):
                                if row_seat == 0:
                                    ids, seeds, mix = [ida, idb], [sa, sb], [ma, mb]
                                else:
                                    ids, seeds, mix = [idb, ida], [sb, sa], [mb, ma]
                                sched.append({
                                    "block": blk,
                                    "cell": {"a": ma, "b": mb, "va": ida, "vb": idb, "rep": rep_idx},
                                    "rowSeat": row_seat,
                                    "ids": ids,
                                    "seeds": seeds,
                                    "battleSeed": bs,
                                    "mixtures": mix,
                                })
                            blk += 1
                            new_pairs_here += 1
                matchup_plan.append({
                    "a": ma,
                    "b": mb,
                    "n_subcells": len(pa["variants"]) * len(pb["variants"]),
                    "existing_pairs": ex_total,
                    "target_pairs": sum(target_alloc.values()),
                    "new_pairs": new_pairs_here,
                    "new_games": 2 * new_pairs_here,
                    "min_subcell_final": min(ex_by_cell[k] + deficits[k] for k in ex_by_cell),
                })
            else:
                # Monte Carlo i.i.d. draw mode from q_A x q_B
                need_total = args.add_pairs if args.add_pairs is not None else max(0, base_pairs - ex_total)
                rng = random.Random(u32_hash(f"nc2000/rr-mc/{args.seed_tag}/{pha}/{phb}"))
                v_pairs = [(va["id"], vb["id"]) for va in pa["variants"] for vb in pb["variants"]]
                v_weights = [va["weight"] * vb["weight"] for va in pa["variants"] for vb in pb["variants"]]
                for r in range(need_total):
                    ida, idb = rng.choices(v_pairs, weights=v_weights, k=1)[0]
                    rep_idx = ex_total + r
                    salt = 0
                    while True:
                        suffix = "" if salt == 0 else f"/s{salt}"
                        tag = f"nc2000/rr-mc/{args.seed_tag}/{pha}/{phb}/r{rep_idx}{suffix}"
                        sa = u32_hash(f"{tag}/a")
                        sb = u32_hash(f"{tag}/b")
                        bs = u32_hash(f"{tag}/battle")
                        sk = (min(pha, phb), max(pha, phb), bs, tuple(sorted((sa, sb))))
                        if sk not in existing_seed_keys:
                            existing_seed_keys.add(sk)
                            break
                        salt += 1
                    for row_seat in (0, 1):
                        if row_seat == 0:
                            ids, seeds, mix = [ida, idb], [sa, sb], [ma, mb]
                        else:
                            ids, seeds, mix = [idb, ida], [sb, sa], [mb, ma]
                        sched.append({
                            "block": blk,
                            "cell": {"a": ma, "b": mb, "va": ida, "vb": idb, "rep": rep_idx},
                            "rowSeat": row_seat,
                            "ids": ids,
                            "seeds": seeds,
                            "battleSeed": bs,
                            "mixtures": mix,
                        })
                    blk += 1
                matchup_plan.append({
                    "a": ma,
                    "b": mb,
                    "n_subcells": len(pa["variants"]) * len(pb["variants"]),
                    "existing_pairs": ex_total,
                    "target_pairs": ex_total + need_total,
                    "new_pairs": need_total,
                    "new_games": 2 * need_total,
                })

    sched_path = args.out_dir / "schedule.jsonl"
    with sched_path.open("w") as fh:
        for row in sched:
            fh.write(json.dumps(row) + "\n")

    meta = {
        "format": "nc2000-rr-v2",
        "pools": str(args.pools.resolve()),
        "ledgers": [str(p.resolve()) for p in (args.ledger or [])],
        "seed_tag": args.seed_tag,
        "mode": args.mode,
        "pairs_per_matchup": args.pairs,
        "min_cell_pairs": args.min_cell_pairs,
        "mixtures": names,
        "profiles": profiles,
        "matchup_plan": matchup_plan,
        "existing_pairs_total": sum(m["existing_pairs"] for m in matchup_plan),
        "new_pairs_total": blk,
        "total_games": len(sched),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "n_shells": len(profiles),
        "mode": args.mode,
        "existing_pairs": meta["existing_pairs_total"],
        "new_pairs": meta["new_pairs_total"],
        "new_games": len(sched),
        "matchups": matchup_plan,
    }))


def cmd_run(args: argparse.Namespace) -> None:
    meta = json.loads((args.dir / "meta.json").read_text())
    sched_path = args.dir / "schedule.jsonl"
    rows = [l.strip() for l in sched_path.open() if l.strip()]
    if not rows:
        print(json.dumps({"dir": str(args.dir), "new_games": 0, "status": "already_complete"}))
        return

    binary = args.binary
    if not binary.exists():
        raise FileNotFoundError(f"nash_refresh binary not found: {binary}")

    chunk_size = max(2, args.chunk_games)
    if chunk_size % 2 == 1:
        chunk_size += 1  # keep seat pairs intact inside chunks

    chunks_dir = args.dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if not any(chunks_dir.glob("chunk-*.jsonl")):
        chunk_size = MINI2.recommend_chunk_size(
            chunk_size,
            colab_nodes=args.colab or None,
            preview_iters=args.preview_iters,
            battle_iters=args.battle_iters,
            battle_mode=args.battle_mode,
            preview_cache=not args.no_preview_cache,
        )
    merged_path = args.dir / "results.jsonl"

    env = os.environ.copy()
    env["NASH_POOLS"] = meta["pools"]
    env["NASH_PREVIEW_CACHE"] = "1" if not args.no_preview_cache else "0"
    if args.preview_iters is not None:
        env["NASH_PREVIEW_ITERS"] = str(args.preview_iters)
    if args.battle_iters is not None:
        env["NASH_BATTLE_ITERS"] = str(args.battle_iters)
    if args.battle_mode is not None:
        env["NASH_BATTLE_MODE"] = args.battle_mode

    t0 = time.monotonic()
    n_chunks = (len(rows) + chunk_size - 1) // chunk_size
    chunk_specs: list[dict[str, Any]] = []
    chunk_result_dirs: list[Path] = []
    for ci in range(n_chunks):
        c_rows = rows[ci * chunk_size : (ci + 1) * chunk_size]
        c_sched = chunks_dir / f"chunk-{ci:03d}.jsonl"
        c_out = chunks_dir / f"chunk-{ci:03d}-out"
        c_done = chunks_dir / f"chunk-{ci:03d}.done.json"
        if not c_sched.exists():
            c_sched.write_text("\n".join(c_rows) + "\n")
        chunk_specs.append({"ci": ci, "sched": c_sched, "out": c_out, "done": c_done, "n_rows": len(c_rows)})
        chunk_result_dirs.append(c_out)

    dispatch = MINI2.run_chunks_dual_node(
        chunk_specs,
        binary=binary,
        pools_path=Path(meta["pools"]),
        local_workers=args.workers,
        mini2_workers=args.mini2_workers,
        use_mini2=args.mini2,
        colab_nodes=args.colab or None,
        preview_iters=args.preview_iters,
        battle_iters=args.battle_iters,
        battle_mode=args.battle_mode,
        preview_cache=not args.no_preview_cache,
    )
    ran_chunks = dispatch["ran_local"] + dispatch["ran_mini2"] + dispatch.get("ran_colab", 0)
    skipped_chunks = dispatch["skipped"]

    # Write consolidated results.jsonl from all chunks
    total_written = 0
    with merged_path.open("w") as out_fh:
        for c_out in chunk_result_dirs:
            for p in sorted(c_out.glob("*.json")):
                if p.name == "manifest.json":
                    continue
                out_fh.write(p.read_text().strip() + "\n")
                total_written += 1

    ledger_added = 0
    if args.merge_ledger is not None:
        merge_proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/nash-ledger.py"),
                "merge",
                "--ledger",
                str(args.merge_ledger),
                "--pools",
                meta["pools"],
                *[str(d) for d in chunk_result_dirs],
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        merge_info = json.loads(merge_proc.stdout.strip().splitlines()[-1])
        ledger_added = int(merge_info.get("added", 0))

    elapsed = time.monotonic() - t0
    print(json.dumps({
        "dir": str(args.dir),
        "total_games": total_written,
        "chunks": n_chunks,
        "chunk_games": chunk_size,
        "ran_chunks": ran_chunks,
        "skipped_chunks": skipped_chunks,
        "elapsed_s": round(elapsed, 2),
        "games_per_hr": round(total_written / max(elapsed, 1e-6) * 3600.0, 1) if ran_chunks > 0 else None,
        "results": str(merged_path),
        "ledger_added": ledger_added,
        "dispatch": dispatch,
    }))


def analyze_matrix(
    profiles: list[dict[str, Any]],
    pair_records: dict[tuple[str, str, str, str, int, tuple[int, int]], dict[int, float]],
    *,
    boot: int = 400,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Compute exact conditional-weighted shell matrix, RM+ Nash, bootstrap, and by_variant EV."""
    n = len(profiles)
    names = [p["name"] for p in profiles]
    ph_to_idx = {p["profile_hash"]: i for i, p in enumerate(profiles)}

    # Organize complete seat pairs by (i, j, set_hash_i, set_hash_j)
    subcell_pairs: dict[tuple[int, int, str, str], list[float]] = defaultdict(list)
    complete_pairs = 0
    incomplete_games = 0
    for (p_lo, h_lo, p_hi, h_hi, _bs, _ss), seats in pair_records.items():
        if p_lo not in ph_to_idx or p_hi not in ph_to_idx:
            continue
        if 0 in seats and 1 in seats:
            ps_lo = 0.5 * (seats[0] + seats[1])
            ilo, ihi = ph_to_idx[p_lo], ph_to_idx[p_hi]
            subcell_pairs[(ilo, ihi, h_lo, h_hi)].append(ps_lo)
            subcell_pairs[(ihi, ilo, h_hi, h_lo)].append(1.0 - ps_lo)
            complete_pairs += 1
        else:
            incomplete_games += len(seats)

    M = [[0.5] * n for _ in range(n)]
    SE = [[0.0] * n for _ in range(n)]
    NP = [[0] * n for _ in range(n)]
    COV = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    MIN_CP = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            qa = profiles[i]["variants"]
            qb = profiles[j]["variants"]
            cov = 0.0
            wsum = 0.0
            vsum = 0.0
            np_tot = 0
            min_cp = 10**9
            for va in qa:
                for vb in qb:
                    xs = subcell_pairs.get((i, j, va["set_hash"], vb["set_hash"]), [])
                    min_cp = min(min_cp, len(xs))
                    if not xs:
                        continue
                    w = va["weight"] * vb["weight"]
                    m = sum(xs) / len(xs)
                    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.25
                    cov += w
                    wsum += w * m
                    vsum += (w * w) * max(v, 0.01) / len(xs)
                    np_tot += len(xs)
            NP[i][j] = NP[j][i] = np_tot
            COV[i][j] = COV[j][i] = round(cov, 6)
            MIN_CP[i][j] = MIN_CP[j][i] = 0 if min_cp == 10**9 else min_cp
            if cov > 0.0:
                score_ij = wsum / cov
                se_ij = math.sqrt(vsum) / cov
                M[i][j] = score_ij
                M[j][i] = 1.0 - score_ij
                SE[i][j] = SE[j][i] = se_ij
            else:
                M[i][j] = M[j][i] = 0.5
                SE[i][j] = SE[j][i] = 0.25

    audit = STATS.nash_uncertainty_audit(names, M, SE, boot=boot, seed=seed)
    p_eq = [float(audit["shells"][name]["weight"]) for name in names]

    # Per-shell summary
    rows = []
    for i, name in enumerate(names):
        sh = audit["shells"][name]
        opp_scores = [M[i][j] for j in range(n) if j != i]
        worst_j = min((j for j in range(n) if j != i), key=lambda j: M[i][j])
        best_j = max((j for j in range(n) if j != i), key=lambda j: M[i][j])
        rows.append({
            "name": name,
            "shell_hash": profiles[i]["shell_hash"],
            "profile_hash": profiles[i]["profile_hash"],
            "n_variants": len(profiles[i]["variants"]),
            "nash_weight": sh["weight"],
            "boot_p10": sh["p10"],
            "boot_med": sh["median"],
            "boot_p90": sh["p90"],
            "ev_vs_nash": sh["ev_vs_eq"],
            "mean_vs_field": round(sum(opp_scores) / len(opp_scores), 4),
            "worst_opp": names[worst_j],
            "worst_score": round(M[i][worst_j], 4),
            "best_opp": names[best_j],
            "best_score": round(M[i][best_j], 4),
        })
    rows.sort(key=lambda r: (-r["nash_weight"], -r["ev_vs_nash"], -r["mean_vs_field"]))

    # Per-variant conditional diagnostics against each opponent shell AND against P*
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for i, prof in enumerate(profiles):
        v_rows = []
        for va in prof["variants"]:
            vs_opp: dict[str, dict[str, Any]] = {}
            ev_nash = p_eq[i] * 0.5
            ev_other_num = 0.0
            ev_other_den = 0.0
            for j in range(n):
                if j == i:
                    continue
                qb = profiles[j]["variants"]
                cov = 0.0
                wsum = 0.0
                vsum = 0.0
                np_v = 0
                for vb in qb:
                    xs = subcell_pairs.get((i, j, va["set_hash"], vb["set_hash"]), [])
                    if not xs:
                        continue
                    wb = vb["weight"]
                    m = sum(xs) / len(xs)
                    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.25
                    cov += wb
                    wsum += wb * m
                    vsum += (wb * wb) * max(v, 0.01) / len(xs)
                    np_v += len(xs)
                if cov > 0.0:
                    sc = wsum / cov
                    se = math.sqrt(vsum) / cov
                else:
                    sc, se = 0.5, 0.25
                vs_opp[names[j]] = {
                    "score": round(sc, 4),
                    "se": round(se, 4),
                    "pairs": np_v,
                    "cov": round(cov, 4),
                }
                ev_nash += p_eq[j] * sc
                if p_eq[j] > 0.0:
                    ev_other_num += p_eq[j] * sc
                    ev_other_den += p_eq[j]
            v_rows.append({
                "id": va["id"],
                "weight": round(va["weight"], 4),
                "ev_vs_nash": round(ev_nash, 4),
                "ev_vs_other_nash": round(ev_other_num / ev_other_den, 4) if ev_other_den > 0 else None,
                "vs": vs_opp,
            })
        by_variant[prof["name"]] = v_rows

    uncovered_cells = [
        {"a": names[i], "b": names[j], "covered_mass": COV[i][j]}
        for i in range(n)
        for j in range(i + 1, n)
        if COV[i][j] < 0.999999
    ]

    return {
        "complete_pairs": complete_pairs,
        "complete_games": 2 * complete_pairs,
        "incomplete_orphan_games": incomplete_games,
        "names": names,
        "matrix": [[round(v, 4) for v in row] for row in M],
        "se_matrix": [[round(v, 4) for v in row] for row in SE],
        "pairs_matrix": NP,
        "covered_mass_matrix": COV,
        "min_subcell_pairs_matrix": MIN_CP,
        "uncovered_cells": uncovered_cells,
        "ranked": rows,
        "by_variant": by_variant,
        "next_cells": audit["next_cells"][:20],
    }


def cmd_analyze(args: argparse.Namespace) -> None:
    meta = json.loads((args.dir / "meta.json").read_text())
    pools_path = Path(meta["pools"])
    names: list[str] = meta["mixtures"]
    teams, profiles = validate_profiles(names, pools_path, verify_frames=False)
    prof_by_name = {p["name"]: p for p in profiles}
    target_hashes = {p["profile_hash"] for p in profiles}

    # Combine ledgers + explicit result sources cleanly without double-counting
    ledger_paths = list(args.ledger or [])
    if not ledger_paths and not args.no_meta_ledgers:
        ledger_paths = [Path(p) for p in meta.get("ledgers", []) if Path(p).exists()]

    combined = load_ledger_pair_records(ledger_paths, target_hashes) if ledger_paths else defaultdict(dict)
    if args.result_sources:
        res_pairs, _ = load_result_pair_records(args.result_sources, prof_by_name, teams, target_hashes)
        for k, seats in res_pairs.items():
            combined[k].update(seats)

    report = analyze_matrix(profiles, combined, boot=args.boot, seed=args.seed)
    report["dir"] = str(args.dir)
    report["pools"] = str(pools_path)
    report["ledgers"] = [str(p) for p in ledger_paths]
    report["result_sources"] = [str(p) for p in (args.result_sources or [])]

    out_path = args.out or (args.dir / "rr-report.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "report": str(out_path),
        "complete_pairs": report["complete_pairs"],
        "complete_games": report["complete_games"],
        "uncovered_cells": len(report["uncovered_cells"]),
        "support_ge_1pct": [
            {
                "name": r["name"],
                "w": r["nash_weight"],
                "p10": r["boot_p10"],
                "p90": r["boot_p90"],
                "ev": r["ev_vs_nash"],
            }
            for r in report["ranked"]
            if r["nash_weight"] >= 0.01
        ],
        "top_next_cell": report["next_cells"][0] if report["next_cells"] else None,
    }))


def cmd_self_test(_args: argparse.Namespace) -> None:
    def mk_team(sp0: str, move0: str) -> list[dict[str, Any]]:
        return [
            {"species": sp0, "level": 55, "gender": "M", "item": "Leftovers", "moves": [move0]},
            {"species": "Zapdos", "level": 50, "gender": "N", "item": "Mint Berry", "moves": ["Thunderbolt"]},
        ]

    teams = {
        "a1": mk_team("Snorlax", "Curse"),
        "a2": mk_team("Snorlax", "Belly Drum"),
        "b1": mk_team("Marowak", "Earthquake"),
        "b2": mk_team("Marowak", "Rock Slide"),
        "b3": mk_team("Marowak", "Swords Dance"),
    }
    # 1. Single-shell validation rejects cross-shell mixture
    try:
        canonical_mixture_profile("bad", [{"id": "a1", "weight": 0.5}, {"id": "b1", "weight": 0.5}], teams)
        raise AssertionError("expected multi-shell mixture rejection")
    except ValueError:
        pass

    # 2. Valid multi-variant profiles
    pa = canonical_mixture_profile("A", [{"id": "a1", "weight": 0.7}, {"id": "a2", "weight": 0.3}], teams)
    pb = canonical_mixture_profile("B", [
        {"id": "b1", "weight": 0.5}, {"id": "b2", "weight": 0.3}, {"id": "b3", "weight": 0.2}
    ], teams)
    assert pa["shell_hash"] != pb["shell_hash"]

    # 3. Stratified allocation covers all 2x3 = 6 sub-cells and sums to target
    alloc = stratified_target_counts(pa["variants"], pb["variants"], 16, 2, False)
    assert len(alloc) == 6 and min(alloc.values()) >= 2 and sum(alloc.values()) == 16

    # 4. Exact conditional-weighted expectation check
    # Give (a1, b1)=1.0, (a1, b2)=0.5, (a1, b3)=0.0, (a2, *)=0.25 ->
    # Expected M[A][B] = 0.7*(0.5*1.0 + 0.3*0.5 + 0.2*0.0) + 0.3*0.25 = 0.7*0.65 + 0.075 = 0.53
    scores = {
        ("a1", "b1"): [1.0, 1.0],
        ("a1", "b2"): [0.5, 0.5],
        ("a1", "b3"): [0.0, 0.0],
        ("a2", "b1"): [0.5, 0.0],
        ("a2", "b2"): [0.5, 0.0],
        ("a2", "b3"): [0.5, 0.0],
    }
    pair_recs: dict[tuple[str, str, str, str, int, tuple[int, int]], dict[int, float]] = {}
    hmap = {tid: LEDGER.sha256_json(sets) for tid, sets in teams.items()}
    pha, phb = pa["profile_hash"], pb["profile_hash"]
    seed_i = 1
    for (va, vb), vals in scores.items():
        for sc in vals:
            if pha < phb:
                pair_recs[(pha, hmap[va], phb, hmap[vb], seed_i, (seed_i, seed_i + 1))] = {0: sc, 1: sc}
            else:
                pair_recs[(phb, hmap[vb], pha, hmap[va], seed_i, (seed_i, seed_i + 1))] = {0: 1.0 - sc, 1: 1.0 - sc}
            seed_i += 10
    rep = analyze_matrix([pa, pb], pair_recs, boot=40, seed=7)
    assert abs(rep["matrix"][0][1] - 0.53) < 1e-6, rep["matrix"]
    assert abs(rep["matrix"][1][0] - 0.47) < 1e-6, rep["matrix"]
    assert rep["covered_mass_matrix"][0][1] == 1.0
    assert rep["min_subcell_pairs_matrix"][0][1] == 2
    assert rep["complete_pairs"] == 12
    # Check duplicate shell rejection through validate_profiles via temp file
    with tempfile.TemporaryDirectory() as td:
        pfile = Path(td) / "pools.json"
        pfile.write_text(json.dumps({
            "teams": [{"id": k, "sets": v} for k, v in teams.items()],
            "mixtures": {
                "A1": [{"id": "a1", "weight": 1.0}],
                "A2": [{"id": "a2", "weight": 1.0}],
            },
        }))
        try:
            validate_profiles(["A1", "A2"], pfile, verify_frames=False)
            raise AssertionError("expected duplicate shell_hash rejection")
        except ValueError:
            pass
    print(json.dumps({"self_test_ok": True, "expected_M_AB": rep["matrix"][0][1]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--pools", type=Path, required=True)
    p_ver.add_argument("--mixtures", type=str, required=True)
    p_ver.add_argument("--no-verify-frames", action="store_true")
    p_ver.set_defaults(func=cmd_verify)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--pools", type=Path, required=True)
    p_prep.add_argument("--mixtures", type=str, required=True)
    p_prep.add_argument("--ledger", type=Path, action="append", default=[])
    p_prep.add_argument("--mode", choices=["stratified", "mc"], default="stratified")
    p_prep.add_argument("--pairs", type=int, default=24)
    p_prep.add_argument("--add-pairs", type=int, default=None)
    p_prep.add_argument("--min-cell-pairs", type=int, default=2)
    p_prep.add_argument("--auto-cover", action="store_true")
    p_prep.add_argument("--pair-overrides", type=str, default=None)
    p_prep.add_argument("--seed-tag", type=str, required=True)
    p_prep.add_argument("--out-dir", type=Path, required=True)
    p_prep.add_argument("--no-verify-frames", action="store_true")
    p_prep.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run")
    p_run.add_argument("--dir", type=Path, required=True)
    p_run.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p_run.add_argument("--workers", type=int, default=10)
    p_run.add_argument("--mini2", action="store_true", help="Enable work-stealing dual-node dispatch with mini2")
    p_run.add_argument("--mini2-workers", type=int, default=11)
    p_run.add_argument("--colab", type=str, default="", help="Colab node pool via colab-barn (e.g. G4, G4,G4, V5E1, auto)")
    p_run.add_argument("--chunk-games", type=int, default=40)
    p_run.add_argument("--merge-ledger", type=Path, default=None)
    p_run.add_argument("--preview-iters", type=int, default=None)
    p_run.add_argument("--battle-iters", type=int, default=None)
    p_run.add_argument("--battle-mode", choices=["adaptive", "fixed"], default=None)
    p_run.add_argument("--no-preview-cache", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze")
    p_an.add_argument("--dir", type=Path, required=True)
    p_an.add_argument("--ledger", type=Path, action="append", default=[])
    p_an.add_argument("--no-meta-ledgers", action="store_true")
    p_an.add_argument("--boot", type=int, default=400)
    p_an.add_argument("--seed", type=int, default=20261120)
    p_an.add_argument("--out", type=Path, default=None)
    p_an.add_argument("result_sources", nargs="*", type=Path)
    p_an.set_defaults(func=cmd_analyze)

    p_st = sub.add_parser("self-test")
    p_st.set_defaults(func=cmd_self_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
