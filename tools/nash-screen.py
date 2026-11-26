#!/usr/bin/env python3
"""Multi-fidelity & production PSRO challenger evaluator (dual-node mini1 + mini2).

Builds deterministic, seat-swapped, stratified schedules for single-shell
(pure or camouflaged) challengers against a disclosed defender pool P^(t),
runs them via native CS-PGO `nash_refresh` locally (and optionally `mini2` via
`tools/mini2_nash.py`) in resumable chunks, merges production results into
`target/nash-ledger.jsonl`, and computes exact conditional-weighted point
estimates + stratified one-sided 95% bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def u32_hash(tag: str) -> int:
    return int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:4], "big")


def allocate_stratum_counts(weights: list[float], total_pairs: int) -> list[int]:
    """Deterministic proportional allocation (every positive weight gets >= 1 when total_pairs >= len)."""
    k = len(weights)
    if total_pairs >= k:
        counts = [1] * k
        rem = total_pairs - k
    else:
        counts = [0] * k
        rem = total_pairs
    raw = [rem * w for w in weights]
    floors = [int(r) for r in raw]
    for i in range(k):
        counts[i] += floors[i]
    leftover = rem - sum(floors)
    order = sorted(range(k), key=lambda i: (raw[i] - floors[i], weights[i]), reverse=True)
    for i in order[:leftover]:
        counts[i] += 1
    return counts


def load_defender(args: argparse.Namespace, base_teams: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return normalized defender members [{id, weight}] and any non-base team sets required."""
    extra: dict[str, Any] = {}
    if args.defender_pools is not None:
        assert args.defender_mixture, "--defender-mixture is required with --defender-pools"
        teams, mixtures = LEDGER.load_catalog_and_mixtures(args.defender_pools)
        assert args.defender_mixture in mixtures, f"mixture {args.defender_mixture!r} not in {args.defender_pools}"
        raw_members = mixtures[args.defender_mixture]
        tot = sum(float(m["weight"]) for m in raw_members)
        members = [{"id": m["id"], "weight": float(m["weight"]) / tot} for m in raw_members]
        for m in members:
            if m["id"] not in base_teams:
                extra[m["id"]] = teams[m["id"]]
        return members, extra

    assert args.defender is not None, "pass either --defender FILE or --defender-pools + --defender-mixture"
    defender_doc = json.loads(args.defender.read_text())
    def_teams = defender_doc["teams"]
    def_total_w = sum(float(t["weight"]) for t in def_teams)
    members = [{"id": t["id"], "weight": float(t["weight"]) / def_total_w} for t in def_teams]
    for t in def_teams:
        if t["id"] not in base_teams:
            extra[t["id"]] = t["sets"]
    return members, extra


def cmd_prepare(args: argparse.Namespace) -> None:
    base_teams, _ = LEDGER.load_catalog_and_mixtures(None)
    base_ids = set(base_teams.keys())
    def_members, extra_teams = load_defender(args, base_teams)

    challengers: list[tuple[str, list[dict[str, Any]]]] = []
    if args.challenger_pools is not None:
        # Multi-variant or singleton mixtures from a pools.json
        ch_teams, ch_mixtures = LEDGER.load_catalog_and_mixtures(args.challenger_pools)
        mix_names = [s.strip() for s in args.challenger_mixtures.split(",") if s.strip()]
        for mname in mix_names:
            assert mname in ch_mixtures, f"challenger mixture {mname!r} not in {args.challenger_pools}"
            raw_m = ch_mixtures[mname]
            tot = sum(float(m["weight"]) for m in raw_m)
            norm_m = [{"id": m["id"], "weight": float(m["weight"]) / tot} for m in raw_m]
            sigs = {LEDGER.shell_signature(ch_teams[m["id"]]) for m in norm_m}
            assert len(sigs) == 1, f"challenger {mname!r} spans {len(sigs)} shells; must be single-shell"
            for m in norm_m:
                if m["id"] not in base_ids:
                    extra_teams[m["id"]] = ch_teams[m["id"]]
            challengers.append((mname, norm_m))
    else:
        assert args.catalog is not None, "pass --catalog or --challenger-pools"
        cat_doc = json.loads(args.catalog.read_text())
        cat_teams = cat_doc["teams"]
        seen_sets: set[str] = set()
        id_filter = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else None
        id_filter_set = set(id_filter) if id_filter is not None else None
        cat_by_id = {tid: {"id": tid, "sets": sets} for tid, sets in base_teams.items()}
        for t in cat_teams:
            cat_by_id[t["id"]] = t
        ordered_teams = [cat_by_id[tid] for tid in id_filter] if id_filter is not None else cat_teams

        for t in ordered_teams:
            tid = t["id"]
            if id_filter_set is not None and tid not in id_filter_set:
                continue
            set_hash = hashlib.sha256(
                json.dumps(t["sets"], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if not args.all_variants and set_hash in seen_sets:
                continue
            seen_sets.add(set_hash)
            if tid not in base_ids:
                extra_teams[tid] = t["sets"]
            challengers.append((tid, [{"id": tid, "weight": 1.0}]))
            if args.limit and len(challengers) >= args.limit:
                break

    mixtures: dict[str, list[dict[str, Any]]] = {"defender": def_members}
    for cname, members in challengers:
        mixtures[f"ch:{cname}"] = members

    # Stratified seat-swapped schedule over (challenger_variant, defender_variant)
    schedule: list[dict[str, Any]] = []
    stratum_summary: dict[str, int] = {}
    for cname, c_members in challengers:
        ch_mix = f"ch:{cname}"
        cell_pairs = [(cm, dm) for cm in c_members for dm in def_members]
        cell_weights = [cm["weight"] * dm["weight"] for cm, dm in cell_pairs]
        counts = allocate_stratum_counts(cell_weights, args.pairs)
        stratum_summary[cname] = sum(counts)
        block_idx = 0
        for s_idx, ((cm, dm), n_pairs) in enumerate(zip(cell_pairs, counts)):
            cid = cm["id"]
            did = dm["id"]
            for rep in range(n_pairs):
                tag = f"nc2000/screen/{args.seed_tag}/{cname if args.independent_seeds else 'shared'}/s{s_idx}/{cid}/{did}/r{rep}"
                seed_ch = u32_hash(f"{tag}/ch/{cname}")
                seed_def = u32_hash(f"{tag}/def/{did}")
                battle_seed = u32_hash(f"{tag}/battle")
                for row_seat in (0, 1):
                    if row_seat == 0:
                        ids = [cid, did]
                        seeds = [seed_ch, seed_def]
                        mix = [ch_mix, "defender"]
                    else:
                        ids = [did, cid]
                        seeds = [seed_def, seed_ch]
                        mix = ["defender", ch_mix]
                    schedule.append({
                        "block": block_idx,
                        "cell": {
                            "challenger": cname,
                            "ch_id": cid,
                            "def_id": did,
                            "stratum": s_idx,
                            "rep": rep,
                        },
                        "rowSeat": row_seat,
                        "ids": ids,
                        "seeds": seeds,
                        "battleSeed": battle_seed,
                        "mixtures": mix,
                    })
                block_idx += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pools_obj = {
        "teams": [{"id": tid, "sets": sets} for tid, sets in extra_teams.items()],
        "mixtures": mixtures,
    }
    (args.out_dir / "pools.json").write_text(json.dumps(pools_obj))
    with (args.out_dir / "schedule.jsonl").open("w") as fh:
        for row in schedule:
            fh.write(json.dumps(row) + "\n")
    meta = {
        "defender": str(args.defender) if args.defender else f"{args.defender_pools}#{args.defender_mixture}",
        "catalog": str(args.catalog or args.challenger_pools),
        "seed_tag": args.seed_tag,
        "pairs_per_challenger": args.pairs,
        "defender_weights": def_members,
        "challengers": [{"name": cname, "members": cm} for cname, cm in challengers],
        "challenger_count": len(challengers),
        "total_games": len(schedule),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "challengers": len(challengers),
        "pairs_per_challenger": args.pairs,
        "total_games": len(schedule),
    }))


def cmd_run(args: argparse.Namespace) -> None:
    sched_path = args.dir / "schedule.jsonl"
    pools_path = args.dir / "pools.json"
    rows = [l.strip() for l in sched_path.open() if l.strip()]
    if not rows:
        print(json.dumps({"dir": str(args.dir), "total_games": 0}))
        return

    chunk_size = max(2, args.chunk_games + (args.chunk_games % 2))
    run_root = args.dir / args.run_name
    chunks_dir = run_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if not any(chunks_dir.glob("chunk-*.jsonl")):
        chunk_size = MINI2.recommend_chunk_size(
            chunk_size,
            colab_nodes=args.colab or None,
            preview_iters=args.preview_iters,
            battle_iters=args.battle_iters,
            battle_mode=args.battle_mode,
            preview_cache=True,
        )

    n_chunks = (len(rows) + chunk_size - 1) // chunk_size
    chunk_specs: list[dict[str, Any]] = []
    chunk_out_dirs: list[Path] = []
    for ci in range(n_chunks):
        c_rows = rows[ci * chunk_size : (ci + 1) * chunk_size]
        c_sched = chunks_dir / f"chunk-{ci:03d}.jsonl"
        c_out = chunks_dir / f"chunk-{ci:03d}-out"
        c_done = chunks_dir / f"chunk-{ci:03d}.done.json"
        if not c_sched.exists():
            c_sched.write_text("\n".join(c_rows) + "\n")
        chunk_specs.append({"ci": ci, "sched": c_sched, "out": c_out, "done": c_done, "n_rows": len(c_rows)})
        chunk_out_dirs.append(c_out)

    t0 = time.monotonic()
    dispatch = MINI2.run_chunks_dual_node(
        chunk_specs,
        binary=args.binary,
        pools_path=pools_path,
        local_workers=args.workers,
        mini2_workers=args.mini2_workers,
        use_mini2=args.mini2,
        colab_nodes=args.colab or None,
        preview_iters=args.preview_iters,
        battle_iters=args.battle_iters,
        battle_mode=args.battle_mode,
        preview_cache=True,
    )

    merged_path = run_root / "results.jsonl"
    total_written = 0
    with merged_path.open("w") as out_fh:
        for c_out in chunk_out_dirs:
            for p in sorted(c_out.glob("*.json")):
                if p.name == "manifest.json":
                    continue
                out_fh.write(p.read_text().strip() + "\n")
                total_written += 1

    ledger_added = 0
    if args.merge_ledger is not None:
        import subprocess, sys
        mproc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/nash-ledger.py"),
                "merge",
                "--ledger",
                str(args.merge_ledger),
                "--pools",
                str(pools_path),
                *[str(d) for d in chunk_out_dirs],
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        ledger_added = int(json.loads(mproc.stdout.strip().splitlines()[-1]).get("added", 0))

    elapsed = time.monotonic() - t0
    print(json.dumps({
        "dir": str(args.dir),
        "run_name": args.run_name,
        "total_games": total_written,
        "chunks": n_chunks,
        **dispatch,
        "elapsed_s": round(elapsed, 2),
        "games_per_hr": round(total_written / max(elapsed, 1e-6) * 3600.0, 1),
        "ledger_added": ledger_added,
    }))


def row_score(outcome: str, row_seat: int) -> float:
    if outcome in ("tie", "cap", "draw"):
        return 0.5
    win_seat = 0 if outcome == "p1" else 1
    return 1.0 if win_seat == row_seat else 0.0


def cmd_summarize(args: argparse.Namespace) -> None:
    meta = json.loads((args.dir / "meta.json").read_text())
    run_dir = args.dir / args.run_name
    def_weights = {m["id"]: float(m["weight"]) for m in meta["defender_weights"]}
    ch_weights_map: dict[str, dict[str, float]] = {}
    for ch in meta.get("challengers", []):
        ch_weights_map[ch["name"]] = {m["id"]: float(m["weight"]) for m in ch["members"]}

    pairs: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    total_games = 0
    merged = run_dir / "results.jsonl"
    if merged.exists():
        for line in merged.open():
            if not line.strip():
                continue
            r = json.loads(line)
            total_games += 1
            cell = r["cell"]
            cid = cell.get("ch_id") or cell["challenger"]
            key = (cell["challenger"], cid, cell["def_id"], int(r["block"]))
            pairs[key].append(row_score(r["outcome"], int(r["rowSeat"])))
    else:
        for p in sorted(run_dir.glob("*.json")):
            if p.name == "manifest.json":
                continue
            r = json.loads(p.read_text())
            total_games += 1
            cell = r["cell"]
            cid = cell.get("ch_id") or cell["challenger"]
            key = (cell["challenger"], cid, cell["def_id"], int(r["block"]))
            pairs[key].append(row_score(r["outcome"], int(r["rowSeat"])))

    by_ch_cell: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for (cname, cid, did, _blk), scs in pairs.items():
        if len(scs) == 2:
            by_ch_cell[cname][(cid, did)].append(0.5 * (scs[0] + scs[1]))

    ranked = []
    for cname, cmap in by_ch_cell.items():
        cw = ch_weights_map.get(cname, {cname: 1.0})
        active_strata = []
        covered_mass = 0.0
        per_def: dict[str, Any] = {}
        for cid, wc in cw.items():
            for did, wd in def_weights.items():
                xs = cmap.get((cid, did), [])
                if xs:
                    w = wc * wd
                    covered_mass += w
                    active_strata.append((w, xs))
        for did, wd in def_weights.items():
            d_xs = [x for cid in cw for x in cmap.get((cid, did), [])]
            if d_xs:
                per_def[did] = {"mean": round(sum(d_xs) / len(d_xs), 4), "pairs": len(d_xs)}
        norm_strata = [(w / covered_mass, xs) for w, xs in active_strata]
        b = STATS.stratified_bounds(norm_strata, alpha=0.05, one_sided=True)
        # Also compute lower one-sided 95% normal bound (to flag statistically significant >50% challengers)
        lower_n95 = max(0.0, b["mean"] - 1.6448536269514722 * b["se"])
        z50 = (b["mean"] - 0.5) / b["se"] if b["se"] > 0 else 0.0
        ranked.append({
            "challenger": cname,
            "pairs": sum(len(xs) for _, xs in norm_strata),
            "covered_mass": round(covered_mass, 4),
            "mean": round(b["mean"], 4),
            "se": round(b["se"], 4),
            "z_vs_50": round(z50, 2),
            "lower_normal_95": round(lower_n95, 4),
            "upper_normal_95": round(b["upper_normal"], 4),
            "upper_eb_95": round(b["upper_eb"], 4),
            "per_defender": per_def,
        })

    ranked.sort(key=lambda r: (-r["mean"], -r["lower_normal_95"], r["challenger"]))
    summary = {
        "run_dir": str(run_dir),
        "total_games": total_games,
        "challengers_scored": len(ranked),
        "above_50_count": sum(1 for r in ranked if r["mean"] > 0.5),
        "sig_above_50_count": sum(1 for r in ranked if r["lower_normal_95"] > 0.5),
        "top": ranked[: args.top],
    }
    out_file = args.out or (args.dir / f"summary-{args.run_name}.json")
    out_file.write_text(json.dumps({"summary": summary, "all": ranked}, indent=2))
    print(json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--defender", type=Path, default=None)
    p_prep.add_argument("--defender-pools", type=Path, default=None)
    p_prep.add_argument("--defender-mixture", type=str, default=None)
    p_prep.add_argument("--catalog", type=Path, default=None)
    p_prep.add_argument("--challenger-pools", type=Path, default=None)
    p_prep.add_argument("--challenger-mixtures", type=str, default=None)
    p_prep.add_argument("--pairs", type=int, default=14)
    p_prep.add_argument("--seed-tag", type=str, required=True)
    p_prep.add_argument("--out-dir", type=Path, required=True)
    p_prep.add_argument("--ids", type=str, default=None, help="Comma-separated challenger IDs")
    p_prep.add_argument("--all-variants", action="store_true")
    p_prep.add_argument("--independent-seeds", action="store_true")
    p_prep.add_argument("--limit", type=int, default=0)
    p_prep.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run")
    p_run.add_argument("--dir", type=Path, required=True)
    p_run.add_argument("--run-name", type=str, default="prod")
    p_run.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    p_run.add_argument("--workers", type=int, default=10)
    p_run.add_argument("--mini2", action="store_true")
    p_run.add_argument("--mini2-workers", type=int, default=11)
    p_run.add_argument("--colab", type=str, default="", help="Colab node pool via colab-barn (e.g. G4, G4,G4, V5E1, auto)")
    p_run.add_argument("--chunk-games", type=int, default=40)
    p_run.add_argument("--merge-ledger", type=Path, default=None)
    p_run.add_argument("--preview-iters", type=int, default=None)
    p_run.add_argument("--battle-iters", type=int, default=None)
    p_run.add_argument("--battle-mode", choices=["adaptive", "fixed"], default=None)
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--dir", type=Path, required=True)
    p_sum.add_argument("--run-name", type=str, default="prod")
    p_sum.add_argument("--top", type=int, default=20)
    p_sum.add_argument("--out", type=Path, default=None)
    p_sum.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
