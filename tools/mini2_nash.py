#!/usr/bin/env python3
"""Project-specific `mini2-barn` dispatch helper for `nc2000-engine` `nash_refresh`.

Runs individual schedule chunks on subordinate node `mini2` via `mini2-barn run`
(hash-verified content-addressed blobs, ephemeral run workspace, atomic collect),
or drives a work-stealing dual-node queue across `mini1` (local) + `mini2`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "target/native-pgo/release/examples/nash_refresh"
DEFAULT_COLAB_BINS = [
    ROOT / "target/colab-x86_64/nash_refresh",
    ROOT / "target/colab-x86_64-musl/x86_64-unknown-linux-musl/release/examples/nash_refresh",
]

# Exact data files read by `crates/bot/examples/nash_refresh.rs` when `mixtures` are present.
NASH_REFRESH_DATA_FILES = [
    "data/gen2stadium2.json",
    "data/learnsets-gen2.json",
    "data/live-search.json",
    "data/search-profiles.json",
    "data/kodoku-beq-v1/novel_evolved_teams.json",
    "data/team-pool-v2/team-pool.json",
    "data/kodoku-r3/conclusion-equilibrium-deme.json",
    "data/nash-refresh-v1/candidate.json",
]


def build_mini2_run_cmd(
    *,
    binary: Path,
    pools_path: Path | None,
    schedule_path: Path,
    out_dir: Path,
    workers: int = 11,
    run_id: str | None = None,
    preview_iters: int | None = None,
    battle_iters: int | None = None,
    battle_mode: str | None = None,
    preview_cache: bool = True,
    preview_dist_cache: Path | None = None,
    preview_dist_k: int | None = None,
    preview_dist_only: bool = False,
    timeout: int | None = None,
) -> list[str]:
    cmd: list[str] = [
        "mini2-barn",
        "run",
        "--run-id",
        run_id or f"nc2000-{int(time.time())}-{secrets.token_hex(3)}",
        "--bin",
        f"bin/nash_refresh={binary.resolve()}",
    ]
    for rel in NASH_REFRESH_DATA_FILES:
        cmd.extend(["--file", f"{rel}={(ROOT / rel).resolve()}"])
    if pools_path is not None:
        cmd.extend(["--file", f"pools.json={pools_path.resolve()}"])
    has_dist_file = preview_dist_cache is not None and preview_dist_cache.exists()
    if has_dist_file:
        cmd.extend(["--file", f"preview-dist.json={preview_dist_cache.resolve()}"])
    cmd.extend([
        "--input",
        f"schedule.jsonl={schedule_path.resolve()}",
        "--env",
        "NC2000_ROOT={run}",
        "--env",
        f"NASH_PREVIEW_CACHE={'1' if preview_cache else '0'}",
    ])
    if pools_path is not None:
        cmd.extend(["--env", "NASH_POOLS={run}/pools.json"])
    if has_dist_file:
        cmd.extend(["--env", "NASH_PREVIEW_DIST_CACHE={run}/preview-dist.json"])
    if preview_dist_k is not None:
        cmd.extend(["--env", f"NASH_PREVIEW_DIST_K={preview_dist_k}"])
    if preview_dist_only:
        cmd.extend(["--env", "NASH_PREVIEW_DIST_ONLY=1"])
    if preview_iters is not None:
        cmd.extend(["--env", f"NASH_PREVIEW_ITERS={preview_iters}"])
    if battle_iters is not None:
        cmd.extend(["--env", f"NASH_BATTLE_ITERS={battle_iters}"])
    if battle_mode is not None:
        cmd.extend(["--env", f"NASH_BATTLE_MODE={battle_mode}"])
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    cmd.extend([
        "--collect",
        f"out={out_dir.resolve()}",
        "--",
        "{run}/bin/nash_refresh",
        "{run}/schedule.jsonl",
        "{run}/out",
        str(workers),
    ])
    return cmd


def run_chunk_mini2(
    *,
    binary: Path,
    pools_path: Path | None,
    schedule_path: Path,
    final_out_dir: Path,
    done_marker: Path,
    workers: int = 11,
    run_id: str | None = None,
    preview_iters: int | None = None,
    battle_iters: int | None = None,
    battle_mode: str | None = None,
    preview_cache: bool = True,
    preview_dist_cache: Path | None = None,
    preview_dist_k: int | None = None,
    preview_dist_only: bool = False,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Execute one chunk on `mini2` and atomically commit `final_out_dir` + `done_marker`."""
    stage_out = final_out_dir.parent / f".{final_out_dir.name}.mini2-{secrets.token_hex(3)}"
    if stage_out.exists():
        shutil.rmtree(stage_out)
    cmd = build_mini2_run_cmd(
        binary=binary,
        pools_path=pools_path,
        schedule_path=schedule_path,
        out_dir=stage_out,
        workers=workers,
        run_id=run_id,
        preview_iters=preview_iters,
        battle_iters=battle_iters,
        battle_mode=battle_mode,
        preview_cache=preview_cache,
        preview_dist_cache=preview_dist_cache,
        preview_dist_k=preview_dist_k,
        preview_dist_only=preview_dist_only,
        timeout=timeout,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        shutil.rmtree(stage_out, ignore_errors=True)
        raise RuntimeError(
            f"mini2-barn run failed (rc={proc.returncode}):\n{proc.stderr or proc.stdout}"
        )
    telemetry = json.loads(proc.stdout.strip().splitlines()[-1])
    if not telemetry.get("ok"):
        shutil.rmtree(stage_out, ignore_errors=True)
        raise RuntimeError(f"mini2-barn remote command failed: {telemetry}")

    if final_out_dir.exists():
        shutil.rmtree(final_out_dir)
    stage_out.rename(final_out_dir)
    done_marker.write_text(json.dumps({
        "host": "mini2",
        "remote_stdout_last": telemetry.get("remote_stdout_last"),
        "remote_wall_ms": telemetry.get("remote_wall_ms"),
        "total_ms": telemetry.get("total_ms"),
    }) + "\n")
    return telemetry


def run_chunk_local(
    *,
    binary: Path,
    pools_path: Path | None,
    schedule_path: Path,
    final_out_dir: Path,
    done_marker: Path,
    workers: int = 10,
    preview_iters: int | None = None,
    battle_iters: int | None = None,
    battle_mode: str | None = None,
    preview_cache: bool = True,
    preview_dist_cache: Path | None = None,
    preview_dist_k: int | None = None,
    preview_dist_only: bool = False,
) -> dict[str, Any]:
    """Execute one chunk locally on `mini1`."""
    env = os.environ.copy()
    if pools_path is not None:
        env["NASH_POOLS"] = str(pools_path.resolve())
    env["NASH_PREVIEW_CACHE"] = "1" if preview_cache else "0"
    if preview_dist_cache is not None and preview_dist_cache.exists():
        env["NASH_PREVIEW_DIST_CACHE"] = str(preview_dist_cache.resolve())
    else:
        env.pop("NASH_PREVIEW_DIST_CACHE", None)
    if preview_dist_k is not None:
        env["NASH_PREVIEW_DIST_K"] = str(preview_dist_k)
    else:
        env.pop("NASH_PREVIEW_DIST_K", None)
    if preview_dist_only:
        env["NASH_PREVIEW_DIST_ONLY"] = "1"
    else:
        env.pop("NASH_PREVIEW_DIST_ONLY", None)
    if preview_iters is not None:
        env["NASH_PREVIEW_ITERS"] = str(preview_iters)
    if battle_iters is not None:
        env["NASH_BATTLE_ITERS"] = str(battle_iters)
    if battle_mode is not None:
        env["NASH_BATTLE_MODE"] = battle_mode

    if final_out_dir.exists():
        shutil.rmtree(final_out_dir)
    if done_marker.exists():
        done_marker.unlink()

    t0 = time.monotonic()
    proc = subprocess.run(
        [str(binary), str(schedule_path), str(final_out_dir), str(workers)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"local nash_refresh failed (rc={proc.returncode}):\n{proc.stderr or proc.stdout}")
    wall_ms = round((time.monotonic() - t0) * 1000)
    stdout_last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    done_marker.write_text(stdout_last + "\n")
    return {"host": "mini1", "wall_ms": wall_ms, "stdout_last": stdout_last}


def find_fresh_colab_binary(mac_binary: Path, explicit: Path | None = None) -> Path | None:
    """Return a Linux x86_64 binary if present and not older (>60s) than `mac_binary`."""
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"colab x86_64 binary not found: {p}")
        return p
    env_bin = os.environ.get("NC2000_COLAB_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"NC2000_COLAB_BIN not found: {p}")
        return p
    mac_mtime = mac_binary.stat().st_mtime if mac_binary.is_file() else 0.0
    for cand in DEFAULT_COLAB_BINS:
        if cand.is_file() and (mac_mtime - cand.stat().st_mtime <= 60.0):
            return cand
    return None


DEFAULT_GRANULARITY_STATE_PATH = ROOT / "target" / "barn-granularity.json"


def _granularity_state_path() -> Path:
    raw = os.environ.get("NC2000_BARN_GRANULARITY_STATE", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_GRANULARITY_STATE_PATH


def _workload_profile_key(
    *,
    preview_iters: int | None,
    battle_iters: int | None,
    battle_mode: str | None,
    preview_cache: bool,
    has_dist_file: bool,
    preview_dist_k: int | None,
    preview_dist_only: bool,
) -> str:
    mode = "dist_only" if preview_dist_only else "battle"
    return (
        f"{mode}|pi={preview_iters}|bi={battle_iters}|bm={battle_mode}|"
        f"pc={int(preview_cache)}|dc={int(has_dist_file)}|dk={preview_dist_k}"
    )


def _align_rows(rows: int, *, pair_aligned: bool, round_up: bool = False) -> int:
    if pair_aligned:
        if rows <= 2:
            return 2
        if rows % 2 == 1:
            return rows + 1 if round_up else rows - 1
        return rows
    return max(1, rows)


def _load_granularity_state(path: Path | None = None) -> dict[str, Any]:
    p = path or _granularity_state_path()
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
                return data
    except Exception:
        pass
    return {"version": 1, "profiles": {}}


def _update_granularity_state(updater: Any, path: Path | None = None) -> dict[str, Any]:
    p = path or _granularity_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_p = p.parent / f".{p.name}.lock"
    with lock_p.open("a+") as lfh:
        fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
        try:
            state = _load_granularity_state(p)
            result = updater(state)
            tmp = p.parent / f".{p.name}.{secrets.token_hex(4)}.tmp"
            tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, p)
            return result
        finally:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_UN)


def query_barn_targets(*, fresh: bool = False) -> dict[str, Any] | None:
    """Query `barn targets` M2M JSON for live `dispatch_hints`."""
    if shutil.which("barn") is None:
        return None
    cmd = ["barn", "targets"]
    if fresh:
        cmd.append("--fresh")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        return data if isinstance(data, dict) and data.get("ok") else None
    except Exception:
        return None


def _effective_duration_bounds(
    targets_resp: dict[str, Any] | None,
    *,
    local_only: bool,
    task_timeout: int | None,
) -> tuple[float, float, dict[str, Any]]:
    hints = dict((targets_resp or {}).get("dispatch_hints") or {})
    default_to = float(hints.get("default_task_timeout_s") or 600.0)
    eff_to = float(task_timeout) if task_timeout is not None else default_to

    if local_only:
        rel = hints.get("worker_rel_dur") or {}
        local_vals = [float(v) for k, v in rel.items() if k in ("mini1", "mini2") and float(v) > 0.0]
        local_ratio = (max(local_vals) / min(local_vals)) if len(local_vals) >= 2 else 1.2
        max_s = round((eff_to * 0.70) / max(local_ratio, 1.0), 1)
        min_s = min(10.0, max_s)
        return min_s, max_s, hints

    ratio = float(hints.get("slowest_to_fastest_ratio") or 1.0)
    if task_timeout is None and hints.get("max_task_s_for_slowest_tier") is not None:
        max_s = float(hints["max_task_s_for_slowest_tier"])
    else:
        max_s = round((eff_to * 0.70) / max(ratio, 1.0), 1)
    rec_range = hints.get("recommended_task_s_on_fastest")
    if isinstance(rec_range, list) and len(rec_range) >= 2:
        min_s = float(rec_range[0])
    else:
        min_s = 30.0
    return min(min_s, max_s), max_s, hints


def _compute_profile_row_bounds(
    profile: dict[str, Any] | None,
    *,
    min_fastest_s: float,
    max_fastest_s: float,
    pair_aligned: bool,
) -> tuple[int | None, int | None, float | None]:
    """Return `(recommended_min_rows, recommended_max_rows, sec_per_row_med)` for current fleet bounds."""
    if not isinstance(profile, dict):
        return None, None, None

    spr_med = profile.get("sec_per_row_med")
    spr_p90 = profile.get("sec_per_row_p90")
    spr_med_f = float(spr_med) if spr_med is not None and float(spr_med) > 0.0 else None
    spr_p90_f = float(spr_p90) if spr_p90 is not None and float(spr_p90) > 0.0 else spr_med_f

    max_rows_candidates: list[int] = []
    if spr_med_f is not None:
        basis_spr = max(spr_med_f, (spr_p90_f or spr_med_f) * 0.78)
        if basis_spr > 0.0:
            max_rows_candidates.append(_align_rows(int(max_fastest_s // basis_spr), pair_aligned=pair_aligned))

    coarse_cap = profile.get("coarse_cap_rows")
    if coarse_cap is not None and int(coarse_cap) > 0:
        cap_val = int(coarse_cap)
        ref_max_s = float(profile.get("coarse_cap_rec_max_s") or 0.0)
        if ref_max_s > 0.0 and max_fastest_s > 0.0:
            scaled = int(math.floor(cap_val * (max_fastest_s / ref_max_s)))
            max_rows_candidates.append(_align_rows(scaled, pair_aligned=pair_aligned))
        else:
            max_rows_candidates.append(_align_rows(cap_val, pair_aligned=pair_aligned))

    rec_max_rows = min(max_rows_candidates) if max_rows_candidates else None

    rec_min_rows: int | None = None
    min_candidates: list[int] = []
    if spr_med_f is not None and min_fastest_s > 0.0:
        min_candidates.append(_align_rows(int(math.ceil(min_fastest_s / spr_med_f)), pair_aligned=pair_aligned, round_up=True))
    fine_floor = profile.get("fine_floor_rows")
    if fine_floor is not None and int(fine_floor) > 0:
        min_candidates.append(_align_rows(int(fine_floor), pair_aligned=pair_aligned, round_up=True))
    if min_candidates:
        rec_min_rows = max(min_candidates)
        if rec_max_rows is not None and rec_min_rows > rec_max_rows:
            rec_min_rows = rec_max_rows

    return rec_min_rows, rec_max_rows, spr_med_f


def _is_local_only(colab_nodes: str | None) -> bool:
    eff_colab = (colab_nodes if colab_nodes is not None else os.environ.get("NC2000_COLAB", "")).strip().lower()
    return eff_colab in ("0", "off", "false", "none", "local")


def recommend_chunk_size(
    requested_rows: int,
    *,
    local_only: bool | None = None,
    colab_nodes: str | None = None,
    preview_iters: int | None = None,
    battle_iters: int | None = None,
    battle_mode: str | None = None,
    preview_cache: bool = True,
    preview_dist_cache: Path | None = None,
    preview_dist_k: int | None = None,
    preview_dist_only: bool = False,
    task_timeout: int | None = None,
    targets_resp: dict[str, Any] | None = None,
    state_path: Path | None = None,
) -> int:
    """Adapt chunk size from live `barn targets` `dispatch_hints` and persisted `granularity_feedback`."""
    pair_aligned = not preview_dist_only
    base = _align_rows(max(1, int(requested_rows)), pair_aligned=pair_aligned, round_up=True)
    if os.environ.get("NC2000_BARN_AUTO_SPLIT", "1").strip().lower() in ("0", "false", "off", "no"):
        return base

    eff_local_only = _is_local_only(colab_nodes) if local_only is None else bool(local_only)
    has_dist_file = preview_dist_cache is not None and preview_dist_cache.exists()
    profile_key = _workload_profile_key(
        preview_iters=preview_iters,
        battle_iters=battle_iters,
        battle_mode=battle_mode,
        preview_cache=preview_cache,
        has_dist_file=has_dist_file,
        preview_dist_k=preview_dist_k,
        preview_dist_only=preview_dist_only,
    )
    state = _load_granularity_state(state_path)
    profile = state.get("profiles", {}).get(profile_key)
    if not isinstance(profile, dict):
        return base

    tg = targets_resp if targets_resp is not None else query_barn_targets()
    min_s, max_s, _ = _effective_duration_bounds(tg, local_only=eff_local_only, task_timeout=task_timeout)
    rec_min_rows, rec_max_rows, _ = _compute_profile_row_bounds(
        profile,
        min_fastest_s=min_s,
        max_fastest_s=max_s,
        pair_aligned=pair_aligned,
    )
    out = base
    if rec_max_rows is not None and out > rec_max_rows:
        out = rec_max_rows
    if rec_min_rows is not None and out < rec_min_rows and profile.get("fine_floor_rows"):
        out = rec_min_rows
    return _align_rows(out, pair_aligned=pair_aligned)


def record_granularity_feedback(
    *,
    profile_key: str,
    dispatched_row_counts: list[int],
    worker_stats: dict[str, Any] | None,
    granularity_feedback: dict[str, Any] | None,
    eff_min_fastest_s: float,
    eff_max_fastest_s: float,
    pair_aligned: bool,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Persist `barn dispatch` `worker_stats` + `granularity_feedback` to adapt future runs."""
    if not dispatched_row_counts:
        return {"recommended_max_rows": None, "recommended_min_rows": None}

    avg_rows = sum(dispatched_row_counts) / max(1, len(dispatched_row_counts))
    max_dispatched_rows = max(dispatched_row_counts)
    fb = granularity_feedback if isinstance(granularity_feedback, dict) else {}
    ws = worker_stats if isinstance(worker_stats, dict) else {}

    fastest_med_s: float | None = fb.get("fastest_med_s") if fb.get("fastest_med_s") is not None else None
    fastest_p90_s: float | None = fb.get("fastest_p90_s") if fb.get("fastest_p90_s") is not None else None
    if fastest_med_s is None and ws:
        completed = [
            (float(st["med_s"]), float(st["p90_s"]) if st.get("p90_s") is not None else float(st["med_s"]))
            for st in ws.values()
            if isinstance(st, dict) and int(st.get("n") or 0) > 0 and st.get("med_s") is not None
        ]
        if completed:
            fastest_med_s, fastest_p90_s = min(completed, key=lambda kv: kv[0])
    if fastest_p90_s is None:
        fastest_p90_s = fastest_med_s

    rec_max_s = float(fb.get("recommended_max_fastest_s") or eff_max_fastest_s)
    status = str(fb.get("status") or "ok")

    def _apply(state: dict[str, Any]) -> dict[str, Any]:
        profiles = state.setdefault("profiles", {})
        ent = dict(profiles.get(profile_key) or {})

        if fastest_med_s is not None and fastest_med_s > 0.0 and avg_rows > 0.0:
            obs_med = fastest_med_s / avg_rows
            obs_p90 = (fastest_p90_s or fastest_med_s) / avg_rows
            prev_med = ent.get("sec_per_row_med")
            prev_p90 = ent.get("sec_per_row_p90")
            if prev_med is not None and float(prev_med) > 0.0:
                ent["sec_per_row_med"] = round(0.5 * float(prev_med) + 0.5 * obs_med, 4)
            else:
                ent["sec_per_row_med"] = round(obs_med, 4)
            if prev_p90 is not None and float(prev_p90) > 0.0:
                ent["sec_per_row_p90"] = round(0.5 * float(prev_p90) + 0.5 * obs_p90, 4)
            else:
                ent["sec_per_row_p90"] = round(obs_p90, 4)

        if status == "too_coarse":
            split_f = max(2, int(fb.get("suggested_split_factor") or 2))
            split_cap = _align_rows(int(max_dispatched_rows // split_f), pair_aligned=pair_aligned)
            ent["coarse_cap_rows"] = split_cap
            ent["coarse_cap_rec_max_s"] = round(rec_max_s, 1)
            ent.pop("fine_floor_rows", None)
        elif status == "too_fine":
            merge_f = max(2, int(fb.get("suggested_merge_factor") or 2))
            ent["fine_floor_rows"] = _align_rows(int(math.ceil(avg_rows * merge_f)), pair_aligned=pair_aligned, round_up=True)
            ent.pop("coarse_cap_rows", None)
            ent.pop("coarse_cap_rec_max_s", None)
        elif status == "ok":
            # Relax prior coarse cap once an actual run completes cleanly with comfortable headroom.
            if fastest_med_s is not None and rec_max_s > 0.0:
                basis_s = max(fastest_med_s, (fastest_p90_s or fastest_med_s) * 0.78)
                if basis_s <= rec_max_s * 0.75:
                    ent.pop("coarse_cap_rows", None)
                    ent.pop("coarse_cap_rec_max_s", None)
            if fastest_med_s is not None and fastest_med_s >= eff_min_fastest_s * 0.5:
                ent.pop("fine_floor_rows", None)

        rec_min_rows, rec_max_rows, _ = _compute_profile_row_bounds(
            ent,
            min_fastest_s=eff_min_fastest_s,
            max_fastest_s=rec_max_s,
            pair_aligned=pair_aligned,
        )
        ent["recommended_max_rows"] = rec_max_rows
        ent["recommended_min_rows"] = rec_min_rows
        ent["last_feedback"] = fb or None
        ent["samples"] = int(ent.get("samples", 0)) + 1
        ent["updated_at"] = round(time.time(), 1)
        profiles[profile_key] = ent
        return {
            "recommended_max_rows": rec_max_rows,
            "recommended_min_rows": rec_min_rows,
            "sec_per_row_med": ent.get("sec_per_row_med"),
            "sec_per_row_p90": ent.get("sec_per_row_p90"),
        }

    return _update_granularity_state(_apply, state_path)


def _merge_subchunk_outputs(
    parent: dict[str, Any],
    sub_items: list[dict[str, Any]],
    sub_records: dict[str, dict[str, Any]],
    *,
    preview_dist_only: bool,
) -> None:
    """Atomically reassemble split sub-chunk outputs into the parent chunk's `out` dir and `done` marker."""
    final_out: Path = parent["out"]
    stage_out = final_out.parent / f".{final_out.name}.merge-{secrets.token_hex(3)}"
    if stage_out.exists():
        shutil.rmtree(stage_out)
    stage_out.mkdir(parents=True, exist_ok=True)
    try:
        if preview_dist_only:
            merged_entries: dict[str, Any] = {}
            fmt = "nc2000-preview-dist-v1"
            policy_epoch: str | None = None
            for sub in sub_items:
                p = sub["out"] / "preview-dist.json"
                doc = json.loads(p.read_text(encoding="utf-8"))
                fmt = doc.get("format", fmt)
                if doc.get("policy_epoch") is not None:
                    policy_epoch = doc["policy_epoch"]
                for khex, ent in (doc.get("entries") or {}).items():
                    prev = merged_entries.get(khex)
                    if prev is None or int(ent.get("k", 0)) >= int(prev.get("k", 0)):
                        merged_entries[khex] = ent
            out_doc: dict[str, Any] = {"format": fmt, "entries": merged_entries}
            if policy_epoch is not None:
                out_doc["policy_epoch"] = policy_epoch
            (stage_out / "preview-dist.json").write_text(
                json.dumps(out_doc, separators=(",", ":")),
                encoding="utf-8",
            )
        else:
            new_dist_entries: dict[str, Any] = {}
            new_dist_fmt = "nc2000-preview-dist-v1"
            new_dist_epoch: str | None = None
            manifest_copied = False
            for sub in sub_items:
                offset = int(sub["offset"])
                n_sub = int(sub["n_rows"])
                sub_out: Path = sub["out"]
                if not manifest_copied and (sub_out / "manifest.json").is_file():
                    shutil.copy2(sub_out / "manifest.json", stage_out / "manifest.json")
                    manifest_copied = True
                for j in range(n_sub):
                    src_f = sub_out / f"{j}.json"
                    rec = json.loads(src_f.read_text(encoding="utf-8"))
                    if isinstance(rec, dict):
                        rec["index"] = offset + j
                    (stage_out / f"{offset + j}.json").write_text(
                        json.dumps(rec, separators=(",", ":")),
                        encoding="utf-8",
                    )
                new_dist_p = sub_out / "preview-dist-new.json"
                if new_dist_p.is_file():
                    nd = json.loads(new_dist_p.read_text(encoding="utf-8"))
                    new_dist_fmt = nd.get("format", new_dist_fmt)
                    if nd.get("policy_epoch") is not None:
                        new_dist_epoch = nd["policy_epoch"]
                    for khex, ent in (nd.get("entries") or {}).items():
                        prev = new_dist_entries.get(khex)
                        if prev is None or int(ent.get("k", 0)) >= int(prev.get("k", 0)):
                            new_dist_entries[khex] = ent
            if new_dist_entries:
                nd_doc: dict[str, Any] = {"format": new_dist_fmt, "entries": new_dist_entries}
                if new_dist_epoch is not None:
                    nd_doc["policy_epoch"] = new_dist_epoch
                (stage_out / "preview-dist-new.json").write_text(
                    json.dumps(nd_doc, separators=(",", ":")),
                    encoding="utf-8",
                )

        if final_out.exists():
            shutil.rmtree(final_out)
        stage_out.rename(final_out)
    except Exception:
        shutil.rmtree(stage_out, ignore_errors=True)
        raise

    workers_used = [str(sub_records[sub["task_id"]].get("worker", "unknown")) for sub in sub_items]
    total_ms = sum(int(sub_records[sub["task_id"]].get("wall_ms") or 0) for sub in sub_items)
    parent["done"].write_text(json.dumps({
        "host": "barn-subsplit",
        "sub_chunks": len(sub_items),
        "workers": workers_used,
        "total_ms": total_ms,
    }) + "\n")


def run_chunks_via_barn(
    pending: list[dict[str, Any]],
    *,
    binary: Path,
    colab_binary: Path | None,
    pools_path: Path | None,
    local_workers: int,
    mini2_workers: int,
    local_only: bool = False,
    colab_threads: int | None = None,
    preview_iters: int | None = None,
    battle_iters: int | None = None,
    battle_mode: str | None = None,
    preview_cache: bool = True,
    preview_dist_cache: Path | None = None,
    preview_dist_k: int | None = None,
    preview_dist_only: bool = False,
    task_timeout: int | None = None,
    skipped: int = 0,
    auto_adapt_granularity: bool = True,
    granularity_state_path: Path | None = None,
) -> dict[str, Any]:
    """Unified transparent work-stealing via `barn dispatch` (governed by ~/.config/barn/policy.json).

    Closes the M2M granularity feedback loop:
    1. Queries `barn targets` `dispatch_hints` and persisted per-workload calibration before dispatch.
    2. Automatically sub-splits oversized pending chunks (pair-aligned) and passes `--est-sec` + `--sort-by-weight`.
    3. Records post-dispatch `worker_stats` + `granularity_feedback` and surfaces them in returned telemetry.
    """
    has_dist_file = preview_dist_cache is not None and preview_dist_cache.exists()
    pair_aligned = not preview_dist_only
    profile_key = _workload_profile_key(
        preview_iters=preview_iters,
        battle_iters=battle_iters,
        battle_mode=battle_mode,
        preview_cache=preview_cache,
        has_dist_file=has_dist_file,
        preview_dist_k=preview_dist_k,
        preview_dist_only=preview_dist_only,
    )
    auto_split_env = os.environ.get("NC2000_BARN_AUTO_SPLIT", "1").strip().lower() not in ("0", "false", "off", "no")
    adapt_enabled = bool(auto_adapt_granularity and auto_split_env)

    targets_resp = query_barn_targets() if adapt_enabled else None
    eff_min_s, eff_max_s, _ = _effective_duration_bounds(
        targets_resp,
        local_only=local_only,
        task_timeout=task_timeout,
    )
    state = _load_granularity_state(granularity_state_path)
    profile = state.get("profiles", {}).get(profile_key)
    _, max_rows_cap, spr_med_f = _compute_profile_row_bounds(
        profile,
        min_fastest_s=eff_min_s,
        max_fastest_s=eff_max_s,
        pair_aligned=pair_aligned,
    )
    if not adapt_enabled:
        max_rows_cap = None

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="nc2000-barn-") as td:
        td_p = Path(td)
        tasks_path = td_p / "tasks.jsonl"
        ledger_path = td_p / "ledger.jsonl"

        dispatch_items: list[dict[str, Any]] = []
        direct_by_task: dict[str, dict[str, Any]] = {}
        split_groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        subsplit_applied = False

        for ch in pending:
            n_rows = int(ch.get("n_rows", 1))
            if max_rows_cap is not None and n_rows > max_rows_cap and ch["sched"].is_file():
                raw_lines = [l for l in ch["sched"].read_text(encoding="utf-8").splitlines() if l.strip()]
                step = _align_rows(max_rows_cap, pair_aligned=pair_aligned)
                if len(raw_lines) > step:
                    sub_items: list[dict[str, Any]] = []
                    for si, off in enumerate(range(0, len(raw_lines), step)):
                        sub_lines = raw_lines[off : off + step]
                        tid = f"{ch['ci']}s{si}"
                        sub_sched = td_p / f"sub-{ch['ci']}-{si}.jsonl"
                        sub_out = td_p / f"sub-{ch['ci']}-{si}-out"
                        sub_sched.write_text("\n".join(sub_lines) + "\n", encoding="utf-8")
                        sub_rec = {
                            "task_id": tid,
                            "sched": sub_sched,
                            "out": sub_out,
                            "n_rows": len(sub_lines),
                            "offset": off,
                        }
                        sub_items.append(sub_rec)
                        dispatch_items.append(sub_rec)
                    split_groups.append((ch, sub_items))
                    subsplit_applied = True
                    continue

            tid = str(ch["ci"])
            direct_by_task[tid] = ch
            dispatch_items.append({
                "task_id": tid,
                "sched": ch["sched"].resolve(),
                "out": ch["out"].resolve(),
                "n_rows": n_rows,
            })

        dispatched_row_counts = [int(it["n_rows"]) for it in dispatch_items]
        avg_dispatched_rows = sum(dispatched_row_counts) / max(1, len(dispatched_row_counts))
        est_sec: float | None = None
        if spr_med_f is not None and avg_dispatched_rows > 0.0:
            est_sec = round(max(1.0, avg_dispatched_rows * spr_med_f), 1)

        lines = [
            json.dumps({
                "task": str(it["task_id"]),
                "sched": str(Path(it["sched"]).resolve()),
                "out": str(Path(it["out"]).resolve()),
                "weight": int(it["n_rows"]),
            }, separators=(",", ":"))
            for it in dispatch_items
        ]
        tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd: list[str] = [
            "barn",
            "dispatch",
            "--bin", f"bin/nash_refresh={binary.resolve()}",
            "--local-threads", str(max(1, local_workers)),
            "--mini2-threads", str(max(1, mini2_workers)),
            "--tasks-file", str(tasks_path),
            "--resume", str(ledger_path),
            "--sort-by-weight",
        ]
        if task_timeout is not None:
            cmd += ["--task-timeout", str(task_timeout)]
        if est_sec is not None:
            cmd += ["--est-sec", str(est_sec)]
        if colab_binary is not None and not local_only:
            cmd += ["--bin", f"linux-x86_64:bin/nash_refresh={colab_binary.resolve()}"]
        if local_only:
            cmd.append("--local-only")
        if colab_threads is not None:
            cmd += ["--colab-threads", str(colab_threads)]
        for rel in NASH_REFRESH_DATA_FILES:
            cmd += ["--file", f"{rel}={(ROOT / rel).resolve()}"]
        if pools_path is not None:
            cmd += ["--file", f"pools.json={pools_path.resolve()}"]
        if has_dist_file:
            cmd += ["--file", f"preview-dist.json={preview_dist_cache.resolve()}"]
        cmd += [
            "--input", "schedule.jsonl={sched}",
            "--collect", "out={out}",
            "--env", "NC2000_ROOT={run}",
            "--env", f"NASH_PREVIEW_CACHE={'1' if preview_cache else '0'}",
        ]
        if pools_path is not None:
            cmd += ["--env", "NASH_POOLS={run}/pools.json"]
        if has_dist_file:
            cmd += ["--env", "NASH_PREVIEW_DIST_CACHE={run}/preview-dist.json"]
        if preview_dist_k is not None:
            cmd += ["--env", f"NASH_PREVIEW_DIST_K={preview_dist_k}"]
        if preview_dist_only:
            cmd += ["--env", "NASH_PREVIEW_DIST_ONLY=1"]
        if preview_iters is not None:
            cmd += ["--env", f"NASH_PREVIEW_ITERS={preview_iters}"]
        if battle_iters is not None:
            cmd += ["--env", f"NASH_BATTLE_ITERS={battle_iters}"]
        if battle_mode is not None:
            cmd += ["--env", f"NASH_BATTLE_MODE={battle_mode}"]
        cmd += [
            "--",
            "{run}/bin/nash_refresh",
            "{run}/schedule.jsonl",
            "{run}/out",
            "{threads}",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok_records: dict[str, dict[str, Any]] = {}
        if ledger_path.is_file():
            for raw in ledger_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                if not rec.get("ok"):
                    continue
                tid = str(rec["task"])
                ok_records[tid] = rec
                ch = direct_by_task.get(tid)
                if ch is None:
                    continue
                worker = str(rec.get("worker", "unknown"))
                if worker == "mini1":
                    ch["done"].write_text(str(rec.get("stdout_last", "")) + "\n")
                else:
                    ch["done"].write_text(json.dumps({
                        "host": worker,
                        "remote_stdout_last": rec.get("stdout_last", ""),
                        "remote_wall_ms": rec.get("remote_wall_ms", rec.get("wall_ms")),
                        "total_ms": rec.get("wall_ms"),
                    }) + "\n")

        for parent, sub_items in split_groups:
            if all(sub["task_id"] in ok_records for sub in sub_items):
                _merge_subchunk_outputs(
                    parent,
                    sub_items,
                    ok_records,
                    preview_dist_only=preview_dist_only,
                )

        if proc.returncode != 0:
            raise RuntimeError(f"barn dispatch failed (rc={proc.returncode}):\n{proc.stderr or proc.stdout}")
        tel = json.loads(proc.stdout.strip().splitlines()[-1])
        next_cal = record_granularity_feedback(
            profile_key=profile_key,
            dispatched_row_counts=dispatched_row_counts,
            worker_stats=tel.get("worker_stats"),
            granularity_feedback=tel.get("granularity_feedback"),
            eff_min_fastest_s=eff_min_s,
            eff_max_fastest_s=eff_max_s,
            pair_aligned=pair_aligned,
            state_path=granularity_state_path,
        )
        by_w: dict[str, int] = tel.get("by_worker", {})
        ran_local = int(by_w.get("mini1", 0))
        ran_mini2 = int(by_w.get("mini2", 0))
        ran_colab = sum(int(v) for k, v in by_w.items() if k.startswith("colab:"))
        out_tel: dict[str, Any] = {
            "ran_local": ran_local,
            "ran_mini2": ran_mini2,
            "ran_colab": ran_colab,
            "skipped": skipped,
            "by_worker": by_w,
            "barn_requeued": tel.get("requeued", 0),
            "timeout_requeues": tel.get("timeout_requeues", 0),
            "timeouts_by_class": tel.get("timeouts_by_class", {}),
            "skipped_infeasible_classes": tel.get("skipped_infeasible_classes", []),
            "hedged": tel.get("hedged", 0),
            "hedged_wins": tel.get("hedged_wins", 0),
            "tail_drain_s": tel.get("tail_drain_s"),
            "worker_stats": tel.get("worker_stats", {}),
            "granularity_feedback": tel.get("granularity_feedback"),
            "barn_dispatched_tasks": len(dispatch_items),
            "barn_next_max_rows": next_cal.get("recommended_max_rows"),
        }
        if subsplit_applied and max_rows_cap is not None:
            out_tel["barn_subsplit_max_rows"] = max_rows_cap
        if est_sec is not None:
            out_tel["barn_est_sec"] = est_sec
        return out_tel


def run_chunks_dual_node(
    chunks: list[dict[str, Any]],
    *,
    binary: Path,
    pools_path: Path | None,
    local_workers: int = 10,
    mini2_workers: int = 11,
    use_mini2: bool = True,
    colab_nodes: str | None = None,
    colab_binary: Path | None = None,
    colab_slots: int | None = None,
    colab_threads: int | None = None,
    colab_cu_budget: float | None = None,
    preview_iters: int | None = None,
    battle_iters: int | None = None,
    battle_mode: str | None = None,
    preview_cache: bool = True,
    preview_dist_cache: Path | None = None,
    preview_dist_k: int | None = None,
    preview_dist_only: bool = False,
    task_timeout: int | None = None,
    auto_adapt_granularity: bool = True,
) -> dict[str, Any]:
    """Dynamic work-stealing execution of chunk list across `mini1` + `mini2` + Colab via `barn`.

    When `use_mini2=True` (and `barn` is installed), delegates transparently to `barn dispatch`
    which reads `~/.config/barn/policy.json` to decide whether to autoscale Colab alongside
    `mini1`+`mini2` (passing `linux-x86_64` binary automatically when fresh on disk).
    Pass `colab_nodes='off'` or `NC2000_COLAB=off` to force `--local-only`.
    """
    pending: list[dict[str, Any]] = []
    skipped = 0
    for ch in chunks:
        if ch["done"].exists() and ch["out"].exists():
            if preview_dist_only:
                if (ch["out"] / "preview-dist.json").exists():
                    skipped += 1
                    continue
            else:
                existing = [p for p in ch["out"].glob("*.json") if p.name not in ("manifest.json", "preview-dist-new.json")]
                if len(existing) == ch["n_rows"]:
                    skipped += 1
                    continue
        pending.append(ch)

    if not pending:
        return {"ran_local": 0, "ran_mini2": 0, "ran_colab": 0, "skipped": skipped}

    eff_colab = (colab_nodes if colab_nodes is not None else os.environ.get("NC2000_COLAB", "")).strip().lower()
    local_only = eff_colab in ("0", "off", "false", "none", "local")
    use_barn = shutil.which("barn") is not None and (use_mini2 or (eff_colab and not local_only))
    if use_barn:
        x86_bin = None if local_only else find_fresh_colab_binary(binary, colab_binary)
        eff_threads = colab_threads
        if eff_threads is None and os.environ.get("NC2000_COLAB_THREADS"):
            eff_threads = int(os.environ["NC2000_COLAB_THREADS"])
        return run_chunks_via_barn(
            pending,
            binary=binary,
            colab_binary=x86_bin,
            pools_path=pools_path,
            local_workers=local_workers,
            mini2_workers=mini2_workers,
            local_only=local_only or (x86_bin is None),
            colab_threads=eff_threads,
            preview_iters=preview_iters,
            battle_iters=battle_iters,
            battle_mode=battle_mode,
            preview_cache=preview_cache,
            preview_dist_cache=preview_dist_cache,
            preview_dist_k=preview_dist_k,
            preview_dist_only=preview_dist_only,
            task_timeout=task_timeout,
            skipped=skipped,
            auto_adapt_granularity=auto_adapt_granularity,
        )

    lock = threading.Lock()
    cursor = [0]
    counts = {"mini1": 0, "mini2": 0}

    def next_chunk() -> dict[str, Any] | None:
        with lock:
            while cursor[0] < len(pending):
                ch = pending[cursor[0]]
                cursor[0] += 1
                if ch["done"].exists() and ch["out"].exists():
                    continue
                return ch
            return None

    def worker_loop(node: str) -> None:
        while True:
            ch = next_chunk()
            if ch is None:
                return
            if node == "mini2":
                run_chunk_mini2(
                    binary=binary,
                    pools_path=pools_path,
                    schedule_path=ch["sched"],
                    final_out_dir=ch["out"],
                    done_marker=ch["done"],
                    workers=mini2_workers,
                    run_id=f"nc2000-c{ch['ci']:03d}-{secrets.token_hex(3)}",
                    preview_iters=preview_iters,
                    battle_iters=battle_iters,
                    battle_mode=battle_mode,
                    preview_cache=preview_cache,
                    preview_dist_cache=preview_dist_cache,
                    preview_dist_k=preview_dist_k,
                    preview_dist_only=preview_dist_only,
                )
            else:
                run_chunk_local(
                    binary=binary,
                    pools_path=pools_path,
                    schedule_path=ch["sched"],
                    final_out_dir=ch["out"],
                    done_marker=ch["done"],
                    workers=local_workers,
                    preview_iters=preview_iters,
                    battle_iters=battle_iters,
                    battle_mode=battle_mode,
                    preview_cache=preview_cache,
                    preview_dist_cache=preview_dist_cache,
                    preview_dist_k=preview_dist_k,
                    preview_dist_only=preview_dist_only,
                )
            with lock:
                counts[node] += 1

    nodes = ["mini1", "mini2"] if (use_mini2 and len(pending) >= 2) else ["mini1"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futs = [pool.submit(worker_loop, n) for n in nodes]
        for fut in futs:
            fut.result()

    return {"ran_local": counts["mini1"], "ran_mini2": counts["mini2"], "skipped": skipped}


def main() -> None:
    """Standalone CLI to offload specific chunk indices of a prepared RR dir to `mini2`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="Prepared RR dir with meta.json & schedule.jsonl")
    ap.add_argument("--chunks", type=str, required=True, help="Comma-separated chunk indices, e.g. 6,5,4")
    ap.add_argument("--chunk-games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--preview-iters", type=int, default=None)
    ap.add_argument("--battle-iters", type=int, default=None)
    ap.add_argument("--battle-mode", choices=["adaptive", "fixed"], default=None)
    args = ap.parse_args()

    meta = json.loads((args.dir / "meta.json").read_text())
    pools_path = Path(meta["pools"])
    rows = [l.strip() for l in (args.dir / "schedule.jsonl").open() if l.strip()]
    chunk_size = max(2, args.chunk_games + (args.chunk_games % 2))
    chunks_dir = args.dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(x.strip()) for x in args.chunks.split(",") if x.strip()]
    t0 = time.monotonic()
    ran = []
    skipped = []
    for ci in indices:
        c_rows = rows[ci * chunk_size : (ci + 1) * chunk_size]
        if not c_rows:
            continue
        c_sched = chunks_dir / f"chunk-{ci:03d}.jsonl"
        c_out = chunks_dir / f"chunk-{ci:03d}-out"
        c_done = chunks_dir / f"chunk-{ci:03d}.done.json"
        # Skip if already done OR already actively being worked on by mini1
        if c_done.exists() or c_out.exists():
            skipped.append(ci)
            continue
        c_sched.write_text("\n".join(c_rows) + "\n")
        run_chunk_mini2(
            binary=args.binary,
            pools_path=pools_path,
            schedule_path=c_sched,
            final_out_dir=c_out,
            done_marker=c_done,
            workers=args.workers,
            run_id=f"nc2000-p0-c{ci:03d}-{secrets.token_hex(3)}",
            preview_iters=args.preview_iters,
            battle_iters=args.battle_iters,
            battle_mode=args.battle_mode,
        )
        ran.append(ci)

    print(json.dumps({
        "host": "mini2",
        "dir": str(args.dir),
        "ran_chunks": ran,
        "skipped_chunks": skipped,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }))


if __name__ == "__main__":
    main()
