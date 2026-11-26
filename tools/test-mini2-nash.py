#!/usr/bin/env python3
"""Unit tests for the M2M barn granularity closed loop in `tools/mini2_nash.py`."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mini2_nash", TOOLS_DIR / "mini2_nash.py")
assert SPEC is not None and SPEC.loader is not None
MINI2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MINI2)


FAKE_HYBRID_TARGETS = {
    "ok": True,
    "targets": [
        {"id": "darwin-arm64", "slots": 2, "cores": 22, "nodes": ["mini1", "mini2"]},
        {"id": "linux-x86_64", "max_slots": 11, "tiers": ["V5E1", "V6E1", "CPU-HM"]},
    ],
    "dispatch_hints": {
        "local_slots": 2,
        "cloud_max_slots": 9,
        "live_cloud_slots": 2,
        "total_max_slots": 13,
        "default_task_timeout_s": 600.0,
        "worker_rel_dur": {
            "mini2": 1.0,
            "mini1": 1.2,
            "colab:V5E1": 2.3,
            "colab:V6E1": 2.4,
            "colab:CPU-HM": 5.2,
        },
        "slowest_to_fastest_ratio": 5.2,
        "max_task_s_for_slowest_tier": 80.8,
        "recommended_task_s_on_fastest": [30.0, 80.8],
    },
}


def _make_chunks(base_dir: Path, row_counts: list[int]) -> list[dict]:
    chunks = []
    for ci, n_rows in enumerate(row_counts):
        sched = base_dir / f"chunk-{ci:03d}.jsonl"
        out = base_dir / f"chunk-{ci:03d}-out"
        done = base_dir / f"chunk-{ci:03d}.done.json"
        rows = [json.dumps({"block": ci * 100 + i, "rowSeat": i % 2, "tag": f"c{ci}-r{i}"}) for i in range(n_rows)]
        sched.write_text("\n".join(rows) + "\n", encoding="utf-8")
        chunks.append({"ci": ci, "sched": sched, "out": out, "done": done, "n_rows": n_rows})
    return chunks


def _extract_flag(cmd: list[str], flag: str) -> str | None:
    if flag in cmd:
        idx = cmd.index(flag)
        if idx + 1 < len(cmd):
            return cmd[idx + 1]
    return None


class Mini2BarnGranularityLoopTests(unittest.TestCase):
    def test_too_coarse_feedback_changes_next_run_subsplit_and_reassembles_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            state_path = td / "granularity.json"
            fake_bin = td / "nash_refresh"
            fake_x86 = td / "nash_refresh_x86"
            fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_x86.write_text("#!/bin/sh\n", encoding="utf-8")

            # --- Run 1: coarse 32-game chunks hit too_coarse (split_factor=3, skewed p90) ---
            run1_dir = td / "run1"
            run1_dir.mkdir()
            chunks1 = _make_chunks(run1_dir, [32, 32])
            seen_cmds: list[list[str]] = []

            def fake_barn_run1(cmd: list[str], **_kwargs):
                seen_cmds.append(list(cmd))
                tasks_file = Path(_extract_flag(cmd, "--tasks-file") or "")
                ledger_file = Path(_extract_flag(cmd, "--resume") or "")
                tasks = [json.loads(line) for line in tasks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                ledger_lines = []
                for t in tasks:
                    out_dir = Path(t["out"])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    rows = [json.loads(r) for r in Path(t["sched"]).read_text(encoding="utf-8").splitlines() if r.strip()]
                    for idx, row in enumerate(rows):
                        row["index"] = idx
                        (out_dir / f"{idx}.json").write_text(json.dumps(row), encoding="utf-8")
                    ledger_lines.append(json.dumps({"task": t["task"], "ok": True, "worker": "mini2", "wall_ms": 110_000}))
                ledger_file.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
                tel = {
                    "cmd": "dispatch",
                    "ok": True,
                    "by_worker": {"mini2": len(tasks)},
                    "requeued": 2,
                    "timeout_requeues": 2,
                    "timeouts_by_class": {"colab:CPU-HM": 2},
                    "skipped_infeasible_classes": [],
                    "hedged": 1,
                    "hedged_wins": 1,
                    "tail_drain_s": 400.0,
                    "worker_stats": {
                        "mini2": {"n": len(tasks), "med_s": 107.6, "p90_s": 242.3, "max_s": 310.0},
                        "colab:CPU-HM": {"n": 0, "med_s": None, "p90_s": None, "max_s": None, "timeouts": 2},
                    },
                    "granularity_feedback": {
                        "status": "too_coarse",
                        "reasons": ["timeout_requeues:2"],
                        "fastest_med_s": 107.6,
                        "fastest_p90_s": 242.3,
                        "recommended_max_fastest_s": 80.8,
                        "suggested_split_factor": 3,
                    },
                }
                return mock.Mock(returncode=0, stdout=json.dumps(tel) + "\n", stderr="")

            with (
                mock.patch.object(MINI2, "query_barn_targets", return_value=FAKE_HYBRID_TARGETS),
                mock.patch.object(MINI2.subprocess, "run", side_effect=fake_barn_run1),
            ):
                res1 = MINI2.run_chunks_via_barn(
                    chunks1,
                    binary=fake_bin,
                    colab_binary=fake_x86,
                    pools_path=None,
                    local_workers=10,
                    mini2_workers=12,
                    preview_iters=240000,
                    battle_iters=243000,
                    battle_mode="adaptive",
                    preview_cache=False,
                    granularity_state_path=state_path,
                )

            # Run 1 surfaced granularity_feedback instead of dropping it, and persisted a smaller cap (<= 10 rows)
            self.assertEqual(res1["barn_dispatched_tasks"], 2)
            self.assertEqual(res1["timeout_requeues"], 2)
            self.assertEqual(res1["timeouts_by_class"], {"colab:CPU-HM": 2})
            self.assertEqual(res1["granularity_feedback"]["status"], "too_coarse")
            self.assertIsNotNone(res1["barn_next_max_rows"])
            self.assertLessEqual(res1["barn_next_max_rows"], 10)
            self.assertEqual(res1["barn_next_max_rows"] % 2, 0)
            self.assertIn("--sort-by-weight", seen_cmds[0])
            self.assertIsNone(_extract_flag(seen_cmds[0], "--task-timeout"))

            # recommend_chunk_size now reacts immediately for fresh hybrid callers, but relaxes for local_only
            rec_hybrid = MINI2.recommend_chunk_size(
                32,
                local_only=False,
                preview_iters=240000,
                battle_iters=243000,
                battle_mode="adaptive",
                preview_cache=False,
                targets_resp=FAKE_HYBRID_TARGETS,
                state_path=state_path,
            )
            rec_local = MINI2.recommend_chunk_size(
                32,
                local_only=True,
                preview_iters=240000,
                battle_iters=243000,
                battle_mode="adaptive",
                preview_cache=False,
                targets_resp=FAKE_HYBRID_TARGETS,
                state_path=state_path,
            )
            self.assertLessEqual(rec_hybrid, 10)
            self.assertGreater(rec_local, rec_hybrid)

            # --- Run 2: caller STILL passes coarse 32-game chunks -> auto sub-splits & passes --est-sec! ---
            run2_dir = td / "run2"
            run2_dir.mkdir()
            chunks2 = _make_chunks(run2_dir, [32, 32])
            dispatched_weights_run2: list[int] = []

            def fake_barn_run2(cmd: list[str], **_kwargs):
                seen_cmds.append(list(cmd))
                tasks_file = Path(_extract_flag(cmd, "--tasks-file") or "")
                ledger_file = Path(_extract_flag(cmd, "--resume") or "")
                tasks = [json.loads(line) for line in tasks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                ledger_lines = []
                for t in tasks:
                    dispatched_weights_run2.append(int(t["weight"]))
                    out_dir = Path(t["out"])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    rows = [json.loads(r) for r in Path(t["sched"]).read_text(encoding="utf-8").splitlines() if r.strip()]
                    for idx, row in enumerate(rows):
                        row["index"] = idx
                        (out_dir / f"{idx}.json").write_text(json.dumps(row), encoding="utf-8")
                    (out_dir / "manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
                    (out_dir / "preview-dist-new.json").write_text(
                        json.dumps({"format": "nc2000-preview-dist-v1", "entries": {t["task"]: {"k": 4, "counts": {"a": 4}}}}),
                        encoding="utf-8",
                    )
                    ledger_lines.append(json.dumps({"task": t["task"], "ok": True, "worker": "colab:CPU-HM:v1", "wall_ms": 140_000}))
                ledger_file.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
                tel = {
                    "cmd": "dispatch",
                    "ok": True,
                    "by_worker": {"colab:CPU-HM:v1": len(tasks)},
                    "requeued": 0,
                    "timeout_requeues": 0,
                    "timeouts_by_class": {},
                    "skipped_infeasible_classes": [],
                    "hedged": 0,
                    "hedged_wins": 0,
                    "tail_drain_s": 40.0,
                    "worker_stats": {
                        "mini2": {"n": 2, "med_s": 34.0, "p90_s": 55.0, "max_s": 62.0},
                    },
                    "granularity_feedback": {
                        "status": "ok",
                        "reasons": [],
                        "fastest_med_s": 34.0,
                        "fastest_p90_s": 55.0,
                        "recommended_max_fastest_s": 80.8,
                    },
                }
                return mock.Mock(returncode=0, stdout=json.dumps(tel) + "\n", stderr="")

            with (
                mock.patch.object(MINI2, "query_barn_targets", return_value=FAKE_HYBRID_TARGETS),
                mock.patch.object(MINI2.subprocess, "run", side_effect=fake_barn_run2),
            ):
                res2 = MINI2.run_chunks_via_barn(
                    chunks2,
                    binary=fake_bin,
                    colab_binary=fake_x86,
                    pools_path=None,
                    local_workers=10,
                    mini2_workers=12,
                    preview_iters=240000,
                    battle_iters=243000,
                    battle_mode="adaptive",
                    preview_cache=False,
                    granularity_state_path=state_path,
                )

            # Verify Run 2 sub-split 2x32g into >= 8 pair-aligned sub-tasks of <= 10 games each!
            self.assertGreaterEqual(res2["barn_dispatched_tasks"], 8)
            self.assertLessEqual(max(dispatched_weights_run2), 10)
            self.assertTrue(all(w % 2 == 0 and w >= 2 for w in dispatched_weights_run2))
            self.assertEqual(res2["barn_subsplit_max_rows"], 10)
            self.assertIsNotNone(res2["barn_est_sec"])
            self.assertIsNotNone(_extract_flag(seen_cmds[1], "--est-sec"))

            # Verify transparent reassembly of parent chunk outputs {0..31}.json with exact indices + preview-dist-new!
            for ci, ch in enumerate(chunks2):
                self.assertTrue(ch["done"].is_file())
                self.assertTrue((ch["out"] / "manifest.json").is_file())
                for j in range(32):
                    rec = json.loads((ch["out"] / f"{j}.json").read_text(encoding="utf-8"))
                    self.assertEqual(rec["index"], j)
                    self.assertEqual(rec["tag"], f"c{ci}-r{j}")
                nd = json.loads((ch["out"] / "preview-dist-new.json").read_text(encoding="utf-8"))
                self.assertGreaterEqual(len(nd["entries"]), 4)

    def test_preview_dist_only_subsplit_merges_preview_dist_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            state_path = td / "granularity.json"
            fake_bin = td / "nash_refresh"
            fake_x86 = td / "nash_refresh_x86"
            fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_x86.write_text("#!/bin/sh\n", encoding="utf-8")

            profile_key = MINI2._workload_profile_key(
                preview_iters=729000,
                battle_iters=243000,
                battle_mode="adaptive",
                preview_cache=True,
                has_dist_file=False,
                preview_dist_k=16,
                preview_dist_only=True,
            )
            MINI2.record_granularity_feedback(
                profile_key=profile_key,
                dispatched_row_counts=[12],
                worker_stats={"mini2": {"n": 4, "med_s": 160.0, "p90_s": 170.0, "max_s": 180.0}},
                granularity_feedback={
                    "status": "too_coarse",
                    "reasons": ["timeout_requeues:2"],
                    "fastest_med_s": 160.0,
                    "fastest_p90_s": 170.0,
                    "recommended_max_fastest_s": 80.8,
                    "suggested_split_factor": 2,
                },
                eff_min_fastest_s=30.0,
                eff_max_fastest_s=80.8,
                pair_aligned=False,
                state_path=state_path,
            )

            chunks = _make_chunks(td, [12])

            def fake_barn_dist(cmd: list[str], **_kwargs):
                tasks_file = Path(_extract_flag(cmd, "--tasks-file") or "")
                ledger_file = Path(_extract_flag(cmd, "--resume") or "")
                tasks = [json.loads(line) for line in tasks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                ledger_lines = []
                for t in tasks:
                    out_dir = Path(t["out"])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    doc = {
                        "format": "nc2000-preview-dist-v1",
                        "policy_epoch": "policy-v3",
                        "entries": {
                            f"k-{t['task']}": {"k": 16, "counts": {"pick0": 16}},
                            "shared": {"k": 16 if t["task"].endswith("s1") else 8, "counts": {"pick1": 16}},
                        },
                    }
                    (out_dir / "preview-dist.json").write_text(json.dumps(doc), encoding="utf-8")
                    ledger_lines.append(json.dumps({"task": t["task"], "ok": True, "worker": "mini1", "wall_ms": 40_000}))
                ledger_file.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
                tel = {
                    "cmd": "dispatch",
                    "ok": True,
                    "by_worker": {"mini1": len(tasks)},
                    "worker_stats": {"mini1": {"n": len(tasks), "med_s": 40.0, "p90_s": 42.0, "max_s": 45.0}},
                    "granularity_feedback": {"status": "ok", "reasons": [], "fastest_med_s": 40.0, "fastest_p90_s": 42.0, "recommended_max_fastest_s": 80.8},
                }
                return mock.Mock(returncode=0, stdout=json.dumps(tel) + "\n", stderr="")

            with (
                mock.patch.object(MINI2, "query_barn_targets", return_value=FAKE_HYBRID_TARGETS),
                mock.patch.object(MINI2.subprocess, "run", side_effect=fake_barn_dist),
            ):
                res = MINI2.run_chunks_via_barn(
                    chunks,
                    binary=fake_bin,
                    colab_binary=fake_x86,
                    pools_path=None,
                    local_workers=10,
                    mini2_workers=12,
                    preview_iters=729000,
                    battle_iters=243000,
                    battle_mode="adaptive",
                    preview_cache=True,
                    preview_dist_k=16,
                    preview_dist_only=True,
                    granularity_state_path=state_path,
                )

            self.assertEqual(res["barn_dispatched_tasks"], 2)
            self.assertEqual(res["barn_subsplit_max_rows"], 6)
            merged = json.loads((chunks[0]["out"] / "preview-dist.json").read_text(encoding="utf-8"))
            self.assertEqual(merged["policy_epoch"], "policy-v3")
            self.assertIn("k-0s0", merged["entries"])
            self.assertIn("k-0s1", merged["entries"])
            self.assertEqual(merged["entries"]["shared"]["k"], 16)

    def test_too_fine_feedback_raises_recommended_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            state_path = Path(td_str) / "granularity.json"
            profile_key = MINI2._workload_profile_key(
                preview_iters=81000,
                battle_iters=9000,
                battle_mode="fixed",
                preview_cache=True,
                has_dist_file=False,
                preview_dist_k=None,
                preview_dist_only=False,
            )
            MINI2.record_granularity_feedback(
                profile_key=profile_key,
                dispatched_row_counts=[4, 4],
                worker_stats={"mini2": {"n": 2, "med_s": 2.0, "p90_s": 2.2, "max_s": 2.3}},
                granularity_feedback={
                    "status": "too_fine",
                    "reasons": ["task_too_short:2.0s"],
                    "fastest_med_s": 2.0,
                    "fastest_p90_s": 2.2,
                    "recommended_max_fastest_s": 80.8,
                    "recommended_min_fastest_s": 30.0,
                    "suggested_merge_factor": 15,
                },
                eff_min_fastest_s=30.0,
                eff_max_fastest_s=80.8,
                pair_aligned=True,
                state_path=state_path,
            )
            rec = MINI2.recommend_chunk_size(
                4,
                local_only=False,
                preview_iters=81000,
                battle_iters=9000,
                battle_mode="fixed",
                preview_cache=True,
                targets_resp=FAKE_HYBRID_TARGETS,
                state_path=state_path,
            )
            self.assertGreaterEqual(rec, 60)


if __name__ == "__main__":
    unittest.main()
