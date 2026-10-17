#!/usr/bin/env python3
"""Execute preregistered scale blocks; stdout retains compact recoverable rows."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol["inputs"].items():
        assert digest(ROOT / name) == expected, f"Input drift: {name}"
    binary = ROOT / "target/release/examples/arena"
    spec = importlib.util.spec_from_file_location("intervals", ROOT / "tools/evaluate-learning.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    result_dir = HERE / "results"
    result_dir.mkdir(exist_ok=True)
    identity = {"protocol_sha256": digest(protocol_path), "binary_sha256": digest(binary),
                "runner_sha256": digest(Path(__file__).resolve()), "platform": platform.platform(),
                "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()}
    emit({"type": "manifest", **identity, "start": args.start, "count": args.count})
    null_path = result_dir / "null.jsonl"
    subprocess.run([str(binary), "blind:100:0.4", "blind:100:0.4", "--teams", protocol["team_pool"],
                    "--belief-pool", protocol["belief_pool"], "--games", "8", "--seed", "202610162999",
                    "--threads", str(args.threads), "--crn-seeds", "--max-turns", "1100",
                    "--jsonl", str(null_path)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    null = json.loads(null_path.read_text())
    assert null["result"]["score"] == 0.5 and null["result"]["split_pairs"] == 0
    assert null["result"]["turn_caps"] == null["result"]["invalid_games"] == 0
    emit({"type": "null", "result": null["result"], "fingerprints": null["fingerprints"]})
    blocks = protocol["blocks"][args.start:args.start+args.count]
    assert len(blocks) == args.count and args.start >= 0 and args.count > 0
    all_pairs = []
    # Resume aggregation only includes prior complete blocks present locally.
    for index in range(args.start):
        previous = json.loads((result_dir / f"block-{index:02}.json").read_text())
        assert previous["protocol_sha256"] == identity["protocol_sha256"]
        all_pairs.extend(previous["result"]["pair_scores"])
    for block in blocks:
        index = block["index"]
        target = result_dir / f"block-{index:02}.json"
        assert not target.exists(), "Do not overwrite a completed block"
        raw_path = result_dir / f"block-{index:02}-raw.jsonl"
        command = [str(binary), protocol["a"], protocol["b"], "--teams", protocol["team_pool"],
                   "--belief-pool", protocol["belief_pool"], "--games", str(protocol["games_per_block"]),
                   "--seed", str(block["seed"]), "--threads", str(args.threads),
                   "--max-turns", str(protocol["max_turns"]), "--jsonl", str(raw_path)]
        emit({"type": "block_start", "index": index, "seed": block["seed"], "unix_time": time.time()})
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([str(binary), "--validate-jsonl", str(raw_path)], check=True, stdout=subprocess.DEVNULL)
        row = json.loads(raw_path.read_text())
        r = row["result"]
        assert r["games"] == protocol["games_per_block"] == 2*len(r["pair_scores"])
        assert len(r["game_scores"]) == r["games"]
        assert r["turn_caps"] == r["invalid_games"] == 0
        assert math.isclose(statistics.mean(r["pair_scores"]), r["score"])
        # Large timing arrays stay in the raw artifact; stdout carries complete
        # outcomes plus summary timing for recovery even if file transfer fails.
        row["timing"] = {k: ({kk: vv for kk, vv in v.items() if kk != "samples_ns"}
                            if isinstance(v, dict) else v) for k, v in row["timing"].items()}
        record = {"type": "block", "index": index, **identity, "raw_sha256": digest(raw_path), **row}
        target.write_text(json.dumps(record, indent=2) + "\n")
        emit(record)
        all_pairs.extend(r["pair_scores"])
        interval = evaluator.betting_interval(all_pairs, alpha=protocol["alpha"])
        result = {"type": "checkpoint", "index": index, "games": 2*len(all_pairs),
                  "score": statistics.mean(all_pairs), "betting_interval": interval,
                  "alpha": protocol["alpha"], "positive": interval[0] > 0.5}
        (result_dir / "checkpoint.json").write_text(json.dumps(result, indent=2) + "\n")
        emit(result)
        if result["positive"]:
            break
    emit({"type": "segment_complete", "last_index": index})


if __name__ == "__main__":
    main()
