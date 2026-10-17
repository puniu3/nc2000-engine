#!/usr/bin/env python3
"""Finish only the already-registered blocks, stopping on a verified positive checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def state(status, **extra):
    value = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
             "controller_pid": os.getpid(), "controller_sha256": SOURCE_SHA256, **extra}
    target = HERE / "controller.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(target)


def audit():
    subprocess.run([sys.executable, str(HERE / "summarize.py")], cwd=ROOT, check=True)
    return read(HERE / "summary.json")


def complete_segment(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows and rows[-1]["type"] == "segment_complete", "Previous segment did not finish cleanly"
    return rows[-1]["last_index"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--previous-stdout", type=Path, required=True)
    args = parser.parse_args()
    previous = args.previous_stdout.resolve()
    assert previous.parent == HERE
    # Wait only while the original, specifically identified driver is live.
    while True:
        check = subprocess.run(["ps", "-p", str(args.wait_pid), "-o", "command="],
                               text=True, capture_output=True)
        if check.returncode != 0:
            break
        assert "headroom-scale-v1/run.py" in check.stdout, "PID now belongs to a different process"
        state("waiting_for_segment", driver_pid=args.wait_pid, previous_stdout=previous.name)
        time.sleep(30)
    previous_last = complete_segment(previous)
    while True:
        summary = audit()
        assert summary["completed_blocks"] == previous_last + 1
        if summary["positive"]:
            state("positive_checkpoint_verified", games=summary["games"], score=summary["score"],
                  betting_interval=summary["betting_interval"])
            return
        protocol = read(HERE / "protocol.json")
        first = read(HERE / "results/block-00.json")
        binary = ROOT / "target/release/examples/arena"
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == first["binary_sha256"], "Measured executable changed"
        start = summary["completed_blocks"]
        remaining = len(protocol["blocks"]) - start
        if remaining == 0:
            state("registered_sample_cap_reached", games=summary["games"],
                  score=summary["score"], betting_interval=summary["betting_interval"])
            return
        assert remaining > 0
        count = min(4, remaining)
        stdout = HERE / f"execution-from-{start:02}.stdout.jsonl"
        stderr = HERE / f"execution-from-{start:02}.log"
        # Exclusive creation prevents a second controller overwriting evidence.
        with stdout.open("x") as out, stderr.open("x") as err:
            command = [sys.executable, str(HERE / "run.py"), "--start", str(start),
                       "--count", str(count), "--threads", "10"]
            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
            state("running_segment", driver_pid=child.pid, start=start, count=count,
                  stdout=stdout.name, stderr=stderr.name)
            code = child.wait()
            assert code == 0, f"Segment exited {code}; preserve artifacts and inspect before resuming"
        previous_last = complete_segment(stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        state("needs_inspection", error=str(error))
        raise
