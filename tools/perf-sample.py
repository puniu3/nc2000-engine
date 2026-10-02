#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    sample = args.directory / "sample.txt"
    with (args.directory / "profile-run.json").open("w") as output:
        process = subprocess.Popen([str(args.binary.resolve()), "blind", "27000", "1"],
                                   stdout=output, stderr=subprocess.PIPE, text=True)
        ready = False
        for line in process.stderr:
            if line.startswith("ready:"):
                ready = True
                subprocess.run(["sample", str(process.pid), "55", "1", "-file", str(sample)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        code = process.wait()
        if not ready or code:
            raise RuntimeError(f"profile workload failed ({code})")
    text = sample.read_text()
    graph = text.split("Call graph:", 1)[1].split("Total number in stack", 1)[0]
    threads = [int(m.group(1)) for line in graph.splitlines()
               if (m := re.match(r"^\s*(\d+) Thread_", line)) and "com.apple.main-thread" not in line]
    assert len(threads) == 1, threads
    hashes = 0
    ancestor_depth = None
    for line in graph.splitlines():
        match = re.match(r"^([ +!:|]*)(\d+) ", line)
        if not match:
            continue
        depth = len(match.group(1))
        if ancestor_depth is not None and depth <= ancestor_depth:
            ancestor_depth = None
        if ancestor_depth is None and ("state_hash_into" in line or "10search_key " in line):
            hashes += int(match.group(2))
            ancestor_depth = depth
    if not hashes:
        raise RuntimeError("No hash frame found; inspect inlined frames before attributing cost")
    fraction = hashes / threads[0]
    assert 0 <= fraction <= 1
    result = {"worker_samples": threads[0], "state_hash_inclusive_samples": hashes,
              "state_hash_fraction": fraction, "free_state_hash_speedup_ceiling": 1 / (1 - fraction)}
    (args.directory / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
