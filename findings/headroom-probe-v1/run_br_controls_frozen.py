#!/usr/bin/env python3
"""Resume the fixed BR-control experiment; never overwrite completed artifacts."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BIN = ROOT / "target/release/examples/arena"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(a, b, games, seed, blind, dest, crn=False):
    args = [str(BIN), a, b, "--teams", "data/team-pool-v2/team-pool.json",
            "--games", str(games), "--seed", str(seed), "--threads", "10",
            "--max-turns", "500", "--jsonl", str(dest)]
    if blind:
        args += ["--belief-pool", "data/belief-pool-v3/belief-pool.json"]
    if crn:
        args += ["--crn-seeds"]
    return args


def run(name, a, b, games, seed, blind, crn=False):
    dest = OUT / "duels" / (name + ".jsonl")
    args = command(a, b, games, seed, blind, dest, crn)
    save(OUT / "br-control-progress.json", {"status": "running", "batch": name,
         "command": args, "updated_at": datetime.now(timezone.utc).isoformat()})
    if not dest.exists():
        with (OUT / (name + ".log")).open("w") as log:
            subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(BIN), "--validate-jsonl", str(dest)], cwd=ROOT, check=True)
    row = read(dest)
    c, r = row["config"], row["result"]
    assert (c["requested_games"], c["base_seed"], c["crn_agent_seeds"]) == (games, seed, crn)
    assert r["games"] == games and r["pairs"] == games // 2
    assert len(r["pair_scores"]) == games // 2 and len(r["game_scores"]) == games
    assert r["turn_caps"] == 0 and r["invalid_games"] == 0
    assert math.isclose(statistics.mean(r["pair_scores"]), r["score"])
    return row


def main():
    targets = [("skuct", 2026101609), ("blind", 2026101610)]
    plan = {
        "schema": "nc2000-headroom-br-control-v1",
        "scope": "Two fixed 100-game oracle/control schedules, preceded by an 8-game CRN null.",
        "targets": targets, "target_c": 0.4, "own_iterations": 9000,
        "oracle_samples": 3, "rollout_status": True, "rollout_switch": True,
        "control": "Same BrAgent; oracle queries disabled; root opponent uses UCB.",
        "comparison": "Paired difference of side-swap means; descriptive normal 95% CI, no acceptance gate.",
        "limits": "Oracle adds 3 target searches; not equal total cost or a formal exploitability certificate. Both A arms see full state; blind B does not.",
        "binary_sha256": sha(BIN),
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT / "data/team-pool-v2/team-pool.json",
            ROOT / "data/belief-pool-v3/belief-pool.json",
            ROOT / "data/gen2stadium2.json",
            ROOT / "crates/bot/src/exploit.rs",
            ROOT / "crates/bot/examples/arena.rs",
            Path(__file__).resolve(),
            *[OUT / "duels" / f"br-exploit-{target}-3k.jsonl" for target, _ in targets]]},
    }
    protocol = OUT / "br-control-protocol.json"
    if protocol.exists():
        assert read(protocol) == plan, "Frozen inputs changed; use a separate experiment."
    else:
        save(protocol, plan)
    null = run("br-control-null", "exploit-control:blind:30:0.4",
               "exploit-control:blind:30:0.4", 8, 2026101611, True, True)
    assert null["result"]["score"] == 0.5 and null["result"]["split_pairs"] == 0
    results = []
    for target, seed in targets:
        spec = f"{target}:3000:0.4"
        oracle = run(f"br-replay-{target}-3k", "exploit:" + spec, spec, 100, seed, target == "blind")
        old = read(OUT / "duels" / f"br-exploit-{target}-3k.jsonl")
        assert oracle["config"] == old["config"], "Replay config drift"
        for key in ("game_scores", "pair_scores", "turns_sum", "turn_caps"):
            assert oracle["result"][key] == old["result"][key], f"Replay drift: {key}"
        for key in ("dex", "pool", "tables"):
            assert oracle["fingerprints"][key] == old["fingerprints"][key]
        control = run(f"br-control-{target}-3k", "exploit-control:" + spec, spec, 100, seed, target == "blind")
        assert oracle["fingerprints"] == control["fingerprints"]
        assert oracle["config"] == control["config"]
        diffs = [a-b for a, b in zip(oracle["result"]["pair_scores"], control["result"]["pair_scores"])]
        delta = statistics.mean(diffs)
        half = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))
        results.append({"target": spec, "games_per_arm": 100, "pairs": len(diffs),
            "oracle_score": oracle["result"]["score"], "control_score": control["result"]["score"],
            "oracle_ci95_half_width": oracle["result"]["ci95"],
            "control_ci95_half_width": control["result"]["ci95"],
            "paired_delta": delta, "paired_ci95": [delta-half, delta+half],
            "paired_differences": diffs, "old_outcomes_reproduced": True,
            "oracle_mean_ms": oracle["timing"]["a"]["mean_ms"],
            "control_mean_ms": control["timing"]["a"]["mean_ms"],
            "turn_caps": 0})
        save(OUT / "br-control-summary.json", {"complete": len(results) == 2, "results": results,
             "null_score": 0.5, "null_split_pairs": 0, "limits": plan["limits"]})
    save(OUT / "br-control-progress.json", {"status": "complete", "games": 408})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
