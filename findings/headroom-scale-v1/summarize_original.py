#!/usr/bin/env python3
"""Independently audit complete scale blocks and recompute the registered interval."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = read(HERE / "protocol.json")
    for name, expected in protocol["inputs"].items():
        assert sha(ROOT / name) == expected, f"Input changed: {name}"
    spec = importlib.util.spec_from_file_location("intervals", ROOT / "tools/evaluate-learning.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    files = sorted((HERE / "results").glob("block-??.json"))
    all_pairs = []
    rows = []
    fingerprints = None
    for index, path in enumerate(files):
        row = read(path)
        assert index == row["index"] and index < len(protocol["blocks"])
        assert row["protocol_sha256"] == sha(HERE / "protocol.json")
        assert row["runner_sha256"] == protocol["inputs"]["findings/headroom-scale-v1/run.py"]
        raw_path = path.with_name(path.stem + "-raw.jsonl")
        assert sha(raw_path) == row["raw_sha256"]
        raw = read(raw_path)
        subprocess.run([str(ROOT / "target/release/examples/arena"), "--validate-jsonl", str(raw_path)],
                       check=True, stdout=subprocess.DEVNULL)
        for field in ("schema", "agent_a", "agent_b", "config", "result", "fingerprints", "root_search"):
            assert row[field] == raw[field], f"Summary/raw mismatch: {field}"
        c, r = row["config"], row["result"]
        assert row["agent_a"] == protocol["a"] + ":16" and row["agent_b"] == protocol["b"] + ":16"
        assert c["base_seed"] == protocol["blocks"][index]["seed"]
        assert c["max_turns"] == protocol["max_turns"] and c["crn_agent_seeds"] is False
        assert c["team_pool_path"] == protocol["team_pool"] and c["belief_pool_path"] == protocol["belief_pool"]
        assert c["teams"] == 24 and c["baked_tables"] == 0
        if fingerprints is None:
            fingerprints = row["fingerprints"]
        assert row["fingerprints"] == fingerprints, "Runtime or data changed between blocks"
        assert c["requested_games"] == r["games"] == protocol["games_per_block"]
        assert len(r["game_scores"]) == r["games"] and r["turn_caps"] == r["invalid_games"] == 0
        assert all(x in (0, 0.5, 1) for x in r["game_scores"])
        pairs = [(a+b)/2 for a, b in zip(r["game_scores"][::2], r["game_scores"][1::2])]
        assert pairs == r["pair_scores"] and len(pairs) == r["pairs"]
        assert [r["game_scores"].count(x) for x in (1, 0, 0.5)] == [r["wins"], r["losses"], r["ties"]]
        assert math.isclose(statistics.mean(pairs), r["score"])
        all_pairs.extend(pairs)
        rows.append({"index": index, "seed": c["base_seed"], "score": r["score"],
                     "games": r["games"], "raw_sha256": row["raw_sha256"], "timing": row["timing"]})
    summary = {"schema": "nc2000-headroom-scale-summary-v1", "protocol_sha256": sha(HERE / "protocol.json"),
               "completed_blocks": len(files), "games": 2*len(all_pairs), "blocks": rows,
               "pair_scores": all_pairs, "alpha": protocol["alpha"], "sample_cap_games": protocol["sample_cap_games"],
               "positive": False, "conclusion": "Awaiting sufficient complete-block evidence"}
    if all_pairs:
        interval = evaluator.betting_interval(all_pairs, alpha=protocol["alpha"])
        summary.update(score=statistics.mean(all_pairs), betting_interval=interval,
                       normal95=evaluator.score_intervals(all_pairs)["normal95"], positive=interval[0] > .5)
        if summary["positive"]:
            summary["conclusion"] = "Demonstrated extra-compute headroom under the registered conditions"
        else:
            summary["conclusion"] = "No positive decision yet; this does not establish saturation"
    output = HERE / "summary.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, indent=2) + "\n")
    temp.replace(output)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("blocks", "pair_scores")}, indent=2))


if __name__ == "__main__":
    main()
