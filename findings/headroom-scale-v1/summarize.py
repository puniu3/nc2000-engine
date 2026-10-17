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


def confirmation_analysis(blocks, amendment, evaluator):
    pairs, history, excluded = [], [], []
    first_positive = None
    for index, values in blocks:
        if index not in amendment["confirmation_block_indices"]:
            excluded.append({"block": index, "pairs": list(range(len(values))), "reason": "discovery block"})
            continue
        omit = amendment["exclude_pair_indices_per_confirmation_block"]
        assert all(0 <= i < len(values) for i in omit)
        pairs.extend(v for i, v in enumerate(values) if i not in omit)
        excluded.append({"block": index, "pairs": omit, "reason": "same-seed first game"})
        interval = evaluator.betting_interval(pairs, alpha=amendment["alpha"])
        checkpoint = {"block": index, "analysis_games": 2*len(pairs),
                      "score": statistics.mean(pairs), "betting_interval": interval}
        history.append(checkpoint)
        if first_positive is None and interval[0] > .5:
            first_positive = checkpoint
    decision = first_positive or (history[-1] if history else None)
    used = pairs[:decision["analysis_games"]//2] if decision else []
    result = {"available_confirmation_games": 2*len(pairs), "analysis_games": len(used)*2,
              "pair_scores": used, "confirmation_checkpoints": history, "exclusions": excluded,
              "first_positive_checkpoint": first_positive, "positive": first_positive is not None}
    if decision:
        result.update(score=decision["score"], betting_interval=decision["betting_interval"],
                      normal95=evaluator.score_intervals(used)["normal95"])
    return result


def main():
    protocol = read(HERE / "protocol.json")
    amendment_path = HERE / "confirmation-amendment.json"
    amendment = read(amendment_path)
    binding = read(HERE / "confirmation-implementation.json")
    assert binding["amendment_sha256"] == sha(amendment_path)
    assert binding["summarizer_sha256"] == sha(Path(__file__).resolve())
    assert amendment["original_protocol_sha256"] == sha(HERE / "protocol.json")
    assert amendment["alpha"] == protocol["alpha"]
    assert (amendment["candidate"], amendment["baseline"]) == (protocol["a"], protocol["b"])
    assert amendment["discovery_block_indices"] == [0]
    assert amendment["confirmation_block_indices"] == list(range(1, len(protocol["blocks"])))
    assert amendment["exclude_pair_indices_per_confirmation_block"] == [0]
    for name, expected in protocol["inputs"].items():
        assert sha(ROOT / name) == expected, f"Input changed: {name}"
    spec = importlib.util.spec_from_file_location("intervals", ROOT / "tools/evaluate-learning.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    files = sorted((HERE / "results").glob("block-??.json"))
    all_pairs = []
    block_pairs = []
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
        block_pairs.append((index, pairs))
        rows.append({"index": index, "seed": c["base_seed"], "score": r["score"],
                     "games": r["games"], "raw_sha256": row["raw_sha256"], "timing": row["timing"]})
    summary = {"schema": "nc2000-headroom-scale-summary-v1", "protocol_sha256": sha(HERE / "protocol.json"),
               "confirmation_amendment_sha256": sha(amendment_path),
               "completed_blocks": len(files), "games": 2*len(all_pairs), "blocks": rows,
               "raw_pair_scores": all_pairs, "raw_score": statistics.mean(all_pairs) if all_pairs else None,
               "alpha": protocol["alpha"], "sample_cap_games": protocol["sample_cap_games"],
               "sample_cap_confirmation_games": amendment["max_confirmation_games"],
               "positive": False, "conclusion": "Awaiting sufficient complete-block evidence"}
    summary.update(confirmation_analysis(block_pairs, amendment, evaluator))
    if summary["analysis_games"]:
        if summary["positive"]:
            summary["conclusion"] = "Demonstrated extra-compute headroom under the registered conditions"
        else:
            summary["conclusion"] = "No positive decision yet; this does not establish saturation"
    output = HERE / "summary.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, indent=2) + "\n")
    temp.replace(output)
    print(json.dumps({k: v for k, v in summary.items()
                     if k not in ("blocks", "pair_scores", "raw_pair_scores", "confirmation_checkpoints", "exclusions")}, indent=2))


if __name__ == "__main__":
    main()
