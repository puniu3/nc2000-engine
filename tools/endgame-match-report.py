import json
import math
import pathlib
import statistics
import sys


def interval(scores, bounded=True):
    if len(scores) < 2:
        return None
    z = 1.959963984540054
    df = len(scores) - 1
    critical = z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2) + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    error = critical * statistics.stdev(scores) / math.sqrt(len(scores))
    mean = statistics.mean(scores)
    return [max(0, mean - error), min(1, mean + error)] if bounded else [mean - error, mean + error]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    protocol = manifest["protocol"]
    pairs = {}
    metrics = [{"milliseconds": [], "iterations": 0, "peak_search_nodes": 0, "peak_solver_entries": 0,
                "calls": 0, "replacements": 0, "decisions_with_replacements": 0, "solver_ms": 0,
                "budget_violations": 0, "node_limit_violations": 0, "illegal": 0,
                "forced": 0, "decisions": 0, "max_probe_ms": 0, "projections": 0, "legality_drift": 0} for _ in range(2)]
    caps = 0
    decisions = 0
    for path in sorted(directory.glob("pair-*.jsonl")):
        lines = path.read_text().splitlines()
        rows = []
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
        if not rows:
            continue
        header = rows[0]
        assert header["type"] == "game"
        game_peaks = [{"projections": 0, "legality_drift": 0} for _ in range(2)]
        for row in rows[1:]:
            if row["type"] != "decision":
                continue
            decisions += 1
            for arm in (0, 1):
                side = header["candidate_side"] ^ arm
                record = row["sides"][side]
                if record.get("wait"):
                    continue
                m = metrics[arm]
                stats = record["stats"]
                m["decisions"] += 1
                m["forced"] += record["forced"]
                m["milliseconds"].append(record["elapsed_ms"])
                m["iterations"] += record["iterations"]
                m["peak_search_nodes"] = max(m["peak_search_nodes"], record["search_nodes"])
                m["peak_solver_entries"] = max(m["peak_solver_entries"], stats["peak_entries"])
                m["calls"] += stats["calls"]
                m["replacements"] += stats["value_replacements"]
                m["decisions_with_replacements"] += stats["value_replacements"] > 0
                m["solver_ms"] += stats["solver_ms"]
                probe = stats.get("max_probe_ms", 0)
                m["max_probe_ms"] = max(m["max_probe_ms"], probe)
                arm_config = protocol["candidate" if arm == 0 else "baseline"]
                m["budget_violations"] += stats["solver_ms"] > protocol["milliseconds"] * arm_config["solver_fraction"] + probe + 0.1
                m["node_limit_violations"] += stats["peak_entries"] > 12000
                m["illegal"] += not record["legal"]
                for name in game_peaks[arm]:
                    game_peaks[arm][name] = max(game_peaks[arm][name], record[name])
        for arm in (0, 1):
            for name, value in game_peaks[arm].items():
                metrics[arm][name] += value
        if rows[-1]["type"] == "result":
            result = rows[-1]
            assert (result["pair"], result["leg"]) == (header["pair"], header["leg"])
            pair = pairs.setdefault(result["pair"], {})
            assert result["leg"] not in pair
            pair[result["leg"]] = result["candidate_score"]
            caps += result["engine_cap"]
    pair_scores = [statistics.mean(pairs[i].values()) for i in sorted(pairs) if len(pairs[i]) == 2]
    complete = len(pair_scores) == protocol["pairs"]
    for m in metrics:
        values = m.pop("milliseconds")
        m.update(mean_ms=statistics.mean(values) if values else None, p95_ms=percentile(values, .95), p99_ms=percentile(values, .99), max_ms=max(values, default=None))
    return {
        "revision": manifest["revision"], "binary_sha256": manifest["binary_sha256"],
        "protocol": protocol, "complete": complete,
        "completed_games": sum(len(pair) for pair in pairs.values()),
        "completed_pairs": len(pair_scores), "expected_games": protocol["pairs"] * 2,
        "decisions": decisions, "engine_caps": caps, "metrics": dict(zip(("candidate", "baseline"), metrics)),
        "pair_scores": pair_scores if complete else None,
        "score": statistics.mean(pair_scores) if complete else None,
        "ci95": interval(pair_scores) if complete else None,
        "interpretation": "Native development evidence only; incomplete runs have no reported score. No publication decision is made here.",
    }


def report(root):
    if (root / "manifest.json").exists():
        return summarize(root)
    arms = {p.name: summarize(p) for p in sorted(root.glob("development-*")) if p.is_dir() and (p / "manifest.json").exists()}
    complete = len(arms) == 4 and all(arm["complete"] for arm in arms.values())
    selected = None
    if complete:
        schedules = [json.loads((root / name / "manifest.json").read_text())["schedule"] for name in arms]
        assert all(schedule == schedules[0] for schedule in schedules)
        off = arms["development-00"]["pair_scores"]
        for arm in arms.values():
            differences = [on - baseline for on, baseline in zip(arm["pair_scores"], off)]
            arm["difference_vs_public_off"] = {"mean": statistics.mean(differences), "ci95": interval(differences, bounded=False), "clusters": len(differences)}
        eligible = []
        for name, arm in arms.items():
            metrics = arm["metrics"]["candidate"]
            if arm["protocol"]["candidate"]["solver_fraction"] == 0:
                continue
            if metrics["illegal"] or metrics["budget_violations"] or metrics["node_limit_violations"]:
                continue
            if metrics["p99_ms"] > 1.1 * arm["protocol"]["milliseconds"]:
                continue
            eligible.append(name)
        if eligible:
            best = max(arms[name]["score"] for name in eligible)
            selected = min((name for name in eligible if arms[name]["score"] >= best - .005), key=lambda name: arms[name]["protocol"]["candidate"]["solver_fraction"])
    memory = {}
    path = root / "rss.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            for process in sample["processes"]:
                memory[process["arm"]] = max(memory.get(process["arm"], 0), process["rss_kib"])
    return {"complete": complete, "arms": arms, "selected_for_confirmation": selected, "sampled_process_peak_rss_kib": memory,
            "interpretation": "Process RSS includes both seats and allocator retention; it is not a per-seat memory measurement. Selection is a development choice, not a confirmed strength claim."}


if __name__ == "__main__":
    print(json.dumps(report(pathlib.Path(sys.argv[1])), indent=2, allow_nan=False))
