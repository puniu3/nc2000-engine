import collections
import importlib.util
import json
import pathlib
import statistics
import sys

spec = importlib.util.spec_from_file_location("paired", pathlib.Path(__file__).with_name("endgame-match-report.py"))
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def distribution(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": paired.percentile(values, .95), "p99": paired.percentile(values, .99), "max": max(values, default=None)}


def timing(directory, protocol):
    rows = [json.loads(line) for line in (directory / "timing.jsonl").read_text().splitlines()]
    games = [row for row in rows if row["type"] == "result"]
    decisions = [row for row in rows if row["type"] == "decision"]
    assert all(row["iterations"] == protocol["iterations"] for row in decisions)
    assert all(row["request_ms"] >= row["search_ms"] for row in decisions)
    complete = len(games) == protocol["games"]
    phases = {phase: {"request_ms": distribution([row["request_ms"] for row in decisions if row["phase"] == phase]),
                      "search_ms": distribution([row["search_ms"] for row in decisions if row["phase"] == phase])}
              for phase in ("preview", "normal", "small-endgame", "forced")}
    nonforced = [row["request_ms"] for row in decisions if row["phase"] != "forced"]
    return {"complete": complete, "games": len(games), "expected_games": protocol["games"], "phases": phases,
            "all_request_ms": distribution([row["request_ms"] for row in decisions]),
            "nonforced_request_ms": distribution(nonforced),
            "nonvisible_decisions": sum(row["visibility"] != "visible" for row in decisions),
            "environment": next(row for row in rows if row["type"] == "environment"),
            "interpretation": "Measured local Chromium waits at the shipped fixed iteration count. Phase differences are retained. This does not establish iPad timing or a matched-time candidate result."}


def matches(directory, protocol):
    pairs = collections.defaultdict(dict)
    samples = [collections.defaultdict(list), collections.defaultdict(list)]
    metrics = [collections.Counter(), collections.Counter()]
    caps = 0
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
        for row in rows[1:]:
            if row["type"] != "decision":
                continue
            for arm in (0, 1):
                record = row["sides"][header["candidate_side"] ^ arm]
                if record.get("wait"):
                    continue
                m = metrics[arm]
                m["decisions"] += 1
                m["forced"] += record["forced"]
                m["illegal"] += not record["legal"]
                m["nonvisible"] += record["visibility"] != "visible"
                m["iterations"] += record["iterations"]
                for name in ("request_ms", "search_ms", "wasm_memory_bytes"):
                    if record.get(name) is not None:
                        samples[arm][name].append(record[name])
                if not record["forced"]:
                    samples[arm]["nonforced_request_ms"].append(record["request_ms"])
                if record["endgame"] is not None:
                    assert record["endgame"]["scope"] == "conditional-determinized"
                    stats = record["endgame"]["stats"]
                    assert stats["root_overrides"] == 0
                    for name in ("calls", "cache_hits", "value_certificates", "value_replacements", "unresolved", "solver_ms"):
                        m[name] += stats[name]
                    m["decisions_with_replacements"] += stats["value_replacements"] > 0
                    m["peak_solver_entries"] = max(m["peak_solver_entries"], stats["peak_entries"])
                    fraction = protocol["candidate" if arm == 0 else "baseline"]["solver_fraction"]
                    m["budget_violations"] += stats["solver_ms"] > protocol["milliseconds"] * fraction + stats["max_probe_ms"] + .1
        if rows[-1]["type"] == "result":
            result = rows[-1]
            assert (result["pair"], result["leg"]) == (header["pair"], header["leg"])
            assert result["leg"] not in pairs[result["pair"]]
            pairs[result["pair"]][result["leg"]] = result["candidate_score"]
            caps += result["engine_cap"]
    scores = [statistics.mean(pairs[index].values()) for index in sorted(pairs) if len(pairs[index]) == 2]
    complete = len(scores) == protocol["pairs"]
    arms = {name: {**metrics[index], **{field: distribution(samples[index][field]) for field in ("request_ms", "search_ms", "nonforced_request_ms", "wasm_memory_bytes")}}
            for index, name in enumerate(("candidate", "baseline"))}
    candidate_mean, baseline_mean = (arms[name]["nonforced_request_ms"]["mean"] for name in ("candidate", "baseline"))
    return {"complete": complete, "completed_games": sum(map(len, pairs.values())), "completed_pairs": len(scores),
            "expected_games": protocol["pairs"] * 2, "score": statistics.mean(scores) if complete else None,
            "ci95": paired.interval(scores) if complete else None, "pair_scores": scores if complete else None,
            "arms": arms, "engine_caps": caps, "nonforced_time_ratio": candidate_mean / baseline_mean if candidate_mean is not None and baseline_mean else None,
            "calibration": protocol["calibration"], "interpretation": "Paired browser outcomes. Timing and available memory measurements are separate from score; no automatic publication decision. Missing memory is unmeasured, not zero. Local Chromium is not iPad Safari."}


def report(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    result = timing(directory, protocol) if protocol["schema"] == "nc2000-web-timing-v1" else matches(directory, protocol)
    result["manifest"] = json.loads((directory / "manifest.json").read_text())
    return result


if __name__ == "__main__":
    print(json.dumps(report(pathlib.Path(sys.argv[1])), indent=2, allow_nan=False))
