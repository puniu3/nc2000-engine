import collections
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import sys


root = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("web_report", root / "tools/endgame-web-report.py")
web_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web_report)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    design_path = root / "findings/endgame-integration-v1/confirmation-design.json"
    design = json.loads(design_path.read_text())
    result_path = directory / "web-result.json"
    result = json.loads(result_path.read_text())
    assert result == web_report.report(directory / "web")
    assert result["complete"] and result["completed_games"] == design["web"]["games"] == 1200
    assert result["completed_pairs"] == design["web"]["pairs"] == 600
    protocol = result["manifest"]["protocol"]
    assert protocol == json.loads((directory / "web-protocol.json").read_text())
    for key in ("pairs", "seed", "threads", "max_nodes"):
        assert protocol[key] == design["web"][key]
    assert protocol["iterations"] == design["web"]["baseline_iterations"]
    selection = json.loads((directory / "candidate-selection.json").read_text())
    selected = selection["arms"][selection["selected_for_confirmation"]]["protocol"]["candidate"]
    assert selection["complete"] and protocol["candidate"]["solver_fraction"] == selected["solver_fraction"]
    assert protocol["candidate"]["information"] == "public" and protocol["baseline"] == {"information": "raw", "solver_fraction": 0}
    timing = json.loads((directory / "timing-validation.json").read_text())
    expected_ms = min(814, math.floor(timing["arms"]["baseline"]["nonforced_request_ms"]["mean"] - (timing["arms"]["candidate"]["nonforced_request_ms"]["mean"] - 814)))
    assert timing["complete"] and protocol["milliseconds"] == expected_ms
    assert sha(directory / "web/protocol.json") == result["manifest"]["protocol_sha256"]
    for name in ("catalog", "prior"):
        assert sha(root / protocol[name]) == protocol[f"{name}_sha256"]
    frozen = json.loads((directory / "launcher.json").read_text())["source_sha256"]
    changed = [name for name, digest in frozen.items() if not (root / name).is_file() or sha(root / name) != digest]
    assert not changed, changed
    assert sha(root / "crates/wasm/pkg-web/nc2000_wasm_bg.wasm") == result["manifest"]["wasm_sha256"]
    catalog = json.loads((root / protocol["catalog"]).read_text())["teams"]
    schedule = result["manifest"]["schedule"]
    assert len(schedule) == 600 and [job["pair"] for job in schedule] == list(range(600))
    files = sorted((directory / "web").glob("pair-*.jsonl"))
    assert len(files) == 1200
    seen = set()
    scores = collections.defaultdict(list)
    counts = {name: collections.Counter() for name in ("candidate", "baseline")}
    forced = {name: [] for name in counts}
    max_probe = 0
    outcomes = collections.Counter()
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        header, *decisions, terminal = rows
        pair, leg = header["pair"], header["leg"]
        assert header["type"] == "game" and terminal["type"] == "result"
        assert (pair, leg) not in seen and leg in (0, 1)
        seen.add((pair, leg))
        job = schedule[pair]
        assert header["candidate_side"] == leg
        assert header["teams"] == [catalog[index]["id"] for index in job["teams"]]
        assert header["seed"] == job["seed"] and header["agent_seed"] == job["agentSeed"]
        assert terminal["pair"] == pair and terminal["leg"] == leg and len(decisions) <= 4000
        score = .5 if terminal["outcome"] == "tie" else int(terminal["outcome"] == f"p{leg + 1}")
        assert terminal["outcome"] in ("tie", "p1", "p2") and terminal["candidate_score"] == score
        scores[pair].append(score)
        outcomes[str(score)] += 1
        for index, row in enumerate(decisions):
            assert row["type"] == "decision" and (row["pair"], row["leg"], row["decision"]) == (pair, leg, index)
            for arm, side in (("candidate", leg), ("baseline", 1 - leg)):
                record = row["sides"][side]
                if record.get("wait"):
                    continue
                count = counts[arm]
                count["decisions"] += 1
                count["illegal"] += not record["legal"]
                count["nonvisible"] += record["visibility"] != "visible"
                memory = record.get("wasm_memory_bytes")
                count["missing_or_invalid_memory"] += not (isinstance(memory, int) and memory > 0 and memory % 65536 == 0)
                assert all(math.isfinite(record[key]) and record[key] >= 0 for key in ("request_ms", "search_ms", "iterations"))
                if record["forced"]:
                    forced[arm].append(record["request_ms"])
                if arm == "baseline":
                    assert record["iterations"] == protocol["iterations"] and record["endgame"] is None
                    continue
                endgame = record["endgame"]
                assert endgame["scope"] == "conditional-determinized"
                stats = endgame["stats"]
                assert all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in stats.values())
                count["public_root_overrides"] += stats["root_overrides"]
                count["public_root_changes"] += stats["root_changes"]
                count["cache_cap_violations"] += stats["peak_entries"] > protocol["max_nodes"]
                count["value_error_violations"] += stats["max_value_error"] > .01 + 1e-12
                count["budget_violations"] += stats["solver_ms"] > protocol["milliseconds"] * protocol["candidate"]["solver_fraction"] + stats["max_probe_ms"] + .1
                max_probe = max(max_probe, stats["max_probe_ms"])
    assert seen == {(pair, leg) for pair in range(600) for leg in (0, 1)}
    pair_scores = [statistics.mean(scores[pair]) for pair in range(600)]
    assert pair_scores == result["pair_scores"] and statistics.mean(pair_scores) == result["score"]
    for arm, count in counts.items():
        assert count["decisions"] == result["arms"][arm]["decisions"]
        assert len(forced[arm]) == result["arms"][arm]["forced"]
    candidate, baseline = (result["arms"][name] for name in ("candidate", "baseline"))
    gates = {
        "complete_fixed_sample": True,
        "minimum_score": result["score"] >= design["web"]["publication_gates"]["minimum_score"],
        "paired_ci_lower": result["ci95"][0] > design["web"]["publication_gates"]["strict_ci95_lower_above"],
        "nonforced_mean_latency": candidate["nonforced_request_ms"]["mean"] <= baseline["nonforced_request_ms"]["mean"],
        "overall_p99_latency": candidate["request_ms"]["p99"] <= baseline["request_ms"]["p99"],
        "complete_memory_measurements": all(count["missing_or_invalid_memory"] == 0 for count in counts.values()),
        "peak_wasm_allocation": candidate["wasm_memory_bytes"]["max"] <= baseline["wasm_memory_bytes"]["max"],
        "recorded_correctness": all(value == 0 for count in counts.values() for key, value in count.items() if key != "decisions"),
        "no_engine_caps": result["engine_caps"] == 0,
    }
    return {
        "schema": "nc2000-endgame-confirmation-audit-v1",
        "complete": True,
        "design_sha256": sha(design_path), "report_sha256": sha(result_path),
        "frozen_inputs_checked": len(frozen), "changed_inputs": changed,
        "games": len(files), "independent_pairs": len(scores), "outcomes_by_candidate_score": dict(outcomes),
        "score": result["score"], "ci95": result["ci95"], "gates": gates,
        "failed_measured_gates": [key for key, passed in gates.items() if not passed],
        "measured_gate_decision": "reject" if not all(gates.values()) else "requires_remaining_adoption_audit",
        "record_audit": {name: dict(count) for name, count in counts.items()},
        "forced_request_ms": {name: web_report.distribution(values) for name, values in forced.items()},
        "maximum_indivisible_probe_ms": max_probe,
        "nonforced_time_ratio": result["nonforced_time_ratio"],
        "peak_wasm_allocation_ratio": candidate["wasm_memory_bytes"]["max"] / baseline["wasm_memory_bytes"]["max"],
        "unresolved_call_fraction": candidate["unresolved"] / candidate["calls"],
        "decisions_with_replacements_fraction": candidate["decisions_with_replacements"] / candidate["decisions"],
        "limitations": [
            "Local Chromium measurements do not establish iPad Safari performance.",
            "Allocated Wasm linear memory is not total browser RSS.",
            "The candidate includes a public-information path change; this comparison does not isolate solver contribution.",
            "The fixed timing calibration did not achieve equal observed nonforced mean wait in confirmation; the candidate lost despite its larger measured time cost.",
            "Recorded correctness checks complement, and do not replace, implementation and information-boundary tests.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(pathlib.Path(sys.argv[1])), indent=2, allow_nan=False))
