import collections
import importlib.util
import json
import pathlib
import statistics
import sys

module = importlib.util.spec_from_file_location("paired", pathlib.Path(__file__).with_name("endgame-match-report.py"))
paired = importlib.util.module_from_spec(module)
module.loader.exec_module(paired)


def guarantee(record, reference, side):
    stats = record["stats"]
    result = {"loss_lower": None, "loss_upper": None, "source": "unresolved"}
    if stats["root_overrides"]:
        result.update(loss_upper=0.02, source="candidate-strategy-certificate")
    bounds = reference.get("bounds")
    if bounds is None:
        return result
    pure = [a for a in record.get("policy") or [] if a["probability"] > 1e-10]
    if len(pure) != 1 or pure[0]["probability"] < 1 - 1e-10:
        return result
    action = next((a for a in reference["actions"][side] if a["action"] == pure[0]["action"]), None)
    if action is None:
        return result
    lo, hi = (bounds["lo"], bounds["hi"]) if side == 0 else (1 - bounds["hi"], 1 - bounds["lo"])
    result["loss_lower"] = max(0, lo - action["hi"])
    upper = max(0, hi - action["lo"])
    result["loss_upper"] = min(upper, result["loss_upper"]) if result["loss_upper"] is not None else upper
    result["source"] = result["source"] + "+reference-pure-action" if stats["root_overrides"] else "reference-pure-action"
    assert result["loss_lower"] <= result["loss_upper"] + 1e-7, "inconsistent policy guarantees"
    return result


def report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    groups = collections.defaultdict(dict)
    metrics = collections.defaultdict(lambda: collections.Counter())
    times = collections.defaultdict(list)
    first = {}
    complete_games = 0
    for path in sorted(directory.glob("game-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        header = rows[0]
        key = (header["information"], header["source_game"], header["source_decision"])
        leg = header["candidate_side"]
        assert header["type"] == "continuation"
        if rows[-1]["type"] == "result":
            assert leg not in groups[key]
            groups[key][leg] = rows[-1]["candidate_score"]
            complete_games += 1
        for row in rows[1:]:
            if row["type"] != "decision":
                continue
            if row["step"] == 0:
                first[(*key, leg)] = row
            for side, record in enumerate(row["sides"]):
                if record.get("wait"):
                    continue
                name = (key[0], "candidate" if side == leg else "baseline")
                m = metrics[name]
                stats = record["stats"]
                times[name].append(record["elapsed_ms"])
                m["decisions"] += 1
                m["forced"] += record["forced"]
                m["illegal"] += not record["legal"]
                m["calls"] += stats["calls"]
                m["unresolved_calls"] += stats["unresolved"]
                m["value_certificates"] += stats["value_certificates"]
                m["replacements"] += stats["value_replacements"]
                m["decisions_with_replacements"] += stats["value_replacements"] > 0
                m["root_overrides"] += stats["root_overrides"]
                m["solver_ms"] += stats["solver_ms"]
                m["peak_solver_entries"] = max(m["peak_solver_entries"], stats["peak_entries"])
    complete = complete_games == manifest["expected_continuations"]
    results = {}
    for information in ("full", "public"):
        clusters = collections.defaultdict(list)
        for (info, game, _), legs in groups.items():
            if info == information and len(legs) == 2:
                clusters[game].append(statistics.mean(legs.values()))
        scores = [statistics.mean(clusters[game]) for game in sorted(clusters)]
        results[information] = {"source_game_clusters": len(scores), "position_pairs": sum(map(len, clusters.values())),
                                "score": statistics.mean(scores) if complete and scores else None,
                                "ci95": paired.interval(scores) if complete else None,
                                "cluster_scores": dict(zip(sorted(clusters), scores)) if complete else None}
        for arm in ("candidate", "baseline"):
            key = (information, arm)
            results[information][arm] = {**metrics[key], "p99_ms": paired.percentile(times[key], .99), "max_ms": max(times[key], default=None)}
    endpoints = []
    if complete:
        for (information, game, decision), legs in sorted(groups.items()):
            if information != "full":
                continue
            reference = json.loads((directory / f"game-{game:03}-d{decision:04}-reference.json").read_text())
            for side in (0, 1):
                candidate = first[(information, game, decision, side)]["sides"][side]
                baseline = first[(information, game, decision, 1 - side)]["sides"][side]
                if candidate.get("wait"):
                    assert baseline.get("wait")
                    continue
                on, off = guarantee(candidate, reference, side), guarantee(baseline, reference, side)
                on_right = on["loss_upper"] is not None and on["loss_upper"] <= .0200000001
                off_right = off["loss_upper"] is not None and off["loss_upper"] <= .0200000001
                off_wrong = off["loss_lower"] is not None and off["loss_lower"] > .0200000001
                endpoints.append({"source_game": game, "source_decision": decision, "side": side,
                                  "candidate": on, "baseline": off, "same_chosen_input": candidate["choice"] == baseline["choice"],
                                  "classification": "proven-error-corrected" if off_wrong and on_right else "already-correct-maintained" if off_right and on_right else "unresolved-comparison"})
    return {"complete": complete, "completed_continuations": complete_games, "expected_continuations": manifest["expected_continuations"],
            "source_games": manifest["source_games"], "source_games_without_samples": manifest["source_games_without_samples"],
            "results": results, "full_information_endpoints": endpoints,
            "endpoint_classes": dict(collections.Counter(row["classification"] for row in endpoints)),
            "interpretation": "Each source game has equal weight; multiple endgame positions from a game are one cluster. Policy loss claims apply only to full-information play. Missing or broad references remain unresolved; public-information strength is judged by its separate continuation outcomes."}


if __name__ == "__main__":
    print(json.dumps(report(pathlib.Path(sys.argv[1])), indent=2, allow_nan=False))
