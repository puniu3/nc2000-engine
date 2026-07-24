#!/usr/bin/env python3
"""Fail-closed validator and merger for M17d off-pool gauntlet shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

GAUNTLET_SCHEMA = "nc2000-m17d-offpool-gauntlet-v1"
MANIFEST_SCHEMA = "nc2000-m17d-offpool-shard-manifest-v1"
MASK64 = (1 << 64) - 1


class ValidationError(RuntimeError):
    pass


def compact_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def fingerprint(tag: str, value: Any) -> str:
    payload = compact_json(value)
    result = 0xCBF29CE484222325
    for byte in tag.encode():
        result = ((result ^ byte) * 0x100000001B3) & MASK64
    for byte in len(payload).to_bytes(8, "little") + payload:
        result = ((result ^ byte) * 0x100000001B3) & MASK64
    return f"fnv1a64:{result:016x}:{tag}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{path}: invalid JSON: {error}") from error


def load_jsonl(path: Path) -> list[Any]:
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise ValidationError(f"{path}: unreadable JSONL: {error}") from error
    require(text.endswith("\n"), f"{path}: truncated JSONL (missing final newline)")
    lines = text.splitlines()
    require(bool(lines), f"{path}: empty JSONL")
    parsed = []
    for line_number, line in enumerate(lines, 1):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValidationError(
                f"{path}:{line_number}: invalid JSON: {error}"
            ) from error
    return parsed


def semantic_run_projection(run: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "schema",
        "build",
        "inputs",
        "selected_custom",
        "selected_pilots",
        "semantic_config",
        "workload_fingerprint",
    ]
    require(all(key in run for key in keys), "run identity is incomplete")
    return {key: run[key] for key in keys}


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    require(isinstance(manifest, dict), f"{path}: manifest must be an object")
    expected_keys = {
        "schema",
        "run_fingerprint",
        "run",
        "total_jobs",
        "jobs",
        "shards",
        "manifest_fingerprint",
    }
    require(set(manifest) == expected_keys, f"{path}: unexpected manifest fields")
    require(manifest["schema"] == MANIFEST_SCHEMA, f"{path}: wrong manifest schema")
    body = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    require(
        manifest["manifest_fingerprint"]
        == fingerprint("m17d-shard-manifest-v1", body),
        f"{path}: manifest fingerprint mismatch",
    )

    run = manifest["run"]
    require(isinstance(run, dict), f"{path}: run identity must be an object")
    require(
        list(run)
        == [
            "schema",
            "build",
            "inputs",
            "selected_custom",
            "selected_pilots",
            "semantic_config",
            "workload_fingerprint",
        ],
        f"{path}: run identity field order/shape mismatch",
    )
    require(run["schema"] == GAUNTLET_SCHEMA, f"{path}: wrong gauntlet schema")
    require(
        manifest["run_fingerprint"] == fingerprint("m17d-run-v1", run),
        f"{path}: semantic run fingerprint mismatch",
    )

    jobs = manifest["jobs"]
    total = manifest["total_jobs"]
    require(isinstance(total, int) and total > 0, f"{path}: invalid total_jobs")
    require(isinstance(jobs, list) and len(jobs) == total, f"{path}: job count mismatch")
    require(
        run["workload_fingerprint"] == fingerprint("m17d-workload-v1", jobs),
        f"{path}: workload fingerprint mismatch",
    )
    for index, job in enumerate(jobs):
        require(isinstance(job, dict), f"{path}: job {index} is not an object")
        require(job.get("index") == index, f"{path}: non-contiguous job index {index}")
        require(
            isinstance(job.get("custom"), int)
            and 0 <= job["custom"] < len(run["selected_custom"]),
            f"{path}: job {index} has invalid custom index",
        )
        require(
            isinstance(job.get("pilot"), int)
            and 0 <= job["pilot"] < len(run["selected_pilots"]),
            f"{path}: job {index} has invalid pilot index",
        )
        require(
            isinstance(job.get("game"), int) and job["game"] >= 0,
            f"{path}: job {index} has invalid game index",
        )
        require(
            isinstance(job.get("custom_is_p1"), bool),
            f"{path}: job {index} has invalid orientation",
        )
        require(
            isinstance(job.get("battle_seed"), str) and job["battle_seed"],
            f"{path}: job {index} has invalid battle seed",
        )
        for field in ("evaluated_agent_seed", "reference_agent_seed"):
            require(
                isinstance(job.get(field), int) and 0 <= job[field] <= MASK64,
                f"{path}: job {index} has invalid {field}",
            )

    ranges = manifest["shards"]
    require(isinstance(ranges, list) and ranges, f"{path}: empty shard plan")
    cursor = 0
    seen_files: set[str] = set()
    for shard in ranges:
        require(
            isinstance(shard, dict) and set(shard) == {"start", "end", "file"},
            f"{path}: invalid shard descriptor",
        )
        start, end, filename = shard["start"], shard["end"], shard["file"]
        require(start == cursor, f"{path}: shard plan has a gap or overlap at {cursor}")
        require(
            isinstance(end, int) and start < end <= total,
            f"{path}: invalid shard range {start}-{end}",
        )
        require(
            filename == f"shard-{start:06}-{end:06}.jsonl",
            f"{path}: non-canonical shard filename {filename!r}",
        )
        require(filename not in seen_files, f"{path}: duplicate shard filename")
        seen_files.add(filename)
        cursor = end
    require(cursor == total, f"{path}: shard plan does not cover the workload")
    return manifest


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        (row["layered"]["score"], row["legacy"]["score"])
        for row in rows
        if row["layered"]["score"] is not None and row["legacy"]["score"] is not None
    ]
    deltas = [new - old for new, old in valid]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def rate(count: int) -> float:
        return count / len(rows) if rows else 0.0

    legacy_capped = sum(row["legacy"]["capped"] for row in rows)
    layered_capped = sum(row["layered"]["capped"] for row in rows)
    legacy_invalid = sum(row["legacy"]["status"] == "invalid" for row in rows)
    layered_invalid = sum(row["layered"]["status"] == "invalid" for row in rows)
    legacy_only = sum(
        row["legacy"]["status"] != "outcome"
        and row["layered"]["status"] == "outcome"
        for row in rows
    )
    layered_only = sum(
        row["legacy"]["status"] == "outcome"
        and row["layered"]["status"] != "outcome"
        for row in rows
    )
    both = sum(
        row["legacy"]["status"] != "outcome"
        and row["layered"]["status"] != "outcome"
        for row in rows
    )
    asymmetric = legacy_only + layered_only
    legacy_cap_rate = rate(legacy_capped)
    layered_cap_rate = rate(layered_capped)
    failures = []
    if legacy_invalid + layered_invalid:
        failures.append(
            f"invalid arms: legacy={legacy_invalid}, layered={layered_invalid}"
        )
    if legacy_cap_rate > 0.01 or layered_cap_rate > 0.01:
        failures.append(
            "cap rate exceeds 1%: "
            f"legacy={legacy_cap_rate:.6f}, layered={layered_cap_rate:.6f}"
        )
    if asymmetric:
        failures.append(f"asymmetric incomplete pairs: {asymmetric}")
    return {
        "pairs": len(rows),
        "valid_pairs": len(valid),
        "excluded_pairs": len(rows) - len(valid),
        "both_incomplete_pairs": both,
        "legacy_only_incomplete_pairs": legacy_only,
        "layered_only_incomplete_pairs": layered_only,
        "asymmetric_incomplete_pairs": asymmetric,
        "legacy_mean": mean([old for _, old in valid]),
        "layered_mean": mean([new for new, _ in valid]),
        "mean_paired_delta": mean(deltas),
        "delta_positive": sum(delta > 0 for delta in deltas),
        "delta_zero": sum(delta == 0 for delta in deltas),
        "delta_negative": sum(delta < 0 for delta in deltas),
        "legacy_invalid": legacy_invalid,
        "layered_invalid": layered_invalid,
        "legacy_invalid_rate": rate(legacy_invalid),
        "layered_invalid_rate": rate(layered_invalid),
        "legacy_capped": legacy_capped,
        "layered_capped": layered_capped,
        "legacy_cap_rate": legacy_cap_rate,
        "layered_cap_rate": layered_cap_rate,
        "certified": not failures,
        "certification_failures": failures,
        "result_fingerprint": fingerprint("m17d-paired-results-v1", rows),
    }


def validate_team_headers(header: dict[str, Any], run: dict[str, Any], label: str) -> None:
    custom = header.get("custom_teams")
    pilots = header.get("pilot_teams")
    require(isinstance(custom, list), f"{label}: missing custom teams")
    require(isinstance(pilots, list), f"{label}: missing pilot teams")
    require(
        len(custom) == len(run["selected_custom"]),
        f"{label}: custom team count mismatch",
    )
    require(
        len(pilots) == len(run["selected_pilots"]),
        f"{label}: pilot team count mismatch",
    )
    for kind, full, selected in (
        ("custom", custom, run["selected_custom"]),
        ("pilot", pilots, run["selected_pilots"]),
    ):
        for index, (team, identity) in enumerate(zip(full, selected, strict=True)):
            require(
                team.get("id") == identity.get("id")
                and team.get("fingerprint") == identity.get("fingerprint"),
                f"{label}: {kind} team {index} identity mismatch",
            )
            require(
                fingerprint("m17d-team-v1", team.get("canonical_team"))
                == team["fingerprint"],
                f"{label}: {kind} team {index} canonical fingerprint mismatch",
            )


def validate_arm(
    arm: dict[str, Any],
    policy: str,
    row: dict[str, Any],
    label: str,
) -> None:
    require(isinstance(arm, dict), f"{label}: arm must be an object")
    require(arm.get("policy") == policy, f"{label}: policy mismatch")
    require(arm.get("status") == "outcome", f"{label}: cap/invalid is not mergeable")
    require(arm.get("capped") is False, f"{label}: capped arm is not mergeable")
    require(arm.get("error") is None, f"{label}: errored arm is not mergeable")
    require(
        arm.get("evaluated_fallback") is True
        and arm.get("reference_fallback") is False,
        f"{label}: fallback contract mismatch",
    )
    outcome = arm.get("outcome")
    require(outcome in {"p1-win", "p2-win", "tie"}, f"{label}: invalid outcome")
    p1_score = {"p1-win": 1.0, "p2-win": 0.0, "tie": 0.5}[outcome]
    expected_score = 1.0 - p1_score if row["custom_is_p1"] else p1_score
    require(arm.get("score") == expected_score, f"{label}: outcome/score mismatch")
    require(
        isinstance(arm.get("turns"), int) and arm["turns"] >= 0,
        f"{label}: invalid turn count",
    )


def validate_shard(
    path: Path,
    manifest: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_jsonl(path)
    start, end = expected["start"], expected["end"]
    label = str(path)
    require(len(records) == end - start + 2, f"{label}: record count mismatch")
    header, trailer = records[0], records[-1]
    rows = records[1:-1]
    require(
        isinstance(header, dict)
        and header.get("schema") == GAUNTLET_SCHEMA
        and header.get("kind") == "run",
        f"{label}: invalid run header",
    )
    require(
        header.get("run_fingerprint") == manifest["run_fingerprint"],
        f"{label}: run fingerprint mismatch",
    )
    require(
        semantic_run_projection(header.get("run", {})) == manifest["run"],
        f"{label}: build/data/team/config lineage mismatch",
    )
    require(
        header.get("shard")
        == {"job_start": start, "job_end": end, "total_jobs": manifest["total_jobs"]},
        f"{label}: shard range mismatch",
    )
    validate_team_headers(header, manifest["run"], label)

    semantic = manifest["run"]["semantic_config"]
    agent = semantic["agent"]
    legacy_policy = agent["evaluated_policy_a"]
    layered_policy = agent["evaluated_policy_b"]
    max_turns = semantic["max_turns"]
    jobs = manifest["jobs"]
    custom = manifest["run"]["selected_custom"]
    pilots = manifest["run"]["selected_pilots"]
    for offset, row in enumerate(rows):
        job_index = start + offset
        job = jobs[job_index]
        row_label = f"{label}: job {job_index}"
        require(
            isinstance(row, dict)
            and row.get("schema") == GAUNTLET_SCHEMA
            and row.get("kind") == "pair",
            f"{row_label}: invalid pair row",
        )
        require(
            row.get("run_fingerprint") == manifest["run_fingerprint"],
            f"{row_label}: run fingerprint mismatch",
        )
        require(row.get("job") == job_index, f"{row_label}: duplicate/gap/reorder")
        expected_fields = {
            "game": job["game"],
            "custom_is_p1": job["custom_is_p1"],
            "battle_seed": job["battle_seed"],
            "evaluated_agent_seed": job["evaluated_agent_seed"],
            "reference_agent_seed": job["reference_agent_seed"],
            "custom_id": custom[job["custom"]]["id"],
            "custom_fingerprint": custom[job["custom"]]["fingerprint"],
            "pilot_id": pilots[job["pilot"]]["id"],
            "pilot_fingerprint": pilots[job["pilot"]]["fingerprint"],
        }
        for field, value in expected_fields.items():
            require(row.get(field) == value, f"{row_label}: {field} lineage mismatch")
        validate_arm(row.get("legacy"), legacy_policy, row, f"{row_label}: legacy")
        validate_arm(row.get("layered"), layered_policy, row, f"{row_label}: layered")
        require(
            row["legacy"]["turns"] <= max_turns
            and row["layered"]["turns"] <= max_turns,
            f"{row_label}: turn count exceeds cap",
        )
        expected_delta = row["layered"]["score"] - row["legacy"]["score"]
        require(
            row.get("delta_layered_minus_legacy") == expected_delta,
            f"{row_label}: paired delta mismatch",
        )

    require(
        isinstance(trailer, dict)
        and trailer.get("schema") == GAUNTLET_SCHEMA
        and trailer.get("kind") == "summary"
        and trailer.get("run_fingerprint") == manifest["run_fingerprint"],
        f"{label}: invalid summary trailer",
    )
    require(trailer.get("summary") == summarize(rows), f"{label}: summary mismatch")
    return header, rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def merge_artifacts(
    manifest_path: Path, manifest: dict[str, Any], output: Path
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    merged_header: dict[str, Any] | None = None
    canonical_teams: tuple[Any, Any] | None = None
    for expected in manifest["shards"]:
        shard_path = manifest_path.parent / expected["file"]
        header, rows = validate_shard(shard_path, manifest, expected)
        teams = (header["custom_teams"], header["pilot_teams"])
        if canonical_teams is None:
            canonical_teams = teams
            merged_header = copy.deepcopy(header)
            merged_header.pop("shard", None)
        else:
            require(
                teams == canonical_teams,
                f"{shard_path}: full team headers differ across shards",
            )
        all_rows.extend(rows)
    require(
        [row["job"] for row in all_rows] == list(range(manifest["total_jobs"])),
        "merged rows contain a duplicate, gap, or reorder",
    )
    require(merged_header is not None, "manifest has no shards")
    summary = summarize(all_rows)
    require(
        summary["valid_pairs"] == manifest["total_jobs"]
        and summary["excluded_pairs"] == 0
        and summary["legacy_invalid"] == 0
        and summary["layered_invalid"] == 0
        and summary["legacy_capped"] == 0
        and summary["layered_capped"] == 0
        and summary["certified"],
        "merged artifact contains cap/invalid/incomplete results",
    )
    trailer = {
        "schema": GAUNTLET_SCHEMA,
        "kind": "summary",
        "run_fingerprint": manifest["run_fingerprint"],
        "summary": summary,
    }
    records = [merged_header, *all_rows, trailer]
    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    atomic_write(output, text)
    return summary


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        custom_team = [{"species": "Custom"}]
        pilot_team = [{"species": "Pilot"}]
        custom_fp = fingerprint("m17d-team-v1", custom_team)
        pilot_fp = fingerprint("m17d-team-v1", pilot_team)
        jobs = [
            {
                "index": index,
                "custom": 0,
                "pilot": 0,
                "game": index,
                "custom_is_p1": bool(index % 2),
                "battle_seed": f"{index},2,3,4",
                "evaluated_agent_seed": index + 10,
                "reference_agent_seed": index + 20,
            }
            for index in range(2)
        ]
        run = {
            "schema": GAUNTLET_SCHEMA,
            "build": {"executable": "build"},
            "inputs": {"dex": "dex"},
            "selected_custom": [{"id": "custom", "fingerprint": custom_fp}],
            "selected_pilots": [{"id": "pilot", "fingerprint": pilot_fp}],
            "semantic_config": {
                "seed": 1,
                "max_turns": 500,
                "agent": {
                    "evaluated_policy_a": "legacy",
                    "evaluated_policy_b": "layered",
                },
            },
            "workload_fingerprint": fingerprint("m17d-workload-v1", jobs),
        }
        run_fp = fingerprint("m17d-run-v1", run)
        shards = [
            {"start": 0, "end": 1, "file": "shard-000000-000001.jsonl"},
            {"start": 1, "end": 2, "file": "shard-000001-000002.jsonl"},
        ]
        body = {
            "schema": MANIFEST_SCHEMA,
            "run_fingerprint": run_fp,
            "run": run,
            "total_jobs": 2,
            "jobs": jobs,
            "shards": shards,
        }
        manifest = {
            **body,
            "manifest_fingerprint": fingerprint("m17d-shard-manifest-v1", body),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
        validated = validate_manifest(manifest_path)

        header_base = {
            "schema": GAUNTLET_SCHEMA,
            "kind": "run",
            "run_fingerprint": run_fp,
            "run": {**run, "execution": {"threads": 1}},
            "custom_teams": [
                {
                    "id": "custom",
                    "fingerprint": custom_fp,
                    "canonical_team": custom_team,
                }
            ],
            "pilot_teams": [
                {
                    "id": "pilot",
                    "fingerprint": pilot_fp,
                    "canonical_team": pilot_team,
                }
            ],
        }
        for shard in shards:
            index = shard["start"]
            job = jobs[index]
            arm = {
                "policy": "legacy",
                "status": "outcome",
                "outcome": "tie",
                "score": 0.5,
                "turns": 1,
                "capped": False,
                "evaluated_fallback": True,
                "reference_fallback": False,
                "error": None,
            }
            row = {
                "schema": GAUNTLET_SCHEMA,
                "kind": "pair",
                "run_fingerprint": run_fp,
                "job": index,
                "custom_id": "custom",
                "custom_fingerprint": custom_fp,
                "pilot_id": "pilot",
                "pilot_fingerprint": pilot_fp,
                "game": job["game"],
                "custom_is_p1": job["custom_is_p1"],
                "battle_seed": job["battle_seed"],
                "evaluated_agent_seed": job["evaluated_agent_seed"],
                "reference_agent_seed": job["reference_agent_seed"],
                "legacy": arm,
                "layered": {**arm, "policy": "layered"},
                "delta_layered_minus_legacy": 0.0,
            }
            header = {
                **header_base,
                "shard": {
                    "job_start": shard["start"],
                    "job_end": shard["end"],
                    "total_jobs": 2,
                },
            }
            trailer = {
                "schema": GAUNTLET_SCHEMA,
                "kind": "summary",
                "run_fingerprint": run_fp,
                "summary": summarize([row]),
            }
            (root / shard["file"]).write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for record in (header, row, trailer)
                )
            )
        summary = merge_artifacts(manifest_path, validated, root / "merged.jsonl")
        require(summary["pairs"] == 2, "self-test merge count mismatch")

        corrupt_path = root / shards[0]["file"]
        records = load_jsonl(corrupt_path)
        records[1]["legacy"]["status"] = "capped"
        corrupt_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            )
        )
        try:
            validate_shard(corrupt_path, validated, shards[0])
        except ValidationError:
            pass
        else:
            raise AssertionError("self-test accepted a capped shard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-shard", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("merge-m17d-shards self-test passed")
        return 0
    require(args.manifest is not None, "--manifest is required")
    manifest = validate_manifest(args.manifest)
    if args.list:
        for shard in manifest["shards"]:
            print(shard["start"], shard["end"], shard["file"], sep="\t")
        return 0
    if args.check_shard is not None:
        require(args.start is not None, "--check-shard requires --start")
        expected = next(
            (shard for shard in manifest["shards"] if shard["start"] == args.start),
            None,
        )
        require(expected is not None, f"manifest has no shard starting at {args.start}")
        validate_shard(args.check_shard, manifest, expected)
        print(f"valid shard {expected['start']}-{expected['end']}: {args.check_shard}")
        return 0
    require(args.out is not None, "--out is required for merge")
    summary = merge_artifacts(args.manifest, manifest, args.out)
    print(
        f"merged {summary['pairs']} certified pairs to {args.out}; "
        f"delta={summary['mean_paired_delta']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(2)
