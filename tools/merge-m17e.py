#!/usr/bin/env python3
"""Fail-closed merger for M17e v3 exact-endgame shard artifacts."""

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import unittest


SHARD_SCHEMA = "nc2000-m17e-exactness-shard-v3"
MERGED_SCHEMA = "nc2000-m17e-exactness-merged-v3"
RUN_KEYS = {
    "profile", "solver_build_fingerprint", "generator_executable_fingerprint",
    "runtime_data", "corpus_fingerprint", "corpus_count", "solver", "selection",
}
RUNTIME_KEYS = {"dex", "meta_pool", "community_rentals", "learnsets"}
SOLVER_KEYS = {
    "work_budget", "node_budget", "cell_cap", "eps", "trial_depth", "descend_floor",
    "dead_damage_quotient", "fold_terminal_nodes", "fold_closed_nodes",
    "monotone_stall_scheduling", "two_sided_resource_scheduling",
    "certified_action_pruning", "support_br_scheduling", "threshold_radius",
}
SELECTION_KEYS = {
    "hp_cap", "max_alive_per_side", "per_battle", "side_filter", "turn_filter",
    "decision_order", "reconstruction_seed",
}
ROW_KEYS = {
    "battle", "decision", "side", "turn", "human", "exact", "width", "stop", "eval",
    "alive0", "alive1", "total_hp", "state_key128", "desc",
}
SUMMARY_KEYS = {
    "row_count", "coordinate_fingerprint", "state_fingerprint", "row_fingerprint",
}


class InputError(Exception):
    pass


def fail(where, message):
    raise InputError(f"{where}: {message}")


def object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def exact_keys(value, expected, where):
    if not isinstance(value, dict):
        fail(where, "must be an object")
    actual = set(value)
    if actual != expected:
        fail(where, f"fields {sorted(actual)!r} != {sorted(expected)!r}")


def integer(value, where, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(where, f"must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        fail(where, f"must be <= {maximum}")
    return value


def finite(value, where, minimum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(where, "must be a finite number")
    number = float(value)
    if strictly_positive and number <= 0:
        fail(where, "must be positive")
    if minimum is not None and number < minimum:
        fail(where, f"must be >= {minimum}")
    return number


def text(value, where):
    if not isinstance(value, str) or not value:
        fail(where, "must be non-empty text")
    return value


def validate_run(run, where):
    exact_keys(run, RUN_KEYS, where)
    text(run["profile"], f"{where}.profile")
    text(run["solver_build_fingerprint"], f"{where}.solver_build_fingerprint")
    text(run["generator_executable_fingerprint"], f"{where}.generator_executable_fingerprint")
    text(run["corpus_fingerprint"], f"{where}.corpus_fingerprint")
    integer(run["corpus_count"], f"{where}.corpus_count", 1)
    exact_keys(run["runtime_data"], RUNTIME_KEYS, f"{where}.runtime_data")
    for field in RUNTIME_KEYS:
        text(run["runtime_data"][field], f"{where}.runtime_data.{field}")

    solver = run["solver"]
    exact_keys(solver, SOLVER_KEYS, f"{where}.solver")
    for field in ("work_budget", "node_budget", "cell_cap", "trial_depth"):
        integer(solver[field], f"{where}.solver.{field}", 1)
    finite(solver["eps"], f"{where}.solver.eps", strictly_positive=True)
    finite(solver["descend_floor"], f"{where}.solver.descend_floor", minimum=0)
    finite(solver["threshold_radius"], f"{where}.solver.threshold_radius", minimum=0)
    for field in SOLVER_KEYS - {
        "work_budget", "node_budget", "cell_cap", "trial_depth", "eps", "descend_floor",
        "threshold_radius",
    }:
        if not isinstance(solver[field], bool):
            fail(f"{where}.solver.{field}", "must be boolean")

    selection = run["selection"]
    exact_keys(selection, SELECTION_KEYS, f"{where}.selection")
    integer(selection["hp_cap"], f"{where}.selection.hp_cap", 1)
    integer(selection["max_alive_per_side"], f"{where}.selection.max_alive_per_side", 1)
    integer(selection["per_battle"], f"{where}.selection.per_battle", 1)
    if selection["side_filter"] is not None:
        integer(selection["side_filter"], f"{where}.selection.side_filter", 0, 1)
    if selection["turn_filter"] is not None:
        integer(selection["turn_filter"], f"{where}.selection.turn_filter", 0, 65535)
    if selection["decision_order"] != "reverse":
        fail(f"{where}.selection.decision_order", "must be 'reverse'")
    integer(selection["reconstruction_seed"], f"{where}.selection.reconstruction_seed", 0)


def validate_row(row, where):
    exact_keys(row, ROW_KEYS, where)
    integer(row["battle"], f"{where}.battle", 0)
    integer(row["decision"], f"{where}.decision", 0)
    integer(row["side"], f"{where}.side", 0, 1)
    integer(row["turn"], f"{where}.turn", 0, 65535)
    if not isinstance(row["human"], str) or not isinstance(row["desc"], str):
        fail(where, "human and desc must be strings")
    finite(row["exact"], f"{where}.exact")
    finite(row["width"], f"{where}.width", minimum=0)
    integer(row["stop"], f"{where}.stop", 0, 65535)
    finite(row["eval"], f"{where}.eval")
    integer(row["alive0"], f"{where}.alive0", 0)
    integer(row["alive1"], f"{where}.alive1", 0)
    integer(row["total_hp"], f"{where}.total_hp", 0)
    state = row["state_key128"]
    if not (isinstance(state, str) and len(state) == 32
            and all(char in "0123456789abcdef" for char in state)):
        fail(f"{where}.state_key128", "must be 32 lowercase hex digits")


def fnv_update(value, data):
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def fingerprint_part(value, data):
    value = fnv_update(value, struct.pack("<Q", len(data)))
    return fnv_update(value, data)


def records_fingerprint(tag, records):
    value = fingerprint_part(0xCBF29CE484222325, tag.encode())
    for record in records:
        value = fingerprint_part(value, b"record")
        for field in record:
            value = fingerprint_part(value, field)
    return f"fnv1a64:{value:016x}:{tag}"


def float_bits(value):
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"{bits:016x}".encode()


def as_bytes(value):
    return str(value).encode()


def coordinate(row):
    return row["battle"], row["decision"]


def row_record(row):
    return [
        as_bytes(row["battle"]), as_bytes(row["decision"]), as_bytes(row["side"]),
        as_bytes(row["turn"]), row["human"].encode(), float_bits(row["exact"]),
        float_bits(row["width"]), as_bytes(row["stop"]), float_bits(row["eval"]),
        as_bytes(row["alive0"]), as_bytes(row["alive1"]), as_bytes(row["total_hp"]),
        row["state_key128"].encode(), row["desc"].encode(),
    ]


def summarize_rows(rows):
    ordered = sorted(rows, key=coordinate)
    coordinates = [[
        as_bytes(row["battle"]), as_bytes(row["decision"]), as_bytes(row["side"]),
        as_bytes(row["turn"]),
    ] for row in ordered]
    states = sorted([[row["state_key128"].encode()] for row in rows])
    return {
        "row_count": len(rows),
        "coordinate_fingerprint": records_fingerprint("m17e-coordinate-v1", coordinates),
        "state_fingerprint": records_fingerprint("m17e-state-v1", states),
        "row_fingerprint": records_fingerprint(
            "m17e-row-v1", [row_record(row) for row in ordered]
        ),
    }


def validate_summary(summary, rows, where):
    exact_keys(summary, SUMMARY_KEYS, where)
    integer(summary["row_count"], f"{where}.row_count", 0)
    for field in SUMMARY_KEYS - {"row_count"}:
        text(summary[field], f"{where}.{field}")
    actual = summarize_rows(rows)
    if summary != actual:
        fail(where, f"declared summary does not match rows: {summary!r} != {actual!r}")


def load_shard(path):
    try:
        with open(path, encoding="utf-8") as source:
            artifact = json.load(source, object_pairs_hook=object_no_duplicates)
    except (OSError, json.JSONDecodeError, InputError) as error:
        raise InputError(f"{path}: {error}") from error
    exact_keys(artifact, {"schema", "run", "shard", "rows"}, path)
    if artifact["schema"] != SHARD_SCHEMA:
        fail(path, f"schema must be {SHARD_SCHEMA!r}")
    validate_run(artifact["run"], f"{path}.run")
    exact_keys(artifact["shard"], {"battle_lo", "battle_hi", "summary"}, f"{path}.shard")
    lo = integer(artifact["shard"]["battle_lo"], f"{path}.shard.battle_lo", 0)
    hi = integer(artifact["shard"]["battle_hi"], f"{path}.shard.battle_hi", 0)
    if lo > hi or hi >= artifact["run"]["corpus_count"]:
        fail(f"{path}.shard", "battle range is outside corpus")
    rows = artifact["rows"]
    if not isinstance(rows, list):
        fail(f"{path}.rows", "must be an array")
    coordinates = set()
    states = set()
    for index, row in enumerate(rows):
        where = f"{path}.rows[{index}]"
        validate_row(row, where)
        if not lo <= row["battle"] <= hi:
            fail(where, f"battle is outside shard range {lo}-{hi}")
        if coordinate(row) in coordinates:
            fail(where, f"duplicate coordinate {coordinate(row)!r}")
        if row["state_key128"] in states:
            fail(where, f"duplicate state_key128 {row['state_key128']}")
        coordinates.add(coordinate(row))
        states.add(row["state_key128"])
    validate_summary(artifact["shard"]["summary"], rows, f"{path}.shard.summary")
    return artifact


def canonical_run(run):
    try:
        return json.dumps(run, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError) as error:
        raise InputError(f"run identity is not canonical JSON: {error}") from error


def merge_shards(paths):
    if not paths:
        raise InputError("at least one shard is required")
    artifacts = [load_shard(path) for path in paths]
    reference = canonical_run(artifacts[0]["run"])
    for path, artifact in zip(paths[1:], artifacts[1:]):
        if canonical_run(artifact["run"]) != reference:
            fail(path, "run identity differs from the first shard")
    artifacts.sort(key=lambda artifact: artifact["shard"]["battle_lo"])
    expected = 0
    for artifact in artifacts:
        shard = artifact["shard"]
        if shard["battle_lo"] != expected:
            fail("merge", f"range gap/overlap: expected battle {expected}, got {shard['battle_lo']}")
        expected = shard["battle_hi"] + 1
    corpus_count = artifacts[0]["run"]["corpus_count"]
    if expected != corpus_count:
        fail("merge", f"range coverage ends at {expected}, corpus_count is {corpus_count}")

    rows = []
    coordinates = set()
    states = set()
    for artifact in artifacts:
        for row in artifact["rows"]:
            coord = coordinate(row)
            if coord in coordinates:
                fail("merge", f"duplicate cross-shard coordinate {coord!r}")
            if row["state_key128"] in states:
                fail("merge", f"duplicate cross-shard state_key128 {row['state_key128']}")
            coordinates.add(coord)
            states.add(row["state_key128"])
            rows.append(row)
    if not rows:
        fail("merge", "complete artifact has no anchor rows")
    rows.sort(key=coordinate)
    return {
        "schema": MERGED_SCHEMA,
        "run": artifacts[0]["run"],
        "merge": {
            "shards": [artifact["shard"] for artifact in artifacts],
            "summary": summarize_rows(rows),
        },
        "rows": rows,
    }


def write_atomic(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


class SelfTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def run_identity(count=2):
        return {
            "profile": "m17e-formal-sweep-v3",
            "solver_build_fingerprint": "build", "generator_executable_fingerprint": "exe",
            "runtime_data": {
                "dex": "dex", "meta_pool": "meta", "community_rentals": "rentals",
                "learnsets": "learnsets",
            },
            "corpus_fingerprint": "corpus", "corpus_count": count,
            "solver": {
                "work_budget": 1, "node_budget": 1, "cell_cap": 1, "eps": 0.02,
                "trial_depth": 1, "descend_floor": 0.1, "dead_damage_quotient": True,
                "fold_terminal_nodes": True, "fold_closed_nodes": True,
                "monotone_stall_scheduling": True, "two_sided_resource_scheduling": True,
                "certified_action_pruning": True, "support_br_scheduling": True,
                "threshold_radius": 0.02,
            },
            "selection": {
                "hp_cap": 150, "max_alive_per_side": 2, "per_battle": 2,
                "side_filter": None, "turn_filter": None, "decision_order": "reverse",
                "reconstruction_seed": 1,
            },
        }

    @staticmethod
    def row(battle, state):
        return {
            "battle": battle, "decision": 3, "side": 0, "turn": 7,
            "human": "move rest", "exact": 0.5, "width": 0.02, "stop": 0,
            "eval": 0.4, "alive0": 1, "alive1": 1, "total_hp": 42,
            "state_key128": f"{state:032x}", "desc": f"b{battle}",
        }

    def write_shard(self, name, lo, hi, rows, run=None):
        artifact = {
            "schema": SHARD_SCHEMA, "run": run or self.run_identity(),
            "shard": {"battle_lo": lo, "battle_hi": hi, "summary": summarize_rows(rows)},
            "rows": rows,
        }
        path = os.path.join(self.temp.name, name)
        with open(path, "w", encoding="utf-8") as output:
            json.dump(artifact, output)
        return path

    def test_valid_complete_merge(self):
        a = self.write_shard("a.json", 0, 0, [self.row(0, 1)])
        b = self.write_shard("b.json", 1, 1, [self.row(1, 2)])
        merged = merge_shards([b, a])
        self.assertEqual([row["battle"] for row in merged["rows"]], [0, 1])
        self.assertEqual(merged["merge"]["summary"]["row_count"], 2)

    def test_deleted_row_breaks_shard_summary(self):
        path = self.write_shard("a.json", 0, 1, [self.row(0, 1), self.row(1, 2)])
        with open(path, encoding="utf-8") as source:
            artifact = json.load(source)
        artifact["rows"].pop()
        with open(path, "w", encoding="utf-8") as output:
            json.dump(artifact, output)
        with self.assertRaisesRegex(InputError, "summary does not match"):
            merge_shards([path])

    def test_gap_or_overlap_is_rejected(self):
        a = self.write_shard("a.json", 0, 0, [])
        b = self.write_shard("b.json", 0, 1, [])
        with self.assertRaisesRegex(InputError, "gap/overlap"):
            merge_shards([a, b])

    def test_run_config_mismatch_is_rejected(self):
        a = self.write_shard("a.json", 0, 0, [])
        changed = self.run_identity()
        changed["solver"]["node_budget"] = 2
        b = self.write_shard("b.json", 1, 1, [], changed)
        with self.assertRaisesRegex(InputError, "run identity differs"):
            merge_shards([a, b])

    def test_cross_shard_state_duplicate_is_rejected(self):
        a = self.write_shard("a.json", 0, 0, [self.row(0, 1)])
        b = self.write_shard("b.json", 1, 1, [self.row(1, 1)])
        with self.assertRaisesRegex(InputError, "duplicate cross-shard state"):
            merge_shards([a, b])

    def test_duplicate_json_key_is_rejected(self):
        path = os.path.join(self.temp.name, "duplicate.json")
        with open(path, "w", encoding="utf-8") as output:
            output.write('{"schema":"a","schema":"b"}')
        with self.assertRaisesRegex(InputError, "duplicate JSON key"):
            load_shard(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="*", help="M17e v3 shard JSON files")
    parser.add_argument("--out", help="merged v3 artifact path")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.out:
        parser.error("--out is required")
    if not args.shards:
        parser.error("at least one shard is required")
    try:
        merged = merge_shards(args.shards)
        write_atomic(args.out, merged)
    except InputError as error:
        parser.error(str(error))
    summary = merged["merge"]["summary"]
    print(
        f"merged {len(merged['merge']['shards'])} shard(s), {summary['row_count']} rows "
        f"over {merged['run']['corpus_count']} battles -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
