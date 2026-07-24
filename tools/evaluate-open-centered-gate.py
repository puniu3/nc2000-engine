#!/usr/bin/env python3
"""Apply the preregistered M17b Web open-sheet centered budget gate."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest


_AGGREGATE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "aggregate-arena.py")
_AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "nc2000_aggregate_arena", _AGGREGATE_PATH)
if _AGGREGATE_SPEC is None or _AGGREGATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load {_AGGREGATE_PATH}")
aggregate = importlib.util.module_from_spec(_AGGREGATE_SPEC)
sys.modules[_AGGREGATE_SPEC.name] = aggregate
_AGGREGATE_SPEC.loader.exec_module(aggregate)


MANIFEST_SCHEMA = "nc2000-arena-centered-tier-gate-v1"
RESULT_SCHEMA = "nc2000-arena-centered-tier-gate-result-v1"
TIERS = [15000, 30000, 60000]
CURRENT_ITERS = 30000
AGENT_FAMILY = "open"

InputError = aggregate.InputError


def evaluator_hash():
    try:
        with open(os.path.realpath(__file__), "rb") as source:
            centered_contents = source.read()
        with open(_AGGREGATE_PATH, "rb") as source:
            aggregate_contents = source.read()
    except OSError as error:
        raise InputError(f"evaluator: cannot hash {__file__}: {error}") from error
    contents = (
        len(centered_contents).to_bytes(8, "big")
        + centered_contents
        + len(aggregate_contents).to_bytes(8, "big")
        + aggregate_contents
    )
    return aggregate.tagged_sha256("arena-centered-tier-gate-evaluator-v1", contents)


def validate_stage(raw, where, lower_iters, higher_iters):
    stage = aggregate.validate_gate_stage(raw, where)
    canonical_a = f"{AGENT_FAMILY}:{higher_iters}:1:16"
    canonical_b = f"{AGENT_FAMILY}:{lower_iters}:1:16"
    if (stage["agent_a"], stage["agent_b"]) != (canonical_a, canonical_b):
        aggregate.fail(
            where,
            f"expected canonical labels {canonical_a!r} vs {canonical_b!r}",
        )
    return stage


def validate_manifest(raw, where="manifest"):
    raw = aggregate.obj(raw, where)
    aggregate.exact_keys(
        raw,
        {
            "schema",
            "agent_family",
            "tiers",
            "current_iters",
            "pool",
            "max_turns",
            "expected",
            "comparisons",
        },
        where,
    )
    if raw.get("schema") != MANIFEST_SCHEMA:
        aggregate.fail(
            f"{where}.schema",
            f"expected {MANIFEST_SCHEMA!r}, got {raw.get('schema')!r}",
        )
    if raw.get("agent_family") != AGENT_FAMILY:
        aggregate.fail(
            f"{where}.agent_family",
            f"expected {AGENT_FAMILY!r}, got {raw.get('agent_family')!r}",
        )

    tiers = aggregate.integer_list(raw.get("tiers"), f"{where}.tiers")
    if tiers != TIERS:
        aggregate.fail(f"{where}.tiers", f"expected exactly {TIERS!r}")
    current_iters = aggregate.integer(
        raw.get("current_iters"), f"{where}.current_iters", 1)
    if current_iters != CURRENT_ITERS:
        aggregate.fail(
            f"{where}.current_iters", f"expected exactly {CURRENT_ITERS}")

    pool = aggregate.text(raw.get("pool"), f"{where}.pool")
    if pool != "meta":
        aggregate.fail(f"{where}.pool", "expected exactly 'meta'")
    max_turns = aggregate.integer(
        raw.get("max_turns"), f"{where}.max_turns", 1, 2**16 - 1)
    if max_turns != 500:
        aggregate.fail(f"{where}.max_turns", "expected exactly 500")
    expected = aggregate.validate_expected_artifact(
        raw.get("expected"), f"{where}.expected")
    if expected["baked_tables"] != 0:
        aggregate.fail(
            f"{where}.expected.baked_tables", "expected exactly 0")

    raw_comparisons = raw.get("comparisons")
    if not isinstance(raw_comparisons, list) or len(raw_comparisons) != 2:
        aggregate.fail(
            f"{where}.comparisons",
            "expected exactly the 30000v15000 and 60000v30000 comparisons",
        )

    comparisons = []
    discovery_seeds = set()
    confirm_seeds = set()
    for index, raw_comparison in enumerate(raw_comparisons):
        cwhere = f"{where}.comparisons[{index}]"
        raw_comparison = aggregate.obj(raw_comparison, cwhere)
        aggregate.exact_keys(
            raw_comparison,
            {"lower_iters", "higher_iters", "discovery", "confirm"},
            cwhere,
        )
        lower = aggregate.integer(
            raw_comparison.get("lower_iters"), f"{cwhere}.lower_iters", 1)
        higher = aggregate.integer(
            raw_comparison.get("higher_iters"), f"{cwhere}.higher_iters", 1)
        expected_pair = (tiers[index], tiers[index + 1])
        if (lower, higher) != expected_pair:
            aggregate.fail(
                cwhere,
                f"expected adjacent tiers {expected_pair[0]} -> {expected_pair[1]}",
            )
        discovery = validate_stage(
            raw_comparison.get("discovery"),
            f"{cwhere}.discovery",
            lower,
            higher,
        )
        confirm = validate_stage(
            raw_comparison.get("confirm"),
            f"{cwhere}.confirm",
            lower,
            higher,
        )
        discovery_seeds.update(discovery["base_seeds"])
        confirm_seeds.update(confirm["base_seeds"])
        comparisons.append(
            {
                "lower_iters": lower,
                "higher_iters": higher,
                "discovery": discovery,
                "confirm": confirm,
            }
        )

    overlap = sorted(discovery_seeds & confirm_seeds)
    if overlap:
        aggregate.fail(
            f"{where}.comparisons",
            f"discovery and confirm base seeds overlap globally: {overlap}",
        )

    return {
        "schema": MANIFEST_SCHEMA,
        "agent_family": AGENT_FAMILY,
        "tiers": tiers,
        "current_iters": current_iters,
        "pool": pool,
        "max_turns": max_turns,
        "expected": expected,
        "comparisons": comparisons,
    }


def read_manifest(path):
    try:
        with open(path, encoding="utf-8") as source:
            raw = json.load(
                source, object_pairs_hook=aggregate.reject_duplicate_keys)
    except (
        OSError,
        json.JSONDecodeError,
        aggregate.DuplicateKeyError,
    ) as error:
        raise InputError(f"{path}: {error}") from error
    return validate_manifest(raw, path)


def evaluate_gate(groups, manifest):
    """Evaluate the centered 15k/30k/60k sequential decision procedure."""
    assessments = []
    manifest_hash = aggregate.canonical_hash(
        "arena-centered-tier-gate-manifest-v1", manifest)

    def consume_group(group, comparison, stage_name):
        if group is None:
            return None
        assessment, _ = aggregate.assess_gate_stage(
            group, manifest, comparison, stage_name)
        assessments.append(assessment)
        return assessment

    def consume(comparison, stage_name):
        base = aggregate.select_gate_stage(
            groups,
            manifest,
            comparison,
            stage_name,
            manifest["max_turns"],
        )
        fallback = aggregate.select_gate_stage(
            groups, manifest, comparison, stage_name, 1000)
        if base is None:
            return None
        assessment = consume_group(base, comparison, stage_name)
        if (assessment["decision"] == "rerun_max_turns_1000"
                and fallback is not None):
            return consume_group(fallback, comparison, stage_name)
        return assessment

    def result(recommended_iters=None, reason=None, rerun=None):
        return {
            "schema": RESULT_SCHEMA,
            "current_iters": manifest["current_iters"],
            "recommended_iters": recommended_iters,
            "inconclusive": reason,
            "rerun_required": rerun,
            "manifest_hash": manifest_hash,
            "evaluator_hash": evaluator_hash(),
            "fingerprints": manifest["expected"]["fingerprints"],
            "assessments": assessments,
        }

    def require_stage(comparison, stage_name):
        assessment = consume(comparison, stage_name)
        if assessment is None:
            return None, result(
                reason=(
                    f"missing {stage_name} data for "
                    f"{comparison['higher_iters']}v{comparison['lower_iters']}"
                )
            )
        if assessment["decision"] == "rerun_max_turns_1000":
            return None, result(
                reason="invalid/cap rate above 1%",
                rerun={
                    "max_turns": 1000,
                    "stage": stage_name,
                    "lower_iters": comparison["lower_iters"],
                    "higher_iters": comparison["higher_iters"],
                    "base_seeds": assessment["base_seeds"],
                },
            )
        return assessment, None

    lower_comparison, upper_comparison = manifest["comparisons"]
    lower_discovery, terminal = require_stage(
        lower_comparison, "discovery")
    if terminal is not None:
        return terminal

    if lower_discovery["decision"] == "stop":
        lower_confirm, terminal = require_stage(
            lower_comparison, "confirm")
        if terminal is not None:
            return terminal
        if lower_confirm["decision"] == "stop":
            return result(recommended_iters=manifest["tiers"][0])
        return result(
            reason=(
                "30k-vs-15k confirm decision: "
                f"{lower_confirm['decision']} (expected stop)"
            )
        )

    if lower_discovery["decision"] != "promote":
        return result(
            reason=f"30k-vs-15k discovery decision: "
                   f"{lower_discovery['decision']}")

    upper_discovery, terminal = require_stage(
        upper_comparison, "discovery")
    if terminal is not None:
        return terminal

    if upper_discovery["decision"] == "promote":
        upper_confirm, terminal = require_stage(
            upper_comparison, "confirm")
        if terminal is not None:
            return terminal
        if upper_confirm["decision"] == "promote":
            return result(recommended_iters=manifest["tiers"][2])
        return result(
            reason=(
                "60k-vs-30k confirm decision: "
                f"{upper_confirm['decision']} (expected promote)"
            )
        )

    if upper_discovery["decision"] == "stop":
        lower_confirm, terminal = require_stage(
            lower_comparison, "confirm")
        if terminal is not None:
            return terminal
        if lower_confirm["decision"] == "promote":
            return result(recommended_iters=manifest["current_iters"])
        return result(
            reason=(
                "30k-vs-15k confirm decision: "
                f"{lower_confirm['decision']} (expected promote)"
            )
        )

    return result(
        reason=f"60k-vs-30k discovery decision: "
               f"{upper_discovery['decision']}")


def write_gate(path, gate):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(gate, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")


def print_gate(gate):
    print("\n== M17b Web open-sheet centered deploy gate ==")
    for item in gate["assessments"]:
        print(
            f"{item['stage']} {item['higher_iters']}v{item['lower_iters']}: "
            f"max-turns {item['max_turns']}  "
            f"score {item['score']:.3f} "
            f"[{item['score95_low']:.3f}, {item['score95_high']:.3f}] "
            f"caps {item['turn_cap_rate']:.2%} "
            f"invalid {item['invalid_rate']:.2%} "
            f"=> {item['decision']}"
        )
    if gate["recommended_iters"] is None:
        print(
            f"recommended_iters null; retain current {gate['current_iters']} "
            f"({gate['inconclusive']})"
        )
    else:
        print(f"recommended_iters {gate['recommended_iters']}")
    if gate["rerun_required"] is not None:
        rerun = gate["rerun_required"]
        print(
            f"rerun required: {rerun['stage']} "
            f"{rerun['higher_iters']}v{rerun['lower_iters']} "
            f"with --max-turns {rerun['max_turns']}"
        )


def fixture_row(
    seed,
    pair_scores,
    higher,
    lower,
    *,
    turn_caps=0,
    max_turns=500,
    baked_tables=0,
):
    row = aggregate.fixture_row(
        seed,
        pair_scores,
        higher=higher,
        lower=lower,
        turn_caps=turn_caps,
        max_turns=max_turns,
        baked_tables=baked_tables,
    )
    row["agent_a"] = f"open:{higher}:1:16"
    row["agent_b"] = f"open:{lower}:1:16"
    row["config"]["pool"] = "meta"
    return row


class SelfTest(unittest.TestCase):
    def write_rows(self, rows):
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False)
        self.addCleanup(
            lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return handle.name

    @staticmethod
    def manifest():
        return validate_manifest(
            {
                "schema": MANIFEST_SCHEMA,
                "agent_family": "open",
                "tiers": [15000, 30000, 60000],
                "current_iters": 30000,
                "pool": "meta",
                "max_turns": 500,
                "expected": {
                    "fingerprints": {
                        "build": (
                            "fnv1a64:1111111111111111:"
                            "arena-build-v1:1parts"
                        ),
                        "dex": (
                            "fnv1a64:aaaaaaaaaaaaaaaa:"
                            "arena-dex-v1:1parts"
                        ),
                        "pool": (
                            "fnv1a64:2222222222222222:"
                            "arena-pool-v1:3parts"
                        ),
                        "tables": (
                            "fnv1a64:3333333333333333:"
                            "arena-tables-v1:2parts"
                        ),
                    },
                    "baked_tables": 0,
                },
                "comparisons": [
                    {
                        "lower_iters": 15000,
                        "higher_iters": 30000,
                        "discovery": {
                            "agent_a": "open:30000:1:16",
                            "agent_b": "open:15000:1:16",
                            "base_seeds": [1],
                            "games_per_seed": 4,
                        },
                        "confirm": {
                            "agent_a": "open:30000:1:16",
                            "agent_b": "open:15000:1:16",
                            "base_seeds": [101],
                            "games_per_seed": 4,
                        },
                    },
                    {
                        "lower_iters": 30000,
                        "higher_iters": 60000,
                        "discovery": {
                            "agent_a": "open:60000:1:16",
                            "agent_b": "open:30000:1:16",
                            "base_seeds": [2],
                            "games_per_seed": 4,
                        },
                        "confirm": {
                            "agent_a": "open:60000:1:16",
                            "agent_b": "open:30000:1:16",
                            "base_seeds": [102],
                            "games_per_seed": 4,
                        },
                    },
                ],
            }
        )

    def evaluate(self, rows, manifest=None):
        path = self.write_rows(rows)
        return evaluate_gate(
            aggregate.read_groups([path]), manifest or self.manifest())

    def test_manifest_rejects_schema_family_label_tier_and_current_drift(self):
        cases = (
            ("schema", "wrong", "schema"),
            ("agent_family", "blind", "agent_family"),
            ("tiers", [10000, 30000, 60000], "tiers"),
            ("current_iters", 15000, "current_iters"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                raw = self.manifest()
                raw[key] = value
                with self.assertRaisesRegex(InputError, message):
                    validate_manifest(raw)

        raw = self.manifest()
        raw["comparisons"][0]["discovery"]["agent_a"] = "blind:30000:1:16"
        with self.assertRaisesRegex(InputError, "canonical labels"):
            validate_manifest(raw)

    def test_manifest_rejects_pool_turn_cap_and_baked_table_drift(self):
        cases = (
            ("pool", "meta:0-9", "pool"),
            ("max_turns", 1000, "max_turns"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                raw = self.manifest()
                raw[key] = value
                with self.assertRaisesRegex(InputError, message):
                    validate_manifest(raw)

        raw = self.manifest()
        raw["expected"]["baked_tables"] = 1
        with self.assertRaisesRegex(InputError, "baked_tables"):
            validate_manifest(raw)

    def test_manifest_rejects_comparison_shape_and_seed_overlap(self):
        raw = self.manifest()
        raw["comparisons"][0]["higher_iters"] = 60000
        with self.assertRaisesRegex(InputError, "expected adjacent tiers"):
            validate_manifest(raw)

        raw = self.manifest()
        raw["comparisons"][1]["confirm"]["base_seeds"] = [1]
        with self.assertRaisesRegex(InputError, "overlap globally"):
            validate_manifest(raw)

    def test_manifest_rejects_fingerprint_and_baked_count_drift(self):
        raw = self.manifest()
        raw["expected"]["fingerprints"]["build"] = "not-a-fingerprint"
        with self.assertRaisesRegex(InputError, "content fingerprint"):
            validate_manifest(raw)

        row = fixture_row(1, [0.5, 0.5], 30000, 15000)
        row["fingerprints"]["build"] = (
            "fnv1a64:deadbeefdeadbeef:arena-build-v1:1parts")
        with self.assertRaisesRegex(
            InputError, "fingerprints.*manifest.expected"
        ):
            self.evaluate([row])

        row = fixture_row(
            1, [0.5, 0.5], 30000, 15000, baked_tables=1)
        with self.assertRaisesRegex(InputError, "baked table count 1"):
            self.evaluate([row])

    def test_recommendation_paths_15k_30k_and_60k(self):
        stop = [0.5, 0.5]
        promote = [1.0, 1.0]
        cases = (
            (
                15000,
                [
                    fixture_row(1, stop, 30000, 15000),
                    fixture_row(101, stop, 30000, 15000),
                ],
            ),
            (
                30000,
                [
                    fixture_row(1, promote, 30000, 15000),
                    fixture_row(2, stop, 60000, 30000),
                    fixture_row(101, promote, 30000, 15000),
                ],
            ),
            (
                60000,
                [
                    fixture_row(1, promote, 30000, 15000),
                    fixture_row(2, promote, 60000, 30000),
                    fixture_row(102, promote, 60000, 30000),
                ],
            ),
        )
        for expected, rows in cases:
            with self.subTest(expected=expected):
                gate = self.evaluate(rows)
                self.assertEqual(gate["schema"], RESULT_SCHEMA)
                self.assertEqual(gate["recommended_iters"], expected)
                self.assertIsNone(gate["inconclusive"])

    def test_confirmation_reversal_keeps_current_without_recommendation(self):
        gate = self.evaluate(
            [
                fixture_row(1, [0.5, 0.5], 30000, 15000),
                fixture_row(101, [1.0, 1.0], 30000, 15000),
            ]
        )
        self.assertIsNone(gate["recommended_iters"])
        self.assertEqual(gate["current_iters"], 30000)
        self.assertRegex(gate["inconclusive"], "expected stop")

        gate = self.evaluate(
            [
                fixture_row(1, [1.0, 1.0], 30000, 15000),
                fixture_row(2, [1.0, 1.0], 60000, 30000),
                fixture_row(102, [0.5, 0.5], 60000, 30000),
            ]
        )
        self.assertIsNone(gate["recommended_iters"])
        self.assertRegex(gate["inconclusive"], "expected promote")

    def test_inconclusive_and_incomplete_inputs_keep_current(self):
        gate = self.evaluate(
            [fixture_row(1, [0.0, 1.0], 30000, 15000)])
        self.assertIsNone(gate["recommended_iters"])
        self.assertRegex(gate["inconclusive"], "inconclusive")

        gate = self.evaluate(
            [fixture_row(1, [1.0, 1.0], 30000, 15000)])
        self.assertIsNone(gate["recommended_iters"])
        self.assertRegex(gate["inconclusive"], "missing discovery.*60000v30000")

    def test_wrong_game_count_fails_closed(self):
        manifest = self.manifest()
        manifest["comparisons"][0]["discovery"]["games_per_seed"] = 200
        with self.assertRaisesRegex(InputError, "expected 200 games"):
            self.evaluate(
                [fixture_row(1, [0.5, 0.5], 30000, 15000)],
                manifest,
            )

    def test_cap_requests_and_consumes_1000_turn_rerun(self):
        capped = fixture_row(
            1,
            [0.5, 0.5],
            30000,
            15000,
            turn_caps=1,
        )
        gate = self.evaluate([capped])
        self.assertIsNone(gate["recommended_iters"])
        self.assertEqual(gate["rerun_required"]["max_turns"], 1000)

        fallback = fixture_row(
            1, [0.5, 0.5], 30000, 15000, max_turns=1000)
        confirm = fixture_row(
            101, [0.5, 0.5], 30000, 15000)
        gate = self.evaluate([capped, fallback, confirm])
        self.assertEqual(gate["recommended_iters"], 15000)
        self.assertEqual(
            [
                (item["decision"], item["max_turns"])
                for item in gate["assessments"]
            ],
            [
                ("rerun_max_turns_1000", 500),
                ("stop", 1000),
                ("stop", 500),
            ],
        )

    def test_1000_turn_shard_cannot_bypass_missing_base(self):
        fallback = fixture_row(
            1, [0.5, 0.5], 30000, 15000, max_turns=1000)
        gate = self.evaluate([fallback])
        self.assertIsNone(gate["recommended_iters"])
        self.assertIsNone(gate["rerun_required"])
        self.assertRegex(gate["inconclusive"], "missing discovery")
        self.assertEqual(gate["assessments"], [])

    def test_cli_returns_nonzero_for_null_recommendation(self):
        rows = self.write_rows(
            [fixture_row(1, [0.0, 1.0], 30000, 15000)])
        manifest_file = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False)
        self.addCleanup(
            lambda: os.path.exists(manifest_file.name)
            and os.unlink(manifest_file.name)
        )
        with manifest_file:
            json.dump(self.manifest(), manifest_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                main([rows, "--gate-manifest", manifest_file.name]), 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="arena JSONL shard(s)")
    parser.add_argument(
        "--gate-manifest",
        default=None,
        help=f"apply {MANIFEST_SCHEMA} manifest",
    )
    parser.add_argument("--gate-out", help="write centered gate result as JSON")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        return (
            0
            if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
            else 1
        )
    if not args.files:
        parser.error("at least one JSONL shard is required")
    if not args.gate_manifest:
        parser.error("--gate-manifest is required")
    if args.gate_out and not args.gate_manifest:
        parser.error("--gate-out requires --gate-manifest")
    try:
        groups = aggregate.read_groups(args.files)
        manifest = read_manifest(args.gate_manifest)
        gate = evaluate_gate(groups, manifest)
    except InputError as error:
        parser.error(str(error))

    aggregate.print_report([group.merged() for group in groups])
    print_gate(gate)
    if args.gate_out:
        write_gate(args.gate_out, gate)
        print(f"gate result -> {args.gate_out}")
    return 0 if gate["recommended_iters"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
