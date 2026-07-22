#!/usr/bin/env python3
"""Aggregate M17a regret-miner JSONL shards."""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from math import exp, inf, isclose, lgamma, log, sqrt


MODE_INFO = {
    "screen": ("offline", "screen"),
    "confirm": ("offline", "confirm"),
    "live-screen": ("live", "screen"),
    "live-confirm": ("live", "confirm"),
}

T95 = [
    0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
    2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
    2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
]


def family(row):
    lineage, _ = MODE_INFO[str(row.get("mode"))]
    default_source = "corpus" if lineage == "offline" else "missing-source"
    source = row.get("source", default_source)
    if not isinstance(source, str) or not source:
        source = "invalid-source"
    fingerprint = row.get("input_fingerprint" if lineage == "live" else "corpus_fingerprint")
    return lineage, source, fingerprint


def family_name(key):
    return "/".join(str(part) for part in key if part is not None)


def number(value, default=float("nan")):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def finite_number(value, where):
    value = number(value)
    if value != value or value in (inf, -inf):
        raise SystemExit(f"{where}: expected finite number")
    return float(value)


def fmt(value, width=6):
    value = number(value)
    return f"{value:{width}.3f}" if value == value else f"{'-':>{width}}"


def location(row):
    if row.get("mode") in ("live-screen", "live-confirm"):
        source = row.get("input_file", "-")
        line = row.get("input_line", "-")
        room = row.get("room", "-")
        rqid = row.get("rqid", "-")
        battle = row.get("battle", "-")
        decision = row.get("decision", "-")
        side = row.get("side", "-")
        turn = row.get("turn", "-")
        return (f"{source}:line{line}/{room}/rqid{rqid}/"
                f"b{battle}/d{decision}/s{side}/t{turn}")
    source = row.get("file", "-")
    battle = row.get("battle", "-")
    decision = row.get("decision", "-")
    side = row.get("side", "-")
    turn = row.get("turn", "-")
    return f"{source}:b{battle}/d{decision}/s{side}/t{turn}"


def tags(row):
    value = row.get("tags", [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return sorted({str(tag) for tag in value})
    return []


def screen_min_visits(row):
    visits = [number(action.get("min_visits"), inf) for action in row.get("actions", [])]
    visits = [visit for visit in visits if visit != inf]
    return min(visits) if visits else float("nan")


def transition(row):
    return f"{row.get('reference_class', '?')}->{row.get('candidate_class', '?')}"


def beta_continued_fraction(a, b, x):
    """Numerical Recipes continued fraction for the incomplete beta."""
    max_iterations, epsilon, floor = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), floor) * (1 if d >= 0 else -1)
    h = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = d if abs(d) >= floor else floor
        c = 1.0 + aa / c
        c = c if abs(c) >= floor else floor
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = d if abs(d) >= floor else floor
        c = 1.0 + aa / c
        c = c if abs(c) >= floor else floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * beta_continued_fraction(a, b, x) / a
    return 1 - front * beta_continued_fraction(b, a, 1 - x) / b


def t95(sample_count):
    return T95[sample_count - 1] if sample_count <= 30 else 1.96


def paired_stats(row, where="confirmation row", validate_serialized=False):
    candidate = row.get("candidate_values")
    reference = row.get("reference_values")
    if not isinstance(candidate, list) or not isinstance(reference, list):
        raise SystemExit(f"{where}: candidate_values/reference_values must be arrays")
    if len(candidate) != len(reference) or len(candidate) < 2:
        raise SystemExit(f"{where}: paired arrays must have equal length >= 2")
    candidate = [finite_number(value, f"{where}: candidate_values[{i}]")
                 for i, value in enumerate(candidate)]
    reference = [finite_number(value, f"{where}: reference_values[{i}]")
                 for i, value in enumerate(reference)]
    deltas = [a - b for a, b in zip(candidate, reference)]
    sample_count = len(deltas)
    mean = sum(deltas) / sample_count
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    se = sqrt(variance / sample_count)
    ci95 = t95(sample_count) * se
    lower95 = mean - ci95
    if variance == 0:
        p_value = 0.0 if mean != 0 else 1.0
    else:
        t_value = abs(mean) / se
        degrees = sample_count - 1
        p_value = regularized_beta(
            degrees / (degrees + t_value * t_value), degrees / 2, 0.5
        )
    stats = {
        "mean": mean, "ci95": ci95, "lower95": lower95,
        "p_value": p_value, "samples": sample_count,
    }
    if validate_serialized:
        for field, expected in [("regret", mean), ("ci95", ci95), ("lower95", lower95)]:
            actual = finite_number(row.get(field), f"{where}: {field}")
            if not isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise SystemExit(
                    f"{where}: {field}={actual!r} disagrees with paired samples ({expected!r})"
                )
        samples = row.get("samples")
        if samples is not None and (
            isinstance(samples, bool) or not isinstance(samples, int) or samples != sample_count
        ):
            raise SystemExit(
                f"{where}: samples={samples!r} disagrees with paired array length {sample_count}"
            )
    return stats


def paired_t_pvalue(row):
    return paired_stats(row)["p_value"]


def bh_qvalues(rows):
    tested = [(paired_t_pvalue(row), index) for index, row in enumerate(rows)]
    tested.sort()
    qvalues = [float("nan")] * len(rows)
    running = 1.0
    total = len(tested)
    for rank in range(total - 1, -1, -1):
        p_value, index = tested[rank]
        running = min(running, p_value * total / (rank + 1))
        qvalues[index] = min(running, 1.0)
    return qvalues


def sort_token(value):
    if value is None:
        return 0, ""
    if isinstance(value, bool):
        return 1, int(value)
    if isinstance(value, (int, float)):
        return 2, float(value)
    if isinstance(value, str):
        return 3, value
    return 4, json.dumps(value, sort_keys=True, separators=(",", ":"))


def row_identity(row):
    """Semantic source coordinate. Rank and results are deliberately excluded."""
    mode = str(row.get("mode", "missing"))
    lineage = MODE_INFO.get(mode, ("unknown", "unknown"))[0]
    fingerprint = row.get("input_fingerprint" if lineage == "live" else "corpus_fingerprint")
    common = (
        mode, row.get("source"), fingerprint,
        row.get("reference"), row.get("candidate"),
    )
    if mode in ("live-screen", "live-confirm"):
        return common + (
            row.get("input_file"), row.get("input_line"), row.get("room"), row.get("rqid"),
            row.get("battle"), row.get("decision"), row.get("side"),
        )
    return common + (
        row.get("file"), row.get("battle"), row.get("decision"), row.get("side"),
    )


def semantic_sort_key(row):
    mode = str(row.get("mode", ""))
    info = MODE_INFO.get(mode, ("unknown", "unknown"))
    lineage = info[0]
    fingerprint = row.get("input_fingerprint" if lineage == "live" else "corpus_fingerprint")
    values = (
        lineage, row.get("source"), fingerprint, mode,
        row.get("input_file", row.get("file")), row.get("input_line"), row.get("room"),
        row.get("rqid"), row.get("battle"), row.get("decision"), row.get("side"),
        row.get("turn"), row.get("reference"), row.get("candidate"), row.get("skip"),
    )
    return tuple(sort_token(value) for value in values)


def comparable_row(row):
    comparable = dict(row)
    comparable.pop("rank", None)
    return comparable


def validate_row(row, where):
    if not isinstance(row, dict):
        raise SystemExit(f"{where}: row must be an object")
    mode = str(row.get("mode", "missing"))
    if mode not in MODE_INFO:
        return
    lineage, stage = MODE_INFO[mode]
    if lineage == "live":
        fingerprint = row.get("input_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise SystemExit(f"{where}: live row requires non-empty input_fingerprint")
    elif "corpus_fingerprint" in row:
        fingerprint = row["corpus_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise SystemExit(f"{where}: corpus_fingerprint must be a non-empty string")
    if "skip" not in row:
        finite_number(row.get("regret"), f"{where}: regret")
        if stage == "confirm":
            paired_stats(row, where, validate_serialized=True)


def print_family_report(key, data, top, qualified):
    screen = data["screen"]
    confirm = data["confirm"]
    suffix = f" [{family_name(key)}]" if qualified else ""

    print(f"\n== screen{suffix}: discovery regret ==")
    for rank, row in enumerate(
        sorted(screen, key=lambda r: (-number(r.get("regret"), -inf), semantic_sort_key(r)))[:top], 1
    ):
        tag_text = ",".join(tags(row)) or "-"
        if key[0] == "live":
            stability = f"oracle-stable {fmt(row.get('oracle_stability'))}"
        else:
            stability = (f"stable {fmt(row.get('product_stability'))}/"
                         f"{fmt(row.get('oracle_stability'))}")
        print(f"{rank:>3} regret {fmt(row.get('regret'))}  {stability}  "
              f"min-vis {fmt(screen_min_visits(row), 7)}  {location(row)}  "
              f"{row.get('reference', '?')} -> {row.get('candidate', '?')}  [{tag_text}]")

    # Multiple-testing correction belongs to a confirmation family. A live
    # decision log must neither help nor penalize the offline corpus (and vice versa).
    stats = [paired_stats(row) for row in confirm]
    qvalues = bh_qvalues(confirm)
    for row, stat, qvalue in zip(confirm, stats, qvalues):
        row["p_value"] = stat["p_value"]
        row["bh_q_value"] = qvalue
    analysed = list(zip(confirm, stats))
    ranked_confirm = sorted(analysed, key=lambda item: (
        not (number(item[0].get("bh_q_value"), inf) <= 0.05 and item[1]["mean"] > 0),
        -item[1]["lower95"], -item[1]["mean"], semantic_sort_key(item[0]),
    ))
    confirmed = [
        (row, stat) for row, stat in analysed
        if number(row.get("bh_q_value"), inf) <= 0.05 and stat["mean"] > 0
    ]
    print(f"\n== confirm{suffix}: paired regret (BH q, lower95, then mean) ==")
    print(f"confirmed(BH q<=0.05, positive effect) {len(confirmed)} / {len(confirm)}")
    for rank, (row, stat) in enumerate(ranked_confirm[:top], 1):
        tag_text = ",".join(tags(row)) or "-"
        print(f"{rank:>3} q {fmt(row.get('bh_q_value'))}  "
              f"lower95 {fmt(stat['lower95'])}  regret {fmt(stat['mean'])}  "
              f"ci95 {fmt(stat['ci95'])}  {location(row)}  "
              f"{row.get('reference', '?')} -> {row.get('candidate', '?')}  [{tag_text}]")

    clusters = defaultdict(list)
    for row, stat in confirmed:
        row_tags = tags(row) or ["(untagged)"]
        for tag in row_tags:
            clusters[(transition(row), tag)].append((row, stat))
    ordered = sorted(
        clusters.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(stat["lower95"] for _, stat in item[1]) / len(item[1]),
            -sum(stat["mean"] for _, stat in item[1]) / len(item[1]),
            item[0][0], item[0][1],
        ),
    )
    print(f"\n== confirmed clusters{suffix}: class transition + tag ==")
    for (pair, tag), rows in ordered[:top]:
        mean_lower = sum(stat["lower95"] for _, stat in rows) / len(rows)
        mean_regret = sum(stat["mean"] for _, stat in rows) / len(rows)
        print(f"n {len(rows):>4}  lower95 {mean_lower:6.3f}  regret {mean_regret:6.3f}  "
              f"{pair:<24} [{tag}]")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="regret-miner JSONL shard(s)")
    parser.add_argument("--top", type=int, default=20, help="rows/clusters to print (default: 20)")
    parser.add_argument("--merge-out", help="also write all rows in deterministic key order")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.files:
        parser.error("at least one regret-miner JSONL shard is required")
    if args.top < 0:
        parser.error("--top must be non-negative")

    families = defaultdict(lambda: {"screen": [], "confirm": [], "skips": Counter()})
    all_rows, skips = [], Counter()
    unknown = Counter()
    total = 0
    duplicates = 0
    seen = {}
    for path in args.files:
        with open(path, encoding="utf-8") as src:
            for lineno, line in enumerate(src, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{lineno}: {error}") from error
                validate_row(row, f"{path}:{lineno}")
                key = row_identity(row)
                if key in seen:
                    first = seen[key]
                    if comparable_row(first) != comparable_row(row):
                        raise SystemExit(f"conflicting duplicate row key {key}: {path}:{lineno}")
                    ranks = [value for value in (first.get("rank"), row.get("rank"))
                             if isinstance(value, int) and not isinstance(value, bool)]
                    if ranks:
                        first["rank"] = min(ranks)
                    duplicates += 1
                    continue
                seen[key] = row
                total += 1
                all_rows.append(row)
                mode = str(row.get("mode", "missing"))
                if "skip" in row:
                    skips[(mode, str(row["skip"]))] += 1
                    if mode in MODE_INFO:
                        families[family(row)]["skips"][(mode, str(row["skip"]))] += 1
                elif mode in MODE_INFO:
                    families[family(row)][MODE_INFO[mode][1]].append(row)
                else:
                    unknown[mode] += 1

    if args.merge_out:
        all_rows.sort(key=semantic_sort_key)
        parent = os.path.dirname(os.path.abspath(args.merge_out))
        os.makedirs(parent, exist_ok=True)
        with open(args.merge_out, "w", encoding="utf-8") as out:
            for row in all_rows:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"merged {len(all_rows)} rows -> {args.merge_out}")

    print("== coverage ==")
    print(f"files {len(args.files)}  rows {total}")
    ordered_families = sorted(families.items(), key=lambda item: family_name(item[0]))
    legacy_only = (
        len(ordered_families) == 1
        and ordered_families[0][0] == ("offline", "corpus", None)
    )
    for key, data in ordered_families:
        screen_mode = "screen" if key[0] == "offline" else "live-screen"
        screen_skips = sum(
            count for (mode, _), count in data["skips"].items() if mode == screen_mode
        )
        attempted = len(data["screen"]) + screen_skips
        prefix = "" if legacy_only else f"{family_name(key)}  "
        suffix = f"  duplicates {duplicates}" if legacy_only else ""
        print(f"{prefix}screen {len(data['screen'])} / attempted {attempted} "
              f"({100 * len(data['screen']) / max(attempted, 1):.1f}%)  "
              f"confirm {len(data['confirm'])}  skips {sum(data['skips'].values())}{suffix}")
    if not legacy_only:
        print(f"duplicates {duplicates}")
    if skips:
        ordered_skips = sorted(skips.items(), key=lambda item: (-item[1], item[0]))
        print("skip reasons: " + ", ".join(
            f"{mode}:{reason}={count}" for (mode, reason), count in ordered_skips
        ))
    if unknown:
        print("unknown modes: " + ", ".join(
            f"{mode}={count}" for mode, count in sorted(unknown.items())
        ))

    for key, data in ordered_families:
        print_family_report(key, data, args.top, not legacy_only)
    return 0


def test_screen(mode, ordinal=0, fingerprint="fnv1a64:aaaaaaaaaaaaaaaa"):
    live = mode == "live-screen"
    row = {
        "mode": mode, "battle": 0, "decision": 0, "side": 0, "turn": 12,
        "reference": "move tackle", "candidate": "move surf",
        "reference_class": "physical", "candidate_class": "special",
        "regret": 0.2, "oracle_stability": 1.0,
        "actions": [{"action": "move tackle", "min_visits": 20}],
        "tags": ["phase:mid"],
    }
    if live:
        row.update({
            "source": "live-decision-log-v2", "input_file": "decisions.jsonl",
            "input_fingerprint": fingerprint,
            "input_line": ordinal + 1, "room": f"battle-live-{ordinal}", "rqid": ordinal + 10,
        })
    else:
        row.update({"file": "offline.raw.log", "product_stability": 1.0})
    return row


def test_confirm(mode, deltas=None, fingerprint="fnv1a64:aaaaaaaaaaaaaaaa"):
    live = mode == "live-confirm"
    deltas = deltas or [0.2, 0.2, 0.2, 0.2]
    row = {
        "mode": mode, "rank": 0, "battle": 0, "decision": 0, "side": 0, "turn": 12,
        "reference": "move tackle", "candidate": "move surf",
        "reference_class": "physical", "candidate_class": "special",
        "reference_values": [0.0] * len(deltas), "candidate_values": deltas,
        "tags": ["phase:mid"],
    }
    stat = paired_stats(row)
    row.update({"regret": stat["mean"], "ci95": stat["ci95"],
                "lower95": stat["lower95"], "samples": len(deltas)})
    if live:
        row.update({
            "source": "live-decision-log-v2", "input_file": "decisions.jsonl",
            "input_fingerprint": fingerprint,
            "input_line": 1, "room": "battle-live-0", "rqid": 10,
        })
    else:
        row["file"] = "offline.raw.log"
    return row


class SelfTest(unittest.TestCase):
    def write_rows(self, rows):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return handle.name

    def run_main(self, rows):
        path = self.write_rows(rows)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([path, "--top", "10"]), 0)
        return output.getvalue()

    def test_legacy_offline_headings_and_stability_are_preserved(self):
        output = self.run_main([test_screen("screen")])
        self.assertIn("== screen: discovery regret ==", output)
        self.assertIn("stable  1.000/ 1.000", output)
        self.assertNotIn("[offline/corpus]", output)

    def test_live_and_offline_coverage_and_clusters_do_not_mix(self):
        live_skip = test_screen("live-screen", 2)
        live_skip["skip"] = "random"
        rows = [
            test_screen("screen"), test_confirm("confirm"),
            test_screen("live-screen", 0), test_screen("live-screen", 1),
            live_skip, test_confirm("live-confirm"),
        ]
        output = self.run_main(rows)
        self.assertIn("offline/corpus  screen 1 / attempted 1", output)
        self.assertIn(
            "live/live-decision-log-v2/fnv1a64:aaaaaaaaaaaaaaaa  screen 2 / attempted 3",
            output,
        )
        self.assertIn("== confirm [offline/corpus]", output)
        self.assertIn(
            "== confirm [live/live-decision-log-v2/fnv1a64:aaaaaaaaaaaaaaaa]", output
        )
        self.assertEqual(output.count("n    1  lower95"), 2)
        self.assertNotIn("n    2  lower95", output)
        self.assertIn("decisions.jsonl:line1/battle-live-0/rqid10", output)

    def test_bh_correction_is_scoped_to_confirmation_family(self):
        offline = test_confirm("confirm", [1.0, 1.0, 1.0, 0.1])
        live = test_confirm("live-confirm", [1.0, -1.0, 1.0, -1.0])
        self.assertLess(paired_t_pvalue(offline), 0.05)
        self.assertGreater(bh_qvalues([offline, live])[0], 0.05)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            print_family_report(("offline", "corpus", None),
                                {"screen": [], "confirm": [offline]}, 10, True)
            print_family_report(("live", "live-decision-log-v2",
                                 "fnv1a64:aaaaaaaaaaaaaaaa"),
                                {"screen": [], "confirm": [live]}, 10, True)
        self.assertLessEqual(offline["bh_q_value"], 0.05)
        self.assertGreater(live["bh_q_value"], 0.05)

    def test_forged_positive_regret_over_negative_raw_deltas_is_rejected(self):
        row = test_confirm("confirm")
        row["reference_values"] = [1.0, 1.0, 1.0, 1.0]
        row["candidate_values"] = [0.0, 0.0, 0.0, 0.0]
        path = self.write_rows([row])
        with self.assertRaisesRegex(SystemExit, "regret=.*disagrees with paired samples"):
            main([path])

    def test_live_fingerprints_form_distinct_families_and_identities(self):
        first = "fnv1a64:1111111111111111"
        second = "fnv1a64:2222222222222222"
        output = self.run_main([
            test_screen("live-screen", fingerprint=first),
            test_confirm("live-confirm", fingerprint=first),
            test_screen("live-screen", fingerprint=second),
            test_confirm("live-confirm", fingerprint=second),
        ])
        self.assertIn(f"live/live-decision-log-v2/{first}  screen 1", output)
        self.assertIn(f"live/live-decision-log-v2/{second}  screen 1", output)
        self.assertEqual(output.count("confirmed(BH q<=0.05, positive effect) 1 / 1"), 2)

    def test_rank_only_difference_is_one_duplicate(self):
        first = test_confirm("confirm")
        second = dict(first)
        second["rank"] = 99
        output = self.run_main([second, first])
        self.assertIn("rows 1", output)
        self.assertIn("duplicates 1", output)

    def test_report_is_invariant_to_input_order_for_ties(self):
        screen_a = test_screen("screen")
        screen_b = test_screen("screen")
        screen_b.update({"battle": 1, "decision": 1, "file": "z.raw.log"})
        confirm_a = test_confirm("confirm")
        confirm_b = test_confirm("confirm")
        confirm_b.update({"battle": 1, "decision": 1, "file": "z.raw.log", "rank": 1})
        rows = [screen_b, confirm_b, screen_a, confirm_a]
        self.assertEqual(self.run_main(rows), self.run_main(list(reversed(rows))))


if __name__ == "__main__":
    sys.exit(main())
