#!/usr/bin/env python3
"""Validate and aggregate nc2000-arena-v1 JSONL shards."""

import argparse
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field


SCHEMA = "nc2000-arena-v1"
OUTPUT_SCHEMA = "nc2000-arena-aggregate-v1"


class InputError(ValueError):
    pass


def fail(where, message):
    raise InputError(f"{where}: {message}")


def obj(value, where):
    if not isinstance(value, dict):
        fail(where, "expected object")
    return value


def text(value, where):
    if not isinstance(value, str) or not value:
        fail(where, "expected non-empty string")
    return value


def boolean(value, where):
    if not isinstance(value, bool):
        fail(where, "expected boolean")
    return value


def integer(value, where, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        fail(where, "expected integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        fail(where, f"expected {bound}, got {value}")
    return value


def number(value, where, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(where, "expected number")
    value = float(value)
    if not math.isfinite(value):
        fail(where, "expected finite number")
    if minimum is not None and value < minimum:
        fail(where, f"expected >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        fail(where, f"expected <= {maximum}, got {value}")
    return value


def close(actual, expected, where):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        fail(where, f"inconsistent value {actual!r}; recomputed {expected!r}")


def mean_ci95(samples):
    mean = sum(samples) / len(samples)
    if len(samples) < 2:
        return mean, None
    variance = sum((sample - mean) ** 2 for sample in samples) / (len(samples) - 1)
    return mean, 1.96 * math.sqrt(variance / len(samples))


def percentile_ns(samples, percentile):
    if not samples:
        return 0
    rank = (len(samples) * percentile + 99) // 100
    return samples[max(rank - 1, 0)]


def elo(score):
    if score <= 0.0 or score >= 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def validate_timing_side(raw, where):
    raw = obj(raw, where)
    moves = integer(raw.get("moves"), f"{where}.moves")
    total_ns = integer(raw.get("total_ns"), f"{where}.total_ns")
    samples = raw.get("samples_ns")
    if not isinstance(samples, list):
        fail(f"{where}.samples_ns", "expected array")
    samples = [integer(value, f"{where}.samples_ns[{i}]") for i, value in enumerate(samples)]
    if len(samples) != moves:
        fail(f"{where}.samples_ns", f"length {len(samples)} != moves {moves}")
    if samples != sorted(samples):
        fail(f"{where}.samples_ns", "samples must be sorted")
    if sum(samples) != total_ns:
        fail(f"{where}.total_ns", f"{total_ns} != sample sum {sum(samples)}")

    mean_ms = number(raw.get("mean_ms"), f"{where}.mean_ms", 0)
    p95_ms = number(raw.get("p95_ms"), f"{where}.p95_ms", 0)
    p99_ms = number(raw.get("p99_ms"), f"{where}.p99_ms", 0)
    close(mean_ms, total_ns / 1e6 / max(moves, 1), f"{where}.mean_ms")
    close(p95_ms, percentile_ns(samples, 95) / 1e6, f"{where}.p95_ms")
    close(p99_ms, percentile_ns(samples, 99) / 1e6, f"{where}.p99_ms")
    return {"moves": moves, "total_ns": total_ns, "samples_ns": samples}


def validate_row(raw, where):
    raw = obj(raw, where)
    if raw.get("schema") != SCHEMA:
        fail(f"{where}.schema", f"expected {SCHEMA!r}, got {raw.get('schema')!r}")
    agent_a = text(raw.get("agent_a"), f"{where}.agent_a")
    agent_b = text(raw.get("agent_b"), f"{where}.agent_b")

    config = obj(raw.get("config"), f"{where}.config")
    requested_games = integer(config.get("requested_games"), f"{where}.config.requested_games", 1)
    base_seed = integer(config.get("base_seed"), f"{where}.config.base_seed", 0, 2**64 - 1)
    threads = integer(config.get("threads"), f"{where}.config.threads", 1)
    max_turns = integer(config.get("max_turns"), f"{where}.config.max_turns", 1, 2**16 - 1)
    pool = text(config.get("pool"), f"{where}.config.pool")
    teams = integer(config.get("teams"), f"{where}.config.teams", 1)
    baked_tables = integer(config.get("baked_tables"), f"{where}.config.baked_tables")
    log_on = boolean(config.get("log_on"), f"{where}.config.log_on")

    result = obj(raw.get("result"), f"{where}.result")
    games = integer(result.get("games"), f"{where}.result.games", 2)
    pairs = integer(result.get("pairs"), f"{where}.result.pairs", 1)
    wins = integer(result.get("wins"), f"{where}.result.wins")
    losses = integer(result.get("losses"), f"{where}.result.losses")
    ties = integer(result.get("ties"), f"{where}.result.ties")
    turn_caps = integer(result.get("turn_caps"), f"{where}.result.turn_caps")
    turns_sum = integer(result.get("turns_sum"), f"{where}.result.turns_sum")
    wall_secs = number(result.get("wall_secs"), f"{where}.result.wall_secs", 0)
    if games != requested_games + requested_games % 2:
        fail(f"{where}.result.games", "does not match rounded-up requested_games")
    if games != pairs * 2:
        fail(f"{where}.result.pairs", f"{pairs} pairs cannot account for {games} games")
    if wins + losses + ties != games:
        fail(f"{where}.result", "wins + losses + ties != games")
    if turn_caps > games:
        fail(f"{where}.result.turn_caps", "exceeds games")
    if turns_sum > games * (max_turns + 1):
        fail(f"{where}.result.turns_sum", "exceeds games * (max_turns + 1)")

    pair_scores = result.get("pair_scores")
    if not isinstance(pair_scores, list):
        fail(f"{where}.result.pair_scores", "expected array")
    pair_scores = [
        number(value, f"{where}.result.pair_scores[{i}]", 0, 1)
        for i, value in enumerate(pair_scores)
    ]
    if len(pair_scores) != pairs:
        fail(f"{where}.result.pair_scores", f"length {len(pair_scores)} != pairs {pairs}")
    if any(score * 4 != round(score * 4) for score in pair_scores):
        fail(f"{where}.result.pair_scores", "scores must be quarter-point side-swap means")
    score, ci95 = mean_ci95(pair_scores)
    close(number(result.get("score"), f"{where}.result.score", 0, 1), score,
          f"{where}.result.score")
    close((wins + 0.5 * ties) / games, score, f"{where}.result wins/losses/ties")
    raw_ci95 = result.get("ci95")
    if ci95 is None:
        if raw_ci95 is not None:
            fail(f"{where}.result.ci95", "must be null for one pair")
    else:
        close(number(raw_ci95, f"{where}.result.ci95", 0), ci95, f"{where}.result.ci95")
    if result.get("ci_unit") != "side_swap_pair":
        fail(f"{where}.result.ci_unit", "expected 'side_swap_pair'")
    close(number(result.get("avg_turns"), f"{where}.result.avg_turns", 0),
          turns_sum / games, f"{where}.result.avg_turns")

    timing = obj(raw.get("timing"), f"{where}.timing")
    if timing.get("unit") != "choose_call":
        fail(f"{where}.timing.unit", "expected 'choose_call'")
    timing_a = validate_timing_side(timing.get("a"), f"{where}.timing.a")
    timing_b = validate_timing_side(timing.get("b"), f"{where}.timing.b")

    return {
        "where": where,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "requested_games": requested_games,
        "base_seed": base_seed,
        "threads": threads,
        "max_turns": max_turns,
        "pool": pool,
        "teams": teams,
        "baked_tables": baked_tables,
        "log_on": log_on,
        "games": games,
        "pairs": pairs,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "turn_caps": turn_caps,
        "pair_scores": pair_scores,
        "turns_sum": turns_sum,
        "wall_secs": wall_secs,
        "timing_a": timing_a,
        "timing_b": timing_b,
    }


@dataclass
class Group:
    agent_a: str
    agent_b: str
    pool: str
    max_turns: int
    rows: list = field(default_factory=list)
    seeds: set = field(default_factory=set)
    invariant: tuple = None

    @property
    def key(self):
        return self.agent_a, self.agent_b, self.pool, self.max_turns

    def add(self, row):
        if row["base_seed"] in self.seeds:
            fail(row["where"], f"duplicate base_seed {row['base_seed']} for group {self.key!r}")
        invariant = row["teams"], row["baked_tables"], row["log_on"]
        if self.invariant is None:
            self.invariant = invariant
        elif invariant != self.invariant:
            fail(row["where"],
                 "same agent/pool/max-turns group has incompatible teams/baked_tables/log_on")
        self.seeds.add(row["base_seed"])
        self.rows.append(row)

    def merged(self):
        rows = sorted(self.rows, key=lambda row: row["base_seed"])
        pair_scores = [score for row in rows for score in row["pair_scores"]]
        score, ci95 = mean_ci95(pair_scores)
        games = sum(row["games"] for row in self.rows)
        turns_sum = sum(row["turns_sum"] for row in self.rows)
        a_samples = sorted(sample for row in self.rows for sample in row["timing_a"]["samples_ns"])
        b_samples = sorted(sample for row in self.rows for sample in row["timing_b"]["samples_ns"])
        score_low = max(0.0, score - (ci95 or 0.0))
        score_high = min(1.0, score + (ci95 or 0.0))

        def timing(samples):
            total = sum(samples)
            return {
                "moves": len(samples),
                "total_ns": total,
                "mean_ms": total / 1e6 / max(len(samples), 1),
                "p95_ms": percentile_ns(samples, 95) / 1e6,
                "p99_ms": percentile_ns(samples, 99) / 1e6,
                "samples_ns": samples,
            }

        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "config": {
                "pool": self.pool,
                "max_turns": self.max_turns,
                "teams": self.invariant[0],
                "baked_tables": self.invariant[1],
                "log_on": self.invariant[2],
                "shards": len(self.rows),
                "base_seeds": sorted(self.seeds),
                "requested_games_sum": sum(row["requested_games"] for row in self.rows),
                "threads": sorted({row["threads"] for row in self.rows}),
            },
            "result": {
                "games": games,
                "pairs": len(pair_scores),
                "wins": sum(row["wins"] for row in self.rows),
                "losses": sum(row["losses"] for row in self.rows),
                "ties": sum(row["ties"] for row in self.rows),
                "turn_caps": sum(row["turn_caps"] for row in self.rows),
                "score": score,
                "ci95": ci95,
                "ci_unit": "side_swap_pair",
                "score95_low": score_low,
                "score95_high": score_high,
                "elo": elo(score),
                "elo95_low": elo(score_low),
                "elo95_high": elo(score_high),
                "pair_scores": pair_scores,
                "turns_sum": turns_sum,
                "avg_turns": turns_sum / games,
                "wall_secs_sum": sum(row["wall_secs"] for row in self.rows),
            },
            "timing": {"unit": "choose_call", "a": timing(a_samples), "b": timing(b_samples)},
        }


def read_groups(paths):
    groups = {}
    for path in paths:
        try:
            source = open(path, encoding="utf-8")
        except OSError as error:
            raise InputError(f"{path}: {error}") from error
        with source:
            for lineno, line in enumerate(source, 1):
                if not line.strip():
                    continue
                where = f"{path}:{lineno}"
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    fail(where, f"invalid JSON: {error}")
                row = validate_row(raw, where)
                key = row["agent_a"], row["agent_b"], row["pool"], row["max_turns"]
                group = groups.setdefault(key, Group(*key))
                group.add(row)
    if not groups:
        fail("input", "no non-empty JSONL rows")
    return [groups[key] for key in sorted(groups)]


def elo_text(value, positive=False):
    if value is None:
        return "+inf" if positive else "-inf"
    return f"{value:+.0f}"


def print_report(merged_groups):
    for index, group in enumerate(merged_groups):
        if index:
            print()
        config, result, timing = group["config"], group["result"], group["timing"]
        ci = result["ci95"]
        ci_text = "n/a" if ci is None else f"{ci:.3f}"
        print(f"== {group['agent_a']} vs {group['agent_b']} | {config['pool']} | "
              f"max-turns {config['max_turns']} ==")
        print(f"shards {config['shards']}  seeds {len(config['base_seeds'])}  "
              f"games {result['games']}  pairs {result['pairs']}")
        print(f"A {result['wins']}W {result['losses']}L {result['ties']}T  caps {result['turn_caps']}  "
              f"score {result['score']:.3f} +/- {ci_text}  "
              f"Elo {elo_text(result['elo'], result['score'] >= 1.0)} "
              f"[{elo_text(result['elo95_low'])}, {elo_text(result['elo95_high'], True)}]")
        print(f"turns {result['turns_sum']} total / {result['avg_turns']:.1f} avg")
        print("think ms/move  "
              f"A {timing['a']['mean_ms']:.1f} p95 {timing['a']['p95_ms']:.1f} "
              f"p99 {timing['a']['p99_ms']:.1f} ({timing['a']['moves']} moves)  |  "
              f"B {timing['b']['mean_ms']:.1f} p95 {timing['b']['p95_ms']:.1f} "
              f"p99 {timing['b']['p99_ms']:.1f} ({timing['b']['moves']} moves)")


def write_json(path, groups):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump({"schema": OUTPUT_SCHEMA, "groups": groups}, output,
                  ensure_ascii=False, separators=(",", ":"))
        output.write("\n")


def fixture_row(seed=1, pair_scores=None, a_samples=None, b_samples=None):
    """Minimal valid row used only by --self-test."""
    pair_scores = pair_scores or [0.75, 0.25]
    a_samples = sorted(a_samples or [1_000_000, 3_000_000])
    b_samples = sorted(b_samples or [2_000_000, 4_000_000])
    score, ci95 = mean_ci95(pair_scores)
    games = len(pair_scores) * 2
    wins = losses = ties = 0
    for pair_score in pair_scores:
        if pair_score == 0:
            losses += 2
        elif pair_score == 0.25:
            losses += 1
            ties += 1
        elif pair_score == 0.5:
            wins += 1
            losses += 1
        elif pair_score == 0.75:
            wins += 1
            ties += 1
        elif pair_score == 1:
            wins += 2

    def side(samples):
        return {
            "moves": len(samples), "total_ns": sum(samples),
            "mean_ms": sum(samples) / 1e6 / len(samples),
            "p95_ms": percentile_ns(samples, 95) / 1e6,
            "p99_ms": percentile_ns(samples, 99) / 1e6, "samples_ns": samples,
        }

    return {
        "schema": SCHEMA, "agent_a": "blind:20000:1:16", "agent_b": "blind:10000:1:16",
        "config": {"requested_games": games, "base_seed": seed, "threads": 2,
                   "max_turns": 500, "pool": "meta:0-9", "teams": 10,
                   "baked_tables": 100, "log_on": True},
        "result": {"games": games, "pairs": len(pair_scores), "wins": wins, "losses": losses,
                   "ties": ties, "turn_caps": 0, "score": score, "ci95": ci95,
                   "ci_unit": "side_swap_pair", "pair_scores": pair_scores,
                   "turns_sum": games * 20, "avg_turns": 20.0, "wall_secs": 1.0},
        "timing": {"unit": "choose_call", "a": side(a_samples), "b": side(b_samples)},
    }


class SelfTest(unittest.TestCase):
    def write_rows(self, rows):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return handle.name

    def test_raw_samples_are_merged_exactly(self):
        path = self.write_rows([
            fixture_row(1, [0.75, 0.25], [1_000_000, 9_000_000], [2_000_000]),
            fixture_row(2, [1.0, 0.5], [3_000_000, 4_000_000], [8_000_000]),
        ])
        merged = read_groups([path])[0].merged()
        self.assertEqual(merged["result"]["pair_scores"], [0.75, 0.25, 1.0, 0.5])
        self.assertEqual(merged["result"]["score"], 0.625)
        self.assertEqual(merged["timing"]["a"]["mean_ms"], 4.25)
        self.assertEqual(merged["timing"]["a"]["p95_ms"], 9.0)

    def test_duplicate_seed_is_rejected_within_group(self):
        path = self.write_rows([fixture_row(7), fixture_row(7)])
        with self.assertRaisesRegex(InputError, "duplicate base_seed 7"):
            read_groups([path])

    def test_corrupt_derived_value_is_rejected(self):
        row = fixture_row()
        row["timing"]["a"]["p95_ms"] = 999
        path = self.write_rows([row])
        with self.assertRaisesRegex(InputError, "p95_ms.*recomputed"):
            read_groups([path])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="arena JSONL shard(s)")
    parser.add_argument("--json-out", "--merge-out", dest="json_out",
                        help="write merged raw samples and statistics as JSON")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.files:
        parser.error("at least one JSONL shard is required")
    try:
        merged = [group.merged() for group in read_groups(args.files)]
    except InputError as error:
        parser.error(str(error))
    print_report(merged)
    if args.json_out:
        write_json(args.json_out, merged)
        print(f"\nmerged {len(merged)} group(s) -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
