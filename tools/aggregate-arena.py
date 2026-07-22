#!/usr/bin/env python3
"""Validate/aggregate arena shards and apply the preregistered M17b tier gate."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field


SCHEMA = "nc2000-arena-v1"
OUTPUT_SCHEMA = "nc2000-arena-aggregate-v1"
GATE_MANIFEST_SCHEMA = "nc2000-arena-tier-gate-v1"
GATE_RESULT_SCHEMA = "nc2000-arena-tier-gate-result-v1"


class InputError(ValueError):
    pass


class DuplicateKeyError(ValueError):
    pass


def fail(where, message):
    raise InputError(f"{where}: {message}")


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def tagged_sha256(tag, contents):
    return f"sha256:{hashlib.sha256(contents).hexdigest()}:{tag}"


def canonical_hash(tag, value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return tagged_sha256(tag, encoded)


def evaluator_hash():
    try:
        with open(os.path.realpath(__file__), "rb") as source:
            contents = source.read()
    except OSError as error:
        raise InputError(f"evaluator: cannot hash {__file__}: {error}") from error
    return tagged_sha256("arena-tier-gate-evaluator-v1", contents)


def obj(value, where):
    if not isinstance(value, dict):
        fail(where, "expected object")
    return value


def exact_keys(value, allowed, where):
    extras = sorted(set(value) - set(allowed))
    if extras:
        fail(where, f"unknown field(s): {extras}")


def text(value, where):
    if not isinstance(value, str) or not value:
        fail(where, "expected non-empty string")
    return value


def content_fingerprint(value, where, tag):
    value = text(value, where)
    parts = value.split(":")
    if (len(parts) != 4 or parts[0] != "fnv1a64" or len(parts[1]) != 16
            or any(char not in "0123456789abcdef" for char in parts[1])
            or parts[2] != tag or not parts[3].endswith("parts")
            or not parts[3][:-5].isdigit()):
        fail(where, f"expected fnv1a64 content fingerprint tagged {tag!r}")
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
    agent_a_iterations = config.get("agent_a_iterations")
    agent_b_iterations = config.get("agent_b_iterations")
    if agent_a_iterations is not None:
        agent_a_iterations = integer(agent_a_iterations,
                                     f"{where}.config.agent_a_iterations", 1)
    if agent_b_iterations is not None:
        agent_b_iterations = integer(agent_b_iterations,
                                     f"{where}.config.agent_b_iterations", 1)
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
    invalid_games = integer(result.get("invalid_games", 0),
                            f"{where}.result.invalid_games")
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
    if invalid_games > games:
        fail(f"{where}.result.invalid_games", "exceeds games")
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

    raw_fingerprints = raw.get("fingerprints")
    fingerprints = None
    if raw_fingerprints is not None:
        raw_fingerprints = obj(raw_fingerprints, f"{where}.fingerprints")
        fingerprints = tuple(
            content_fingerprint(raw_fingerprints.get(name),
                                f"{where}.fingerprints.{name}", tag)
            for name, tag in (("build", "arena-build-v1"),
                              ("dex", "arena-dex-v1"),
                              ("pool", "arena-pool-v1"),
                              ("tables", "arena-tables-v1"))
        )

    return {
        "where": where,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "requested_games": requested_games,
        "base_seed": base_seed,
        "threads": threads,
        "max_turns": max_turns,
        "agent_a_iterations": agent_a_iterations,
        "agent_b_iterations": agent_b_iterations,
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
        "invalid_games": invalid_games,
        "pair_scores": pair_scores,
        "turns_sum": turns_sum,
        "wall_secs": wall_secs,
        "timing_a": timing_a,
        "timing_b": timing_b,
        "fingerprints": fingerprints,
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
        invariant = (row["teams"], row["baked_tables"], row["log_on"],
                     row["agent_a_iterations"], row["agent_b_iterations"],
                     row["threads"], row["fingerprints"])
        if self.invariant is None:
            self.invariant = invariant
        elif invariant != self.invariant:
            fail(row["where"],
                 "same agent/pool/max-turns group has incompatible "
                 "teams/baked_tables/log_on/iterations/threads/fingerprints")
        self.seeds.add(row["base_seed"])
        self.rows.append(row)

    def subset(self, seeds):
        """Exact manifest-selected seed subset, retaining all invariants."""
        selected = [row for row in self.rows if row["base_seed"] in seeds]
        present = {row["base_seed"] for row in selected}
        missing = sorted(set(seeds) - present)
        if missing:
            return None
        subgroup = Group(self.agent_a, self.agent_b, self.pool, self.max_turns)
        for row in selected:
            subgroup.add(row)
        return subgroup

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
                "agent_a_iterations": self.invariant[3],
                "agent_b_iterations": self.invariant[4],
                "shards": len(self.rows),
                "base_seeds": sorted(self.seeds),
                "requested_games_sum": sum(row["requested_games"] for row in self.rows),
                "threads": [self.invariant[5]],
            },
            "result": {
                "games": games,
                "pairs": len(pair_scores),
                "wins": sum(row["wins"] for row in self.rows),
                "losses": sum(row["losses"] for row in self.rows),
                "ties": sum(row["ties"] for row in self.rows),
                "turn_caps": sum(row["turn_caps"] for row in self.rows),
                "invalid_games": sum(row["invalid_games"] for row in self.rows),
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
            "fingerprints": None if self.invariant[6] is None else {
                "build": self.invariant[6][0],
                "dex": self.invariant[6][1],
                "pool": self.invariant[6][2],
                "tables": self.invariant[6][3],
            },
        }


def read_groups(paths):
    groups = {}
    artifact_generation = None
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
                raw_config = raw.get("config") if isinstance(raw, dict) else None
                raw_result = raw.get("result") if isinstance(raw, dict) else None
                current = (isinstance(raw, dict) and raw.get("fingerprints") is not None
                           or isinstance(raw_config, dict)
                           and ("agent_a_iterations" in raw_config
                                or "agent_b_iterations" in raw_config)
                           or isinstance(raw_result, dict) and "invalid_games" in raw_result)
                generation = "current" if current else "legacy"
                if artifact_generation is None:
                    artifact_generation = generation
                elif artifact_generation != generation:
                    fail(where, "legacy/current arena artifacts cannot be mixed")
                row = validate_row(raw, where)
                key = row["agent_a"], row["agent_b"], row["pool"], row["max_turns"]
                group = groups.setdefault(key, Group(*key))
                group.add(row)
    if not groups:
        fail("input", "no non-empty JSONL rows")
    return [groups[key] for key in sorted(groups)]


def integer_list(value, where):
    if not isinstance(value, list) or not value:
        fail(where, "expected non-empty integer array")
    values = [integer(item, f"{where}[{i}]", 0, 2**64 - 1)
              for i, item in enumerate(value)]
    if len(set(values)) != len(values):
        fail(where, "duplicate seed")
    return values


def validate_gate_stage(raw, where):
    raw = obj(raw, where)
    exact_keys(raw, {"agent_a", "agent_b", "base_seeds", "games_per_seed"}, where)
    stage = {
        "agent_a": text(raw.get("agent_a"), f"{where}.agent_a"),
        "agent_b": text(raw.get("agent_b"), f"{where}.agent_b"),
        "base_seeds": integer_list(raw.get("base_seeds"), f"{where}.base_seeds"),
        "games_per_seed": integer(raw.get("games_per_seed"),
                                  f"{where}.games_per_seed", 2),
    }
    if stage["games_per_seed"] % 2:
        fail(f"{where}.games_per_seed", "must be even for side-swap pairs")
    return stage


def validate_expected_artifact(raw, where):
    raw = obj(raw, where)
    exact_keys(raw, {"fingerprints", "baked_tables"}, where)
    raw_fingerprints = obj(raw.get("fingerprints"), f"{where}.fingerprints")
    exact_keys(raw_fingerprints, {"build", "dex", "pool", "tables"},
               f"{where}.fingerprints")
    fingerprints = {
        name: content_fingerprint(raw_fingerprints.get(name),
                                  f"{where}.fingerprints.{name}", tag)
        for name, tag in (("build", "arena-build-v1"),
                          ("dex", "arena-dex-v1"),
                          ("pool", "arena-pool-v1"),
                          ("tables", "arena-tables-v1"))
    }
    baked_tables = integer(raw.get("baked_tables"), f"{where}.baked_tables")
    return {"fingerprints": fingerprints, "baked_tables": baked_tables}


def validate_gate_manifest(raw, where="manifest"):
    raw = obj(raw, where)
    exact_keys(raw, {"schema", "tiers", "pool", "max_turns", "expected", "comparisons"},
               where)
    if raw.get("schema") != GATE_MANIFEST_SCHEMA:
        fail(f"{where}.schema",
             f"expected {GATE_MANIFEST_SCHEMA!r}, got {raw.get('schema')!r}")
    tiers = integer_list(raw.get("tiers"), f"{where}.tiers")
    if len(tiers) != 4 or any(higher != 2 * lower
                              for lower, higher in zip(tiers, tiers[1:])):
        fail(f"{where}.tiers",
             "expected exactly [1x, 2x, 4x, 8x] positive iteration tiers")
    if tiers[0] == 0:
        fail(f"{where}.tiers[0]", "iteration tier must be positive")
    pool = text(raw.get("pool"), f"{where}.pool")
    max_turns = integer(raw.get("max_turns"), f"{where}.max_turns", 1, 2**16 - 1)
    expected = validate_expected_artifact(raw.get("expected"), f"{where}.expected")
    raw_comparisons = raw.get("comparisons")
    if not isinstance(raw_comparisons, list) or len(raw_comparisons) != 3:
        fail(f"{where}.comparisons",
             "expected the three adjacent 2x/1x, 4x/2x, and 8x/4x comparisons")
    comparisons = []
    discovery_seeds, confirm_seeds = set(), set()
    for index, raw_comparison in enumerate(raw_comparisons):
        cwhere = f"{where}.comparisons[{index}]"
        raw_comparison = obj(raw_comparison, cwhere)
        exact_keys(raw_comparison,
                   {"lower_iters", "higher_iters", "discovery", "confirm"}, cwhere)
        lower = integer(raw_comparison.get("lower_iters"), f"{cwhere}.lower_iters", 1)
        higher = integer(raw_comparison.get("higher_iters"), f"{cwhere}.higher_iters", 1)
        if (lower, higher) != (tiers[index], tiers[index + 1]):
            fail(cwhere, f"expected adjacent tiers {tiers[index]} -> {tiers[index + 1]}")
        discovery = validate_gate_stage(raw_comparison.get("discovery"),
                                        f"{cwhere}.discovery")
        canonical_a = f"blind:{higher}:1:16"
        canonical_b = f"blind:{lower}:1:16"
        if (discovery["agent_a"], discovery["agent_b"]) != (canonical_a, canonical_b):
            fail(f"{cwhere}.discovery",
                 f"expected canonical labels {canonical_a!r} vs {canonical_b!r}")
        discovery_seeds.update(discovery["base_seeds"])
        confirm = None
        if raw_comparison.get("confirm") is not None:
            confirm = validate_gate_stage(raw_comparison["confirm"], f"{cwhere}.confirm")
            if (confirm["agent_a"], confirm["agent_b"]) != (canonical_a, canonical_b):
                fail(f"{cwhere}.confirm",
                     f"expected canonical labels {canonical_a!r} vs {canonical_b!r}")
            confirm_seeds.update(confirm["base_seeds"])
        comparisons.append({
            "lower_iters": lower,
            "higher_iters": higher,
            "discovery": discovery,
            "confirm": confirm,
        })
    overlap = sorted(discovery_seeds & confirm_seeds)
    if overlap:
        fail(f"{where}.comparisons",
             f"discovery and confirm base seeds overlap: {overlap}")
    return {"schema": GATE_MANIFEST_SCHEMA,
            "tiers": tiers, "pool": pool, "max_turns": max_turns,
            "expected": expected,
            "comparisons": comparisons}


def read_gate_manifest(path):
    try:
        with open(path, encoding="utf-8") as source:
            raw = json.load(source, object_pairs_hook=reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, DuplicateKeyError) as error:
        raise InputError(f"{path}: {error}") from error
    return validate_gate_manifest(raw, path)


def select_gate_stage(groups, manifest, comparison, stage_name, max_turns):
    stage = comparison[stage_name]
    if stage is None:
        return None
    matches = [group for group in groups
               if group.agent_a == stage["agent_a"]
               and group.agent_b == stage["agent_b"]
               and group.pool == manifest["pool"]
               and group.max_turns == max_turns]
    if len(matches) > 1:
        raise AssertionError("group key must be unique")
    if not matches:
        return None
    subgroup = matches[0].subset(set(stage["base_seeds"]))
    if subgroup is None:
        return None
    wrong_sizes = [(row["base_seed"], row["games"]) for row in subgroup.rows
                   if row["games"] != stage["games_per_seed"]]
    if wrong_sizes:
        fail(stage_name,
             f"expected {stage['games_per_seed']} games per base seed, got {wrong_sizes}")
    return subgroup


def assess_gate_stage(group, manifest, comparison, stage_name):
    merged = group.merged()
    config, result = merged["config"], merged["result"]
    where = f"{stage_name} {comparison['higher_iters']}v{comparison['lower_iters']}"
    if config["agent_a_iterations"] != comparison["higher_iters"]:
        fail(where, "agent A artifact iteration budget does not match higher_iters")
    if config["agent_b_iterations"] != comparison["lower_iters"]:
        fail(where, "agent B artifact iteration budget does not match lower_iters")
    if merged["fingerprints"] is None:
        fail(where, "artifact lacks build/dex/pool/table content fingerprints")
    if merged["fingerprints"] != manifest["expected"]["fingerprints"]:
        fail(where, "artifact fingerprints do not match manifest.expected")
    if config["baked_tables"] != manifest["expected"]["baked_tables"]:
        fail(where,
             f"baked table count {config['baked_tables']} does not match manifest.expected "
             f"{manifest['expected']['baked_tables']}")

    games = result["games"]
    cap_rate = result["turn_caps"] / games
    invalid_rate = result["invalid_games"] / games
    assessment = {
        "stage": stage_name,
        "lower_iters": comparison["lower_iters"],
        "higher_iters": comparison["higher_iters"],
        "max_turns": config["max_turns"],
        "base_seeds": config["base_seeds"],
        "games": games,
        "score": result["score"],
        "score95_low": result["score95_low"],
        "score95_high": result["score95_high"],
        "turn_cap_rate": cap_rate,
        "invalid_rate": invalid_rate,
        "decision": None,
    }
    if cap_rate > 0.01 or invalid_rate > 0.01:
        assessment["decision"] = ("rerun_max_turns_1000"
                                  if config["max_turns"] < 1000
                                  else "invalid_or_cap_rate_still_above_1pct")
        return assessment, merged["fingerprints"]
    if result["ci95"] is None:
        assessment["decision"] = "insufficient_pairs"
        return assessment, merged["fingerprints"]

    promote = result["score95_low"] > 0.5 and result["score"] >= 0.53
    stop = result["score95_high"] < 0.55
    # The futility boundary is sovereign when both asymmetric rules happen
    # to hold: a tier whose entire interval is below 0.55 is not deployable.
    if stop:
        assessment["decision"] = "stop"
    elif promote:
        assessment["decision"] = "promote"
    else:
        assessment["decision"] = "inconclusive"
    return assessment, merged["fingerprints"]


def evaluate_gate(groups, manifest):
    """Sequential 1x/2x/4x/8x discovery, then fresh-seed knee confirmation."""
    assessments = []
    manifest_fingerprint = canonical_hash("arena-tier-gate-manifest-v1", manifest)
    evaluator_fingerprint = evaluator_hash()
    lineage = manifest["expected"]["fingerprints"]

    def consume_group(group, comparison, stage_name):
        if group is None:
            return None
        assessment, _ = assess_gate_stage(group, manifest, comparison, stage_name)
        assessments.append(assessment)
        return assessment

    def consume(comparison, stage_name):
        base = select_gate_stage(groups, manifest, comparison, stage_name,
                                 manifest["max_turns"])
        fallback = None
        if manifest["max_turns"] < 1000:
            fallback = select_gate_stage(groups, manifest, comparison, stage_name, 1000)
        if base is None:
            return consume_group(fallback, comparison, stage_name)
        assessment = consume_group(base, comparison, stage_name)
        if assessment["decision"] == "rerun_max_turns_1000" and fallback is not None:
            return consume_group(fallback, comparison, stage_name)
        return assessment

    def result(recommended_iters=None, reason=None, rerun=None):
        return {
            "schema": GATE_RESULT_SCHEMA,
            "recommended_iters": recommended_iters,
            "inconclusive": reason,
            "rerun_required": rerun,
            "manifest_hash": manifest_fingerprint,
            "evaluator_hash": evaluator_fingerprint,
            "fingerprints": lineage,
            "assessments": assessments,
        }

    promoted = []
    for comparison in manifest["comparisons"]:
        assessment = consume(comparison, "discovery")
        if assessment is None:
            return result(reason=(f"missing discovery data for "
                                  f"{comparison['higher_iters']}v{comparison['lower_iters']}"))
        decision = assessment["decision"]
        if decision == "rerun_max_turns_1000":
            return result(reason="invalid/cap rate above 1%",
                          rerun={"max_turns": 1000,
                                 "stage": "discovery",
                                 "lower_iters": comparison["lower_iters"],
                                 "higher_iters": comparison["higher_iters"],
                                 "base_seeds": assessment["base_seeds"]})
        if decision != "promote":
            if decision == "stop" and not promoted:
                return result(recommended_iters=manifest["tiers"][0])
            if decision == "stop":
                break
            return result(reason=f"discovery decision: {decision}")
        promoted.append(comparison)

    # The highest promoted tier is the knee candidate. It is deployable only
    # after the same adjacent comparison passes on globally fresh seeds.
    candidate = promoted[-1]
    assessment = consume(candidate, "confirm")
    if assessment is None:
        return result(reason=(f"missing fresh-seed confirmation for "
                              f"{candidate['higher_iters']}v{candidate['lower_iters']}"))
    decision = assessment["decision"]
    if decision == "rerun_max_turns_1000":
        return result(reason="invalid/cap rate above 1%",
                      rerun={"max_turns": 1000,
                             "stage": "confirm",
                             "lower_iters": candidate["lower_iters"],
                             "higher_iters": candidate["higher_iters"],
                             "base_seeds": assessment["base_seeds"]})
    if decision == "promote":
        return result(recommended_iters=candidate["higher_iters"])
    return result(reason=f"confirm decision: {decision}")


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
        print(f"A {result['wins']}W {result['losses']}L {result['ties']}T  "
              f"caps {result['turn_caps']} invalid {result['invalid_games']}  "
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


def write_gate_json(path, gate):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(gate, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")


def print_gate(gate):
    print("\n== M17b 1x/2x/4x/8x deploy gate ==")
    for item in gate["assessments"]:
        print(f"{item['stage']} {item['higher_iters']}v{item['lower_iters']}: "
              f"max-turns {item['max_turns']}  "
              f"score {item['score']:.3f} "
              f"[{item['score95_low']:.3f}, {item['score95_high']:.3f}] "
              f"caps {item['turn_cap_rate']:.2%} invalid {item['invalid_rate']:.2%} "
              f"=> {item['decision']}")
    if gate["recommended_iters"] is not None:
        print(f"recommended_iters {gate['recommended_iters']}")
    else:
        print(f"inconclusive: {gate['inconclusive']}")
    if gate["rerun_required"] is not None:
        rerun = gate["rerun_required"]
        print(f"rerun required: {rerun['stage']} "
              f"{rerun['higher_iters']}v{rerun['lower_iters']} "
              f"with --max-turns {rerun['max_turns']}")


def fixture_row(seed=1, pair_scores=None, a_samples=None, b_samples=None,
                higher=20000, lower=10000, turn_caps=0, max_turns=500,
                baked_tables=100, threads=2):
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
        "schema": SCHEMA,
        "agent_a": f"blind:{higher}:1:16", "agent_b": f"blind:{lower}:1:16",
        "config": {"requested_games": games, "base_seed": seed, "threads": threads,
                   "max_turns": max_turns, "pool": "meta:0-9", "teams": 10,
                   "agent_a_iterations": higher, "agent_b_iterations": lower,
                   "baked_tables": baked_tables, "log_on": True},
        "result": {"games": games, "pairs": len(pair_scores), "wins": wins, "losses": losses,
                   "ties": ties, "turn_caps": turn_caps, "invalid_games": 0,
                   "score": score, "ci95": ci95,
                   "ci_unit": "side_swap_pair", "pair_scores": pair_scores,
                   "turns_sum": games * 20, "avg_turns": 20.0, "wall_secs": 1.0},
        "timing": {"unit": "choose_call", "a": side(a_samples), "b": side(b_samples)},
        "fingerprints": {
            "build": "fnv1a64:1111111111111111:arena-build-v1:1parts",
            "dex": "fnv1a64:aaaaaaaaaaaaaaaa:arena-dex-v1:1parts",
            "pool": "fnv1a64:2222222222222222:arena-pool-v1:3parts",
            "tables": "fnv1a64:3333333333333333:arena-tables-v1:2parts",
        },
    }


class SelfTest(unittest.TestCase):
    def write_rows(self, rows):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return handle.name

    def write_json(self, value):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with handle:
            json.dump(value, handle)
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

    @staticmethod
    def gate_manifest(confirm0=True, confirm1=True, confirm2=True):
        comparison0 = {
            "lower_iters": 10000, "higher_iters": 20000,
            "discovery": {"agent_a": "blind:20000:1:16",
                          "agent_b": "blind:10000:1:16", "base_seeds": [1],
                          "games_per_seed": 4},
        }
        comparison1 = {
            "lower_iters": 20000, "higher_iters": 40000,
            "discovery": {"agent_a": "blind:40000:1:16",
                          "agent_b": "blind:20000:1:16", "base_seeds": [2],
                          "games_per_seed": 4},
        }
        comparison2 = {
            "lower_iters": 40000, "higher_iters": 80000,
            "discovery": {"agent_a": "blind:80000:1:16",
                          "agent_b": "blind:40000:1:16", "base_seeds": [3],
                          "games_per_seed": 4},
        }
        if confirm0:
            comparison0["confirm"] = {
                "agent_a": "blind:20000:1:16", "agent_b": "blind:10000:1:16",
                "base_seeds": [101], "games_per_seed": 4,
            }
        if confirm1:
            comparison1["confirm"] = {
                "agent_a": "blind:40000:1:16", "agent_b": "blind:20000:1:16",
                "base_seeds": [102], "games_per_seed": 4,
            }
        if confirm2:
            comparison2["confirm"] = {
                "agent_a": "blind:80000:1:16", "agent_b": "blind:40000:1:16",
                "base_seeds": [103], "games_per_seed": 4,
            }
        return validate_gate_manifest({
            "schema": GATE_MANIFEST_SCHEMA,
            "tiers": [10000, 20000, 40000, 80000],
            "pool": "meta:0-9", "max_turns": 500,
            "expected": {
                "fingerprints": {
                    "build": "fnv1a64:1111111111111111:arena-build-v1:1parts",
                    "dex": "fnv1a64:aaaaaaaaaaaaaaaa:arena-dex-v1:1parts",
                    "pool": "fnv1a64:2222222222222222:arena-pool-v1:3parts",
                    "tables": "fnv1a64:3333333333333333:arena-tables-v1:2parts",
                },
                "baked_tables": 100,
            },
            "comparisons": [comparison0, comparison1, comparison2],
        })

    def test_gate_recommends_confirmed_knee(self):
        path = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000),
            fixture_row(2, [0.5, 0.5], higher=40000, lower=20000),
            fixture_row(101, [0.75, 0.75], higher=20000, lower=10000),
        ])
        gate = evaluate_gate(read_groups([path]), self.gate_manifest())
        self.assertEqual(gate["recommended_iters"], 20000)
        self.assertIsNone(gate["inconclusive"])
        self.assertEqual([item["decision"] for item in gate["assessments"]],
                         ["promote", "stop", "promote"])
        self.assertEqual(gate["manifest_hash"],
                         canonical_hash("arena-tier-gate-manifest-v1",
                                        self.gate_manifest()))
        self.assertEqual(gate["evaluator_hash"], evaluator_hash())

    def test_gate_requires_fresh_confirmation(self):
        path = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000),
            fixture_row(2, [0.5, 0.5], higher=40000, lower=20000),
        ])
        gate = evaluate_gate(read_groups([path]), self.gate_manifest(confirm0=False))
        self.assertIsNone(gate["recommended_iters"])
        self.assertRegex(gate["inconclusive"], "fresh-seed confirmation")

    def test_gate_can_recommend_confirmed_8x_tier(self):
        path = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000),
            fixture_row(2, [1.0, 1.0], higher=40000, lower=20000),
            fixture_row(3, [1.0, 1.0], higher=80000, lower=40000),
            fixture_row(103, [0.75, 0.75], higher=80000, lower=40000),
        ])
        gate = evaluate_gate(read_groups([path]), self.gate_manifest())
        self.assertEqual(gate["recommended_iters"], 80000)
        self.assertEqual([item["decision"] for item in gate["assessments"]],
                         ["promote", "promote", "promote", "promote"])

    def test_gate_rejects_discovery_confirm_seed_overlap(self):
        manifest = self.gate_manifest()
        raw = {
            "schema": GATE_MANIFEST_SCHEMA, "tiers": manifest["tiers"],
            "pool": manifest["pool"], "max_turns": manifest["max_turns"],
            "expected": manifest["expected"],
            "comparisons": manifest["comparisons"],
        }
        raw["comparisons"][0]["confirm"]["base_seeds"] = [1]
        with self.assertRaisesRegex(InputError, "seeds overlap"):
            validate_gate_manifest(raw)

    def test_gate_fails_closed_on_lineage_mismatch(self):
        confirm = fixture_row(101, [0.75, 0.75], higher=20000, lower=10000)
        confirm["fingerprints"]["build"] = "fnv1a64:deadbeefdeadbeef:arena-build-v1:1parts"
        path = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000),
            fixture_row(2, [0.5, 0.5], higher=40000, lower=20000),
            confirm,
        ])
        with self.assertRaisesRegex(InputError, "incompatible.*fingerprints"):
            evaluate_gate(read_groups([path]), self.gate_manifest())

    def test_gate_requests_1000_turn_rerun_above_one_percent(self):
        path = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000, turn_caps=1),
        ])
        gate = evaluate_gate(read_groups([path]), self.gate_manifest())
        self.assertIsNone(gate["recommended_iters"])
        self.assertEqual(gate["rerun_required"]["max_turns"], 1000)

    def test_gate_consumes_only_affected_stage_1000_turn_rerun(self):
        path = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000, turn_caps=1),
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000,
                        max_turns=1000),
            fixture_row(2, [0.5, 0.5], higher=40000, lower=20000),
            fixture_row(101, [0.75, 0.75], higher=20000, lower=10000),
        ])
        gate = evaluate_gate(read_groups([path]), self.gate_manifest())
        self.assertEqual(gate["recommended_iters"], 20000)
        self.assertEqual([(item["decision"], item["max_turns"])
                          for item in gate["assessments"]],
                         [("rerun_max_turns_1000", 500), ("promote", 1000),
                          ("stop", 500), ("promote", 500)])

    def test_gate_pins_expected_fingerprint_and_baked_count(self):
        wrong_fingerprint = fixture_row(1, [1.0, 1.0], higher=20000, lower=10000)
        wrong_fingerprint["fingerprints"]["dex"] = (
            "fnv1a64:bbbbbbbbbbbbbbbb:arena-dex-v1:1parts")
        with self.assertRaisesRegex(InputError, "fingerprints.*manifest.expected"):
            evaluate_gate(read_groups([self.write_rows([wrong_fingerprint])]),
                          self.gate_manifest())

        wrong_count = fixture_row(1, [1.0, 1.0], higher=20000, lower=10000,
                                  baked_tables=99)
        with self.assertRaisesRegex(InputError, "baked table count 99"):
            evaluate_gate(read_groups([self.write_rows([wrong_count])]),
                          self.gate_manifest())

    def test_group_rejects_thread_count_drift(self):
        path = self.write_rows([fixture_row(1), fixture_row(2, threads=3)])
        with self.assertRaisesRegex(InputError, "incompatible.*threads"):
            read_groups([path])

    def test_legacy_current_mix_is_explicitly_rejected(self):
        legacy = fixture_row(9)
        legacy["agent_a"] = "random"
        legacy["agent_b"] = "maxdamage"
        del legacy["fingerprints"]
        del legacy["config"]["agent_a_iterations"]
        del legacy["config"]["agent_b_iterations"]
        del legacy["result"]["invalid_games"]
        path = self.write_rows([fixture_row(1), legacy])
        with self.assertRaisesRegex(InputError, "legacy/current.*cannot be mixed"):
            read_groups([path])

    def test_manifest_duplicate_json_key_is_rejected(self):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with handle:
            handle.write('{"schema":"first","schema":"second"}')
        with self.assertRaisesRegex(InputError, "duplicate JSON key 'schema'"):
            read_gate_manifest(handle.name)

    def test_manifest_rejects_noncanonical_agent_label(self):
        manifest = self.gate_manifest()
        manifest["comparisons"][0]["discovery"]["agent_a"] = "blind:20000"
        with self.assertRaisesRegex(InputError, "expected canonical labels"):
            validate_gate_manifest(manifest)

    def test_cli_returns_nonzero_for_missing_inconclusive_and_rerun(self):
        missing = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000),
        ])
        manifest = self.write_json(self.gate_manifest())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([missing, "--gate-manifest", manifest]), 1)

        inconclusive = self.write_rows([
            fixture_row(1, [0.75, 0.25], higher=20000, lower=10000),
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([inconclusive, "--gate-manifest", manifest]), 1)

        rerun = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000, turn_caps=1),
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([rerun, "--gate-manifest", manifest]), 1)

    def test_gate_futility_rule_wins_when_boundaries_overlap(self):
        scores = ([0.75] * 16 + [0.5] * 84) * 10  # point .54, upper95 < .55
        path = self.write_rows([
            fixture_row(1, scores, higher=20000, lower=10000),
        ])
        manifest = self.gate_manifest()
        manifest["comparisons"][0]["discovery"]["games_per_seed"] = 2000
        gate = evaluate_gate(read_groups([path]), manifest)
        self.assertEqual(gate["assessments"][0]["decision"], "stop")
        self.assertEqual(gate["recommended_iters"], 10000)

    def test_gate_rejects_incomplete_seed_shard(self):
        path = self.write_rows([
            fixture_row(1, [1.0, 1.0], higher=20000, lower=10000),
        ])
        manifest = self.gate_manifest()
        manifest["comparisons"][0]["discovery"]["games_per_seed"] = 200
        with self.assertRaisesRegex(InputError, "expected 200 games per base seed"):
            evaluate_gate(read_groups([path]), manifest)

    def test_legacy_v1_still_merges_without_gate_lineage(self):
        row = fixture_row()
        del row["fingerprints"]
        del row["config"]["agent_a_iterations"]
        del row["config"]["agent_b_iterations"]
        del row["result"]["invalid_games"]
        merged = read_groups([self.write_rows([row])])[0].merged()
        self.assertIsNone(merged["fingerprints"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="arena JSONL shard(s)")
    parser.add_argument("--json-out", "--merge-out", dest="json_out",
                        help="write merged raw samples and statistics as JSON")
    parser.add_argument("--gate-manifest",
                        help="apply nc2000-arena-tier-gate-v1 manifest")
    parser.add_argument("--gate-out", help="write tier-gate result as JSON")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.files:
        parser.error("at least one JSONL shard is required")
    if args.gate_out and not args.gate_manifest:
        parser.error("--gate-out requires --gate-manifest")
    try:
        groups = read_groups(args.files)
        merged = [group.merged() for group in groups]
        manifest = read_gate_manifest(args.gate_manifest) if args.gate_manifest else None
        gate = evaluate_gate(groups, manifest) if manifest else None
    except InputError as error:
        parser.error(str(error))
    print_report(merged)
    if args.json_out:
        write_json(args.json_out, merged)
        print(f"\nmerged {len(merged)} group(s) -> {args.json_out}")
    if gate:
        print_gate(gate)
        if args.gate_out:
            write_gate_json(args.gate_out, gate)
            print(f"gate result -> {args.gate_out}")
        if gate["recommended_iters"] is None:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
