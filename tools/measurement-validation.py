#!/usr/bin/env python3
"""Independent historical accounting and an information-boundary falsifier.

Does not run matches, change policies, or validate engine outcomes from scores.
Exit 2 means the confirmation gate failed, not a tool/infrastructure failure.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB
ROOT = Path(__file__).resolve().parents[1]
HISTORY = "872d8ed8"
REPORT_PATH = "data/blind-boundary-repair-v1/challenger.json"
PARTY_PATH = "data/kodoku-r3/conclusion-equilibrium-deme.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mix(state):
    state = ((state ^ (state >> 30)) * M1) & MASK
    state = ((state ^ (state >> 27)) * M2) & MASK
    return state ^ (state >> 31)


def undo_right_xor(value, shift):
    result = value
    for offset in range(shift, 64, shift):
        result ^= value >> offset
    return result


def unmix(value):
    value = undo_right_xor(value, 31)
    value = (value * pow(M2, -1, 1 << 64)) & MASK
    value = undo_right_xor(value, 27)
    value = (value * pow(M1, -1, 1 << 64)) & MASK
    return undo_right_xor(value, 30)


def battle_seed(value):
    return ",".join(str((value >> shift) & 65535) for shift in (48, 32, 16, 0))


def recover_secrets(own_seed, own_side):
    """Attack inputs are ONLY Red's own u64 seed and physical side (0/1).

    No report, opponent seed, engine seed, run master seed, block index,
    true team, observation or future choice is supplied to this function.
    The public scheduling algorithm assigns consecutive outputs:
    battle seed, physical seat 0 seed, physical seat 1 seed.
    """
    require(own_side in (0, 1), "invalid side")
    require(type(own_seed) is int and 0 <= own_seed <= MASK, "invalid own seed")
    state = unmix(own_seed)
    engine_state = (state - (own_side + 1) * GAMMA) & MASK
    opponent_state = (state + (1 if own_side == 0 else -1) * GAMMA) & MASK
    opponent_seed = mix(opponent_state)
    return {
        "battle_seed": battle_seed(mix(engine_state)),
        "blue_seed_u64": opponent_seed,
        "blue_effective_seed_u32": opponent_seed & 0xFFFFFFFF,
    }


def historical_jobs(report, parties):
    """Recompute draw/seat/RNG schedule independently in Python."""
    teams = parties["teams"]
    weights = [t["weight"] for t in teams]
    require(len(teams) == 4 and weights == [.30, .25, .25, .20], "wrong pool")
    cumulative = []
    total = 0.0
    for weight in weights:
        total += weight
        cumulative.append(total)
    match_state = report["seed"] ^ 0xD6E8FEB86659FD93
    jobs = []
    for block in range(report["pairs_per_tier"]):
        draws = []
        for _ in range(2):
            match_state = (match_state + GAMMA) & MASK
            draw = (mix(match_state) >> 11) / (1 << 53) * total
            draws.append(next(i for i, edge in enumerate(cumulative) if edge >= draw))
        blue, red = draws
        state = report["seed"] ^ (((block + 1) * GAMMA) & MASK) ^ (((blue * 7 + red) * M1) & MASK)
        values = []
        for _ in range(3):
            state = (state + GAMMA) & MASK
            values.append(mix(state))
        engine, first, second = values
        for mirrored in (False, True):
            jobs.append({
                "index": len(jobs), "blue_is_p1": not mirrored,
                "blue_team": teams[red if mirrored else blue]["id"],
                "red_team": teams[blue if mirrored else red]["id"],
                "battle_seed": battle_seed(engine),
                "blue_seed": second if mirrored else first,
                "red_seed": first if mirrored else second,
            })
    return jobs


def accounting(report, parties):
    require(report["corrected_product"] is True, "not corrected-product history")
    require(report["base_iters"] == 81000 and report["k_oracle"] == 2
            and report["same_budget"] is True, "unexpected experiment configuration")
    tier = next(t for t in report["duel_tiers"] if t["tier"] == "T2_Rules_Bayesian_Oracle_BR")
    games = tier["game_records"]
    require(len(games) == 64 and report["pairs_per_tier"] == 32, "incomplete history")
    require(tier["games"] == report["games_per_tier"] == len(games), "game count mismatch")
    expected = historical_jobs(report, parties)
    for game, job in zip(games, expected):
        for key, value in job.items():
            require(game[key] == value, f"game {job['index']}: wrong {key}")
        require(game["blue_score"] in (0, .5, 1), "invalid score")
        require(type(game["turns"]) is int and game["turns"] >= 0, "invalid turn")
        require(type(game["solver_interventions"]) is int and game["solver_interventions"] >= 0,
                "invalid solver count")
    red_scores = [1 - game["blue_score"] for game in games]
    blocks = [statistics.mean(red_scores[i:i+2]) for i in range(0, len(games), 2)]
    mean = statistics.mean(blocks)
    se = statistics.stdev(blocks) / math.sqrt(len(blocks))
    require(tier["paired_red_scores"] == blocks, "wrong block scores")
    checks = {
        "red_wr": mean, "blue_wr": 1 - mean, "se_red_wr": se,
        "gain_over_shipped_bot": mean - .5,
        "avg_turns": statistics.mean(g["turns"] for g in games),
        "solver_interventions": sum(g["solver_interventions"] for g in games),
        "turn_capped_or_natural_ties": red_scores.count(.5),
    }
    for key, value in checks.items():
        require(math.isclose(tier[key], value, rel_tol=1e-12, abs_tol=1e-15), f"wrong aggregate {key}")
    interval = [max(0., mean - 1.96 * se), min(1., mean + 1.96 * se)]
    require(all(math.isclose(a, b, abs_tol=1e-15) for a, b in
                zip(interval, tier["red_wr_95pct_normal_approx"])), "wrong descriptive interval")
    exact = [g for g in games if g["blue_team"] == g["red_team"]]
    return {
        "status": "pass_record_accounting_only", "games": len(games), "blocks": len(blocks),
        "red_wins": red_scores.count(1), "blue_wins": red_scores.count(0), "draws": red_scores.count(.5),
        "paired_red_scores": blocks, "paired_score_histogram": dict(collections.Counter(blocks)),
        "red_mean": mean, "paired_standard_error": se, "descriptive_normal_95pct": interval,
        "max_recorded_turn": max(g["turns"] for g in games), "harness_turn_cap": report["max_turns"],
        "records_at_or_above_cap": [g["index"] for g in games if g["turns"] >= report["max_turns"]],
        "solver_interventions": checks["solver_interventions"],
        "exact_same_type_games": len(exact),
        "exact_same_type_red_wins": sum(g["blue_score"] == 0 for g in exact),
        "exact_same_type_blue_wins": sum(g["blue_score"] == 1 for g in exact),
        "limitation": "Scores are reported labels; no historical action logs exist to independently re-score engine terminals.",
    }


def boundary(report):
    games = next(t for t in report["duel_tiers"] if t["tier"] == "T2_Rules_Bayesian_Oracle_BR")["game_records"]
    witnesses = []
    for game in games:
        own_side = 1 if game["blue_is_p1"] else 0
        # The attack is evaluated before looking up referee-only truth below.
        guessed = recover_secrets(game["red_seed"], own_side)
        engine_match = guessed["battle_seed"] == game["battle_seed"]
        opponent_match = guessed["blue_seed_u64"] == game["blue_seed"]
        witnesses.append({
            "index": game["index"],
            "attack_inputs": {"own_seed": game["red_seed"], "own_side": own_side},
            "recovered": guessed,
            "engine_seed_matches": engine_match, "opponent_seed_matches": opponent_match,
        })
    exposed = sum(w["engine_seed_matches"] and w["opponent_seed_matches"] for w in witnesses)
    return {
        "status": "fail" if exposed else "no_recovery_by_this_probe_only",
        "recovered_games": exposed, "tested_games": len(games), "witnesses": witnesses,
        "attack_uses": ["own Red u64 seed", "own physical seat", "public scheduling algorithm"],
        "attack_does_not_use": ["master experiment seed", "block index", "true team", "opponent seed", "engine seed", "future actions"],
        "causal_scope": "Recoverability is demonstrated. This does not show that historical T2 used the inverse or that this defect caused its 42 wins.",
    }


def git_blob(path):
    return subprocess.check_output(["git", "show", f"{HISTORY}:{path}"], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_report, raw_parties = git_blob(REPORT_PATH), git_blob(PARTY_PATH)
    report, parties = json.loads(raw_report), json.loads(raw_parties)
    counts = accounting(report, parties)
    leak = boundary(report)
    paths = [
        "crates/bot/src/rng.rs", "crates/bot/examples/omniscient_audit.rs",
        "data/gen2stadium2.json", "data/belief-pool-v3/belief-pool.json",
        "data/compiled-preview-v1.json", "data/live-search.json", PARTY_PATH,
    ]
    frozen = {
        path: {
            "historical_sha256": sha(git_blob(path)),
            "working_sha256": sha((ROOT / path).read_bytes()) if (ROOT / path).exists() else None,
        }
        for path in paths
    }
    result = {
        "schema": "measurement-gates-v1", "historical_revision": HISTORY,
        "working_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "audit_script_sha256": sha(Path(__file__).read_bytes()),
        "historical_report_sha256": sha(raw_report), "frozen_input_hashes": frozen,
        "historical_accounting": counts, "seed_boundary": leak,
        "confirmation_status": "not_started_gate_failed" if leak["status"] == "fail" else "not_started_remaining_gates_unverified",
        "remaining_gates": {
            "full_manifest_and_binaries": "not_frozen_for_confirmation",
            "isolated_worker_and_hidden_state_probes": "not_implemented",
            "actual_worker_parity_and_independent_terminal_replay": "not_run",
            "T2_W_clock_controls": "not_run", "K_compatible_preview": "not_built",
            "prospective_D_K_matches": "not_run",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "historical-report.json").write_bytes(raw_report)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accounting": counts, "secret_recovery_games": leak["recovered_games"],
                      "confirmation_status": result["confirmation_status"]}, indent=2))
    return 2 if leak["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
