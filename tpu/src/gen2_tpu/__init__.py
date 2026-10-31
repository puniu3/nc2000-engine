"""Gen 2 (`gen2stadium2` / NC 2000) TPU Emulation & Branchless JAX Kernel."""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp

from .tpu_emu import (
    TPU_V5E,
    TPU_V5P,
    TPU_V6E,
    TPUAuditReport,
    TPUSpec,
    init_tpu_emulator,
    make_tpu_mesh,
    shard_batch,
    verify_tpu_compatibility,
)
from .prng import (
    prng_next_u32,
    prng_random,
    prng_random_chance,
    seed_from_str,
    seed_from_u64_int,
    seed_to_str,
    seed_to_u64_int,
)
from .state import (
    PACKED_BATTLE_BYTES,
    PACKED_BATTLE_WORDS,
    ActionPhase,
    CompactBattle,
    CompactWeather,
    ItemFlag,
    ItemStatKind,
    MatchTable,
    MoveFlag,
    PackedBattle,
    Status,
    VolBit,
    pack_battle,
    unpack_battle,
)
from .dex_loader import (
    DEFAULT_DEX_PATH,
    DexData,
    build_compact_battle,
    load_dex,
    toid,
)
from .kernel import (
    ACC_LUT,
    CRIT_DEN,
    SIMD16_ROLLS,
    STAT_NUM,
    Simd16HitOutcome,
    action_sort_key,
    action_sort_key_new,
    active_stat,
    apply_actions,
    apply_damage_to_mon,
    apply_roll_branchless,
    best_hit_fraction,
    check_berry_triggers,
    check_faints,
    effectiveness_ext,
    eval_leaf,
    evaluate_hit_simd16,
    exec_after_move_self,
    exec_move,
    exec_residuals,
    exec_status_move,
    exec_switch,
    expected_hit_fraction,
    get_damage_base_branchless,
    get_stat_branchless,
    legal_actions_mask,
    lookup_accuracy_threshold,
    mon_stat,
    resolve_turn_order,
    sample_uniform_legal_action,
    step_switch,
    step_turn,
    try_inflict_status,
)
from .rollout import (
    RolloutResult,
    clone_battle_batch,
    pallas_evaluate_hits,
    rollout_single,
    sharded_tpu_rollout,
    vmap_rollout,
)
from .residual import (
    EVAL_FEAT_DIM,
    EVAL_HIDDEN_DIM,
    ROLLOUT_ACT_DIM,
    ROLLOUT_RANK,
    ROLLOUT_STATE_DIM,
    EvalResidualTrainParams,
    EvalResidualWeights,
    ResidualBundle,
    ResidualTrainingMetrics,
    RolloutResidualWeights,
    compact_eval_residual_sym,
    compute_rollout_action_residuals,
    default_residual_bundle,
    effective_eval_residual_weights,
    eval_residual_forward,
    export_residual_bundle_json,
    extract_eval_features,
    extract_eval_features_with_eff,
    extract_rollout_state_and_actions,
    load_residual_bundle_json,
    residual_bundle_from_dict,
    residual_bundle_to_dict,
    train_residual_bundle_jax,
    zero_eval_residual_weights,
    zero_residual_bundle,
    zero_rollout_residual_weights,
)
from .search import (
    Depth2OracleResult,
    EvalWeights,
    HeavyRolloutResult,
    LabeledPositionBatch,
    RootEquilibriumResult,
    clone_packed_battle_batch,
    compact_base_logit_and_eff,
    compact_race_margin,
    default_eval_weights,
    eval_compact_01,
    eval_compact_leaf,
    evaluate_depth2_counterfactual_oracle,
    evaluate_root_payoff_matrix,
    heavy_rollout_packed_single,
    label_positions_batch_tpu,
    sample_heavy_playout_action,
    sharded_heavy_rollout,
    solve_rm_plus_jax,
)
from .kodoku import (
    MAX_TOTAL_LEVEL_NC2000,
    PREVIEW_ACTIONS_0BASED_NP,
    PREVIEW_ACTIONS_1BASED,
    PREVIEW_CORE_MASKS_60_NP,
    SPACE_VERSION,
    CoreNashPairResult,
    KodokuGauntletSummary,
    RectEquilibriumResult,
    RosterTable6,
    SharedDexTable,
    assemble_3v3_from_rosters_jax,
    build_shared_dex_table,
    compile_roster6,
    core_nash_6v6_pair_tpu,
    evaluate_3v3_picks_batch_tpu,
    evaluate_kodoku_gauntlet_tpu,
    export_dominance_audit_cells_json,
    export_pair_table_dict,
    export_pair_table_json,
    legal_action_indices,
    legal_preview_mask_60,
    preview_actions_1based,
    recompute_population_wars_np,
    replicator_dynamics_jax,
    select_top_k_orthogonal,
    simulate_multi_table_heavy_games,
    solve_rect_rm_plus_jax,
)

__all__ = [
    "ACC_LUT",
    "CRIT_DEN",
    "DEFAULT_DEX_PATH",
    "PACKED_BATTLE_BYTES",
    "PACKED_BATTLE_WORDS",
    "SIMD16_ROLLS",
    "STAT_NUM",
    "TPU_V5E",
    "TPU_V5P",
    "TPU_V6E",
    "ActionPhase",
    "CompactBattle",
    "CompactWeather",
    "DexData",
    "EvalWeights",
    "HeavyRolloutResult",
    "ItemFlag",
    "ItemStatKind",
    "MatchTable",
    "MoveFlag",
    "PackedBattle",
    "RolloutResult",
    "RootEquilibriumResult",
    "Simd16HitOutcome",
    "Status",
    "TPUAuditReport",
    "TPUSpec",
    "VolBit",
    "action_sort_key",
    "action_sort_key_new",
    "active_stat",
    "apply_actions",
    "apply_damage_to_mon",
    "apply_roll_branchless",
    "best_hit_fraction",
    "build_compact_battle",
    "check_berry_triggers",
    "check_faints",
    "clone_battle_batch",
    "clone_packed_battle_batch",
    "compact_race_margin",
    "default_eval_weights",
    "effectiveness_ext",
    "eval_compact_01",
    "eval_compact_leaf",
    "eval_leaf",
    "evaluate_hit_simd16",
    "evaluate_root_payoff_matrix",
    "exec_after_move_self",
    "exec_move",
    "exec_residuals",
    "exec_status_move",
    "exec_switch",
    "expected_hit_fraction",
    "get_damage_base_branchless",
    "get_stat_branchless",
    "heavy_rollout_packed_single",
    "init_tpu_emulator",
    "legal_actions_mask",
    "load_dex",
    "lookup_accuracy_threshold",
    "make_tpu_mesh",
    "mon_stat",
    "pack_battle",
    "pallas_evaluate_hits",
    "prng_next_u32",
    "prng_random",
    "prng_random_chance",
    "resolve_turn_order",
    "rollout_single",
    "sample_heavy_playout_action",
    "sample_uniform_legal_action",
    "seed_from_str",
    "seed_from_u64_int",
    "seed_to_str",
    "seed_to_u64_int",
    "shard_batch",
    "sharded_heavy_rollout",
    "sharded_tpu_rollout",
    "solve_rm_plus_jax",
    "step_switch",
    "step_turn",
    "toid",
    "try_inflict_status",
    "unpack_battle",
    "verify_tpu_compatibility",
    "vmap_rollout",
]


def _sample_teams() -> tuple[list[dict], list[dict]]:
    def mk(species: str, level: int, item: str, moves: list[str]) -> dict:
        return {
            "name": species,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "level": level,
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255,
        }

    p1 = [
        mk("Snorlax", 55, "Leftovers", ["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        mk("Zapdos", 50, "Mint Berry", ["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        mk("Cloyster", 50, "Leftovers", ["Surf", "Ice Beam", "Spikes", "Explosion"]),
    ]
    p2 = [
        mk("Marowak", 55, "Thick Club", ["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        mk("Skarmory", 50, "Leftovers", ["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        mk("Raikou", 50, "Miracle Berry", ["Thunderbolt", "Crunch", "Reflect", "Rest"]),
    ]
    return p1, p2


def _run_benchmark() -> None:
    devices = init_tpu_emulator()
    print("=== Gen 2 TPU Emulation Environment ===")
    print(f"Emulated TPU devices : {len(devices)} ({[str(d) for d in devices]})")
    print(f"jax_enable_x64       : {jax.config.read('jax_enable_x64')}")

    p1, p2 = _sample_teams()
    table, base_state = build_compact_battle("1,2,3,4", p1, p2)

    batch_size = 1024  # 8 TPU cores x 128 VPU lanes
    horizon = 32
    batched_state, policy_seeds = clone_battle_batch(base_state, batch_size)

    # Audit TPU HLO compatibility
    report = verify_tpu_compatibility(
        lambda st, tbl, ps: vmap_rollout(st, tbl, ps, horizon=horizon),
        batched_state,
        table,
        policy_seeds,
        spec=TPU_V6E,
    )
    print(f"\n--- TPU Compatibility Audit ({report.spec_name}) ---")
    print(f"Passed               : {report.passed}")
    print(f"Forbidden 64b dtypes : {report.forbidden_dtypes}")
    print(f"Host callbacks       : {report.has_host_callbacks}")
    print(f"Dynamic shapes       : {report.has_dynamic_shapes}")
    print(
        f"State + I/O memory   : {(report.arg_memory_bytes + report.output_memory_bytes) / 1024:.1f} KiB "
        f"(HLO temp buffers: {report.temp_memory_bytes / 1024:.1f} KiB, fits in {TPU_V6E.vmem_mib} MiB VMEM: {report.fits_in_vmem})"
    )
    print(f"HLO lines            : {report.hlo_line_count}")

    # Run sharded rollout across 8 virtual TPU cores
    mesh = make_tpu_mesh(axis_names=("core",))
    res = sharded_tpu_rollout(batched_state, table, policy_seeds, horizon=horizon, mesh=mesh)
    jax.block_until_ready(res.leaf_values)

    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        res = sharded_tpu_rollout(batched_state, table, policy_seeds, horizon=horizon, mesh=mesh)
    jax.block_until_ready(res.leaf_values)
    elapsed = time.perf_counter() - t0

    total_steps = batch_size * horizon * iters
    p1_wins = int(jnp.sum(res.outcomes == 1))
    p2_wins = int(jnp.sum(res.outcomes == 2))
    ties = int(jnp.sum(res.outcomes == 3))
    ongoing = int(jnp.sum(res.outcomes == 0))
    mean_val = float(jnp.mean(res.leaf_values))

    print(f"\n--- 8-Core Sharded Rollout Benchmark (B={batch_size}, H={horizon}) ---")
    print(f"Mean P1 leaf value   : {mean_val:.4f}")
    print(f"Terminal outcomes    : P1Win={p1_wins}, P2Win={p2_wins}, Tie={ties}, Ongoing={ongoing}")
    print(f"Throughput (CPU emu) : {total_steps / elapsed:,.0f} turn-steps/sec ({elapsed / iters * 1000:.2f} ms/batch)")

    # PackedBattle + Playout::Heavy + Root Payoff Matrix RM+ Benchmark
    packed_batch, heavy_seeds = clone_packed_battle_batch(base_state, batch_size)
    heavy_res = sharded_heavy_rollout(packed_batch, table, heavy_seeds, turns=8, eps=0.2, mesh=mesh)
    jax.block_until_ready(heavy_res.leaf_values)

    t1 = time.perf_counter()
    for _ in range(iters):
        heavy_res = sharded_heavy_rollout(
            packed_batch, table, heavy_seeds, turns=8, eps=0.2, mesh=mesh
        )
    jax.block_until_ready(heavy_res.leaf_values)
    heavy_elapsed = time.perf_counter() - t1

    eq = evaluate_root_payoff_matrix(base_state, table, samples_per_cell=32, rollout_turns=8)
    jax.block_until_ready(eq.game_value)

    print(f"\n--- PackedBattle (120B) + Playout::Heavy (8-turn + EvalWeights) ---")
    print(f"State compression    : 516 B -> {PACKED_BATTLE_BYTES} B ({PACKED_BATTLE_WORDS} x uint32, 4.3x reduction)")
    print(f"Heavy leaf mean (P1) : {float(jnp.mean(heavy_res.leaf_values)):.4f}")
    print(
        f"Heavy throughput     : {(batch_size * 8 * iters) / heavy_elapsed:,.0f} turn-steps/sec "
        f"({heavy_elapsed / iters * 1000:.2f} ms/batch of {batch_size} heavy rollouts)"
    )
    print(f"Root RM+ game value  : {float(eq.game_value):.4f}")
    print(f"Root RM+ sigma0      : {[round(float(x), 3) for x in eq.sigma0]}")
    print(f"Root RM+ sigma1      : {[round(float(x), 3) for x in eq.sigma1]}")


def _load_pool_teams(path: str) -> list[dict]:
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "teams" in raw:
        return raw["teams"]
    if isinstance(raw, dict) and "sets" in raw:
        return [raw]
    raise ValueError(f"Unrecognized team pool schema in {path}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import sys
    import tarfile
    from pathlib import Path

    args_list = sys.argv[1:] if argv is None else argv
    if not args_list:
        _run_benchmark()
        return

    parser = argparse.ArgumentParser(
        prog="gen2-tpu",
        description="Gen 2 NC2000 JAX TPU Engine, Oracle & 6v6 Kodoku CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("benchmark", help="Run TPU HLO audit and rollout/oracle benchmark")

    p_pair = sub.add_parser("pair-table", help="Evaluate a 6v6 team pair and export PairTable JSON")
    p_pair.add_argument("--team-a", required=True, help="Path to Team A JSON (6 sets or pool entry)")
    p_pair.add_argument("--team-b", required=True, help="Path to Team B JSON (6 sets or pool entry)")
    p_pair.add_argument("--out", required=True, help="Output PairTable JSON path")
    p_pair.add_argument("--screen-games", type=int, default=2)
    p_pair.add_argument("--support-k", type=int, default=3)
    p_pair.add_argument("--refine-games", type=int, default=6)
    p_pair.add_argument("--screen-turns", type=int, default=40)
    p_pair.add_argument("--refine-turns", type=int, default=60)
    p_pair.add_argument("--seed", type=int, default=20260405)

    p_kod = sub.add_parser(
        "kodoku-cells",
        help="Evaluate a 6v6 team pool gauntlet on TPU and export meta_eco cells JSON",
    )
    p_kod.add_argument("--pool-p", required=True, help="Path to pool P JSON")
    p_kod.add_argument("--pool-q", default=None, help="Optional path to pool Q JSON (default: symmetric P x P)")
    p_kod.add_argument("--out", required=True, help="Output cells JSON path")
    p_kod.add_argument("--screen-games", type=int, default=2)
    p_kod.add_argument("--support-k", type=int, default=3)
    p_kod.add_argument("--refine-games", type=int, default=6)
    p_kod.add_argument("--screen-turns", type=int, default=40)
    p_kod.add_argument("--refine-turns", type=int, default=60)
    p_kod.add_argument("--seed", type=int, default=20260405)

    p_bnd = sub.add_parser(
        "bundle-colab",
        help="Create self-contained tarball (including gen2stadium2.json) for Colab TPU v6e1",
    )
    p_bnd.add_argument("--out", required=True, help="Output .tar.gz path")

    ns = parser.parse_args(args_list)
    if ns.cmd == "benchmark":
        _run_benchmark()
    elif ns.cmd == "pair-table":
        from .kodoku import build_shared_dex_table

        raw_a = json.loads(Path(ns.team_a).read_text(encoding="utf-8"))
        raw_b = json.loads(Path(ns.team_b).read_text(encoding="utf-8"))
        sets_a = raw_a["sets"] if isinstance(raw_a, dict) and "sets" in raw_a else raw_a
        sets_b = raw_b["sets"] if isinstance(raw_b, dict) and "sets" in raw_b else raw_b
        id_a = raw_a.get("id", Path(ns.team_a).stem) if isinstance(raw_a, dict) else Path(ns.team_a).stem
        id_b = raw_b.get("id", Path(ns.team_b).stem) if isinstance(raw_b, dict) else Path(ns.team_b).stem

        shared_dex = build_shared_dex_table()
        r_a = compile_roster6(sets_a)
        r_b = compile_roster6(sets_b)
        res = core_nash_6v6_pair_tpu(
            r_a,
            r_b,
            shared_dex,
            screen_games=ns.screen_games,
            support_k=ns.support_k,
            refine_games=ns.refine_games,
            screen_turns=ns.screen_turns,
            refine_turns=ns.refine_turns,
            seed=ns.seed,
        )
        export_pair_table_json(
            ns.out,
            res,
            team_a=id_a,
            team_b=id_b,
            screen_games=ns.screen_games,
            support_k=ns.support_k,
            refine_games=ns.refine_games,
            screen_turns=ns.screen_turns,
            refine_turns=ns.refine_turns,
            seed=ns.seed,
        )
        print(f"Wrote PairTable ({id_a} vs {id_b}, v*={res.value:.4f}) to {ns.out}")
    elif ns.cmd == "kodoku-cells":
        teams_p = _load_pool_teams(ns.pool_p)
        teams_q = _load_pool_teams(ns.pool_q) if ns.pool_q else None
        summary = evaluate_kodoku_gauntlet_tpu(
            teams_p,
            teams_q=teams_q,
            screen_games=ns.screen_games,
            support_k=ns.support_k,
            refine_games=ns.refine_games,
            screen_turns=ns.screen_turns,
            refine_turns=ns.refine_turns,
            seed=ns.seed,
        )
        export_dominance_audit_cells_json(
            ns.out,
            summary,
            games_per_cell=float(max(1, ns.refine_games)),
        )
        print(
            f"Wrote {len(summary.p_ids)}x{len(summary.q_ids)} Kodoku cells to {ns.out}"
        )
    elif ns.cmd == "bundle-colab":
        pkg_root = Path(__file__).resolve().parents[2]
        out_path = Path(ns.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_path, "w:gz") as tf:
            tf.add(pkg_root / "src" / "gen2_tpu", arcname="src/gen2_tpu")
            tf.add(pkg_root / "tests", arcname="tests")
            if DEFAULT_DEX_PATH.exists():
                tf.add(DEFAULT_DEX_PATH, arcname="src/gen2_tpu/data/gen2stadium2.json")
        print(f"Created self-contained Colab TPU bundle at {out_path}")


