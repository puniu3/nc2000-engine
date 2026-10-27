#!/usr/bin/env python3
"""GCP Cloud Batch orchestrator for NC2000 PSRO x Quality-Diversity metagame runs.

Uses native Google Cloud Batch (`gcloud batch jobs submit`) with SPOT VMs
(`provisioningModel: SPOT`) and GCS bucket mounting. Never uploads credentials
or home directories.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import time

PROJECT = "focus-cairn-458511-f1"
DEFAULT_REGION = "us-central1"
BUCKETS = {
    "us-central1": "focus-cairn-458511-f1-spot-batch-us-central1",
    "asia-northeast1": "focus-cairn-458511-f1-spot-batch-asia-northeast1",
    "asia-northeast3": "focus-cairn-458511-f1-spot-batch-asia-northeast3",
}
GCS_PREFIX = "nc2000-psro-v1"
RUST_IMAGE = "rust:1.94-bookworm"

# 16 diverse strategic anchor niches (Trap/Perish, Stall/Rest, Baton Pass,
# Spikes/Phaze, Belly Drum, Off-Meta Sweepers) for Quality-Diversity lineages:
QD_ANCHORS = [
    "misdreavus",
    "umbreon",
    "suicune",
    "raikou",
    "tyranitar",
    "marowak",
    "heracross",
    "skarmory",
    "blissey",
    "vaporeon",
    "espeon",
    "smeargle",
    "forretress",
    "quagsire",
    "porygon2",
    "machamp",
    "tentacruel",
    "houndoom",
    "steelix",
    "jolteon",
    "miltank",
    "clefable",
    "alakazam",
    "dragonite",
]

STAGE_PATHS = [
    "Cargo.toml",
    "Cargo.lock",
    "crates/engine/Cargo.toml",
    "crates/engine/build.rs",
    "crates/engine/src",
    "crates/bot/Cargo.toml",
    "crates/bot/build.rs",
    "crates/bot/src",
    "crates/bot/examples",
    "crates/conformance/Cargo.toml",
    "crates/conformance/src",
    "crates/wasm/Cargo.toml",
    "crates/wasm/src",
    "data/gen2stadium2.json",
    "data/learnsets-gen2.json",
    "data/legal-move-sources-gen2.json",
    "data/compiled-preview-v1.json",
    "data/search-profiles.json",
    "data/community-rentals-v0/teams.json",
    "data/meta-pool-v0/meta-pool.json",
    "data/team-pool-v2/team-pool.json",
    "data/belief-pool-v3/belief-pool.json",
    "data/meta-nash-v3/pool-artifact.json",
    "data/meta-nash-v3/solution.json",
    "data/team-selection-v2/panel.json",
    "data/team-inventory-v1/eval/teams.json",
]


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], capture: bool = False) -> str:
    print("+", " ".join(cmd), file=sys.stderr)
    if capture:
        res = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return res.stdout.strip()
    subprocess.run(cmd, check=True)
    return ""


def build_source_tarball(root: pathlib.Path, out_tar: pathlib.Path) -> str:
    h = hashlib.sha256()
    with tarfile.open(out_tar, "w:gz") as tf:
        for rel in STAGE_PATHS:
            p = root / rel
            if not p.exists():
                raise FileNotFoundError(f"missing staged path: {p}")
            if p.is_file():
                files = [p]
            else:
                files = sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                arcname = f.relative_to(root).as_posix()
                h.update(arcname.encode())
                h.update(b"\0")
                h.update(f.read_bytes())
                h.update(b"\0")
                tf.add(f, arcname=arcname)
    return h.hexdigest()[:16]


def cmd_stage(args: argparse.Namespace) -> str:
    root = repo_root()
    bucket = BUCKETS[args.region]
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = pathlib.Path(tmp) / "src-data.tar.gz"
        src_hash = build_source_tarball(root, tar_path)
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"Packaged src-data.tar.gz ({size_mb:.2f} MiB, sha256[:16]={src_hash})", file=sys.stderr)
        gcs_uri = f"gs://{bucket}/{GCS_PREFIX}/src-{src_hash}.tar.gz"
        latest_uri = f"gs://{bucket}/{GCS_PREFIX}/src-latest.tar.gz"
        run_cmd(["gcloud", "storage", "cp", str(tar_path), gcs_uri])
        run_cmd(["gcloud", "storage", "cp", gcs_uri, latest_uri])
        return src_hash


def submit_batch_job(
    job_id: str,
    region: str,
    machine_type: str,
    task_count: int,
    parallelism: int,
    cpu_milli: int,
    memory_mib: int,
    max_run_duration: str,
    script: str,
) -> None:
    bucket = BUCKETS[region]
    config = {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": RUST_IMAGE,
                                "commands": ["/bin/bash", "-c", script],
                            }
                        }
                    ],
                    "computeResource": {
                        "cpuMilli": cpu_milli,
                        "memoryMib": memory_mib,
                    },
                    "maxRetryCount": 3,
                    "maxRunDuration": max_run_duration,
                    "volumes": [
                        {
                            "gcs": {"remotePath": bucket},
                            "mountPath": "/mnt/disks/gcs",
                        }
                    ],
                },
                "taskCount": str(task_count),
                "parallelism": str(parallelism),
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "policy": {
                        "machineType": machine_type,
                        "provisioningModel": "SPOT",
                    }
                }
            ],
            "network": {
                "networkInterfaces": [
                    {
                        "network": "global/networks/spot-batch",
                        "subnetwork": f"regions/{region}/subnetworks/spot-batch",
                    }
                ]
            },
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        cfg_path = f.name
    try:
        run_cmd(
            [
                "gcloud",
                "batch",
                "jobs",
                "submit",
                job_id,
                "--location",
                region,
                "--config",
                cfg_path,
            ]
        )
    finally:
        os.unlink(cfg_path)


def cmd_submit_build(args: argparse.Namespace) -> None:
    src_hash = cmd_stage(args)
    job_id = f"nc2000-build-{src_hash[:8]}-{int(time.time()) % 100000}"
    script = f"""set -euo pipefail
mkdir -p /tmp/work /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}
if [ -f /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/research_meta ] && [ -f /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/team_eval ]; then
  echo "Binaries already cached for {src_hash}"
  exit 0
fi
tar -xzf /mnt/disks/gcs/{GCS_PREFIX}/src-{src_hash}.tar.gz -C /tmp/work
cd /tmp/work
CARGO_TARGET_DIR=/tmp/target cargo build --release -p nc2000-bot --example research_meta --example team_eval
cp /tmp/target/release/examples/research_meta /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/research_meta.tmp
cp /tmp/target/release/examples/team_eval /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/team_eval.tmp
mv /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/research_meta.tmp /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/research_meta
mv /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/team_eval.tmp /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/team_eval
mkdir -p /mnt/disks/gcs/{GCS_PREFIX}/bin-latest
cp /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/research_meta /mnt/disks/gcs/{GCS_PREFIX}/bin-latest/research_meta
cp /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/team_eval /mnt/disks/gcs/{GCS_PREFIX}/bin-latest/team_eval
echo "Built and cached binaries for {src_hash}"
"""
    submit_batch_job(
        job_id=job_id,
        region=args.region,
        machine_type="c2d-highcpu-16",
        task_count=1,
        parallelism=1,
        cpu_milli=16000,
        memory_mib=30000,
        max_run_duration="1800s",
        script=script,
    )
    print(f"Submitted build job: {job_id} (src_hash={src_hash})")


def generate_round_lineages(round_num: int, base_seed: int, target_gauntlet: str) -> list[dict]:
    """Generate 32 diverse PSRO x Quality-Diversity lineages for round `round_num`."""
    lineages = []
    # Group A: 6 Unconstrained Best-Response lineages from random non-dominated teams
    for i in range(6):
        seed = base_seed + round_num * 1000 + i
        lineages.append({
            "name": f"r{round_num}-br-rand-{i:02d}",
            "mode": "random",
            "anchors": "",
            "max_staples": "",
            "seed_idx": "",
            "weaken": 0,
            "seed": seed,
            "gauntlet": target_gauntlet,
        })
    # Group B: 8 Staple-Capped Novelty Best-Response lineages (at most 1 or 2 meta staples)
    for i in range(8):
        seed = base_seed + round_num * 1000 + 100 + i
        max_s = 1 if i < 4 else 2
        lineages.append({
            "name": f"r{round_num}-novel-cap{max_s}-{i:02d}",
            "mode": "random",
            "anchors": "",
            "max_staples": str(max_s),
            "seed_idx": "",
            "weaken": 0,
            "seed": seed,
            "gauntlet": target_gauntlet,
        })
    # Group C: 18 Niche-Anchored Quality-Diversity lineages (each pins a distinct species in slot 0)
    for i in range(18):
        anchor = QD_ANCHORS[(i + (round_num - 1) * 6) % len(QD_ANCHORS)]
        seed = base_seed + round_num * 1000 + 200 + i
        lineages.append({
            "name": f"r{round_num}-qd-{anchor}-{i:02d}",
            "mode": "anchor",
            "anchors": anchor,
            "max_staples": "2",
            "seed_idx": "",
            "weaken": 0,
            "seed": seed,
            "gauntlet": target_gauntlet,
        })
    return lineages


def cmd_submit_evolve(args: argparse.Namespace) -> None:
    src_hash = cmd_stage(args)
    bucket = BUCKETS[args.region]
    lineages = generate_round_lineages(args.round, args.seed_base, args.gauntlet_file)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(lineages, f, indent=2)
        lin_path = f.name
    try:
        run_cmd(
            [
                "gcloud",
                "storage",
                "cp",
                lin_path,
                f"gs://{bucket}/{GCS_PREFIX}/round-{args.round}/lineages-manifest.json",
            ]
        )
    finally:
        os.unlink(lin_path)

    task_count = len(lineages)
    # Use 6 x c2d-highcpu-16 (96 vCPUs, 6 external IPs <= 8 limit) or n2-highcpu-16
    parallelism = min(task_count, args.parallelism)
    job_id = f"nc2000-psro-r{args.round}-{int(time.time()) % 100000}"

    script = f"""set -euo pipefail
mkdir -p /tmp/work /mnt/disks/gcs/{GCS_PREFIX}/round-{args.round}/checkpoints
tar -xzf /mnt/disks/gcs/{GCS_PREFIX}/src-{src_hash}.tar.gz -C /tmp/work
cd /tmp/work

BIN=/mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/research_meta
mkdir -p /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}
if [ -s "$BIN" ] && cp "$BIN" /tmp/research_meta 2>/dev/null; then
  chmod +x /tmp/research_meta
  BIN=/tmp/research_meta
else
  echo "Building research_meta locally on worker..."
  CARGO_TARGET_DIR=/tmp/target cargo build --release -p nc2000-bot --example research_meta
  if [ "${{BATCH_TASK_INDEX:-0}}" = "0" ]; then
    cp /tmp/target/release/examples/research_meta "${{BIN}}.tmp" && mv "${{BIN}}.tmp" "$BIN" || true
  fi
  BIN=/tmp/target/release/examples/research_meta
fi

export BIN_PATH="$BIN"

python3 - << 'PY'
import json, os, subprocess, sys

idx = int(os.environ.get("BATCH_TASK_INDEX", "0"))
manifest_path = "/mnt/disks/gcs/{GCS_PREFIX}/round-{args.round}/lineages-manifest.json"
lineages = json.load(open(manifest_path))
if idx >= len(lineages):
    sys.exit(0)
cfg = lineages[idx]
cmd = [
    os.environ["BIN_PATH"],
    "--lineage", cfg["name"],
    "--gauntlet-file", cfg["gauntlet"],
    "--agent", "{args.agent}",
    "--iters", "{args.iters}",
    "--lambda", "{args.lam}",
    "--eval-games", "{args.eval_games}",
    "--seed", str(cfg["seed"]),
    "--threads", "{args.threads_per_task}",
    "--out", "/mnt/disks/gcs/{GCS_PREFIX}/round-{args.round}/checkpoints",
    "--resume",
]
if cfg["anchors"]:
    cmd += ["--anchors", cfg["anchors"]]
else:
    cmd += ["--random-team"]
if cfg["max_staples"]:
    cmd += ["--max-staples", cfg["max_staples"]]

print("Running:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
PY
"""
    submit_batch_job(
        job_id=job_id,
        region=args.region,
        machine_type=args.machine_type,
        task_count=task_count,
        parallelism=parallelism,
        cpu_milli=args.threads_per_task * 1000,
        memory_mib=args.threads_per_task * 1800,
        max_run_duration="14400s",
        script=script,
    )
    print(f"Submitted PSRO evolution job: {job_id} ({task_count} lineages, parallelism={parallelism})")


def cmd_submit_phase0(args: argparse.Namespace) -> None:
    """Shard the 103-candidate x 78-panel cells across Spot VMs and run team_eval."""
    root = repo_root()
    src_hash = cmd_stage(args)
    bucket = BUCKETS[args.region]

    panel_data = json.loads((root / "data/team-selection-v2/panel.json").read_text())
    candidates: list[str] = panel_data["candidates"]
    panel_Ids: list[str] = [p["id"] for p in panel_data["members"]]
    panel_set = set(panel_Ids)

    # Construct unordered pairs so no cell is duplicated:
    # - every pair of panel members (p1 < p2)
    # - every non-panel candidate against every panel member
    cells: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for c in candidates:
        for p in panel_Ids:
            if c == p:
                continue
            a, b = (c, p) if (c in panel_set and p in panel_set and c > p) else (c, p)
            if a in panel_set and b in panel_set and a > b:
                a, b = b, a
            key = (min(a, b), max(a, b)) if (a in panel_set and b in panel_set) else (c, p)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            cells.append(f"{a} {b} 0 {args.seeds}")

    num_shards = args.shards
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for i, line in enumerate(cells):
        shards[i % num_shards].append(line)

    with tempfile.TemporaryDirectory() as tmp:
        for idx, sh in enumerate(shards):
            p = pathlib.Path(tmp) / f"cells-{idx:02d}.txt"
            p.write_text("\n".join(sh) + "\n")
        run_cmd(
            [
                "gcloud",
                "storage",
                "cp",
                f"{tmp}/cells-*.txt",
                f"gs://{bucket}/{GCS_PREFIX}/phase0-{args.tag}/shards/",
            ]
        )

    parallelism = min(num_shards, args.parallelism)
    job_id = f"nc2000-phase0-{args.tag}-{int(time.time()) % 100000}"
    script = f"""set -euo pipefail
mkdir -p /tmp/work /mnt/disks/gcs/{GCS_PREFIX}/phase0-{args.tag}/cells
tar -xzf /mnt/disks/gcs/{GCS_PREFIX}/src-{src_hash}.tar.gz -C /tmp/work
cd /tmp/work

BIN=/mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}/team_eval
mkdir -p /mnt/disks/gcs/{GCS_PREFIX}/bin-{src_hash}
if [ -s "$BIN" ] && cp "$BIN" /tmp/team_eval 2>/dev/null; then
  chmod +x /tmp/team_eval
  BIN=/tmp/team_eval
else
  echo "Building team_eval on worker..."
  CARGO_TARGET_DIR=/tmp/target cargo build --release -p nc2000-bot --example team_eval
  if [ "${{BATCH_TASK_INDEX:-0}}" = "0" ]; then
    cp /tmp/target/release/examples/team_eval "${{BIN}}.tmp" && mv "${{BIN}}.tmp" "$BIN" || true
  fi
  BIN=/tmp/target/release/examples/team_eval
fi

SHARD=$(printf "%02d" "${{BATCH_TASK_INDEX:-0}}")
CELLS_FILE="/mnt/disks/gcs/{GCS_PREFIX}/phase0-{args.tag}/shards/cells-${{SHARD}}.txt"
if [ ! -f "$CELLS_FILE" ]; then
  echo "No shard file $CELLS_FILE"
  exit 0
fi

"$BIN" \\
  --teams data/team-inventory-v1/eval/teams.json \\
  --cells "$CELLS_FILE" \\
  --agent "{args.agent}" \\
  --belief-pool data/belief-pool-v3/belief-pool.json \\
  --seeds {args.seeds} \\
  --seed-base {args.seed_base} \\
  --threads {args.threads_per_task} \\
  --out "/mnt/disks/gcs/{GCS_PREFIX}/phase0-{args.tag}/cells"
"""
    submit_batch_job(
        job_id=job_id,
        region=args.region,
        machine_type=args.machine_type,
        task_count=num_shards,
        parallelism=parallelism,
        cpu_milli=args.threads_per_task * 1000,
        memory_mib=args.threads_per_task * 1800,
        max_run_duration="14400s",
        script=script,
    )
    print(f"Submitted Phase 0 job: {job_id} ({len(cells)} cells across {num_shards} shards, parallelism={parallelism})")


def cmd_collect(args: argparse.Namespace) -> None:
    bucket = BUCKETS[args.region]
    dest = pathlib.Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    run_cmd(["gcloud", "storage", "rsync", "-r", f"gs://{bucket}/{GCS_PREFIX}/{args.remote_subpath}", str(dest)])


def cmd_status(args: argparse.Namespace) -> None:
    regions = list(BUCKETS.keys()) if getattr(args, "all_regions", True) else [args.region]
    for reg in regions:
        print(f"=== {reg} ===")
        run_cmd(
            [
                "gcloud",
                "batch",
                "jobs",
                "list",
                "--location",
                reg,
                "--format=table(name.basename(),status.state,status.statusEvents[-1].description,createTime)",
            ]
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", default=DEFAULT_REGION, choices=list(BUCKETS.keys()))
    sub = p.add_subparsers(dest="sub", required=True)

    sp_stage = sub.add_parser("stage", help="Stage source + data tarball to GCS")
    sp_stage.set_defaults(func=cmd_stage)

    sp_build = sub.add_parser("submit-build", help="Prebuild Linux x86_64 binaries on Cloud Batch")
    sp_build.set_defaults(func=cmd_submit_build)

    sp_ev = sub.add_parser("submit-evolve", help="Submit PSRO x QD 32-lineage array job")
    sp_ev.add_argument("--round", type=int, default=1)
    sp_ev.add_argument("--seed-base", type=int, default=20261028)
    sp_ev.add_argument("--gauntlet-file", default="data/meta-nash-v3/pool-artifact.json")
    sp_ev.add_argument("--agent", default="blind:1000:0.4")
    sp_ev.add_argument("--iters", type=int, default=30)
    sp_ev.add_argument("--lam", type=int, default=6)
    sp_ev.add_argument("--eval-games", type=int, default=8)
    sp_ev.add_argument("--machine-type", default="c2d-highcpu-16")
    sp_ev.add_argument("--threads-per-task", type=int, default=16)
    sp_ev.add_argument("--parallelism", type=int, default=6)
    sp_ev.set_defaults(func=cmd_submit_evolve)

    sp_p0 = sub.add_parser("submit-phase0", help="Submit Phase 0 103-candidate panel evaluation")
    sp_p0.add_argument("--tag", default="stage-a-3000")
    sp_p0.add_argument("--agent", default="blind:3000:0.4")
    sp_p0.add_argument("--seeds", type=int, default=2)
    sp_p0.add_argument("--seed-base", type=int, default=20261111)
    sp_p0.add_argument("--shards", type=int, default=12)
    sp_p0.add_argument("--machine-type", default="c2d-highcpu-16")
    sp_p0.add_argument("--threads-per-task", type=int, default=16)
    sp_p0.add_argument("--parallelism", type=int, default=6)
    sp_p0.set_defaults(func=cmd_submit_phase0)

    sp_col = sub.add_parser("collect", help="Sync results from GCS to local directory")
    sp_col.add_argument("--remote-subpath", required=True)
    sp_col.add_argument("--out", required=True)
    sp_col.set_defaults(func=cmd_collect)

    sp_st = sub.add_parser("status", help="List Cloud Batch jobs in region")
    sp_st.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
