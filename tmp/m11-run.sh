#!/usr/bin/env bash
# M11 full research run driver (launched 2026-07-18, in parallel with the
# full-pool re-bake — bake keeps ~11 threads, research gets 5, both nice'd).
#
# 12 lineages, strictly sequential (one research process at a time so only
# one competes with the bake): the 8 T1 pool seeds, then 4 random-team
# lineages per the README M11a launch plan. Every invocation passes
# --resume, so rerunning this script after a crash/reboot skips completed
# lineages and continues the interrupted one from its checkpoint.
#
#   nohup bash tmp/m11-run.sh >> tmp/m11-full.log 2>&1 &
#
# Writes only to data/research-v0/ (never data/preview-tables-v0/).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/examples/research_meta"
OUT="$ROOT/data/research-v0"
THREADS=5
NICENESS=10
mkdir -p "$OUT"

run() {
  echo "=== $(date -Is) start: $* ==="
  nice -n "$NICENESS" "$BIN" \
    --budget-profile full --threads "$THREADS" --out "$OUT" --resume "$@"
  echo "=== $(date -Is) exit $?: $* ==="
}

for i in 0 1 2 3 4 5 6 7; do
  run --seed-team "$i"
done
for k in 0 1 2 3; do
  run --random-team --seed "10$k" --lineage "rand-$k"
done
echo "=== $(date -Is) ALL 12 LINEAGES COMPLETE ==="
