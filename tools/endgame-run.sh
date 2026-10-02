#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--help" ]; then
    echo 'usage: bash tools/endgame-run.sh [new-output-directory]'
    exit 0
fi
out="${1:-tmp/selfplay-endgames-repro}"
mkdir -p "$out"
if [ -e "$out/strong.jsonl" ]; then
    echo 'Choose a new output directory; the existing corpus will not be overwritten.' >&2
    exit 1
fi
cargo build --release -p nc2000-bot --example selfplay_endgames
target/release/examples/selfplay_endgames generate "$out/strong.jsonl" 16 27000 20261004
target/release/examples/selfplay_endgames measure "$out/strong.jsonl" 1000000 4096 27000 --milliseconds 10000 --nodes 120000 --reuse > "$out/strong-measured.jsonl"
