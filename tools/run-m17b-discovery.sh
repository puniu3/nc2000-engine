#!/usr/bin/env bash
set -euo pipefail

higher_iters=${1:?usage: run-m17b-discovery.sh HIGHER LOWER [THREADS] [GAMES]}
lower_iters=${2:?usage: run-m17b-discovery.sh HIGHER LOWER [THREADS] [GAMES]}
threads=${3:-$(nproc)}
games=${4:-200}
output_dir=${CX_OUT:-tmp/m17b-discovery}

mkdir -p "$output_dir"

if [[ -x ./arena ]]; then
	arena=./arena
else
	cargo build --release -p nc2000-bot --example arena
	arena=target/release/examples/arena
fi

for seed in 1700000001 1700000002 1700000003 1700000004; do
	"$arena" \
		"blind:${higher_iters}:1:16" "blind:${lower_iters}:1:16" \
		--games "$games" \
		--seed "$seed" \
		--threads "$threads" \
		--max-turns 500 \
		--pool meta \
		--jsonl "$output_dir/m17b-${higher_iters}v${lower_iters}-discovery-${seed}.jsonl"
done
