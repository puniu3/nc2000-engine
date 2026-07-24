#!/usr/bin/env bash
set -euo pipefail

higher_iters=${1:?usage: run-m17b-stage.sh HIGHER LOWER [THREADS] [GAMES] [discovery|confirm]}
lower_iters=${2:?usage: run-m17b-stage.sh HIGHER LOWER [THREADS] [GAMES] [discovery|confirm]}
threads=${3:-$(nproc)}
games=${4:-200}
stage=${5:-discovery}
output_dir=${CX_OUT:-tmp/m17b-discovery}

mkdir -p "$output_dir"

if [[ -x ./arena ]]; then
	arena=./arena
else
	cargo build --release -p nc2000-bot --example arena
	arena=target/release/examples/arena
fi

case "$stage" in
	discovery) seeds=(1700000001 1700000002 1700000003 1700000004) ;;
	confirm) seeds=(2700000001 2700000002 2700000003 2700000004) ;;
	*) echo "stage must be discovery or confirm" >&2; exit 2 ;;
esac

for seed in "${seeds[@]}"; do
	"$arena" \
		"blind:${higher_iters}:1:16" "blind:${lower_iters}:1:16" \
		--games "$games" \
		--seed "$seed" \
		--threads "$threads" \
		--max-turns 500 \
		--pool meta \
		--jsonl "$output_dir/m17b-${higher_iters}v${lower_iters}-${stage}-${seed}.jsonl"
done
