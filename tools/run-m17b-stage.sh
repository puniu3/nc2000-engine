#!/usr/bin/env bash
set -euo pipefail

higher_iters=${1:?usage: run-m17b-stage.sh HIGHER LOWER [THREADS] [GAMES] [discovery|confirm]}
lower_iters=${2:?usage: run-m17b-stage.sh HIGHER LOWER [THREADS] [GAMES] [discovery|confirm]}
threads=${3:-$(nproc)}
games=${4:-200}
stage=${5:-discovery}
output_dir=${CX_OUT:-tmp/m17b-discovery}

mkdir -p "$output_dir"

current_tmp=
cleanup() {
	if [[ -n "$current_tmp" ]]; then
		rm -f -- "$current_tmp"
	fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -x ./arena ]]; then
	arena=./arena
else
	cargo build --release -p nc2000-bot --example arena
	arena=target/release/examples/arena
fi
export NC2000_REPO_ROOT=${NC2000_REPO_ROOT:-$PWD}

case "$stage" in
	discovery) seeds=(1700000001 1700000002 1700000003 1700000004) ;;
	confirm) seeds=(2700000001 2700000002 2700000003 2700000004) ;;
	*) echo "stage must be discovery or confirm" >&2; exit 2 ;;
esac

validate_shard() {
	local path=$1
	local seed=$2
	"$arena" --validate-jsonl "$path" &&
		grep -Fq "\"agent_a\":\"blind:${higher_iters}:1:16\"" "$path" &&
		grep -Fq "\"agent_b\":\"blind:${lower_iters}:1:16\"" "$path" &&
		grep -Fq "\"requested_games\":${games}" "$path" &&
		grep -Fq "\"base_seed\":${seed}" "$path" &&
		grep -Fq "\"threads\":${threads}" "$path" &&
		grep -Fq '"max_turns":500' "$path" &&
		grep -Fq '"pool":"meta"' "$path" &&
		grep -Fq '"fingerprints":{"build":"fnv1a64:' "$path"
}

for seed in "${seeds[@]}"; do
	output="$output_dir/m17b-${higher_iters}v${lower_iters}-${stage}-${seed}.jsonl"
	if [[ -e "$output" ]]; then
		if [[ -s "$output" ]] && validate_shard "$output" "$seed"; then
			echo "skipping completed shard: $output" >&2
			continue
		fi
		echo "refusing to overwrite invalid or mismatched shard: $output" >&2
		exit 3
	fi

	current_tmp=$(mktemp "$output_dir/.m17b-${higher_iters}v${lower_iters}-${stage}-${seed}.jsonl.tmp.XXXXXX")
	"$arena" \
		"blind:${higher_iters}:1:16" "blind:${lower_iters}:1:16" \
		--games "$games" \
		--seed "$seed" \
		--threads "$threads" \
		--max-turns 500 \
		--pool meta \
		--jsonl "$current_tmp"

	if [[ ! -s "$current_tmp" ]] || ! validate_shard "$current_tmp" "$seed"; then
		echo "arena produced an invalid or mismatched shard for seed $seed" >&2
		exit 4
	fi
	mv -n -- "$current_tmp" "$output"
	if [[ -e "$current_tmp" ]]; then
		echo "refusing to overwrite shard created concurrently: $output" >&2
		exit 3
	fi
	current_tmp=
done
