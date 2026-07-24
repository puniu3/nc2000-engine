#!/usr/bin/env bash
set -euo pipefail

higher_iters=${1:?usage: run-m17b-stage.sh HIGHER LOWER [THREADS] [GAMES] [discovery|confirm] [blind|open]}
lower_iters=${2:?usage: run-m17b-stage.sh HIGHER LOWER [THREADS] [GAMES] [discovery|confirm] [blind|open]}
threads=${3:-$(nproc)}
games=${4:-200}
stage=${5:-discovery}
agent_family=${6:-blind}
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

case "$stage" in
	discovery) seeds=(1700000001 1700000002 1700000003 1700000004) ;;
	confirm) seeds=(2700000001 2700000002 2700000003 2700000004) ;;
	*) echo "stage must be discovery or confirm" >&2; exit 2 ;;
esac

case "$agent_family" in
	blind) file_prefix=m17b ;;
	open) file_prefix=m17b-open ;;
	*) echo "agent family must be blind or open" >&2; exit 2 ;;
esac
agent_a="${agent_family}:${higher_iters}:1:16"
agent_b="${agent_family}:${lower_iters}:1:16"

if [[ -x ./arena ]]; then
	arena=./arena
else
	cargo build --release -p nc2000-bot --example arena
	arena=target/release/examples/arena
fi
export NC2000_REPO_ROOT=${NC2000_REPO_ROOT:-$PWD}

validate_shard() {
	local path=$1
	local seed=$2
	"$arena" --validate-jsonl "$path" &&
		grep -Fq "\"agent_a\":\"${agent_a}\"" "$path" &&
		grep -Fq "\"agent_b\":\"${agent_b}\"" "$path" &&
		grep -Fq "\"requested_games\":${games}" "$path" &&
		grep -Fq "\"base_seed\":${seed}" "$path" &&
		grep -Fq "\"threads\":${threads}" "$path" &&
		grep -Fq '"max_turns":500' "$path" &&
		grep -Fq '"pool":"meta"' "$path" &&
		grep -Fq '"fingerprints":{"build":"fnv1a64:' "$path"
}

for seed in "${seeds[@]}"; do
	output="$output_dir/${file_prefix}-${higher_iters}v${lower_iters}-${stage}-${seed}.jsonl"
	if [[ -e "$output" ]]; then
		if [[ -s "$output" ]] && validate_shard "$output" "$seed"; then
			echo "skipping completed shard: $output" >&2
			continue
		fi
		echo "refusing to overwrite invalid or mismatched shard: $output" >&2
		exit 3
	fi

	current_tmp=$(mktemp "$output_dir/.${file_prefix}-${higher_iters}v${lower_iters}-${stage}-${seed}.jsonl.tmp.XXXXXX")
	"$arena" \
		"$agent_a" "$agent_b" \
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
