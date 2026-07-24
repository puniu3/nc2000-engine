#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: run-m17d-shards.sh OUT_DIR [JOBS_PER_SHARD] [PROFILE] [GAUNTLET_ARGS...]}
shard_jobs=${2:-64}
profile=${3:-full}
shift "$(( $# < 3 ? $# : 3 ))"
extra_args=("$@")

case "$shard_jobs" in
	''|*[!0-9]*) echo "JOBS_PER_SHARD must be a positive integer" >&2; exit 2 ;;
esac
if (( shard_jobs == 0 )); then
	echo "JOBS_PER_SHARD must be positive" >&2
	exit 2
fi
case "$profile" in
	smoke|full) ;;
	*) echo "PROFILE must be smoke or full" >&2; exit 2 ;;
esac
fixed_args=()
if [[ "$profile" == full ]]; then
	fixed_args=(--max-turns 1000)
fi
for arg in "${extra_args[@]}"; do
	case "$arg" in
		--profile|--shard-size|--manifest-out|--job-start|--job-end|--out)
			echo "wrapper owns $arg" >&2
			exit 2
			;;
		--max-turns)
			if [[ "$profile" == full ]]; then
				echo "full wrapper fixes --max-turns at 1000" >&2
				exit 2
			fi
			;;
	esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
merger="$script_dir/merge-m17d-shards.py"
repo_root=${NC2000_REPO_ROOT:-$(cd -- "$script_dir/.." && pwd)}
export NC2000_REPO_ROOT=$repo_root
if [[ -n ${NC2000_M17D_BIN:-} ]]; then
	gauntlet=$NC2000_M17D_BIN
elif [[ -x ./offpool_fallback_gauntlet ]]; then
	gauntlet=./offpool_fallback_gauntlet
else
	cargo build \
		--manifest-path "$repo_root/Cargo.toml" \
		--release \
		-p nc2000-bot \
		--example offpool_fallback_gauntlet
	gauntlet="$repo_root/target/release/examples/offpool_fallback_gauntlet"
fi
if [[ ! -x "$gauntlet" ]]; then
	echo "M17d gauntlet binary is not executable: $gauntlet" >&2
	exit 2
fi

mkdir -p "$output_dir"
manifest="$output_dir/manifest.json"
candidate=$(mktemp "$output_dir/.manifest.candidate.XXXXXX")
current_tmp=
cleanup() {
	rm -f -- "$candidate"
	if [[ -n "$current_tmp" ]]; then
		rm -f -- "$current_tmp"
	fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"$gauntlet" \
	--profile "$profile" \
	"${fixed_args[@]}" \
	"${extra_args[@]}" \
	--shard-size "$shard_jobs" \
	--manifest-out "$candidate"

if [[ -e "$manifest" ]]; then
	if ! cmp -s -- "$candidate" "$manifest"; then
		echo "refusing to resume: $manifest does not match the current build/data/team/config/seed workload" >&2
		exit 3
	fi
else
	mv -n -- "$candidate" "$manifest"
	if [[ -e "$candidate" ]]; then
		echo "refusing concurrent manifest creation: $manifest" >&2
		exit 3
	fi
	candidate=
fi

while IFS=$'\t' read -r start end filename; do
	shard="$output_dir/$filename"
	if [[ -e "$shard" ]]; then
		if python3 "$merger" \
			--manifest "$manifest" \
			--check-shard "$shard" \
			--start "$start" >/dev/null
		then
			echo "skipping completed shard: $shard" >&2
			continue
		fi
		echo "refusing to overwrite invalid or mismatched shard: $shard" >&2
		exit 3
	fi
	current_tmp=$(mktemp "$output_dir/.$filename.tmp.XXXXXX")
	"$gauntlet" \
		--profile "$profile" \
		"${fixed_args[@]}" \
		"${extra_args[@]}" \
		--job-start "$start" \
		--job-end "$end" \
		--out "$current_tmp"
	python3 "$merger" \
		--manifest "$manifest" \
		--check-shard "$current_tmp" \
		--start "$start" >/dev/null
	mv -n -- "$current_tmp" "$shard"
	if [[ -e "$current_tmp" ]]; then
		echo "refusing concurrent shard creation: $shard" >&2
		exit 3
	fi
	current_tmp=
	echo "completed shard: $shard" >&2
done < <(python3 "$merger" --manifest "$manifest" --list)

python3 "$merger" \
	--manifest "$manifest" \
	--out "$output_dir/merged.jsonl"
