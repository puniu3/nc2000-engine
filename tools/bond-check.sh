#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=tmp/destiny-bond-check
mkdir -p "$out"
cargo test --release -p nc2000-bot --lib --test destiny_bond_wait --test perish_deadline --test mask_sleep_talk --test blind --test import
cargo build --release -p nc2000-bot --example destiny_bond_gate
target/release/examples/destiny_bond_gate --fast --blind --proof --seed 71001 --seeds 8 > "$out/blind.jsonl"
target/release/examples/destiny_bond_gate --fast --seed 71001 --seeds 8 > "$out/pinned.jsonl"
if [[ ${1:-} == --corpus ]]; then
    target/release/examples/destiny_bond_gate --corpus > "$out/corpus.jsonl"
fi
python3 - "$out" <<'PY'
import collections
import json
import pathlib
import sys

for path in sorted(pathlib.Path(sys.argv[1]).glob('*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = [row for row in rows if row['type'] == 'paired_search']
    print(path.name, dict(collections.Counter((row['case'], row['before'], row['after']) for row in pairs)))
    for row in rows:
        if row['type'] == 'proof':
            print(row['case'], row['result']['actions'])
        elif row['type'] == 'corpus_summary':
            print(row)
PY
