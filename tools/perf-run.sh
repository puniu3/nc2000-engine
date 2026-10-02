#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
if [ "${1:-}" = "--help" ]; then
    echo 'usage: bash tools/perf-run.sh [output-directory]'
    exit 0
fi
out="${1:-$root/tmp/perf-current}"
case "$out" in /*) ;; *) out="$root/$out" ;; esac
mkdir -p "$out/source" "$out/report"
base=f37bf28
python3 - "$out" "$base" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1])/'provenance.json'
current=hashlib.sha256(subprocess.check_output(['git','diff',sys.argv[2],'--','crates/engine','crates/bot/src'])).hexdigest()
if p.exists() and json.loads(p.read_text())['candidate_diff_sha256'] != current:
    raise SystemExit('Source changed; choose a new output directory: bash tools/perf-run.sh tmp/perf-next')
PY
if [ ! -f "$out/source/Cargo.toml" ]; then
    git archive "$base" | tar -x -C "$out/source"
fi
cp crates/bot/examples/perf_probe.rs "$out/source/crates/bot/examples/perf_probe.rs"
cp -R crates/bot/examples/combo_common "$out/source/crates/bot/examples/"
export NC2000_REPO_ROOT="$root"
cargo build --release --manifest-path "$out/source/Cargo.toml" -p nc2000-bot --example perf_probe
cp "$out/source/target/release/examples/perf_probe" "$out/base"
cargo build --release -p nc2000-bot --example perf_probe
cp target/release/examples/perf_probe "$out/candidate"
python3 - "$out" "$base" <<'PY'
import hashlib,json,pathlib,subprocess,sys
out=pathlib.Path(sys.argv[1]); paths=['data/gen2stadium2.json','data/belief-pool-v3/belief-pool.json','data/search-profiles.json','crates/bot/examples/perf_probe.rs']
record={'baseline_commit':subprocess.check_output(['git','rev-parse',sys.argv[2]],text=True).strip(),
        'candidate_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff',sys.argv[2],'--','crates/engine','crates/bot/src'])).hexdigest(),
        'rustc':subprocess.check_output(['rustc','-Vv'],text=True),
        'files':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}}
(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
PY
python3 tools/perf-check.py "$out/base" "$out/candidate" --out "$out/blind"
python3 tools/perf-check.py "$out/base" "$out/candidate" --mode skuct --out "$out/skuct"
for mode in clone hash step determinize rollout; do
    python3 tools/perf-check.py "$out/base" "$out/candidate" --mode "$mode" --iterations 3000 --out "$out/$mode"
done
swiftc -O tools/metal-launch-probe.swift -o "$out/metal-probe"
python3 tools/perf-gpu.py "$out"
for seconds in 10 60; do
    if [ ! -s "$out/budget-$seconds.json" ]; then
        /usr/bin/time -l "$out/candidate" budget "$seconds" 1 > "$out/budget-$seconds.json" 2> "$out/budget-$seconds.txt"
    fi
done
python3 - "$out" <<'PY'
import json,os,pathlib,subprocess,sys
out=pathlib.Path(sys.argv[1]); n=max(1,(os.cpu_count() or 1)-1)
for threads in sorted({1,min(4,n),min(8,n),n}):
    p=out/f'threads-{threads}.json'
    if not p.exists():
        result=subprocess.run(['/usr/bin/time','-l',str(out/'candidate'),'blind','27000',str(threads)],capture_output=True,text=True,check=True)
        p.write_text(result.stdout)
        (out/f'threads-{threads}.txt').write_text(result.stderr)
PY
if [ ! -s "$out/profile.json" ]; then
    python3 tools/perf-sample.py "$out/candidate" "$out"
fi
python3 tools/perf-report.py "$out"
