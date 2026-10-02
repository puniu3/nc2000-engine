#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('baseline', type=pathlib.Path)
p.add_argument('candidate', type=pathlib.Path)
p.add_argument('--out', type=pathlib.Path, required=True)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
assert a.repeats > 0
root = pathlib.Path(__file__).resolve().parent.parent
modules = {k: getattr(a, k).resolve() for k in ['baseline', 'candidate']}
manifest = {'repeats': a.repeats, 'sha256': {
    f'{k}/{name}': hashlib.sha256((v / name).read_bytes()).hexdigest()
    for k, v in modules.items()
    for name in ['nc2000_wasm_bg.wasm', 'nc2000_wasm.js']}}
for name in ['tools/perf-wasm.cjs', 'data/belief-pool-v3/belief-pool.json']:
    manifest['sha256'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
a.out.mkdir(parents=True, exist_ok=True)
mp = a.out / 'manifest.json'
if mp.exists():
    assert json.loads(mp.read_text()) == manifest
else:
    mp.write_text(json.dumps(manifest, indent=2) + '\n')
runs = {k: [] for k in modules}
reference = None
for repeat in range(a.repeats):
    for arm in (list(modules) if repeat % 2 == 0 else list(reversed(modules))):
        output = a.out / f'{arm}-{repeat}.json'
        if output.exists():
            run = json.loads(output.read_text())
        else:
            result = subprocess.run(['node', str(root / 'tools/perf-wasm.cjs'), str(modules[arm])],
                                    text=True, capture_output=True, check=True)
            run = json.loads(result.stdout)
            output.write_text(json.dumps(run, indent=2) + '\n')
        signature = [{k: v for k, v in row.items() if k != 'ms'} for row in run['rows']]
        if reference is None:
            reference = signature
        assert signature == reference, f'exploration diverged: {output}'
        runs[arm].append(run['total_ms'])
        print(arm, repeat, run['total_ms'], flush=True)
medians = {k: statistics.median(v) for k, v in runs.items()}
result = {'median_ms': medians, 'time_reduction': 1 - medians['candidate'] / medians['baseline'],
          'exact_match': True, 'positions': len(reference), 'runs_ms': runs,
          'runtime': subprocess.check_output(['node', '--version'], text=True).strip()}
(a.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
