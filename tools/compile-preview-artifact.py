#!/usr/bin/env python3
"""Assemble complete, versioned live-policy previews; never publish partial data.
Usage: python3 tools/compile-preview-artifact.py JOBS OUTPUT SOURCE_DIR...
Run compile_preview --prepare and compile_preview before this build step.
"""
import hashlib
import json
from pathlib import Path
import sys
import subprocess

jobs_path, output, *folders = map(Path, sys.argv[1:])
jobs = json.loads(jobs_path.read_text())
assert json.loads(Path('data/search-profiles.json').read_text())['blind']['c'] == 0.4, 'This bake runner is fixed at c=0.4; update and recompute for a new coefficient'
policies, build = {}, None
for job in jobs:
    candidates = [folder / (job['id'] + '.json') for folder in folders]
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise SystemExit('Missing completed preview: ' + job['id'])
    rows = [json.loads(p.read_text()) for p in found]
    for row in rows:
        assert row['job'] == job, job['id']
        assert build is None or build == row['build'], 'Mixed engine builds'
        build = row['build']
    semantic = [{k:v for k,v in row.items() if k != 'samples'} | {'samples':[{k:v for k,v in s.items() if k != 'increment_ns'} for s in row['samples']]} for row in rows]
    assert all(row == semantic[0] for row in semantic), 'Conflicting duplicate: ' + job['id']
    row = rows[0]
    assert not row['fallback'], 'Known catalog public roster absent from belief'
    samples = row['samples']
    assert [(s['seed'], s['budget']) for s in samples] == [(seed, b) for seed in job['seeds'] for b in job['budgets']]
    assert row['key'] not in policies, 'Duplicate public-information key'
    policies[row['key']] = [s['choice'] for s in samples if s['budget'] == job['budgets'][-1]]

# Ask a freshly compiled executable, so old rows cannot be labelled with
# hashes of changed sources merely because the assembler ran afterwards.
current_build = subprocess.check_output(['cargo', 'run', '--quiet', '--release', '-p', 'nc2000-bot', '--example', 'compile_preview', '--', '--identity'], text=True).strip()
assert build == current_build, 'Stale preview computation; recompute with current engine sources'

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = ['data/search-profiles.json', 'tools/compile-preview-artifact.py', 'data/learnsets-gen2.json', 'data/community-rentals-v0/teams.json', 'Cargo.toml', 'Cargo.lock', 'crates/bot/Cargo.toml', 'crates/bot/build.rs', 'crates/engine/Cargo.toml', 'crates/engine/build.rs', 'crates/conformance/Cargo.toml', 'crates/bot/examples/compile_preview.rs', 'crates/bot/examples/anchor_gate.rs', 'crates/bot/examples/endgame_exactness_corpus.rs', 'data/gen2stadium2.json', 'data/belief-pool-v3/belief-pool.json', 'data/team-pool-v2/team-pool.json']
for folder in ['crates/bot/src', 'crates/engine/src', 'crates/conformance/src']:
    inputs.extend(str(p) for p in sorted(Path(folder).rglob('*.rs')))
artifact = dict(format='nc2000-compiled-preview-v1', build=build, c=0.4,
    priorSha256=sha('data/belief-pool-v3/belief-pool.json'), dexSha256=sha('data/gen2stadium2.json'),
    matchups=len(jobs), iterations=jobs[0]['budgets'][-1], sampleSeeds=jobs[0]['seeds'], jobsSha256=sha(jobs_path),
    method='Precomputed blind MCTS argmax at each fixed search seed, over all legal ordered selections. Approximation, not an exact equilibrium certificate.',
    inputs={p:sha(p) for p in sorted(inputs)}, policies=policies)
output.parent.mkdir(parents=True, exist_ok=True)
tmp = output.with_suffix('.tmp')
tmp.write_text(json.dumps(artifact, separators=(',', ':'))+'\n')
tmp.replace(output)
print(f'{len(policies)} complete matchups -> {output}')
