#!/usr/bin/env python3
"""Observe the existing frozen runs and assemble results when both finish.

Never starts/restarts experiments and never treats a saved PID as liveness.
Completion here means ready for review, not that the user goal is achieved.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('findings/mixed-policy-strength-v1')
state = json.loads((root / 'RUNNING.json').read_text())

def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def inspect():
    runs = []
    for run in state['runs']:
        directory = Path(run['directory'])
        plan = json.loads((directory / 'plan.json').read_text())
        wanted = {j['id'] for j in plan['jobs']}
        actual = {p.stem for p in (directory / 'rows').glob('*.json')}
        assert actual <= wanted, f'unexpected result ids: {actual - wanted}'
        result = subprocess.run(['ps', '-p', str(run['pid']), '-o', 'command='], capture_output=True, text=True)
        command = result.stdout.strip()
        live = result.returncode == 0 and state['executable'] in command and run['directory'] in command
        runs.append({'kind': run['kind'], 'pid': run['pid'], 'live': live,
            'observed_command': command, 'completed': len(actual), 'planned': len(wanted),
            'all_rows_present': actual == wanted})
    return {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs': runs}

if '--once' in sys.argv:
    print(json.dumps(inspect(), indent=2))
    raise SystemExit()

lock = root / 'watch.lock'
with lock.open('x') as f:
    f.write(str(os.getpid()) + '\n')
try:
    previous = None
    while True:
        evidence = inspect()
        atomic_json(root / 'process-observation.json', evidence)
        counts = tuple(r['completed'] for r in evidence['runs'])
        if counts != previous:
            subprocess.run([sys.executable, 'tools/mixed-policy-strength-report.py'], check=True)
            print(json.dumps(evidence), flush=True)
            previous = counts
        if all(r['all_rows_present'] for r in evidence['runs']):
            for run in state['runs']:
                with (Path(run['directory']) / 'analysis.log').open('w') as log:
                    subprocess.run([sys.executable, 'tools/analyze-mixed-policy.py', run['directory']],
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, 'tools/mixed-policy-strength-report.py'], check=True)
            summary = root / 'main/summary.json'
            atomic_json(root / 'ready-for-review.json', {
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'status': 'ready_for_review',
                'summary_sha256': hashlib.sha256(summary.read_bytes()).hexdigest(),
                'main_games': len(json.loads((root / 'main/plan.json').read_text())['jobs']),
                'goal_completed': False,
            })
            print('All planned results audited and rendered; manual interpretation/completion audit remains.', flush=True)
            break
        if any(not r['all_rows_present'] and not r['live'] for r in evidence['runs']):
            atomic_json(root / 'run-interruption.json', evidence)
            raise RuntimeError('A specific experiment process is absent with rows missing; no automatic restart.')
        time.sleep(30)
finally:
    lock.unlink(missing_ok=True)
