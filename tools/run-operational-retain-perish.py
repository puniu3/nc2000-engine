import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=Path('tmp/retain-perish') / datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
parser.add_argument('--workers', type=int, default=10)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
raw = args.out / 'jobs'
raw.mkdir()
source = Path('findings/operational-retain-perish-v1/protocol.json')
protocol = json.loads(source.read_text())
for name, sha in protocol['inputs'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
shutil.copy2(source, args.out / 'protocol.json')
binary = args.out.resolve() / 'gate'
shutil.copy2('target/release/examples/operational_combo_gate', binary)
(args.out / 'execution.json').write_text(json.dumps({
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'workers': args.workers,
}, indent=2) + '\n')
env = dict(os.environ, NC2000_REPO_ROOT=str(Path.cwd()))

def run(cell, job, arm):
    spec = protocol['cells'][cell]
    path = raw / f'{cell}-{job}-{arm}.jsonl'
    command = [str(binary), '--retain-perish', '--mode', 'stress' if cell == 'same' else cell,
               '--arm', arm, '--games', '1', '--start', str(job), '--seed', str(spec['seed']),
               '--iters', str(spec.get('iterations', protocol['iterations'])), '--threads', '1', '--out', str(path)]
    if cell == 'same':
        command.append('--same')
    with path.with_suffix('.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['job'] == job
    return cell, job, arm, rows[0]

completed = {}
counts = {cell: 0 for cell in protocol['cells']}
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = [pool.submit(run, cell, job, arm)
               for cell in ['same', 'natural', 'adaptive', 'stress', 'onpool']
               for job in range(protocol['cells'][cell]['games']) for arm in ['a', 'b']]
    for future in concurrent.futures.as_completed(futures):
        cell, job, arm, row = future.result()
        completed.setdefault((cell, job), {})[arm] = row
        counts[cell] += 1
        print(f'{cell} {counts[cell]}/{2 * protocol["cells"][cell]["games"]} job={job} arm={arm}', flush=True)
        pair = completed[cell, job]
        if len(pair) == 2:
            meta = lambda r: {k: v for k, v in r.items() if k not in ['a', 'b']}
            assert meta(pair['a']) == meta(pair['b'])
            merged = dict(meta(row), a=pair['a']['a'], b=pair['b']['b'])
            with (args.out / f'{cell}.jsonl').open('a') as out:
                out.write(json.dumps(merged) + '\n')
for cell in protocol['cells']:
    path = args.out / f'{cell}.jsonl'
    rows = sorted((json.loads(line) for line in path.read_text().splitlines()), key=lambda r: r['job'])
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
subprocess.run([sys.executable, 'tools/analyze-operational-retain-perish.py', '--dir', str(args.out)], check=True)
