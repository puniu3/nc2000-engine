import argparse
import concurrent.futures
import json
import datetime
import sys
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=Path('tmp/loadout-draw') / datetime.datetime.now().strftime('rerun-%Y%m%d-%H%M%S'))
parser.add_argument('--workers',type=int,default=10)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
if any(args.out.glob('*.jsonl')):
    parser.error('output already contains results; choose a new --out directory')
raw=args.out/'jobs';raw.mkdir(exist_ok=True)
protocol=json.loads(Path('findings/operational-loadout-draw-v1/protocol.json').read_text())
binary='target/release/examples/operational_combo_gate'
def run(cell, job):
    spec=protocol['cells'][cell]
    path=raw/f'{cell}-{job}.jsonl'
    cmd=[binary,'--triad','--mode','stress' if cell=='same' else cell,'--games','1','--start',str(job),
         '--seed',str(spec['seed']),'--iters',str(spec.get('iterations',27000)),'--threads','1','--out',str(path)]
    if cell=='same':cmd.append('--same')
    with (raw/f'{cell}-{job}.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=log,check=True)
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['job']==job
    return cell,rows[0]
rows={k:[] for k in protocol['cells']}
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures=[pool.submit(run,cell,job) for cell in ['same','natural','adaptive','stress','onpool']
             for job in range(protocol['cells'][cell]['games'])]
    for future in concurrent.futures.as_completed(futures):
        cell,row=future.result();rows[cell].append(row)
        with (args.out/f'{cell}.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
        print(f'{cell} {len(rows[cell])}/{protocol["cells"][cell]["games"]} job={row["job"]}',flush=True)
for cell,rs in rows.items():
    assert len(rs)==protocol['cells'][cell]['games']
    (args.out/f'{cell}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sorted(rs,key=lambda r:r['job'])))

subprocess.run([sys.executable, 'tools/analyze-operational-loadout-draw.py', '--dir', str(args.out)],check=True)
