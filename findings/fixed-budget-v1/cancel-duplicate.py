"""Stop only a shard whose entire fixed assignment is already preserved."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['strength','continuation']);parser.add_argument('name');args=parser.parse_args()
run=p/'cloud'/f'{args.name}-run';jobs=json.loads((p/f'{args.name}-jobs.json').read_text())
if args.mode=='strength':paths=[p/'strength-merged'/j for j in jobs]
else:paths=[p/'continuation-merged'/f'r{j//32:02}-s{(j//16)%2}-t{j%16:02}.json' for j in jobs]
assert jobs and all(f.exists() for f in paths),'Unique work is still missing'
for f in paths:json.loads(f.read_text())
if (run/'receipt.json').exists():
 assert json.loads((run/'receipt.json').read_text())['cleanup']=='verified';print('Already completed and cleaned');raise SystemExit(0)
if (run/'cancellation-audit.json').exists():print('Already cancelled and cleaned');raise SystemExit(0)
output=f'--output findings/fixed-budget-v1/cloud/{args.name}-run '
processes=[]
for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
 pid,_,cmd=line.strip().partition(' ')
 if 'spot_batch.py exec ' in cmd and output in cmd+' ':processes.append(int(pid))
assert len(processes)==1,processes
reason=dict(mode=args.mode,reason='Every assigned condition is complete in the canonical merged study; stop duplicate computation only.',jobs=jobs)
(p/'cloud'/f'{args.name}-cancel-reason.json').write_text(json.dumps(reason,indent=2)+'\n')
os.kill(processes[0],signal.SIGTERM)
for _ in range(40):
 try:os.kill(processes[0],0)
 except ProcessLookupError:break
 time.sleep(.25)
with (run/'explicit-cleanup.log').open('w') as log:
 subprocess.run(['spot-batch','cleanup','--run-dir',str(run/'execution')],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['python3',str(p/'record-cancellation.py'),args.name],check=True)
