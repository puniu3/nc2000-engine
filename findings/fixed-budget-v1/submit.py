"""Submit one bounded shard using the installed Cloud Batch interface."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parent.parent;cloud=p/'cloud'
a=argparse.ArgumentParser();a.add_argument('mode',choices=['strength','bake','continuation']);a.add_argument('name');a.add_argument('region');a.add_argument('jobs',type=Path);a.add_argument('--resume',type=Path);a.add_argument('--minutes',type=int,default=60);a.add_argument('--cpus',type=int,choices=[2,4,8],default=8);args=a.parse_args()
assert args.name and all(c.isascii() and (c.isalnum() or c=='-') for c in args.name)
assert 3<=args.minutes<=60
rel=lambda path:str(path.resolve().relative_to(root))
jobs=rel(args.jobs)
inputs=json.loads((cloud/'inputs.json').read_text())
if args.mode=='strength':
 proof=json.loads((p/'resume-validation.json').read_text());real=json.loads((p/'resume-real-canary.json').read_text())
 assert proof['passed'] and real['passed']
 assert proof['source_sha256']==hashlib.sha256((root/'crates/bot/examples/budget_resume.rs').read_bytes()).hexdigest()
 setup=rel(cloud/'setup-resume.sh');worker=rel(cloud/'strength_resume.py')
 inputs+=['crates/bot/examples/budget_resume.rs',rel(p/'strength-protocol.json')]
elif args.mode=='continuation':
 setup=rel(cloud/'setup-continuation.sh');worker=rel(cloud/'continuation_once.py')
 inputs+=['crates/bot/examples/preview_continuation.rs',rel(p/'unknown')]
else:
 setup=rel(cloud/'setup.sh');worker=rel(cloud/'bake_once.py')
inputs += [setup,worker,jobs]
worker_args=[jobs]
if args.resume:
 assert args.mode in ['strength','continuation'];resume=rel(args.resume);inputs.append(resume);worker_args.append(resume)
cmd=['spot-batch','exec','--gcp-location',args.region,'--machine-type',f'e2-standard-{args.cpus}','--minutes',str(args.minutes),'--budget-yen','200','--setup','sh '+setup,'--output',rel(cloud/(args.name+'-run'))]
for f in dict.fromkeys(inputs):cmd+=['--input',f]
cmd+=['--','env',f'NC2000_BUDGET_THREADS={args.cpus}','python3',worker,*worker_args]
(cloud/(args.name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
with (cloud/(args.name+'-submission.log')).open('w') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
print(args.name,'exit',r.returncode)
raise SystemExit(r.returncode)
