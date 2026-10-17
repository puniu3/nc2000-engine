#!/usr/bin/env python3
import argparse,concurrent.futures,json,subprocess,time,os,fcntl
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--trials',type=int,nargs='+',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();assert all(0<=n<16 for n in a.trials) and 1<=a.workers<=8
root=Path(__file__).resolve().parents[1];out=root/'findings/deployed-endurance-v1/continuations';out.mkdir(exist_ok=True)
def run(job):
 trial,arm=job;key=f'g4-d12-{arm}-{trial}';dest=out/(key+'.json')
 with (out/(key+'.lock')).open('w') as lock:
  # Other local queues may claim a case first. Wait for their result instead
  # of duplicating the expensive game or mistaking a busy lock for failure.
  fcntl.flock(lock,fcntl.LOCK_EX)
  if dest.exists():
   result=json.loads(dest.read_text());assert result['score'] is not None
   return {'trial':trial,'arm':arm,'existing':True}
  args=['node','tests/endurance-continuation.mjs','g4','12','1',arm,str(trial),str(dest)];start=time.time()
  with (out/(key+'.stdout')).open('w') as stdout,(out/(key+'.stderr')).open('w') as stderr:
   proc=subprocess.Popen(args,cwd=root/'web',stdout=stdout,stderr=stderr)
   live={'pid':proc.pid,'trial':trial,'arm':arm,'started':start,'command':args};(out/(key+'.live.json')).write_text(json.dumps(live)+'\n');code=proc.wait()
  record={**live,'returncode':code,'seconds':time.time()-start};(out/(key+'.process.json')).write_text(json.dumps(record,indent=2)+'\n')
  if code:raise RuntimeError(record)
  result=json.loads(dest.read_text());assert result['score'] is not None
  return record
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 futures=[pool.submit(run,(t,arm)) for t in a.trials for arm in ['baseline','conserve-heal']]
 for f in concurrent.futures.as_completed(futures):print(json.dumps(f.result()),flush=True)
