#!/usr/bin/env python3
"""Resume only completed files; one launcher owns trials 1..15."""
import concurrent.futures,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'findings/deployed-endurance-v1/continuations'
def run(job):
 trial,arm=job;dest=out/f'g4-d12-{arm}-{trial}.json'
 if dest.exists():
  result=json.loads(dest.read_text());assert result['score'] is not None
  return {'trial':trial,'arm':arm,'existing':True}
 args=['node','tests/endurance-continuation.mjs','g4','12','1',arm,str(trial),str(dest)]
 start=time.time();p=subprocess.run(args,cwd=root/'web',text=True,capture_output=True)
 record={'trial':trial,'arm':arm,'returncode':p.returncode,'seconds':time.time()-start,'stdout':p.stdout,'stderr':p.stderr}
 (out/f'g4-d12-{arm}-{trial}.process.json').write_text(json.dumps(record,indent=2)+'\n')
 if p.returncode:raise RuntimeError(record)
 result=json.loads(dest.read_text());assert result['score'] is not None
 return record
jobs=[(trial,arm) for trial in range(1,16) for arm in ['baseline','conserve-heal']]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for result in pool.map(run,jobs):print(json.dumps(result),flush=True)
