#!/usr/bin/env python3
import concurrent.futures,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'findings/deployed-endurance-v1/continuations';out.mkdir(parents=True,exist_ok=True)
subprocess.run(['node','tests/endurance-canary.mjs'],cwd=root/'web',check=True)
def run(job):
 trial,arm=job;dest=out/f'g4-d12-{arm}-{trial}.json';start=time.time()
 p=subprocess.run(['node','tests/endurance-continuation.mjs','g4','12','1',arm,str(trial),str(dest)],cwd=root/'web',text=True,capture_output=True)
 record={'trial':trial,'arm':arm,'returncode':p.returncode,'seconds':time.time()-start,'stdout':p.stdout,'stderr':p.stderr};(out/f'g4-d12-{arm}-{trial}.process.json').write_text(json.dumps(record)+'\n')
 if p.returncode:raise RuntimeError(record)
 result=json.loads(dest.read_text());assert result['score'] is not None
 return record
jobs=[(trial,arm) for trial in range(2,16) for arm in ['baseline','conserve-heal']]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
