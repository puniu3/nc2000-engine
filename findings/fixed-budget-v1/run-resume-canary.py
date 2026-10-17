import json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parent.parent
base=[str(root/'target/release/examples/budget_resume'),str(p/'strength-protocol.json')]
first=p/'resume-interrupted';second=p/'strength-local-resume-canary';jobs=p/'resume-real-canary-jobs.json'
env=dict(os.environ,NC2000_BUDGET_THREADS='1')
with (p/'resume-real-canary.log').open('w') as log:
 process=subprocess.Popen([*base,str(first),str(jobs)],cwd=root,env=env,stdout=log,stderr=log)
 checkpoint=first/'battle243-p000-s1.partial'
 while process.poll() is None:
  if checkpoint.exists() and json.loads(checkpoint.read_text())['turn']>=4:
   process.terminate();process.wait();break
  time.sleep(.5)
 assert checkpoint.exists(), 'canary did not create a checkpoint'
 subprocess.run([*base,str(second),str(jobs),str(first)],cwd=root,env=env,stdout=log,stderr=log,check=True)
name='battle243-p000-s1.json'
a=json.loads((p/'strength-canary'/name).read_text());b=json.loads((second/name).read_text())
for row in [a,b]:
 for d in row['result']['decisions']:del d['ns']
assert a==b,'real-budget restart changed trajectory'
(p/'resume-real-canary.json').write_text(json.dumps({'passed':True,'job':name,'checkpoint_turn':json.loads(checkpoint.read_text())['turn'],'checks':'Every decision and final outcome identical to the original uninterrupted 81k/243k harness, excluding elapsed times.'},indent=2)+'\n')
print('Real-budget resumed trajectory matches original harness')
