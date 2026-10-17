import os,json,subprocess
from pathlib import Path
p=Path('findings/fixed-budget-v1');out=Path(os.environ['SPOT_BATCH_OUTPUT']);binary='/tmp/nc2000-budget-target/release/examples/compile_preview'
subprocess.run([binary,str(p/'canary-jobs.json'),str(out/'canary'),'2'],check=True)
for f in (p/'canary').glob('*.json'):
 a=json.loads(f.read_text()); b=json.loads((out/'canary'/f.name).read_text())
 for row in [a,b]:
  for sample in row['samples']: del sample['increment_ns']
 assert a==b,f.name
(out/'parity.json').write_text(json.dumps({'passed':True,'canaries':2}))
subprocess.run([binary,str(p/'bake-jobs.json'),str(out/'bake'),'16'],check=True)
