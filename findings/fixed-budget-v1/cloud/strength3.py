import os,subprocess
from pathlib import Path
p=Path('findings/fixed-budget-v1');out=Path(os.environ['SPOT_BATCH_OUTPUT'])
subprocess.run(['/tmp/nc2000-budget-target/release/examples/budget_allocation',str(p/'strength-protocol.json'),str(out/'games'),str(p/'strength-last-jobs.json')],check=True,env=dict(os.environ,NC2000_BUDGET_THREADS='8'))
