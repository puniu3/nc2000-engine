import os,subprocess,sys
from pathlib import Path
out=Path(os.environ['SPOT_BATCH_OUTPUT'])
args=['/tmp/nc2000-budget-target/release/examples/budget_resume','findings/fixed-budget-v1/strength-protocol.json',str(out/'games'),*sys.argv[1:]]
subprocess.run(args,check=True,env=dict(os.environ,NC2000_BUDGET_THREADS=os.environ.get('NC2000_BUDGET_THREADS','8')))
