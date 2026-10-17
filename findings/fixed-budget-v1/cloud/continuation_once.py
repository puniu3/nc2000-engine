import os,subprocess,sys,shutil
from pathlib import Path
out=Path(os.environ['SPOT_BATCH_OUTPUT'])/'continuations'
if len(sys.argv)>2:shutil.copytree(sys.argv[2],out,dirs_exist_ok=True)
subprocess.run(['/tmp/nc2000-budget-target/release/examples/preview_continuation','findings/fixed-budget-v1/unknown',str(out),sys.argv[1]],check=True,env=dict(os.environ,NC2000_CONTINUATION_THREADS=os.environ.get('NC2000_BUDGET_THREADS','8')))
