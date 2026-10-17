import os,subprocess,sys
from pathlib import Path
out=Path(os.environ['SPOT_BATCH_OUTPUT'])
subprocess.run(['/tmp/nc2000-budget-target/release/examples/compile_preview',sys.argv[1],str(out/'bake'),os.environ.get('NC2000_BUDGET_THREADS','8')],check=True)
