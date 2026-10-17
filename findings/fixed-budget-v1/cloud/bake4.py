import os,subprocess
from pathlib import Path
p=Path('findings/fixed-budget-v1');out=Path(os.environ['SPOT_BATCH_OUTPUT']);bins=Path('/tmp/nc2000-budget-target/release/examples')
subprocess.run([str(bins/'compile_preview'),str(p/'bake-tail-remaining.json'),str(out/'bake'),'8'],check=True)
subprocess.run([str(bins/'preview_continuation'),str(p/'unknown'),str(out/'continuations'),str(p/'continuation-tail-jobs.json')],check=True,env=dict(os.environ,NC2000_CONTINUATION_THREADS='8'))
