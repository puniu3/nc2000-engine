import os,subprocess
from pathlib import Path
p=Path('findings/fixed-budget-v1');out=Path(os.environ['SPOT_BATCH_OUTPUT'])
subprocess.run(['/tmp/nc2000-budget-target/release/examples/compile_preview',str(p/'bake-remaining-2.json'),str(out/'bake'),'8'],check=True)
