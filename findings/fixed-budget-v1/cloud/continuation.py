import os,subprocess,shutil
from pathlib import Path
p=Path('findings/fixed-budget-v1');out=Path(os.environ['SPOT_BATCH_OUTPUT'])
resume=p/'continuation-resume'
if resume.exists(): shutil.copytree(resume,out/'continuations',dirs_exist_ok=True)
subprocess.run(['/tmp/nc2000-budget-target/release/examples/preview_continuation',str(p/'unknown'),str(out/'continuations'),str(p/'continuation-cloud-jobs.json')],check=True,env=dict(os.environ,NC2000_CONTINUATION_THREADS='8'))
