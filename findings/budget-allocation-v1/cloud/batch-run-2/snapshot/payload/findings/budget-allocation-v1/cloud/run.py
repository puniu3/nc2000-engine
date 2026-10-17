import json
import os
from pathlib import Path
import platform
import subprocess

p = Path('findings/budget-allocation-v1')
c = p / 'cloud'
out = Path('results')
out.mkdir(exist_ok=True)
env = dict(os.environ, NC2000_BUDGET_THREADS='16')
binary = '/tmp/nc2000-budget-target/release/examples/budget_allocation'
def run(output, jobs):
    subprocess.run([binary, str(p/'protocol.json'), str(out/output), str(c/jobs)], env=env, check=True)

run('canary', 'canary-jobs.json')
for name in json.loads((c/'canary-jobs.json').read_text()):
    a = json.loads((c/'expected'/name).read_text())
    b = json.loads((out/'canary'/name).read_text())
    for r in [a,b]:
        for d in r['result']['decisions']:
            del d['ns']
    assert a == b, 'Cross-platform trajectory mismatch: '+name
(out/'platform-parity.json').write_text(json.dumps(dict(passed=True, canary_games=2, platform=platform.platform(), machine=platform.machine()), indent=2)+'\n')
print('Cross-platform full-trajectory canaries passed', flush=True)
run('games-remote', 'cloud-jobs.json')
print('All remote allocation games completed', flush=True)
