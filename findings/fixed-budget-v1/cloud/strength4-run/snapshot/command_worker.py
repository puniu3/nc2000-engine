"""Execute one submitted computation in its private Linux working directory."""
import json
import os
from pathlib import Path
import subprocess
import time


def run(root):
    root = Path(root).resolve()
    request = json.loads((root/'request.json').read_text())
    output = root/'results'
    output.mkdir(exist_ok=True)
    files = output/'files'
    files.mkdir(exist_ok=True)
    env = {**os.environ, 'SPOT_BATCH_OUTPUT':str(files), 'PYTHONUNBUFFERED':'1'}
    started = time.monotonic()
    commands = ([['bash', '-euo', 'pipefail', '-c', request['setup']]] if request.get('setup') else []) + [request['command']]
    code = 0
    with (output/'stdout.txt').open('wb') as stdout, (output/'stderr.txt').open('wb') as stderr:
        for command in commands:
            proc = subprocess.Popen(command, cwd=root/'payload', env=env, stdout=stdout, stderr=stderr)
            while proc.poll() is None:
                if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > request['output_limit_bytes']:
                    proc.kill()
                    proc.wait()
                    raise RuntimeError('Computation exceeded the result size limit')
                time.sleep(.2)
            code = proc.returncode
            if code:
                break
    (output/'command.json').write_text(json.dumps({'command':request['command'], 'returncode':code,
                                                 'wall_seconds':time.monotonic()-started}, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(run(Path.cwd()))
