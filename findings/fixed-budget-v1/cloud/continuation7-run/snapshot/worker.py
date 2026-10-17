"""Dependency-free bounded parallel command runner, shared by Mac and Linux."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import threading
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def run(job_path, output, binary=None, concurrency=None, low_priority=False):
    spec = json.loads(Path(job_path).read_text())
    root = Path(job_path).resolve().parent
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise RuntimeError('Output already complete; use a fresh directory')
    if low_priority:
        os.nice(10)
    stop = threading.Event()
    samples = []
    def sample():
        while not stop.is_set():
            row = {'time': time.time(), 'load': os.getloadavg()}
            if Path('/proc/meminfo').exists():
                row['meminfo'] = Path('/proc/meminfo').read_text()
                row['cpu'] = Path('/proc/stat').read_text().splitlines()[0]
                row['network'] = Path('/proc/net/dev').read_text()
            samples.append(row)
            stop.wait(2)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    groups = []
    try:
        for n in concurrency or spec['concurrency']:
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            started = time.time()
            def one(item):
                rep, task = item
                ident = f"p{n}-r{rep}-{task['id']}"
                result_path = out / (ident + '.json')
                if result_path.exists():
                    return json.loads(result_path.read_text())
                argv = [str(binary) if a == '{binary}' and binary else a for a in task['argv']]
                t = time.time()
                # Cap output of each process without keeping an unbounded pipe in memory.
                stdout_path, stderr_path = out / (ident + '.stdout'), out / (ident + '.stderr')
                with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
                    proc = subprocess.Popen(argv, cwd=root, stdout=stdout, stderr=stderr, start_new_session=True)
                    deadline = time.monotonic() + spec.get('task_timeout_seconds', 90)
                    while proc.poll() is None:
                        if time.monotonic() > deadline or stdout_path.stat().st_size + stderr_path.stat().st_size > 1024 * 1024:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            raise RuntimeError('Task timeout/output limit: ' + ident)
                        time.sleep(.1)
                row = {'id': task['id'], 'parallel': n, 'repeat': rep, 'wall_seconds': time.time() - t, 'returncode': proc.returncode}
                row['stdout'] = stdout_path.read_text()
                row['stderr'] = stderr_path.read_text()
                if spec.get('adapter') == 'mcts':
                    row['search'] = json.loads(row.pop('stdout'))
                    s = row['search']
                    if proc.returncode or s.get('error') or not s.get('read_only') or s.get('completed', 0) <= 0:
                        raise RuntimeError('MCTS failed: ' + ident)
                elif proc.returncode:
                    raise RuntimeError('Command failed: ' + ident)
                write_json(result_path, row)
                stdout_path.unlink()
                stderr_path.unlink()
                return row
            jobs = [(r, t) for r in range(spec.get('repeats', 1)) for t in spec['tasks']]
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                rows = list(pool.map(one, jobs))
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            group = {'parallel': n, 'started': started, 'wall_seconds': time.time() - started,
                     'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                     'max_child_rss_bytes': after.ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
                     'tasks': len(rows), 'completed': sum(r.get('search', {}).get('completed', 0) for r in rows)}
            group['rollouts_per_second'] = group['completed'] / group['wall_seconds']
            groups.append(group)
            write_json(out / 'progress.json', groups)
            print(json.dumps(group), flush=True)
    finally:
        stop.set()
        thread.join()
        write_json(out / 'samples.json', samples)
    for rel, expected in spec.get('input_sha256', {}).items():
        if digest(root / rel) != expected:
            raise RuntimeError('Input changed: ' + rel)
    summary = {'platform': platform.platform(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count(),
               'job_sha256': digest(job_path), 'groups': groups, 'input_unchanged': True}
    write_json(out / 'summary.json', summary)
    write_json(out / 'checksums.json', {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('job'); p.add_argument('output'); p.add_argument('--binary')
    p.add_argument('--concurrency', nargs='+', type=int); p.add_argument('--low-priority', action='store_true')
    args = p.parse_args()
    run(args.job, args.output, args.binary, args.concurrency, args.low_priority)
