import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parents[1]
destination = pathlib.Path(sys.argv[1]).resolve()
destination.mkdir(parents=True, exist_ok=False)
binary = destination / "endgame_matches"
shutil.copy2(root / "target/release/examples/endgame_matches", binary)
protocols = sorted((root / "findings/endgame-integration-v1").glob("development-*-protocol.json"))
assert len(protocols) == 4
processes = []
for protocol in protocols:
    arm = protocol.stem.removesuffix("-protocol")
    frozen_protocol = destination / protocol.name
    shutil.copy2(protocol, frozen_protocol)
    log = (destination / f"{arm}.log").open("x")
    process = subprocess.Popen([str(binary), str(frozen_protocol), str(destination / arm)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    processes.append((arm, process, log))
(destination / "launcher.json").write_text(json.dumps({
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "pids": {name: process.pid for name, process, _ in processes},
    "started_unix": time.time(),
}, indent=2) + "\n")
with (destination / "rss.jsonl").open("x") as samples:
    while any(process.poll() is None for _, process, _ in processes):
        records = []
        for name, process, _ in processes:
            if process.poll() is None:
                result = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True)
                if result.stdout.strip():
                    records.append({"arm": name, "pid": process.pid, "rss_kib": int(result.stdout.strip())})
        samples.write(json.dumps({"unix": time.time(), "processes": records}) + "\n")
        samples.flush()
        failed = [(name, process.returncode) for name, process, _ in processes if process.poll() not in (None, 0)]
        if failed:
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            raise RuntimeError(f"match process failed; partial evidence preserved: {failed}")
        time.sleep(10)
for name, process, log in processes:
    log.close()
    assert process.returncode == 0, (name, process.returncode)
print("Completed all 400 native development games.")
