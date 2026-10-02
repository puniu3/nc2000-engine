#!/usr/bin/env python3
import json
import pathlib
import plistlib
import subprocess
import sys
import time


def utilization():
    data = subprocess.check_output(["ioreg", "-a", "-r", "-c", "AGXAccelerator"])
    return max((d.get("PerformanceStatistics", {}).get("Device Utilization %", 100)
                for d in plistlib.loads(data)), default=100)


out = pathlib.Path(sys.argv[1])
records = []
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    before = utilization()
    if before >= 10:
        time.sleep(2)
        continue
    result = json.loads(subprocess.check_output([str((out / "metal-probe").resolve())], text=True))
    time.sleep(2)
    after = utilization()
    result["utilization_before"] = before
    result["utilization_after"] = after
    result["host_idle_screen_passed"] = before < 10 and after < 10
    records.append(result)
    if result["host_idle_screen_passed"]:
        (out / "metal.json").write_text(json.dumps(result, indent=2) + "\n")
        break
    time.sleep(2)
else:
    raise RuntimeError("no quiet GPU measurement window within 300 seconds")
(out / "metal-attempts.json").write_text(json.dumps(records, indent=2) + "\n")
