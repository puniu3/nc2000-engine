import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def fingerprint():
    paths = subprocess.check_output(["git", "ls-files", "crates", "web", "data/gen2stadium2.json", "tools/endgame-confirmation-run.py", "tools/endgame-match-report.py", "tools/endgame-web-report.py", "tools/endgame-holdout-report.py"], cwd=ROOT, text=True).splitlines()
    paths += ["crates/wasm/pkg-web/nc2000_wasm_bg.wasm", "crates/wasm/pkg-web/nc2000_wasm.js"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(set(paths)) if (ROOT / name).is_file()}


def require_live(pid, expected_command):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    assert result.returncode == 0 and expected_command in result.stdout, f"prerequisite process {pid} is missing or changed; partial evidence retained"


def protocols(design, selection, timing, timing_protocol, selection_path):
    assert selection["complete"] and timing["complete"]
    name = selection["selected_for_confirmation"]
    assert name is not None, "no development candidate passed the frozen eligibility rules"
    chosen = selection["arms"][name]
    fraction = chosen["protocol"]["candidate"]["solver_fraction"]
    for arm in timing["arms"].values():
        assert not arm.get("illegal", 0) and not arm.get("nonvisible", 0)
    assert timing_protocol["milliseconds"] == 814
    assert timing_protocol["pairs"] == 16
    candidate = timing["arms"]["candidate"]["nonforced_request_ms"]["mean"]
    baseline = timing["arms"]["baseline"]["nonforced_request_ms"]["mean"]
    milliseconds = min(814, math.floor(baseline - (candidate - 814)))
    assert 1 <= milliseconds <= 814
    web = {key: timing_protocol[key] for key in ("schema", "catalog", "catalog_sha256", "prior", "prior_sha256", "c")}
    settings = design["web"]
    web.update(calibration=False, candidate_selection=str(selection_path), pairs=settings["pairs"],
               threads=settings["threads"], iterations=settings["baseline_iterations"], seed=settings["seed"],
               milliseconds=milliseconds, max_nodes=settings["max_nodes"],
               candidate={"information": "public", "solver_fraction": fraction},
               baseline={"information": "raw", "solver_fraction": 0},
               interpretation=json.dumps(settings, ensure_ascii=False))
    settings = design["independent_endgames"]
    holdout = {key: settings[key] for key in ("corpus", "corpus_sha256", "selection", "milliseconds", "reference_ms", "seed", "threads")}
    holdout.update(schema="nc2000-endgame-holdout-v1", prior=web["prior"], prior_sha256=web["prior_sha256"],
                   calibration=False, candidate_selection=str(selection_path), solver_fraction=fraction,
                   interpretation=json.dumps(settings, ensure_ascii=False))
    return web, holdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("development", type=pathlib.Path)
    parser.add_argument("timing", type=pathlib.Path)
    parser.add_argument("timing_pid", type=int)
    parser.add_argument("destination", type=pathlib.Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    design_path = ROOT / "findings/endgame-integration-v1/confirmation-design.json"
    design = json.loads(design_path.read_text())
    shutil.copy2(design_path, destination / "design.json")
    binary = destination / "endgame_holdout"
    shutil.copy2(ROOT / "target/release/examples/endgame_holdout", binary)
    frozen = fingerprint()
    write(destination / "launcher.json", {"pid": os.getpid(), "source_sha256": frozen, "holdout_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
          "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "started_unix": time.time()})
    native_report, web_report = module("endgame-match-report"), module("endgame-web-report")
    launcher = json.loads((args.development / "launcher.json").read_text())
    last_counts = None
    while True:
        selection = native_report.report(args.development)
        timing = web_report.report(args.timing)
        counts = ([arm["completed_games"] for arm in selection["arms"].values()], timing["completed_games"])
        if counts != last_counts:
            print(json.dumps({"prerequisite_completed_games": counts}), flush=True)
            last_counts = counts
        if selection["complete"] and timing["complete"]:
            break
        for name, pid in launcher["pids"].items():
            if not selection["arms"][name]["complete"]:
                require_live(pid, str(args.development.resolve() / "endgame_matches"))
        if not timing["complete"]:
            require_live(args.timing_pid, "web-budget-control-v3-protocol.json")
        time.sleep(30)
    assert fingerprint() == frozen, "evaluation inputs changed while awaiting prerequisites"
    selection_path = destination / "candidate-selection.json"
    write(selection_path, selection)
    write(destination / "timing-validation.json", timing)
    web, holdout = protocols(design, selection, timing, json.loads((args.timing / "protocol.json").read_text()), selection_path)
    write(destination / "web-protocol.json", web)
    write(destination / "holdout-protocol.json", holdout)
    subprocess.run(["ports", "--help"], check=True)
    commands = {
        "web": ["node", str(ROOT / "web/tests/endgame-matches.mjs"), str(destination / "web-protocol.json"), str(destination / "web")],
        "holdout": [str(binary), str(destination / "holdout-protocol.json"), str(destination / "holdout")],
    }
    processes, logs = {}, []
    try:
        for name, command in commands.items():
            log = (destination / f"{name}.log").open("x")
            logs.append(log)
            processes[name] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(destination / "evaluation-pids.json", {name: process.pid for name, process in processes.items()})
        while any(process.poll() is None for process in processes.values()):
            assert all(process.poll() in (None, 0) for process in processes.values()), "evaluation process failed; partial evidence retained"
            time.sleep(30)
        assert all(process.returncode == 0 for process in processes.values())
        assert fingerprint() == frozen, "evaluation inputs changed during confirmation"
        write(destination / "web-result.json", web_report.report(destination / "web"))
        write(destination / "holdout-result.json", module("endgame-holdout-report").report(destination / "holdout"))
        print("Confirmation and independent endgame evaluations complete; adoption audit remains required.", flush=True)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
