#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q + 0.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=pathlib.Path)
    p.add_argument("candidate", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--mode", default="blind")
    p.add_argument("--iterations", type=int, default=27000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    assert min(args.iterations, args.threads, args.repeats) > 0
    args.out.mkdir(parents=True, exist_ok=True)
    bins = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    manifest = {"binaries": {k: digest(v) for k, v in bins.items()},
                "mode": args.mode, "iterations": args.iterations,
                "threads": args.threads, "repeats": args.repeats}
    root = pathlib.Path(__file__).resolve().parent.parent
    manifest["inputs"] = {p: digest(root / p) for p in ["data/gen2stadium2.json", "data/belief-pool-v3/belief-pool.json", "data/search-profiles.json"]}
    mf = args.out / "manifest.json"
    if mf.exists():
        assert json.loads(mf.read_text()) == manifest, "output belongs to another experiment"
    else:
        mf.write_text(json.dumps(manifest, indent=2) + "\n")
    runs = {k: [] for k in bins}
    reference = None
    for repeat in range(args.repeats):
        for arm in (list(bins) if repeat % 2 == 0 else list(reversed(bins))):
            path = args.out / f"{arm}-{repeat}.json"
            if path.exists():
                run = json.loads(path.read_text())
            else:
                result = subprocess.run([str(bins[arm]), args.mode, str(args.iterations), str(args.threads)],
                                        capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                run = json.loads(result.stdout)
                for row in run["rows"]:
                    row["signature"] = hashlib.sha256(str(row["signature"]).encode()).hexdigest()
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(run, indent=2) + "\n")
                temporary.replace(path)
            signature = [(r["id"], r["tag"], r["state"], r["signature"]) for r in run["rows"]]
            if reference is None:
                reference = signature
            assert reference == signature, f"state or search statistics diverged: {path}"
            runs[arm].append(run)
            print(f"{arm} {repeat + 1}/{args.repeats}: {run['wall_seconds']:.3f}s", flush=True)
    def stats(arm):
        rr = runs[arm]
        tags = sorted({row["tag"] for row in rr[0]["rows"]})
        return {"median_seconds": statistics.median(r["wall_seconds"] for r in rr),
                "p95_seconds": {tag: statistics.median(percentile([x["seconds"] for x in r["rows"] if x["tag"] == tag], .95) for r in rr) for tag in tags}}
    a, b = stats("baseline"), stats("candidate")
    reduction = 1 - b["median_seconds"] / a["median_seconds"]
    ratios = {tag: b["p95_seconds"][tag] / a["p95_seconds"][tag] for tag in a["p95_seconds"]}
    result = {"baseline": a, "candidate": b, "time_reduction": reduction,
              "p95_ratios": ratios, "exact_match": True, "positions": len(reference),
              "accepted": reduction >= .05 and max(ratios.values()) <= 1.05}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
