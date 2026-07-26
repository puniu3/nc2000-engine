#!/usr/bin/env python3
"""Two questions the raw M16b agreement number cannot answer.

1. Is the switch cluster a real disagreement, or is it the bot having no
   opinion?  `top1` is the bot's top action's share of root visits.  A 6-action
   root at 1/6 is a uniform policy, and comparing its argmax against a human
   choice measures a coin flip.  If agreement rises with confidence but the
   switch/move gap survives inside every confidence band, the cluster is real.

2. Do the structural threats a per-mon eval cannot see (Encore lock, Mean Look
   trap, a Belly Drum/Curse sweeper across the field) mark the positions where
   the bot disagrees most?  Tags come from engine volatiles and boosts, never
   from imputed movesets.

Usage: tools/agreement-by-confidence.py tmp/ha-tagged.jsonl
"""
import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r.get("skip") or not r.get("in_set"):
            continue
        if r.get("top1") is None:
            continue
        rows.append(r)
    return rows


def rate(rows):
    if not rows:
        return float("nan"), 0
    return sum(1 for r in rows if r["agree1"]) / len(rows), len(rows)


def band(t):
    for lo in (0.9, 0.7, 0.5, 0.35, 0.25):
        if t >= lo:
            return lo
    return 0.0


BAND_LABEL = {
    0.9: "0.90-1.00  (decided)",
    0.7: "0.70-0.90",
    0.5: "0.50-0.70",
    0.35: "0.35-0.50",
    0.25: "0.25-0.35",
    0.0: "0.00-0.25  (no opinion)",
}


def tags_of(r):
    out = []
    sv = set(r.get("self_vols") or [])
    fv = set(r.get("foe_vols") or [])
    if "encore" in sv:
        out.append("self encored")
    if "trapped" in sv or "partiallytrapped" in sv:
        out.append("self trapped")
    if "perishsong" in sv or "perishsong" in fv:
        out.append("perish song up")
    if "confusion" in sv:
        out.append("self confused")
    if "leechseed" in sv:
        out.append("self seeded")
    if (r.get("foe_boost") or 0) >= 2:
        out.append("foe set up (+2 or more)")
    if (r.get("self_boost") or 0) >= 2:
        out.append("self set up (+2 or more)")
    if (r.get("self_hp") or 100) <= 25:
        out.append("self below 25% hp")
    if r.get("self_status"):
        out.append("self statused")
    if not out:
        out.append("(plain position)")
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tmp/ha-tagged.jsonl"
    rows = load(path)
    moves = [r for r in rows if r["kind"] == "move"]
    switches = [r for r in rows if r["kind"] == "switch"]

    print(f"scorable decisions: {len(rows)}  (move {len(moves)}, switch {len(switches)})\n")

    ar, _ = rate(rows)
    am, _ = rate(moves)
    asw, _ = rate(switches)
    print(f"top-1 agreement   all {ar:.3f}   move {am:.3f}   switch {asw:.3f}"
          f"   gap {am - asw:+.3f}\n")

    print("=== 1. is the bot even deciding? ===")
    print("mean top-1 visit share:")
    for name, group in (("move", moves), ("switch", switches)):
        if not group:
            continue
        t = sorted(r["top1"] for r in group)
        n = len(t)
        print(f"  {name:7s} mean {sum(t)/n:.3f}  median {t[n//2]:.3f}"
              f"  p10 {t[n//10]:.3f}  p90 {t[9*n//10]:.3f}"
              f"  share below 0.25: {sum(1 for x in t if x < 0.25)/n:.1%}")

    print("\nagreement inside each confidence band (the gap must survive here):")
    print(f"  {'band':24s} {'move':>16s} {'switch':>16s} {'gap':>8s}")
    groups = defaultdict(lambda: {"move": [], "switch": []})
    for r in rows:
        groups[band(r["top1"])][r["kind"]].append(r)
    for lo in (0.9, 0.7, 0.5, 0.35, 0.25, 0.0):
        g = groups[lo]
        mrate, mn = rate(g["move"])
        srate, sn = rate(g["switch"])
        gap = mrate - srate if mn and sn else float("nan")
        print(f"  {BAND_LABEL[lo]:24s} {mrate:>8.3f} n={mn:<5d} {srate:>8.3f} n={sn:<5d} {gap:>+8.3f}")

    print("\n=== 2. do the unseeable threats mark the disagreements? ===")
    print(f"  {'tag':26s} {'n':>6s} {'agree':>7s} {'human switched':>15s} {'bot switch mass':>16s} {'top1':>7s}")
    bytag = defaultdict(list)
    for r in rows:
        for t in tags_of(r):
            bytag[t].append(r)
    base_sw = sum(1 for r in rows if r["kind"] == "switch") / max(len(rows), 1)
    base_mass = sum(r["switch_mass"] for r in rows) / max(len(rows), 1)
    base_a, _ = rate(rows)
    print(f"  {'ALL':26s} {len(rows):>6d} {base_a:>7.3f} {base_sw:>15.1%} {base_mass:>16.3f}"
          f" {sum(r['top1'] for r in rows)/len(rows):>7.3f}")
    for t, g in sorted(bytag.items(), key=lambda kv: -len(kv[1])):
        a, n = rate(g)
        hs = sum(1 for r in g if r["kind"] == "switch") / n
        bm = sum(r["switch_mass"] for r in g) / n
        t1 = sum(r["top1"] for r in g) / n
        print(f"  {t:26s} {n:>6d} {a:>7.3f} {hs:>15.1%} {bm:>16.3f} {t1:>7.3f}")


if __name__ == "__main__":
    main()
