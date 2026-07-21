#!/usr/bin/env python3
"""M16b aggregator: tmp/human-agreement.jsonl -> agreement report.

Usage: python3 tools/aggregate-human-agreement.py [tmp/human-agreement.jsonl]

Denominator discipline: agreement rates are over SCORABLE decisions
(in_set && n_actions > 1). Exclusion (out-of-set) rates are reported
separately per stratum — they measure imputation coverage, not the bot.
"""
import json, sys
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "tmp/human-agreement.jsonl"
rows, skips = [], Counter()
for ln in open(path):
    r = json.loads(ln)
    if "skip" in r:
        skips[r["skip"]] += 1
    else:
        rows.append(r)

scorable = [r for r in rows if r["in_set"] and r["n_actions"] > 1]
print(f"decisions {len(rows)}  skips {dict(skips)}")
print(f"scorable {len(scorable)} ({100*len(scorable)/max(len(rows),1):.1f}%)  "
      f"out-of-set {sum(not r['in_set'] for r in rows)}  "
      f"trivial(n=1) {sum(r['in_set'] and r['n_actions']==1 for r in rows)}")

def rate(rs, k=1):
    if not rs: return float('nan')
    return sum(1 for r in rs if r["rank"] is not None and r["rank"] <= k) / len(rs)

def block(label, rs):
    print(f"  {label:<28} n {len(rs):>6}  top1 {100*rate(rs,1):5.1f}%  "
          f"top2 {100*rate(rs,2):5.1f}%  top3 {100*rate(rs,3):5.1f}%")

print("\n== agreement (scorable only) ==")
block("ALL", scorable)
for kind in ("move", "switch"):
    block(f"kind={kind}", [r for r in scorable if r["kind"] == kind])
for rev in range(5):
    block(f"revealed={rev}", [r for r in scorable if r["kind"] == "move" and r["revealed"] == rev])
for prov in ("cand-full", "cand-fill", "learnset-pad"):
    block(f"own_prov={prov}", [r for r in scorable if r["own_prov"] == prov])

print("\n== exclusion (imputation coverage) ==")
for kind in ("move", "switch"):
    rs = [r for r in rows if r["kind"] == kind]
    ex = sum(not r["in_set"] for r in rs)
    print(f"  kind={kind:<8} n {len(rs):>6}  out-of-set {100*ex/max(len(rs),1):5.1f}%")

print("\n== class confusion (human rows x bot cols, scorable, disagreements only) ==")
mat = defaultdict(Counter)
for r in scorable:
    if not r["agree1"]:
        mat[r["human_class"]][r["bot_class"]] += 1
classes = ["Physical", "Special", "Status", "switch"]
print("  " + " ".join(f"{c:>9}" for c in ["human\\bot"] + classes))
for h in classes:
    print("  " + f"{h:>9}" + " ".join(f"{mat[h][b]:>9}" for b in classes))

print("\n== top disagreement pairs (human -> bot, scorable) ==")
pairs = Counter((r["human"], r["bot"]) for r in scorable if not r["agree1"])
for (h, b), n in pairs.most_common(15):
    print(f"  {n:>4}  human {h:<28} bot {b}")

print("\n== human action in bot's bottom half (strong disagreement) ==")
strong = [r for r in scorable if r["rank"] is not None and r["rank"] > max(2, r["n_actions"] // 2)]
print(f"  n {len(strong)} ({100*len(strong)/max(len(scorable),1):.1f}% of scorable)")
byk = Counter(r["kind"] for r in strong)
print(f"  by kind: {dict(byk)}")
