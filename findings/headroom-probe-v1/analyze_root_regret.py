#!/usr/bin/env python3
"""Finite-sample root noise of the shipped 27k blind search, read off existing records.

Input: findings/release-strength-extension-v1/records.jsonl.gz (1,536 games of
blind:27000:0.4 self-play, every frame carries per-action visits and means).

For each decision with >= 2 actions, each action's value is approximated as
N(mean, mean*(1-mean)/visits). K posterior draws estimate
  - flip rate: P(a fresh draw's argmax != the played action)
  - simple regret: E[max_a Q_a - Q_played]
Caveat: UCB means are biased (not independent unbiased estimates), so these are
rough indicators of search noise, not certified losses.

Usage: python3 findings/headroom-probe-v1/analyze_root_regret.py [K]
"""
import gzip
import json
import math
import random
import sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SRC = "findings/release-strength-extension-v1/records.jsonl.gz"
rng = random.Random(20261016)
bands = {"<0.50": [0, 0.0, 0.0], "0.50-0.70": [0, 0.0, 0.0],
         "0.70-0.90": [0, 0.0, 0.0], ">=0.90": [0, 0.0, 0.0]}
tot = {"battle": [0, 0.0, 0.0], "preview": [0, 0.0, 0.0]}

with gzip.open(SRC, "rt") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue  # manifest
        for fr in json.loads(line)["frames"]:
            acts = fr["response"]["actions"]
            if len(acts) <= 1:
                continue
            played = fr["response"]["action"]
            acts = sorted(acts, key=lambda a: (a["input"] == played, a["visits"]), reverse=True)
            mus = [a["mean"] for a in acts]
            ses = [math.sqrt(max(1e-9, m * (1 - m)) / max(1, a["visits"])) for m, a in zip(mus, acts)]
            flips, regret = 0, 0.0
            for _ in range(K):
                d = [rng.gauss(m, s) for m, s in zip(mus, ses)]
                b = max(range(len(d)), key=d.__getitem__)
                flips += b != 0
                regret += d[b] - d[0]
            kind = "preview" if acts[0]["input"].startswith("team ") else "battle"
            t = tot[kind]
            t[0] += 1; t[1] += flips / K; t[2] += regret / K
            if kind == "battle":
                share = acts[0]["visits"] / fr["response"]["iterations"]
                key = "<0.50" if share < .5 else "0.50-0.70" if share < .7 else "0.70-0.90" if share < .9 else ">=0.90"
                bands[key][0] += 1; bands[key][1] += flips / K; bands[key][2] += regret / K

for kind, (n, fl, rg) in tot.items():
    print(f"{kind}: n={n} flip={100*fl/n:.2f}% regret={100*rg/n:.3f}% win-prob/decision")
for key, (n, fl, rg) in bands.items():
    print(f"  battle top1-share {key:9s}: n={n} flip={100*fl/n:.2f}% regret={100*rg/n:.4f}%")
