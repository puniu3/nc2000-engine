#!/usr/bin/env python3
"""Which of the three causes makes a flat root flat?

A quarter of corpus roots sit within 1.5x of a uniform visit distribution, and
agreement collapses there.  Three different defects produce that one symptom
and they need different fixes:

  (a) decision-rule noise — UCB spreads visits when values are close, so the
      visit argmax is a coin flip even though the value estimates separate.
      Signature: argmax(visits) != argmax(means), and the answer flips between
      seeds.
  (b) eval resolution — the values really are equal because the evaluator
      cannot see the difference.  Signature: tiny `q_spread`.
  (c) genuine indifference — the position is a coin flip and flatness is
      correct.  Signature: tiny `q_spread` too, so (b) and (c) separate only by
      whether an expert thinks the position is decided; this script bounds
      their combined share and leaves the split to inspection.

Also separates "when to retreat" from "which mon to bring in", which the
aggregate switch mass cannot.

Usage: tools/switch-diagnosis.py tmp/ha-s1.jsonl [tmp/ha-s2.jsonl]
"""
import json
import sys
from collections import defaultdict


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("skip") or not r.get("in_set") or r.get("top1") is None:
            continue
        out[(r["battle"], r["turn"], r["side"])] = r
    return out


def band(r):
    u = 1.0 / max(r["n_actions"], 1)
    if r["top1"] >= 3 * u:
        return "sharp   (>=3x uniform)"
    if r["top1"] >= 1.5 * u:
        return "middle  (1.5-3x)"
    return "flat    (<1.5x uniform)"


BANDS = ["sharp   (>=3x uniform)", "middle  (1.5-3x)", "flat    (<1.5x uniform)"]


def pct(x):
    return f"{x:.3f}"


def main():
    s1 = load(sys.argv[1] if len(sys.argv) > 1 else "tmp/ha-s1.jsonl")
    rows = list(s1.values())
    n = len(rows)
    print(f"n = {n} scorable decisions\n")

    # ---- the metric itself: argmax vs policy mass -------------------------
    print("=== the metric: argmax hides the action-set shape ===")
    for kind in ("move", "switch"):
        g = [r for r in rows if r["kind"] == kind]
        share = [r["human_share"] for r in g if r.get("human_share") is not None]
        agree = sum(1 for r in g if r["agree1"]) / len(g)
        print(f"  {kind:7s} n={len(g):6d}  argmax agreement {pct(agree)}"
              f"   mean policy mass on the human's action {sum(share)/len(share):.3f}")
    print("  (policy mass does not care that ~4 of 6 actions are moves; argmax does)\n")

    # ---- (a) decision rule ------------------------------------------------
    print("=== (a) is the decision rule the problem? ===")
    have_mean = [r for r in rows if r.get("bot_mean")]
    if have_mean:
        flip = sum(1 for r in have_mean if r["bot_mean"] != r["bot"]) / len(have_mean)
        a_vis = sum(1 for r in have_mean if r["agree1"]) / len(have_mean)
        a_mean = sum(1 for r in have_mean if r["bot_mean"] == r["human"]) / len(have_mean)
        print(f"  argmax(visits) != argmax(mean value): {flip:.1%} of roots")
        print(f"  agreement by visits {pct(a_vis)}   by mean value {pct(a_mean)}"
              f"   delta {a_mean - a_vis:+.3f}")
        print("  by band:")
        by = defaultdict(list)
        for r in have_mean:
            by[band(r)].append(r)
        for b in BANDS:
            g = by.get(b) or []
            if not g:
                continue
            f = sum(1 for r in g if r["bot_mean"] != r["bot"]) / len(g)
            av = sum(1 for r in g if r["agree1"]) / len(g)
            am = sum(1 for r in g if r["bot_mean"] == r["human"]) / len(g)
            print(f"    {b:24s} n={len(g):6d}  disagree {f:5.1%}"
                  f"   agree visits {pct(av)}  mean {pct(am)}  delta {am - av:+.3f}")

    if len(sys.argv) > 2:
        s2 = load(sys.argv[2])
        both = [(s1[k], s2[k]) for k in s1.keys() & s2.keys()]
        print(f"\n  seed stability on {len(both)} shared decisions:")
        by = defaultdict(list)
        for a, b in both:
            by[band(a)].append((a, b))
        allflip = sum(1 for a, b in both if a["bot"] != b["bot"]) / len(both)
        print(f"    top-1 changes when only the seed changes: {allflip:.1%} overall")
        for bnd in BANDS:
            g = by.get(bnd) or []
            if not g:
                continue
            f = sum(1 for a, b in g if a["bot"] != b["bot"]) / len(g)
            kf = sum(1 for a, b in g if a["bot"].split()[0] != b["bot"].split()[0]) / len(g)
            print(f"    {bnd:24s} n={len(g):6d}  action flips {f:5.1%}"
                  f"   move/switch class flips {kf:5.1%}")
        # a two-seed committee: does agreeing with either seed help?
        either = sum(1 for a, b in both if a["agree1"] or b["agree1"]) / len(both)
        base = sum(1 for a, _ in both if a["agree1"]) / len(both)
        print(f"    agreement with one seed {pct(base)}, with either of two {pct(either)}"
              f"  (spread {either - base:+.3f} = how much is coin flip)")

    # ---- (b)/(c) value spread --------------------------------------------
    print("\n=== (b)/(c) can the evaluator separate the actions at all? ===")
    sp = sorted(r["q_spread"] for r in rows if r.get("q_spread") is not None)
    if sp:
        m = len(sp)
        print(f"  root value spread (max-min over visited actions): median {sp[m//2]:.3f}"
              f"  p10 {sp[m//10]:.3f}  p90 {sp[9*m//10]:.3f}")
        print(f"  share of roots whose whole action set fits in 0.05 of value:"
              f" {sum(1 for x in sp if x < 0.05)/m:.1%}")
        by = defaultdict(list)
        for r in rows:
            if r.get("q_spread") is not None:
                by[band(r)].append(r)
        for b in BANDS:
            g = by.get(b) or []
            if not g:
                continue
            s = sorted(r["q_spread"] for r in g)
            print(f"    {b:24s} n={len(g):6d}  median spread {s[len(s)//2]:.3f}"
                  f"   under 0.05: {sum(1 for x in s if x < 0.05)/len(s):5.1%}")

    # ---- when vs which ----------------------------------------------------
    print("\n=== when to retreat vs which mon to bring in ===")
    sw = [r for r in rows if r["kind"] == "switch" and (r.get("n_switch") or 0) >= 2]
    if sw:
        hit = sum(1 for r in sw if r.get("best_switch") == r["human"]) / len(sw)
        chance = sum(1.0 / r["n_switch"] for r in sw) / len(sw)
        print(f"  human-switch decisions with a real choice of target: {len(sw)}")
        print(f"  the bot's favourite switch IS the human's: {hit:.3f}"
              f"   (chance {chance:.3f})")
        by = defaultdict(list)
        for r in sw:
            by[band(r)].append(r)
        for b in BANDS:
            g = by.get(b) or []
            if not g:
                continue
            h = sum(1 for r in g if r.get("best_switch") == r["human"]) / len(g)
            c = sum(1.0 / r["n_switch"] for r in g) / len(g)
            print(f"    {b:24s} n={len(g):5d}  target hit {h:.3f}  (chance {c:.3f})")


if __name__ == "__main__":
    main()
