#!/usr/bin/env python3
"""Can a duel resolve this eval candidate at all?  Ask before spending VM hours.

The exchange term (data/exchange-term-verdict.txt) cost a full round of
instruments to learn this: it changed the bot's top-1 choice on 13.8% of corpus
decisions while changing the RNG SEED ALONE changed it on 15.5% at the same
budget.  The 800-game seed-paired duel therefore read parity by construction,
and every agreement metric read flat by construction.  Nothing was wrong with
the instruments; the candidate's behavioural footprint was under the search's
own noise, and no downstream measurement can resolve a signal there.

This is the cheap pre-check.  Ten minutes of corpus replay:

    cargo run --release -p nc2000-bot --example human_agreement -- \
        --corpus tmp/corpus-spectator --battles 0-59 --iters 30000 --seed 1 \
        --out tmp/cand.jsonl            # and the same with --seed 2 for the floor

    tools/eval-candidate-screen.py BASE.jsonl CAND.jsonl [BASE-OTHER-SEED.jsonl]

BASE and CAND must be the same corpus, budget and seed, differing only in the
eval change under test.  BASE-OTHER-SEED is BASE's config re-run at a different
seed; it is what supplies the noise floor.  Without it there is no floor and
this tool refuses to issue a verdict -- an unanchored footprint number is
exactly the mistake that produced the exchange-term dead end.

Rows pair on the coordinate (battle, side, turn).  Coordinates absent from any
supplied file are dropped and counted.

Denominator: the neighbours' scorable filter (`in_set`, `top1` present), so the
count matches tools/agreement-by-confidence.py and the recorded 1,984-decision
runs.  `--all` widens it to every decision the bot made, human-scorable or not
(measured to move the rates by <0.2pp on the 30k artifacts).

THRESHOLDS (top-1 change rate; footprint F, floor N, ratio R = F/N)

    MEASURABLE     R >= 2.00  and  F >= 0.05
    INCONCLUSIVE   1.25 <= R < 2.00, or R >= 2.00 with F < 0.05
    UNMEASURABLE   R < 1.25

Why those numbers.  A same-seed rerun of the same eval is bit-identical (it was
checked: ha-30k-base2 vs ha-30k-s1, same config and seed, 0/1984 flips), so all
baseline-vs-candidate divergence really is caused by the eval.  But it is
TRANSMITTED through a chaotic search: a candidate with no systematic preference
change still flips coin-flip roots at roughly the re-seed rate.  So the floor is
the null hypothesis, not an error bar.  R = 2.0 means half the divergence is
more than re-seeding can produce.  R = 1.25 is where the gap stops being
statistically real at all: at n ~ 2000 and p ~ 0.15 the two-proportion sigma is
~0.011, so 0.25 x 0.155 = 0.039 is only ~3.4 sigma -- below that the footprint
and the floor are not even distinguishable on this corpus, let alone in a duel.
F >= 0.05 is an absolute guard: with a small floor a large ratio can still
describe a candidate that touches almost nothing.

Confidence bands are agreement-by-confidence.py's, banded on the BASELINE's
`top1` visit share.  A candidate whose footprint sits only in the flat bands is
moving the search's coin flips, not its opinions.

Exit code doubles as a gate for shell drivers:
    0 MEASURABLE   1 INCONCLUSIVE   2 UNMEASURABLE   3 no floor given
"""
import argparse
import json
import sys
from collections import defaultdict

# thresholds -- see the docstring; changing these changes what ships
RATIO_MEASURABLE = 2.00
RATIO_UNMEASURABLE = 1.25
ABS_FOOTPRINT_MIN = 0.05

MASS_FIELDS = ("switch_mass", "status_mass")


def load(path, keep_all):
    """coordinate -> row, applying the neighbours' scorable filter."""
    out, total, dropped = {}, 0, 0
    for line in open(path):
        r = json.loads(line)
        total += 1
        if r.get("skip") or r.get("top1") is None:
            dropped += 1
            continue
        if not keep_all and not r.get("in_set"):
            dropped += 1
            continue
        out[(r["battle"], r["side"], r["turn"])] = r
    return out, total, dropped


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
BANDS = (0.9, 0.7, 0.5, 0.35, 0.25, 0.0)


def diff(a, b, coords):
    """Behavioural distance between two arms over shared coordinates."""
    n = len(coords)
    d = {"n": n}
    d["top1"] = sum(1 for k in coords if a[k]["bot"] != b[k]["bot"]) / n
    d["class"] = sum(1 for k in coords if a[k]["bot_class"] != b[k]["bot_class"]) / n
    d["kind"] = sum(
        1 for k in coords if a[k]["bot"].split()[0] != b[k]["bot"].split()[0]
    ) / n
    for f in MASS_FIELDS:
        have = [k for k in coords if f in a[k] and f in b[k]]
        d[f] = (
            sum(abs(a[k][f] - b[k][f]) for k in have) / len(have) if have else None
        )
        d[f + "_n"] = len(have)
    return d


def sigma(p1, p2, n):
    """Two-proportion sigma for the footprint-minus-floor gap.

    Rough on purpose: the two rates share the baseline arm, so they are
    correlated and this over-states the spread.  It is a sanity guide for
    reading the ratio, not a test.
    """
    v = p1 * (1 - p1) / n + p2 * (1 - p2) / n
    return v ** 0.5 if v > 0 else float("nan")


def fmt(x, w=6):
    return "     -" if x is None else f"{x:{w}.4f}"


def rate_line(label, d):
    print(f"  {label:28s} {d['top1']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="human_agreement .jsonl for the shipped eval")
    ap.add_argument("candidate", help="same corpus/budget/seed, candidate eval")
    ap.add_argument("seed_arm", nargs="?",
                    help="baseline config at a DIFFERENT seed (supplies the floor)")
    ap.add_argument("--all", action="store_true",
                    help="score every bot decision, not only the in_set ones")
    args = ap.parse_args()

    arms = [("baseline", args.baseline), ("candidate", args.candidate)]
    if args.seed_arm:
        arms.append(("seed arm", args.seed_arm))

    loaded = {}
    print("eval-candidate-screen: can a duel resolve this candidate?")
    for name, path in arms:
        rows, total, dropped = load(path, args.all)
        loaded[name] = rows
        iters = sorted({r.get("iters") for r in rows.values()})
        print(f"  {name:10s} {path}")
        print(f"             rows {total}  scorable {len(rows)}"
              f"  unscorable {dropped}  iters {iters}")

    base, cand = loaded["baseline"], loaded["candidate"]
    seed_arm = loaded.get("seed arm")

    shared = set(base) & set(cand)
    if seed_arm is not None:
        shared &= set(seed_arm)
    coords = sorted(shared)
    if not coords:
        sys.exit("no shared coordinates: are these the same corpus and battle range?")
    print(f"\npaired decisions {len(coords)}   unpaired dropped:", end="")
    for name, rows in loaded.items():
        print(f"  {name} {len(rows) - len(coords)}", end="")
    print()
    budgets = {name: sorted({r.get("iters") for r in rows.values()})
               for name, rows in loaded.items()}
    if len({tuple(v) for v in budgets.values()}) > 1:
        print("  WARNING: arms differ in `iters` -- the comparison mixes budgets")

    fp = diff(base, cand, coords)
    floor = diff(base, seed_arm, coords) if seed_arm is not None else None

    print("\n=== candidate footprint (baseline -> candidate, eval differs) ===")
    print(f"  top-1 action changed         {fp['top1']:.4f}"
          f"   ({round(fp['top1'] * fp['n'])}/{fp['n']})")
    print(f"  action class changed         {fp['class']:.4f}"
          f"   (bot_class: Physical/Special/Status/switch)")
    print(f"  move <-> switch changed      {fp['kind']:.4f}")
    for f in MASS_FIELDS:
        note = "" if fp[f] is not None else "   (field missing from an arm)"
        print(f"  {f:12s} mean |delta|   {fmt(fp[f])}{note}")

    if floor is None:
        print("\n=== noise floor ===")
        print("  UNKNOWN -- no second-seed baseline was supplied.")
        print("\nVERDICT: NONE.  A footprint without a floor is not evidence:")
        print("  the exchange term's 13.8% top-1 change rate looked like a signal")
        print("  until the seed-only rate came back 15.5%.  Re-run the baseline at")
        print("  another --seed and pass it as the third argument.")
        sys.exit(3)

    print("\n=== noise floor (baseline -> same eval, other seed) ===")
    print(f"  top-1 action changed         {floor['top1']:.4f}"
          f"   ({round(floor['top1'] * floor['n'])}/{floor['n']})")
    print(f"  action class changed         {floor['class']:.4f}")
    print(f"  move <-> switch changed      {floor['kind']:.4f}")
    for f in MASS_FIELDS:
        note = "" if floor[f] is not None else "   (field missing from an arm)"
        print(f"  {f:12s} mean |delta|   {fmt(floor[f])}{note}")

    print("\n=== footprint / floor ===")
    for key, label in (("top1", "top-1"), ("class", "action class"),
                       ("kind", "move <-> switch")) + tuple(
                           (f, f) for f in MASS_FIELDS):
        f_v, n_v = fp[key], floor[key]
        if f_v is None or n_v is None:
            print(f"  {label:16s}      -    (floor unmeasurable: field missing)")
            continue
        ratio = f_v / n_v if n_v > 0 else float("inf")
        extra = ""
        if key in ("top1", "class", "kind"):
            s = sigma(f_v, n_v, fp["n"])
            extra = f"   gap {f_v - n_v:+.4f} = {(f_v - n_v) / s:+.1f} sigma"
        print(f"  {label:16s} {ratio:5.2f}x   ({f_v:.4f} vs {n_v:.4f}){extra}")

    ratio = fp["top1"] / floor["top1"] if floor["top1"] > 0 else float("inf")
    if ratio >= RATIO_MEASURABLE and fp["top1"] >= ABS_FOOTPRINT_MIN:
        verdict, code = "MEASURABLE", 0
        why = ("most of the divergence is the candidate, not the dice: a "
               "seed-paired duel\n  and the agreement metrics can resolve this "
               "arm.  Spend the VM hours.")
    elif ratio < RATIO_UNMEASURABLE:
        verdict, code = "UNMEASURABLE", 2
        why = ("the eval change moves no more decisions than re-rolling the "
               "seed does.\n  Every behavioural instrument will read parity by "
               "construction -- a duel cannot\n  resolve this arm at any sample "
               "size a duel can afford.  Do not run one; either\n  make the "
               "change bigger or judge it on an offline statistic and say so.")
    elif fp["top1"] < ABS_FOOTPRINT_MIN:
        verdict, code = "INCONCLUSIVE", 1
        why = ("the ratio clears the floor but the absolute footprint is tiny "
               "(<5% of\n  decisions), so a duel's power comes from very few "
               "divergent games.  Widen the\n  corpus or the candidate before "
               "committing VM hours.")
    else:
        verdict, code = "INCONCLUSIVE", 1
        why = ("above the floor but not clearly: expect weak duel power.  "
               "Budget for a much\n  larger duel than the standard 800 games, "
               "or find an instrument that reads the\n  specific decisions this "
               "arm changes (see the band table below).")
    print(f"\nVERDICT: {verdict}   top-1 ratio {ratio:.2f}x"
          f"   (MEASURABLE >= {RATIO_MEASURABLE:.2f}x and footprint"
          f" >= {ABS_FOOTPRINT_MIN:.2f};"
          f" UNMEASURABLE < {RATIO_UNMEASURABLE:.2f}x)")
    print(f"  {why}")

    print("\n=== footprint by baseline confidence (top1 visit share) ===")
    print("  a candidate that only moves the flat bands is moving coin flips")
    print(f"  {'band':24s} {'n':>6s} {'cand top1':>10s} {'floor':>8s} {'ratio':>7s}"
          f" {'cand cls':>9s} {'floor cls':>10s}")
    by = defaultdict(list)
    for k in coords:
        by[band(base[k]["top1"])].append(k)
    for lo in BANDS:
        g = by.get(lo) or []
        if not g:
            continue
        f_d = diff(base, cand, g)
        n_d = diff(base, seed_arm, g)
        r = f_d["top1"] / n_d["top1"] if n_d["top1"] > 0 else float("inf")
        print(f"  {BAND_LABEL[lo]:24s} {len(g):>6d} {f_d['top1']:>10.4f}"
              f" {n_d['top1']:>8.4f} {r:>7.2f} {f_d['class']:>9.4f}"
              f" {n_d['class']:>10.4f}")

    flat = [k for k in coords if base[k]["top1"] < 0.25]
    changed = [k for k in coords if base[k]["bot"] != cand[k]["bot"]]
    changed_flat = [k for k in changed if base[k]["top1"] < 0.25]
    if changed:
        print(f"\n  of the candidate's {len(changed)} top-1 changes,"
              f" {len(changed_flat) / len(changed):.1%} sit below 0.25 confidence"
              f" ({len(flat) / len(coords):.1%} of decisions are there)")
        t1 = sum(base[k]["top1"] for k in changed) / len(changed)
        print(f"  mean baseline confidence where the candidate changed the pick:"
              f" {t1:.3f}"
              f"  (all decisions {sum(base[k]['top1'] for k in coords) / len(coords):.3f})")

    sys.exit(code)


if __name__ == "__main__":
    main()
