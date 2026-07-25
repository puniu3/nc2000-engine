#!/usr/bin/env python3
"""Do stronger humans switch MORE or LESS? — the cheap discriminator for the
M16b switching cluster.

The bot disagrees with humans on voluntary switching far more than on moves
(top-1 25% vs 39%), and two independent attempts to close that gap came back
null: the M16c rollout switch policy (2026-07-21) and the Spikes eval term
(2026-07-25), one from L2 and one from L1. So the disagreement may not be a
bot defect at all. Two readings survive:

  1. humans switch correctly and the bot is genuinely suboptimal;
  2. humans switch too much and the bot is right to switch less.

This separates them without running the bot at all: if switching is skill,
stronger players should do it more. Two views, both from the spectator logs:

  * WITHIN-GAME, paired inside each battle: winner's voluntary-switch rate vs
    loser's. Same length, same matchup, same metagame. **But it does not
    control the confound that matters: a losing player switches BECAUSE they
    are losing** — hunting a favourable matchup, retreating a weakened mon,
    stalling. The causality runs outcome -> switching, so the full-game
    version is contaminated by reverse causality and cannot be read as
    "switching loses games".
  * EARLY-WINDOW, the same pairing restricted to the opening turns, before
    either side has a decided advantage to react to. This is the cheap fix for
    that confound: whatever difference survives here is closer to style than
    to desperation.
  * CROSS-PLAYER: per-player win rate vs per-player switch rate, for players
    with enough games to rank. Noisier, but it is the version that speaks to
    "do the strong players in this pool play a switch-heavy style".

Voluntary means a `|switch|` that is not answering a faint and not the opening
send-out. Roar/Whirlwind drags arrive as `|drag|` and are already excluded.

Usage: python3 tools/switch-rate-by-skill.py [tmp/corpus-spectator] [--min-games N]
"""
import sys
import re
from collections import defaultdict
from pathlib import Path


EARLY_WINDOWS = (5, 10, 15)


def parse(path: Path):
    """-> (names, winner, per_side {turns, voluntary, forced})"""
    names = {}
    winner = None
    started = False
    faint_pending = {0: False, 1: False}
    turns = 0
    cur_turn = 0
    vol = {0: 0, 1: 0}
    forced = {0: 0, 1: 0}
    early = {w: {0: 0, 1: 0} for w in EARLY_WINDOWS}
    for ln in path.read_text(errors="replace").splitlines():
        if not ln.startswith("|"):
            continue
        f = ln.split("|")
        if len(f) < 2:
            continue
        tag = f[1]
        if tag == "player" and len(f) > 3:
            side = 0 if f[2].startswith("p1") else 1
            if f[3]:
                names[side] = f[3]
        elif tag == "turn":
            started = True
            cur_turn = int(f[2]) if f[2].isdigit() else cur_turn
            turns = max(turns, cur_turn)
        elif tag == "win" and len(f) > 2:
            winner = f[2]
        elif tag == "tie":
            winner = "__tie__"
        elif tag == "faint" and len(f) > 2:
            side = 0 if f[2].startswith("p1") else 1
            faint_pending[side] = True
        elif tag == "switch" and len(f) > 2:
            side = 0 if f[2].startswith("p1") else 1
            if not started:
                continue  # opening send-out
            if faint_pending[side]:
                faint_pending[side] = False
                forced[side] += 1
            else:
                vol[side] += 1
                for w in EARLY_WINDOWS:
                    if cur_turn <= w:
                        early[w][side] += 1
    return names, winner, turns, vol, forced, early


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/corpus-spectator")
    min_games = 10
    if "--min-games" in sys.argv:
        min_games = int(sys.argv[sys.argv.index("--min-games") + 1])

    files = sorted(root.glob("*.log"))
    if not files:
        print(f"no logs under {root}")
        return

    # within-game paired
    pairs = []          # (winner_rate, loser_rate)
    early_pairs = {w: [] for w in EARLY_WINDOWS}
    per_player = defaultdict(
        lambda: {"g": 0, "w": 0, "turns": 0, "vol": 0, "wturns": 0, "wvol": 0}
    )
    ties = decided = noname = 0

    for p in files:
        names, winner, turns, vol, forced, early = parse(p)
        if turns == 0 or len(names) < 2:
            noname += 1
            continue
        for s in (0, 1):
            st = per_player[names[s]]
            st["g"] += 1
            st["turns"] += turns
            st["vol"] += vol[s]
        if winner in (None, "__tie__"):
            ties += 1
            continue
        wside = 0 if names.get(0) == winner else (1 if names.get(1) == winner else None)
        if wside is None:
            continue
        decided += 1
        per_player[winner]["w"] += 1
        # Switch rate measured ONLY in games this player won. Everyone is then
        # scored in the same game state, so switching induced by losing cannot
        # drive the correlation -- the confound the early window leaves behind.
        per_player[winner]["wturns"] += turns
        per_player[winner]["wvol"] += vol[wside]
        lside = 1 - wside
        pairs.append((vol[wside] / turns, vol[lside] / turns))
        for w in EARLY_WINDOWS:
            denom = min(turns, w)
            if denom > 0:
                early_pairs[w].append((early[w][wside] / denom, early[w][lside] / denom))

    print(f"battles {len(files)}  decided {decided}  ties/unknown {ties}  unusable {noname}")
    print()
    print("== WITHIN-GAME (paired: winner vs loser, voluntary switches per turn) ==")
    wr = sum(a for a, _ in pairs) / len(pairs)
    lr = sum(b for _, b in pairs) / len(pairs)
    diffs = [a - b for a, b in pairs]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    wins_more = sum(1 for d in diffs if d > 0)
    loses_more = sum(1 for d in diffs if d < 0)
    print(f"  winner {wr:.4f}/turn   loser {lr:.4f}/turn")
    print(f"  paired diff (winner - loser) {mean:+.4f}  95% [{mean - 1.96 * se:+.4f}, {mean + 1.96 * se:+.4f}]")
    print(f"  winner switched more in {wins_more}/{n} battles, less in {loses_more}")
    verdict = (
        "winner switches MORE -> reading 1 (switching is skill)"
        if mean - 1.96 * se > 0
        else "winner switches LESS -> reading 2 (humans over-switch)"
        if mean + 1.96 * se < 0
        else "no detectable difference -> neither reading supported"
    )
    print(f"  => {verdict}")
    print()

    print("== EARLY-WINDOW (same pairing, opening turns only: reverse causality removed) ==")
    for w in EARLY_WINDOWS:
        ps = early_pairs[w]
        if len(ps) < 2:
            continue
        ds = [a - b for a, b in ps]
        m = sum(ds) / len(ds)
        v = sum((d - m) ** 2 for d in ds) / (len(ds) - 1)
        e = (v / len(ds)) ** 0.5
        lo, hi = m - 1.96 * e, m + 1.96 * e
        tag = "winner MORE" if lo > 0 else "winner LESS" if hi < 0 else "no difference"
        wmean = sum(a for a, _ in ps) / len(ps)
        lmean = sum(b for _, b in ps) / len(ps)
        print(
            f"  turns 1-{w:<3} n {len(ps):>4}  winner {wmean:.4f}  loser {lmean:.4f}  "
            f"diff {m:+.4f} 95% [{lo:+.4f}, {hi:+.4f}]  -> {tag}"
        )
    print()

    ranked = [
        (nm, st["w"] / st["g"], st["vol"] / st["turns"], st["g"])
        for nm, st in per_player.items()
        if st["g"] >= min_games and st["turns"] > 0
    ]
    print(f"== CROSS-PLAYER (>= {min_games} games): win rate vs switch rate ==")
    if len(ranked) < 3:
        print(f"  only {len(ranked)} players qualify — not enough to rank")
        return
    ranked.sort(key=lambda r: -r[1])
    print(f"  {'player':<16} {'games':>5} {'winrate':>8} {'sw/turn':>8}")
    for nm, w, s, g in ranked[:8]:
        print(f"  {nm:<16} {g:>5} {w:>8.3f} {s:>8.4f}")
    if len(ranked) > 16:
        print("   ...")
        for nm, w, s, g in ranked[-8:]:
            print(f"  {nm:<16} {g:>5} {w:>8.3f} {s:>8.4f}")
    wonly = [
        (nm, st["w"] / st["g"], st["wvol"] / st["wturns"], st["g"])
        for nm, st in per_player.items()
        if st["g"] >= min_games and st["wturns"] > 0
    ]
    if len(wonly) >= 3:
        ax = [r[1] for r in wonly]
        ay = [r[2] for r in wonly]
        amx, amy = sum(ax) / len(ax), sum(ay) / len(ay)
        acov = sum((x - amx) * (y - amy) for x, y in zip(ax, ay))
        asx = sum((x - amx) ** 2 for x in ax) ** 0.5
        asy = sum((y - amy) ** 2 for y in ay) ** 0.5
        ar = acov / (asx * asy) if asx > 0 and asy > 0 else float("nan")
        print(
            f"\n  WINS-ONLY (same outcome for everyone): players {len(wonly)}  "
            f"r(winrate, switch rate in won games) = {ar:+.3f}"
        )

    xs = [r[1] for r in ranked]
    ys = [r[2] for r in ranked]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    r = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    print(f"\n  players {len(ranked)}  Pearson r(winrate, switch rate) = {r:+.3f}")
    print("  (positive = stronger players switch more; a cross-player r also picks up")
    print("   team archetype, since stall teams switch more regardless of who pilots them)")


if __name__ == "__main__":
    main()
