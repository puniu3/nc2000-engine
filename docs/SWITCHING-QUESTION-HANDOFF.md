# Switching question — handoff

Updated: 2026-07-26. Written to survive a session exit: there is a CX job in
flight whose result is the whole point.

## The question

M16b measures the bot disagreeing with humans on voluntary switching far more
than on moves — `kind=switch` top-1 **24.9%** vs `kind=move` **42.8%**. Two
readings survive:

1. humans switch correctly and the bot is genuinely suboptimal;
2. humans switch too much and the bot is right to switch less.

## What has been ruled out

**Both obvious fixes are done and null, from opposite layers.**

- **L2 rollout** — `RmConfig::rollout_m16c` (default **false**) carries
  bad-matchup voluntary switching + `status_pseudo_score`. Measured 2026-07-21:
  corpus agreement 39.3%→38.7% overall, **switches 25.0%→23.8%** (worse),
  self-play parity. Recorded diagnosis: at product budgets the tree, not the
  rollout tail, owns root values. **Do not re-attack here.**
- **L1 eval** — the Spikes term shipped 2026-07-25 is a switching tax by
  construction. Matched A/B on 141 battles, same seeds, 4,571 scorable each
  (`human_agreement --spikes 1.5` vs `--spikes 0.0`): ALL top-1 37.4%→36.9%,
  **switch 25.5%→25.9%** (+0.4pp ≈ 3 decisions on n=733). No detectable
  movement.

Side finding worth keeping: **agreement and calibration are not aligned
metrics.** The Spikes term improved calibration on three criteria and held
strength parity, yet agreement fell slightly. Optimising M16b agreement may
cost calibration. Do not treat agreement as a proxy for eval correctness.

## Instruments that cannot answer it

- **Human replays.** `tools/switch-rate-by-skill.py` (committed, runnable).
  Within-battle paired, winner vs loser voluntary switches per turn: winner
  switches **less**, −0.0130 [−0.0236, −0.0025], and it survives restriction to
  the opening turns at a stable −0.019 across the 1-5/1-10/1-15 windows, all
  three significant. But the early window does not remove the **matchup**
  confound — species are visible at preview, so a bad matchup causes both more
  switching and more losing, which is rational response, not over-switching.
  Cross-player, measured only in games each player won so everyone is scored in
  the same state: **r(winrate, switch rate) = +0.409**, the opposite direction,
  at n=20 with a CI including zero. Reconciliation, not contradiction:
  switching looks like skill as a *style* while remaining a distress signal
  *inside* a game. The corpus cannot do better — 20 rankable players, three
  heavy accounts.
- **A perturbed-bot duel** (add a switch bonus, measure strength). Rejected on
  the owner's argument, which is correct: neither outcome identifies. Weaker →
  is switching bad, or does this bot switch badly? Stronger → is switching
  good, or can the bot merely not punish switches?

## The instrument that can (owner's design)

Build a position small enough to **solve exactly to the end**, read the
**equilibrium switch probability** off the root LP, compare the bot's root
switch mass in the same position. The reference is right by construction, so a
gap is the bot's and nothing else.

Built for it:

- **`ExactSolver::solve_root`** (`crates/bot/src/exact.rs`) → `RootEquilibrium`
  with both sides' mixtures, `switch_mass(side)`, and `entry_gap`.
- **`crates/bot/examples/switch_equilibrium.rs`** — 2v2 positions where every
  mon carries exactly ONE move, so the action set is {attack, switch} and the
  root is 2x2. That removes "switched to the wrong target" and "picked the
  wrong move" as alternative explanations: with one move and one bench mon
  there is no such freedom, only the rate.

**A row means nothing unless BOTH hold:**

- `entry_gap ≈ 0` — the root payoff matrix is exact. A mixture solved off a
  bracketed matrix is not an equilibrium of anything. This is strictly stronger
  than the root value converging: the value can be pinned by dominant entries
  while the rest are still [0, 1].
- `0 < value < 1` — the position is undecided. A decided position makes *every*
  strategy optimal, so its switch mass is LP degeneracy.

Then: **delta < 0** (bot switches less than equilibrium) → reading 1, the
cluster is the bot's error. **delta ≈ 0** → reading 2, and the human switch
rate is what needs explaining.

## Why it is not answered yet — structural, not tuning

`--hp` scales every live mon's HP, and both ends fail for opposite reasons:

- **1 HP**: solves exactly (`entry_gap` 0.0000 measured) but switching concedes
  a free KO, so equilibrium never switches and the value is a decided 0.0/1.0 —
  degenerate.
- **25–60%**: the trade-off is real (the incoming mon survives its hit) but the
  tree is unsolved within budget (`entry_gap` ≈ 0.92), and 40% OOMed the 8 GB
  box.

**Nothing in between collapses chance, provably.** One hit never KOing needs
HP above CRIT damage (~2x normal); two hits always KOing needs HP at most 2x
the MINIMUM roll; max roll > min roll. So any position with a real
stay-or-retreat decision keeps chance nodes. That is what M17e is for — it just
needs more budget and RAM than this box has.

The OOM lever is **`--leaf-cap`**, not the state budget: `enumerate_step` holds
a full `Battle` clone per chance leaf, so the old 100k default is gigabytes.
Default is now 20k.

## RESULT of the first real sweep — it ran, and it answered nothing

```
CX task 20260725-221714   switch-eq-v3   16 vCPU / 32 GB   exit 0, 9818s
results in ~/cx/results/20260725-221714/out/hp{20,30,40}.txt
```

All **9 rows UNUSABLE** (`entry_gap` 0.05–0.97, none near 0). But the run
reclassifies the constraint, which is the value in it:

- **RAM was not binding.** 32 GB held at `--leaf-cap 20000`; no OOM, no FAIL.
- **Wall clock was not binding.** 9818 s of a 21600 s cap.
- **The `--budget`/`--work` caps were.** Every arm terminated at its cap with
  the matrix still bracketed. So "throw a bigger box at it" is already spent as
  a move; the handoff's own advice against blindly raising budgets stands.

**The real structure: tractability and undecidedness are anti-correlated across
SCENARIOS, not along HP.**

| scenario | gap @20/30/40 | value |
|---|---|---|
| electric/ground vs water/grass | 0.336 / 0.335 / **0.050** | **0.0000 at all three** — decided |
| ice/fighting vs dragon/psychic | 0.289 / 0.239 / 0.173 | 0.29 / 0.94 / **0.32** |
| fire/water vs grass/water | 0.725 / 0.968 / 0.766 | 0.94 / 1.00 / 0.62 |

The one position that nearly solves is a certain loss for p1, so its mixture is
LP degeneracy; the positions that are genuinely undecided are the intractable
ones. The two conditions fight each other along the axes now exposed — HP scale
is not the lever, and neither is compute.

Closest to admissible, and worth stating precisely because it is NOT an answer:
`ice/fighting vs dragon/psychic` at 40% HP — value 0.3158 (undecided),
`entry_gap` 0.1734, `eq_sw` 0.840 vs `bot_sw` 0.222, **delta −0.618**. Both
undecided rows in the sweep have delta < 0 (−0.372, −0.618) and every positive
delta sits in a decided row. That is a hint toward reading 1, on a bracketed
matrix, at n=2 — do not record it as evidence.

Side observation: at 20/30% HP the bot puts 0.36–0.46 root mass on switch in
positions certified 0.0000. Every strategy is optimal there so it is not an
error, but the bot plainly does not know it is lost.

## IN FLIGHT — the HP-skew ladder

```
CX task 20260726-051101   switch-eq-v5-ladder   16 vCPU / 32 GB   -T 21600 (6h)
results land in ~/cx/results/20260726-051101/out/p<h1>_<h2>.txt
```

### v4 was preempted, and its timings redesigned the ladder

`20260726-011409` (same six arms) got Spot-preempted 3.5 h into its last arm and
returned **no rows** — `cp` into `$CX_OUT` only ran after the loop. Its per-arm
start markers survive in `~/cx/results/20260726-011409/stdout` and are worth more
than the rows would have been:

| arm | wall | |
|---|---|---|
| `p40_40` | 3 m 48 s | |
| `p40_36` | 2 m 14 s | weakening p2 shrinks the tree, monotonically |
| `p40_32` | 1 m 18 s | |
| `p40_28` | 36 s | |
| `p48_40` | 16 m 00 s | strengthening p1 grows it |
| `p56_40` | **> 3 h 30 m, unfinished** | and then explodes |

So the two directions are not symmetric in cost, and the v3 estimate of ~18 min
per position was wrong in both directions. Three changes follow, all in v5:

- **Each arm is wrapped in `timeout 1800`**, so one deep position cannot eat the
  task's 6 h — that is exactly how v4 died.
- **`cp` into `$CX_OUT` after every arm**, not once at the end, so a preemption
  costs one arm instead of all of them.
- **The p2 ladder is refined to steps of 2** (40:40 → 40:24) since those arms
  cost 1-4 minutes each, and the p1 direction is cut back to 44 and 48. The
  worry the timings raise: the fast arms may be fast because the position went
  *decided the other way* (a certified 1.0 for p1), skipping over the undecided
  band — a fine ladder is what catches the crossing if it is narrow.

The lever: keep the scenario that nearly solves and remove the reason it is
degenerate. `scale_hp` scaled BOTH sides uniformly, so "p1 is simply losing"
was unreachable by `--hp`. `ecffb96` adds `--hp1`/`--hp2` and a `--scenario`
filter, so an arm spends its whole budget on one position instead of another
uniform pass over three.

Eleven arms, sequentially, all on `electric/ground vs water/grass`, everything
else identical to v3 (`--iters 30000 --budget 4000000 --work 400000000
--leaf-cap 20000`, seed 1):

| arm | p1 % | p2 % | what it tests |
|---|---|---|---|
| `p40_40` | 40 | 40 | baseline repro — must reproduce v3's `gap 0.0500 / value 0.0000` |
| `p40_38` … `p40_24` | 40 | 38, 36, 34, 32, 30, 28, 26, 24 | weaken p2: value rises off 0 AND the tree shrinks |
| `p44_40` / `p48_40` | 44 / 48 | 40 | strengthen p1: value rises, tree grows |

Both ladders raise p1's value, so the crossing into `0 < value < 1` should be
bracketed from two directions; lowering p2 is the preferred direction because it
also shortens the game. Each arm is wrapped in `/usr/bin/time -v`, so the arm
files carry peak RSS — v3 never measured it, and knowing it is what decides
whether future arms can run concurrently (the solver is single-threaded, so 16
vCPU is otherwise idle; RAM is the only reason not to).

Why the position is degenerate in the first place, worth knowing before reading
the rows: p1 is Electabuzz(Thunderbolt)/Sandslash(Earthquake) against
Quagsire(Surf)/Victreebel(Giga Drain). Thunderbolt is a **0x** immunity into
Quagsire's Ground, and both foes hit Sandslash for 2x. That one-sidedness is
also why it solves — the branching is small precisely because half of p1's
options do nothing.

Reading order when it lands: `p40_40` first (if the baseline does not reproduce,
the refactor moved the position and nothing else in the file is comparable), then
the ladder for the first row with `entry_gap ≈ 0` and `0 < value < 1`.

Second candidate axis if the skew does not produce a usable row: lower levels
(less absolute HP, so fewer roll-distinct outcomes).

### Submission postmortem — cx hygiene, read before submitting anything

**Two dead submissions precede v3; both died in under 20 s, neither is a
result.** Ignore `20260725-211409` and `20260725-212400`.

- `-211409` is a **false success** — `exit 0` after 15 s having run nothing. The
  command was `PATH=... mkdir -p ... && for h in ...; do cargo ...`, and a
  variable-assignment prefix applies to that ONE command, so `cargo` ran without
  it and died 127 each iteration while the trailing `cp` succeeded and set the
  task's exit code. Fixed by `export PATH=...;` rather than a prefix, plus an
  `rc=1` per failing iteration so the task's exit code is not decided by its
  last command.
- `-212400` died `exit 1`, all three arms `exit 101`:
  `could not find Cargo.toml in ~/ws/a1a35629b92a`. The PATH fix worked; the
  submission was simply made from the wrong cwd, and `cx` takes the **source dir
  from cwd** — so it synced `~/agents`, which has no crate. v3 passes
  `-d /home/puniu/nc2000-engine` explicitly and asserts `test -f Cargo.toml`
  (exit 66) before the loop.

Lesson for any future `cx` submission from this repo: pass `-d` explicitly and
make the command fail loudly on its own preconditions. Three of the last four
failures were 30-second environment deaths, not experiments.

v3's arms were `--hp 20 30 40`, `--budget 4000000`, `--work 400000000`,
`--leaf-cap 20000`, `--iters 30000`; the results are read above.

```bash
~/cx/cx status                       # DONE / FAIL / still RUN?
~/cx/cx logs 20260725-221714         # stdout+stderr
ls ~/cx/results/20260725-221714/out/ # hp20.txt hp30.txt hp40.txt (+ FAILURES if any arm died)
```

Still-untried axes beyond the per-side HP scale proposed above: a move pair with
no type resistance, so the damage lattice is coarser; or accepting a small
`entry_gap` ε and reporting the mixture as approximate with the ε stated. If a
future arm OOMs rather than exhausting budget, drop `--leaf-cap` to 5000 and run
one HP value per task so one position's peak memory is the whole box.

## Repo state

Committed on `master`, **not pushed** (owner: push not needed).

- `49bbb34` — `tools/switch-rate-by-skill.py`, the human-replay archaeology.
- `e0ff123` — `solve_root` + the harness + a horizon bug in `solve_root`
  (`solve` leaves `t_max` past the horizon that certified the value, so the
  re-walk re-entered unfinished territory; symptom was a tight
  `certified.width()` beside an `entry_gap` near 1.0, which cannot be true).
- `36807d1` — `--leaf-cap`.
- `de193dc` — `reconstruct_context_with_cfg` + `human_agreement --spikes`, the
  instrument behind the L1 A/B above. It was still uncommitted when the previous
  session exited.

Everything else from this session is pushed and deployed; see
`docs/M17-HANDOFF.md`.
