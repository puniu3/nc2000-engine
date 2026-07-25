# Switching question — handoff

Updated: 2026-07-25. Written to survive a session exit: there is a CX job in
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

## IN FLIGHT — pick this up first

```
CX task 20260725-212400   name switch-eq-v2   16 vCPU / 32 GB   -T 21600 (6h)
```

The first submission (`20260725-211409`) is a **false success** — ignore it. It
reported `exit 0` after 15 s having run nothing: the command was written
`PATH=... mkdir -p ... && for h in ...; do cargo ...`, and a variable-assignment
prefix applies to that ONE command, so `cargo` ran without it and died 127 each
iteration while the trailing `cp` succeeded and set the task's exit code. Two
lessons, both now in the resubmission: `export PATH=...;` rather than a prefix,
and make per-iteration failures set the task's exit code instead of letting the
last command decide it. `~/cx/README.md` already warns that cargo is not on a
non-interactive PATH; the prefix form defeats the fix.

Runs `switch_equilibrium` at `--hp 20 30 40`, `--budget 4000000`,
`--work 400000000`, `--leaf-cap 20000`, `--iters 30000`.

```bash
~/cx/cx status                       # DONE / FAIL / still RUN?
~/cx/cx logs 20260725-212400         # stdout+stderr
ls ~/cx/results/20260725-212400/out/ # hp20.txt hp30.txt hp40.txt (+ FAILURES if any arm died)
```

Read each row against the two conditions above. Expected outcomes and what to
do with each:

- **A row with `entry_gap ≈ 0` and `0 < value < 1`** — the experiment worked.
  Read `delta` and settle the question. Add more scenarios at that HP setting
  for a second and third data point before concluding.
- **All rows still bracketed** — 32 GB was not the binding constraint either.
  Do not keep raising budgets blindly; instead shrink the position further
  along an axis that is not HP. Candidates not yet tried: lower levels (less
  HP in absolute terms, so fewer damage-roll-distinct outcomes), a move pair
  with no type resistance so the damage lattice is coarser, or accepting a
  small `entry_gap` ε and reporting the mixture as approximate with the ε
  stated.
- **OOM / FAIL again** — drop `--leaf-cap` to 5000 and run one HP value per
  task so a single position's peak memory is the whole budget.

## Repo state

Committed on `master`, **not pushed** (owner: push not needed).

- `49bbb34` — `tools/switch-rate-by-skill.py`, the human-replay archaeology.
- `e0ff123` — `solve_root` + the harness + a horizon bug in `solve_root`
  (`solve` leaves `t_max` past the horizon that certified the value, so the
  re-walk re-entered unfinished territory; symptom was a tight
  `certified.width()` beside an `entry_gap` near 1.0, which cannot be true).
- `36807d1` — `--leaf-cap`.

Everything else from this session is pushed and deployed; see
`docs/M17-HANDOFF.md`.
