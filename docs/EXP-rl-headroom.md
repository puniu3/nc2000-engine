# EXP — RL headroom: does the search have a usable policy-prior slot?

Owner question (2026-08-14): *is there a prospect of strengthening the bot
further with RL?*

This is the measurement that comes before any learning machinery. It does not
train anything. A learned policy net is, mechanically, "a cheap
position-dependent action prior evaluated at every node", and a learned value
net is "a leaf value that replaces the truncated rollout". So both halves of
the RL proposal can be probed with hand-built stand-ins **before** paying for
data generation, training, and wasm inference:

- **Does the prior slot transmit at all?** Bound it from below with an
  adversarial prior. If a deliberately bad prior cannot hurt the bot, no
  learned prior can help it.
- **Does the best available prior information help?** Use the repo's own
  action-quality heuristic (`expected_hit_fraction` + M16c
  `status_pseudo_score`), which is the strongest cheap prior in-tree.
- **Does the M16b cluster-2 direction help?** Push the prior toward status /
  multi-turn plans, the one disagreement cluster still open.
- **How does any effect scale with budget?** The product runs 30,000
  iterations. An effect that search finds on its own by 300 is worthless.
- **What would a learned value have to beat?** Compare the shipped 8-turn
  truncated rollout against an eval-only leaf at equal iterations and equal
  wall clock.

## Scope caveat, stated up front

**Positions and opponents are self-play.** The 570-battle human spectator
corpus is not redistributable (`.gitignore`: real player handles), so
`human_agreement`'s corpus arm could not run and no human-agreement number
appears here. M16a's warning applies in full: the eval correlates 0.580 on
corpus positions against ~0.78 on self-play ones, so self-play is the
distribution the bot's own blind spots produced. Every "no gain" below is a
no-gain **against itself**.

## Machinery (default-off, shipped inert)

- `smmcts::PriorKind` = `Off | Uniform | Greedy | Inverted`, plus
  `RmConfig::{puct, prior_tau, prior_status_bonus}`.
- `select_puct` — `Q(a) + puct·P(a)·√N/(1+n(a))`, FPU = the side's visited
  mean. It deliberately drops UCB1's untried-first rule: *which* action
  deserves a first visit is exactly the prior's claim, so untried-first would
  erase the effect under test.
- `mcts::action_scores` — the prior's scores. Switches take the mean of the
  move scores, so the prior never shifts move-vs-switch mass (M16b cluster 1
  closed voluntary switching as "not a defect — do not optimise switch
  top-1", and re-opening it here would confound the probe).
- `examples/prior_probe.rs` — footprint-vs-seed-floor screen and action-class
  mechanism.
- Arena specs `skuctp:ITERS:PUCT:KIND[:TAU[:STATUS_BONUS[:BUCKETS]]]` and
  `skuctv:ITERS:TURNS`.

`PriorKind::Off` is the shipped rule bit-for-bit — `tests/prior.rs` traces
whole games move-for-move against a config whose (unread) prior fields are set
to values that would change play if the slot ever leaked. Independently:
`arena skuct:300 maxdamage --games 20 --seed 1` still returns **14W/6L**, the
figure recorded in the M15a README entry.

## Results

All duels seed-paired, agent seeds derived from the game index only, so the
score is thread-count invariant. `±` is the 95% CI over side-swapped blocks.

### 1. The slot transmits — hard

Adversarial prior vs shipped `skuct:1000`, 100 games, seed 7:

| c_puct | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|
| inverted prior | 0.130 ±0.067 | 0.080 ±0.058 | 0.060 ±0.045 | 0.030 ±0.033 |

Monotone in c_puct, down to 0.03. The control — the same PUCT rule on a
**uniform** prior — is neutral at every constant (0.520 / 0.560 / 0.480 /
0.540, all straddling 0.5) at identical think time, so this is the prior's
content talking, not the rule change.

**Reading: a policy prior can steer this search almost arbitrarily. Nothing
structural blocks a learned policy from mattering.**

### 2. The best available prior information is worth zero

Informed prior (damage + M16c status pseudo-scores) vs uniform prior, same
c_puct, 800 games, fresh seed 21, 1,000 iterations:

| c_puct | 1.0 | 2.0 |
|---|---|---|
| greedy vs uniform | 0.511 ±0.028 | 0.507 ±0.025 |

Parity, at tight intervals, at identical think time — while changing behaviour
enormously (average game length 30 → 20 turns).

### 3. …and what it *does* change is the wrong direction

`prior_probe`, 30 self-play games/arm, 1,000 iterations:

| arm | damage | status | switch |
|---|---|---|---|
| baseline skuct | 0.654 | 0.184 | 0.161 |
| PUCT uniform prior | 0.639 | 0.140 | 0.221 |
| PUCT greedy prior | 0.792 | **0.047** | 0.161 |

The "informed" prior is a **damage amplifier**: it cuts status-move play from
18.4% to 4.7%, and the footprint's class shifts run Status→Damage 43 with
Damage→Status outside the top six. Cause: `expected_hit_fraction` runs
0.25–0.5 where `status_pseudo_score` tops out at 0.30 and is 0.0 for every
move it does not name — Double Team, Substitute, Toxic, Perish Song, Mean
Look, i.e. most of M16b cluster 2.

Footprint screen, 300 self-play positions: candidate top-1 change rate
**0.540** against a seed-flip floor of **0.357** = **1.51x the floor**. Unlike
every additive eval term this repo has screened, a prior *is* resolvable by
duel — and the duel resolved it at parity.

### 4. The cluster-2 direction is monotonically harmful

Flat prior bonus on every Status-category move, vs uniform prior, 300
iterations, 400 games, seed 21:

| status bonus | 0.0 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|
| score vs uniform | 0.495 ±0.029 | 0.472 ±0.043 | 0.446 ±0.039 | **0.254 ±0.040** |
| avg turns | 22.8 | 25.1 | 33.0 | 43.2 |

Steering the tree toward multi-turn plans loses monotonically and stalls the
games out. This does **not** show the human plans are bad — it shows a flat
class-level push toward them is, and that the M16b cluster-2 gap is not
convertible by re-weighting the action class. Same shape as cluster 1's
resolution.

### 5. The one place the prior pays is below the budget the product uses

Informed prior vs uniform prior, c_puct 2.0, 800 games, seed 21:

| iterations | 100 | 300 | 1,000 | 3,000 |
|---|---|---|---|---|
| score | **0.576 ±0.029** | 0.495 ±0.029 | 0.507 ±0.025 | **0.453 ±0.028** |

The prior is worth about +53 Elo at 100 iterations, is entirely gone by 300,
and by 3,000 it is a **liability** (upper CI bound 0.481, below parity). The
shipped Web game runs **30,000** plus ponder.

**This is the load-bearing number.** The prior's information is not wrong, it
is *shallow*: 300 iterations of search already find it, and past that the
prior's narrowing of the tree costs more than its ranking is worth. For a
learned prior to pay at the product's budget it would have to carry
information 30,000 iterations cannot find — two orders of magnitude past where
this one's value went negative.

### 6. Leaf value: the opposite trend

Eval-only leaf (`turns=0`) vs the shipped 8-turn truncated rollout. At equal
iterations the eval-only side scores 0.338 ±0.042 (1,000 iters, 400 games)
but costs half as much per move (25.1 vs 52.2 ms), so the honest comparison is
equal wall clock — eval-only at 2x the iterations, think time matched to ~1%:

| rung | 600:0 vs 300:8 | 2000:0 vs 1000:8 |
|---|---|---|
| score | 0.477 ±0.045 | **0.385 ±0.043** |
| think ms/move | 14.2 vs 14.6 | 51.5 vs 50.9 |

The rollout is worth **+0.115 ≈ +81 Elo** at the 1,000-iteration rung, and its
advantage **grows** with budget (near-parity at the 300 rung). This is the
mirror image of the prior's curve.

### 7. Where the leaf's value actually lives: lookahead, not the evaluator

Two measurements, added after the owner asked whether `eval01` is on the
critical path at all.

**Leaf provenance** (`--features leafstats`, shipped `skuct:1000` self-play,
20 games, 874k leaf evaluations):

| leaf produced by | share |
|---|---|
| terminal outcome — the rollout played the battle out | **0.701** |
| 8-turn truncation → `eval01` | 0.299 |
| tree turn cap → `eval01` | 0.000 |

The rollout starts at a tree leaf, which deepens as the tree grows, and
ε-greedy max-damage play ends battles fast — so 70% of the time the "leaf
value" is a real game result, not an estimate. `eval01` is consulted on the
remaining 30%.

**Eval degradation** — strip `EvalWeights` to HP + alive (no threat, no status
penalties, no boosts, no PP, no Spikes/Substitute, no sleep clock), everything
else identical. This is the leaf-side twin of the inverted prior:

| arm | 300 iters | 1,000 iters |
|---|---|---|
| HP-only vs shipped eval | 0.496 ±0.043 | 0.480 ±0.042 |
| threat term removed only | — | 0.485 ±0.041 |

**Every feature `eval.rs` has accumulated beyond "HP + alive" is worth no
measurable strength.** Not a footprint problem — play visibly changes (average
game length moves to 44.4 turns at 300) — the changed decisions simply do not
convert.

This is the mechanism behind the repo's standing finding that "the bot's
confident choices are invariant to the eval's formulation". `eval01` is asked
an **easy question, rarely**: 30% of leaves, and those leaves sit 8 turns
downstream of the tree, where HP and how many mons are still alive already
carry nearly all the signal. Improving the answer to an easy question buys
nothing.

It also sharpens §6. The +81 Elo there is the value of the **lookahead** — 8
turns of simulation that reach a terminal state 70% of the time — and not the
value of the static evaluator sitting behind it. A learned value competing for
that slot is not competing against `eval01`; it is competing against actually
playing the game out.

### 8. Is the eval slot the *stall* slot? (owner hypothesis, 2026-08-14)

The natural objection to §7: `eval01` may be consulted mainly in heal-stalled
positions, so an average-effect measurement could be null while the eval
matters a great deal inside that stratum. Two parts, and the hypothesis is
directionally right but the prize is small.

**Is the cutoff stratum stall-shaped?** Partly. Over the same 874k baseline
leaves, splitting by whether any living mon on either side still has a
recovery move with PP or Leftovers:

| | recovery live | mean turn |
|---|---|---|
| terminal leaves | 0.440 | 18.8 |
| cutoff leaves (`eval01`) | **0.602** | **16.1** |

A real 1.93x odds enrichment — but 40% of eval leaves have no recovery at all,
so "ばかり" overstates it. And the mean turn runs the other way: eval leaves
fire *earlier*, because a cutoff means the rollout **started shallow**, not
that the game dragged.

**Does eval quality convert inside that stratum?** Stratification is a-priori
and team-level, using the arena's existing `--heal-min 4` (preview picks 3 of
6, so 4 healers guarantee every legal triple carries one; 8 of the 32 meta
teams qualify). Filtering on realised game length would be post-treatment
selection — length is affected by the thing under test.

| arm (1,000 iters) | fixtures | full meta | stall pool |
|---|---|---|---|
| HP+alive vs shipped | 0.480 ±0.042 | 0.505 ±0.037 | 0.513 ±0.039 |
| **constant 0.5** vs shipped | — | 0.482 ±0.051 | **0.453 ±0.046** |

The second row is the eval-side twin of the inverted prior: every weight zero,
so `eval01` returns a constant and the 30% of leaves that call it carry **no
information at all**. That costs 0.047 in the stall pool and is not resolvable
from zero on the full pool.

So the whole eval channel is worth ~0.047 (≈33 Elo) where it matters most —
and **HP+alive already collects essentially all of it** (0.513, i.e. no loss
against shipped). The curve flat → HP+alive → shipped saturates immediately.
The hypothesis is right that the effect concentrates in stall play; it is the
size that kills it, not the location.

Strictly, flat→shipped bounds the channel from below, not shipped→perfect from
above. But two points on a curve that has already flattened, applied 8 turns
downstream and then averaged over hundreds of rollouts per root action, leave
very little room for an oracle evaluator to find above shipped **in this
slot**.

And for stall positions specifically the repo already owns the right
instrument, and it is not a network: M17e's **exact endgame solver** with
certified bounds (Phase C product use still gated). A heal war is a long
resource race — PP and Toxic counters decided dozens of turns out. That wants
exact depth, not a better static guess.

### 9. The endgame solver: does exactness convert? (owner question)

§8 ended by pointing at M17e's certified solver as the right instrument for
stall play. That pointer needed testing too. `examples/solver_reach.rs` walks
shipped-`skuct` self-play games and, at each decision point, asks
`BoundSolver` whether it can certify the root inside a fixed budget. Positions
are the ones the product actually reaches, and the solver is handed the true
full-information state — which is what the shipped open-team-sheet product has
once everything has been revealed.

**Reach and cost.** Ungated, work budget 30,000 engine runs (5 games, 210
decisions):

| mons left | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|
| certify rate | 0.000 | 0.000 | 0.000 | 0.061 | 0.222 |
| ms if it fails | 2154 | 1959 | 2139 | 1738 | 1499 |

4.3% of decisions certified, and each failure burns ~2 s. Raising the budget to
200,000 and gating to ≤3 mons remaining (the only region with any hit rate)
lifts reach to 18.2% — at **3.4 s per success and 16.3 s per failure**, on a
machine ~18x faster than the README's baseline and far faster than the
certified iPad the product's 2.3 s/move budget was set on.

**And then the finding that settles it.** Every certified value:

```
0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 1.00 1.00 1.00 1.00 1.00
```

**14 of 14 already decided** (0/14 have 0.5 inside the bracket). The solver
certifies exactly when the position has collapsed to a forced win — which is
structural, not incidental: a subgame is cheap to prove precisely when the
branching has died, and it has died precisely when the outcome is no longer in
doubt. Contested positions are the ones that do not collapse, so they are the
ones the solver cannot afford.

That is the opposite of the shape a strength gain needs. Exactness is
available where it changes nothing, and unavailable where it would change
something. It corroborates M17e's own artifact from the other direction: 72
eligible rows across all 570 corpus battles, against ~20,719 valid decision
rows.

**Not measured:** the direct win-rate delta of a hybrid agent (solver when it
certifies, `skuct` otherwise). Given 14/14 certified positions sitting at
0.99+, such an agent would be paying seconds per move to confirm moves the
search already plays, so it was not built. The residual case for it is narrow —
a bot that blunders a 0.99 position back to contention — and it is not worth
3.4–16 s per decision to insure against.

**Where the solver still earns its keep** is exactly where it already is: as
an offline oracle (the M17e anchor gate, `endgame_exactness_corpus`) that
proves the eval wrong on certified rows. That is measurement infrastructure,
not a player.

## Verdict

**The two halves of the RL proposal do not have the same prospects, and the
measurement separates them cleanly.**

*Policy net — negative, and the evidence is strong.* The slot transmits
(adversarial prior: 0.5 → 0.03), so nothing structural stands in the way. But
the best prior information available in-tree is worth 0.507/0.511 at 1,000
iterations, the M16b cluster-2 direction is monotonically harmful (0.254 at
bonus 0.8), and the whole channel's value is negative by 3,000 iterations
against a product that runs 30,000. A learned policy would have to be better
than the heuristic by enough to survive a budget regime where the heuristic's
entire contribution has already gone negative. This is the third independent
instance of the repo's standing finding — SPSA weights, additive eval terms,
and now the action prior — and unlike the first two it is *not* a
measurability problem: the prior's footprint is 1.51x the seed floor, well
inside duel resolution. It was measured, and it lost.

*Value net — the live target.* The leaf-value channel carries +81 Elo at the
1,000-iteration rung, and unlike the prior its value increases with budget.
That is the channel a learned value would occupy. What this experiment does
**not** show is that a learned value would beat the current leaf; it shows the
channel matters and fixes the bar precisely:

- beat **8-turn ε-greedy rollout + `eval01`**, not `eval01` (the eval alone
  loses by 81 Elo — the rollout is not a weak baseline, and §7 shows why: 70%
  of its returns are real game outcomes);
- fit in **~27 µs/iteration** on this machine (52.2 − 25.1 ms per 1,000
  iterations), i.e. about half the current per-iteration cost, or it pays for
  itself in lost iterations;
- and survive the extrapolation from 1,000 to 30,000, which is measured only
  in trend direction here (favourable) and not at the product budget.

*And the corollary from §7, which is the sharpest practical rule here:*
**incremental eval accuracy is worthless, and this is now measured rather than
inferred.** Gutting `eval01` to HP+alive costs nothing, so no amount of making
it more accurate — by RL, regression, or hand — pays while it stays in its
current slot, where it is asked an easy question on 30% of leaves. A learned
value is worth building only as the thing that **deletes the rollout**, which
moves it to 100% of leaves and asks it the hard question. Half-measures in
this slot are provably null.

§8 adds the bound: even an oracle evaluator in the current slot is playing for
about 33 Elo in the stall stratum and roughly nothing elsewhere, because a
constant leaf only costs that much. The slot, not the accuracy, is the
binding constraint.

§9 closes the endgame-solver route the previous section pointed at: it
certifies 4.3% of decisions ungated, and every position it can certify is
already won. Exactness is cheap only where the answer no longer matters.

*Recommended order if this line is opened:* value first, policy not at all.
Start from `eval_calibration`'s existing labelling path (GT `skuct:300`, 32
playouts) — that is already a value-net training-data generator — and treat
`skuctv` as the standing baseline to beat. Do not start with behavioural
cloning of the human corpus: the one direction the corpus nominated
(cluster 2, more multi-turn plans) is the direction that measured worst here.

*Standing caveat:* every number is self-play. The corpus arm could not run,
and a policy prior that loses in self-play could still be the one that reads
as competent to a human — which is the criterion the Spikes term shipped on.
Nothing here is evidence about play against humans.

## Reproduction

```
cargo test --release -p nc2000-bot --test prior
cargo run --release -p nc2000-bot --example arena -- \
    skuctp:1000:2.0:inverted skuct:1000 --games 100 --seed 7
cargo run --release -p nc2000-bot --example arena -- \
    skuctp:1000:2.0:greedy skuctp:1000:2.0:uniform --games 800 --seed 21
cargo run --release -p nc2000-bot --example arena -- \
    skuctp:300:2.0:greedy:0.15:0.8 skuctp:300:2.0:uniform --games 400 --seed 21
cargo run --release -p nc2000-bot --example prior_probe -- \
    --games 30 --iters 1000 --puct 2.0 --seed 5 --positions 300
cargo run --release -p nc2000-bot --example arena -- \
    skuctv:2000:0 skuctv:1000:8 --games 400 --seed 21
cargo run --release -p nc2000-bot --example arena -- \
    skuctw:1000:hponly skuctw:1000:shipped --games 400 --seed 21
cargo run --release -p nc2000-bot --features leafstats --example prior_probe -- \
    --games 20 --iters 1000 --seed 5 --positions 1 --arms base
```

Machine: 4 vCPU container, `skuct:300` = 15.1 ms/move (the README's baseline
machine measures 280 ms for the same spec, so wall-clock figures here are not
comparable to the ones quoted elsewhere in the repo; scores and think-time
*ratios* are).
