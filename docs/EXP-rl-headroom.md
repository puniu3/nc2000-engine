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
  loses by 81 Elo — the rollout is not a weak baseline);
- fit in **~27 µs/iteration** on this machine (52.2 − 25.1 ms per 1,000
  iterations), i.e. about half the current per-iteration cost, or it pays for
  itself in lost iterations;
- and survive the extrapolation from 1,000 to 30,000, which is measured only
  in trend direction here (favourable) and not at the product budget.

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
```

Machine: 4 vCPU container, `skuct:300` = 15.1 ms/move (the README's baseline
machine measures 280 ms for the same spec, so wall-clock figures here are not
comparable to the ones quoted elsewhere in the repo; scores and think-time
*ratios* are).
