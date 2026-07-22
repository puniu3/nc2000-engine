# Heal/stall reachability census: results

Date: 2026-07-22
Harness: `crates/bot/examples/stall_census.rs` (a9d6c4e)
Root: b455 s0 T39 — Snorlax 53/292 (doubleedge/earthquake/selfdestruct/curse)
vs Skarmory 97/171 (toxic/whirlwind/drillpeck/rest), the battle-455 stall
anchor. Note: this imputation gives Snorlax NO heal (the human's actual Rest
was not yet revealed at the reconstruction cut) — this is the same
determinization every solver/eval measurement uses, and it makes the
position a ONE-SIDED-heal stall.

## Question

Does the reachable decision-state set of a heal/stall position diverge or
converge, and where are the chokepoints for an exact solver?

## Method

BFS by joint decision steps from the root, successors via `enumerate_step`,
dedupe/expansion on a selectable quotient:

- `--quot turn`: `state_key128` with `Battle::turn` zeroed;
- `--quot proj`: structural projection of both actives (species, level, HP,
  status + Time/Counter ints, boosts, PP/disabled per slot, volatile
  ids+ints) — drops turn AND the damage-bookkeeping fields
  (`last_damage`/`attacked_by`/...) that `state_key128` hashes. Behavior-
  preserving only when neither side carries a bookkeeping observer
  (Counter/Mirror Coat/Bide/Flail/Reversal/Rage); the harness warns
  otherwise. True for this matchup.

Per layer: distinct states, quotient-recurrence hits, post-Rest states
(full-HP + asleep proxy), PP-total range, uniform-over-cells absorbed mass
(weak proxy — Selfdestruct dominates it; labeled, not strategic).

## Measured (proj quotient, 12M engine runs, frontier cap 60k)

| depth | distinct/layer | growth | post-Rest | absorbed mass cum |
|---:|---:|---:|---:|---:|
| 1 | 155 | — | 0 | 0.275 |
| 2 | 2,278 | ×14.7 | 2 | 0.555 |
| 3 | 18,830 | ×8.3 | 73 | 0.756 |
| 4 | ≥60,000 (cap) | ×≥3.2 | 443 | 0.849 PARTIAL |
| 5 | ≥60,000 (cap) | — | 167 | 0.854 PARTIAL |

Raw `state_key128` layers (turn mode, 3M runs): 310 / 9,704 / ≥30,000.

## Findings

1. **Dead-field factor ×2–×4.3.** Raw-vs-proj layer ratios: 310/155,
   9704/2278. The solver key splits states on damage bookkeeping that is
   provably unobservable in matchups without observer moves. Immediate,
   sound node-count win for `bot::bounds` node memory (guard = "no observer
   move in either moveset", checkable at the root).
2. **Turn-quotient recurrence is zero at shallow depth — necessarily.**
   PP decreases monotonically (~1/side/turn), so a state can never recur
   at a different turn with equal PP. Turn is layer-redundant GIVEN PP
   (+ bounded sleep interleavings); the unroll dimension that matters is
   PP, not turn.
3. **One-sided-heal stalls are monotone-coordinate DAGs with a tiny
   recurrent fiber.** In this anchor every dimension except the healer's
   own HP × sleep clock is monotone: both PP vectors ↓, no-heal side's HP ↓,
   Toxic counter ↑, Curse boosts ↑ (bounded). Non-monotone fiber ≈
   Skarmory-HP × slp-clock ≤ ~500 points. A solver sweeping in monotone
   order (PP, then no-heal HP, then tox) never needs a whole layer in
   memory and can free resolved generations safely — the 15k–60k/layer
   census counts are traversal volume, not resident-set size.
4. **Per-layer census grows but decelerates** (×14.7 → ×8.3 → ×≥3.2).
   Semantic space is bounded (HP lattice × bounded clocks/boosts × PP
   paths) but the transient fan is far beyond eager per-layer solving at
   depth ≥4 — consistent with the e-5 healthy-fan wall.
5. **Post-Rest compression ×3.4 (this anchor) to ×12 (turn-mode run).**
   Dropping HP from the projection collapses post-Rest states — but only
   the healer's HP resets; the opponent's HP dimension survives in
   one-sided-heal stalls. Full "Rest is a state merger" collapse needs
   both sides healing.

## Caveats

- Depth ≥4 counts are lower bounds (frontier cap + work budget).
- TWO-SIDED-heal stalls weaken finding 3: both HP dims become
  non-monotone, fiber grows toward the HP-pair lattice (~50k). Not yet
  measured — next census target.
- proj quotient soundness is per-matchup (observer-move check).

## Implications for ENDGAME-SOLVER-ALGORITHM.md

- Phase B node memory (open e-5 item): add the dead-field-quotiented key
  behind the observer-move guard, and generation-free resolved nodes by
  monotone coordinates in heal/stall class positions.
- The "PP=0 / Struggle endgame" boundary remains a valid chokepoint
  (decision-free, HP-sum-monotone) but the monotone layering above reaches
  the same effect earlier in one-sided-heal positions.
- Complementary to the merged probe-refine result (THRESHOLD-DAMAGE-
  RESULTS.md): probe-refine fixes the abstraction's policy at the root;
  the census addresses how deep exact/bounded solving can be scheduled.

## Implemented follow-up: guarded semantic key

Commit following this memo adds a certificate-domain-tagged
`NoDamageBookkeeping` key to `bot::bounds`. It omits the behavior-dead
`Battle::last_damage` and Pokemon `last_damage`/`hurt_this_turn`/
`times_attacked` fields, plus `attacked_by` only when a scan of every
roster's base and current move slots proves Counter and Mirror Coat absent.
Raw and semantic fingerprints are distinct key domains; an unsafe corpus
root can never cross-merge with a safe one when a solver is reused.

Correctness gates cover individual key equality/inequality, direct and
copied observer moves, tagged-domain isolation, and an exhaustive 2x2
turn-1000 successor/value comparison between full-key-distinct states.

Measured on the same b455 s0 T39 root with identical 200,000-run budgets:

| key | stop | interval | expansions | live nodes | wall |
|---|---|---|---:|---:|---:|
| raw | WorkExhausted | [0.003, 0.940] | 11,730 | 25,848 | 23 s |
| guarded semantic | WorkExhausted | [0.003, 0.940] | 4,156 | 22,005 | 24 s |

At a 23,000-node limit the raw key stopped at `NodeBudget` after 174,037
runs and 23,067 nodes. The guarded key completed the full 200,004 runs with
21,252 nodes and the same interval. Thus the immediate win is 64.6% fewer
expansions and avoiding the node-budget wall, not a wall-time reduction.
This is the prerequisite for closed-generation folding; hashing alone does
not tighten the healthy-stall bracket.

## Reproduction

```bash
cargo run --release -p nc2000-bot --example stall_census -- \
  --quot proj --work 12000000 --frontier-cap 60000 --max-depth 8 \
  --out tmp/stall-census-b455-s0-proj.csv
```
