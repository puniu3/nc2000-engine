# Heal/stall reachability census: results

Date: 2026-07-22
Harness: `crates/bot/examples/stall_census.rs` (a9d6c4e)
Root: b455 s0 T39 — Snorlax 53/292 @ Leftovers
(doubleedge/earthquake/selfdestruct/curse) vs Skarmory 97/171
(toxic/whirlwind/drillpeck/rest), the battle-455 stall anchor. This
imputation gives Snorlax no healing *move*, but Leftovers still restores HP;
the position is therefore TWO-SIDED healing. The earlier one-sided label was
wrong. The layer counts remain valid, but the one-sided monotonicity inference
below is superseded by the implemented classifier measurements.

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
3. **The b455 census does not prove the one-sided-heal DAG claim.** Snorlax's
   Leftovers makes its HP non-monotone alongside Skarmory's Rest. A separate,
   conservative classifier now proves the claim state-by-state: exactly one
   side may regain HP, PP never increases, move identities/max HP/active mons
   remain fixed, the non-healer's HP never increases, and every nonterminal
   edge strictly decreases `PP total + non-healer HP + turns remaining`.
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
- b455 is already a two-sided-heal case. Both HP dimensions are non-monotone;
  PP/resource/SCC treatment is still required after strategic pruning.
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

## Implemented follow-up: terminal/closed folding and monotone scheduler

Terminal successors now contribute directly to their parent cells. Exact
non-root nodes are swept into a persistent closed memo; every live edge to a
folded node is replaced by the same fixed interval contribution. No unresolved
descendant is pruned. On b455 at 200,000 runs, terminal folding reduced live
nodes 22,652→16,477 and wall 24→21 s at the same `[0.003, 0.940]` interval.
With a 17,000-node cap, closed folding completed 200,002 runs with 16,267 live
nodes; disabling it stopped at the node cap after 192,028 runs and 17,209
nodes, again with the same interval.

The one-sided-heal classifier scans active move slots and items, including
Rest, static heal/drain, weather heals, Leech Seed, Leftovers, and healing
berries; it rejects move/PP mutation and HP-transfer paths. Existing Leech
Seed follows `source_slot`, the field used by mechanics, rather than a stale
object reference from reconstruction. Every generated nonterminal edge is
checked before its rank can affect scheduling; any violation disables only
the optimization, never certification. A 570-corpus endgame coverage pass
admitted 24/72 roots with zero invalidations. On classified b51 at a 12k-node
cap, the scheduler completed the 200k work budget with 11,696 live nodes; the
ordinary scheduler hit the cap at 198k runs/12,003 nodes. The gain is modest;
closed folding is the larger memory win.

## Implemented follow-up: two-sided resource DAG

The classifier now covers the actual b455 shape: Leftovers on one side and
Rest on the other. It does not assert HP monotonicity. Instead, fixed active
ids/move ids/max HP plus componentwise nonincreasing PP and increasing turn
prove a lexicographically decreasing `(total PP, turns remaining)` rank.
Healing PP is exposed only as a scheduling coordinate. Branches that later
become one-sided are preferred, then lower healing/total PP generations.

At a 7,800-node cap, b455 with resource ordering reached the existing
evaluation-threshold proof `[0.004, 0.865]` after 56,696 runs with 7,695 live
nodes. Disabling only the two-sided resource scheduler stopped at the node
cap after 53,118 runs/7,921 nodes with `[0.004, 0.874]`, short of the proof.
The 0–100 corpus slice classified 5/7 roots as two-sided and 1/7 as one-sided,
with zero edge invalidations. No SCC pass is planned: retaining absolute turn
makes the exact graph acyclic, while dropping it would unsoundly merge states
with different distance to the turn-1000 tie.

## Reproduction

```bash
cargo run --release -p nc2000-bot --example stall_census -- \
  --quot proj --work 12000000 --frontier-cap 60000 --max-depth 8 \
  --out tmp/stall-census-b455-s0-proj.csv
```
