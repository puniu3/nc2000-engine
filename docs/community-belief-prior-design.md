# Community-Data Belief Prior — Design Plan

Status: **PLANNED, NOT STARTED.** Stretch goal, sequenced strictly *after*
open-sheet play is declared complete. Candidate M18. Additive to the M10/M15
hidden-set substrate; refines the M15 "community belief prior" idea
(`data/community-rentals-v0/ASSESSMENT.md`) from full-set/uniform to
per-move marginal.

Audience: AI agents / the owner, strengthening genuinely-hidden-set play
(real ladder / hidden-team servers) after M17.

## Executive summary

Ship the **code** (belief interpreter + imputation behavior) once, bug-free
and frozen; push the **data** (the construction prior) out to plain,
community-editable files. The prior is per-`(species, move)` marginal
probabilities — "Machamp knows Encore with p=0.20". New off-meta "landmine"
teams are handled by editing that data, never by touching code. The whole
scheme is safe because **certified code dominates the data**: revealed facts,
legality, and HP imputation are code and always override the prior, so the
worst any data edit can do is play suboptimally against *unrevealed* sets — it
can never cause a crash or a tactically-visible blunder.

Investigation (2026-07-23) found the substrate already built and duel-tested;
the delta is roughly **one data file + one function** (`fallback_set`) plus a
verification gate. No search or engine changes.

## Scope & premise

- **Post-open-sheet stretch.** Open-sheet is the certified product and stays
  the completion line; this is strictly additive and must not degrade it.
- **Local execution, owner-trusted data.** The bot runs on the user's machine
  and eats whatever data that user chooses to load (a shared community file is
  an opt-in download). Contributors are **well-intentioned and non-technical**.
  There is therefore **no adversarial-data threat model** — only malformed-data
  robustness (don't crash on a typo). Data poisoning is self-harm on a local
  box and is out of scope; shared-store integrity is the community admin's
  problem, not this code's.
- **Goal:** handle hidden opponent teams without *tactically-detectable*
  blunders, while **accepting strategic exploitability by landmine teams** as a
  declared property.

## The two failure standards (the completion bar)

Everything hinges on splitting "detectable failure" into two standards:

- **Class A — tactical blunder.** A single-decision error a competent observer
  *sharing the same information horizon* can point to: staying into a
  guaranteed KO with a free switch available; **ignoring an already-revealed
  move**; refusing a guaranteed kill. Cheap to detect (one game, shallow
  oracle). **Ship bar = zero class-A.**
- **Class B — strategic exploitability.** A repeat adversary models the
  deterministic policy and farms it (landmine baiting). Detectable only over
  many games with a seed-marginal oracle. **Accepted here as a property.**

Key line: *being surprised by hidden info you could not have known is class B*
(even a Nash player loses to specific hidden sets) — **NOT** class A — **as long
as you do not keep walking into it after it is revealed.** Losing to a landmine
is fine; re-stepping the same landmine after the reveal is a class-A bug. That
boundary is exactly what the reveal-dominance invariant (below) enforces.

## Why this is a coherent ship line (the open-sheet asymmetry)

- **Open sheet** has no persistent hidden state → each turn is a
  simultaneous-move matrix game → local unexploitability (class B) is
  *tractable and certifiable* via the existing RM+ root solve + endgame solver.
  That is precisely why open-sheet is the completion line.
- **Hidden teams** reintroduce a belief-over-team → class-B unexploitability
  collapses to CFR-scale and is not affordable. So B is *deliberately dropped*
  in this regime; only A is defended.
- Graceful recovery: a hidden battle *reveals progressively* and converges to
  open sheet, so the class-B guarantee is lost mainly in the **opening** and
  the certified core reasserts as reveals accumulate.

## Mechanism / policy split (the core architecture)

- **Code = certified invariants.** Frozen, bug-free, shipped once.
- **Data = the belief prior.** Plain declarative files, community-tunable,
  adapting to meta drift / new landmines with **no code change or redeploy**.
- **Safety rests on ONE invariant — certified code dominates the data.**
  Revealed facts, legality, and HP/PP/status imputation are code and *always*
  win over the prior. Consequence: the worst any data edit can do is class-B
  suboptimality against *unrevealed* slots; it can never reach the class-A
  surface or crash. **This is what makes hand-edited data safe without review.**
- Under the well-intentioned-user premise this reduces further to: a **total,
  malformed-robust interpreter** — clamp/normalize probabilities, ignore
  unknown keys, fall back to a built-in default on a missing/garbage entry,
  never crash on a typo. No adversarial hardening.

## Model B: belief representation & consumption

- **Data = per-`(species, move)` marginal probabilities** (optionally
  per-species item marginals, lead probability). Declarative numbers only.
  **No correlations, no conditional logic.** Dropping correlations costs only
  the *quality of the guess about still-unrevealed moves after a partial
  reveal* — squarely inside accepted class-B — and keeps the format dumb enough
  for a non-technical human to edit. (E.g. revealing Curse cannot lower the
  Encore marginal; that Bayesian coupling is knowingly forgone.)
- **Open-sheet is the degenerate case where the 4 moves sit at p=1.0.** One
  unified belief; revealed information is just marginals pinned to 1.0 / 0.0.
- **Consumption = sampled determinization (ISMCTS), NOT search-layer
  weighting.** The engine is a pure `(state, action) → state`; opponent
  uncertainty *must* resolve to a concrete state before a step. So Model B is
  realized by **sampling a concrete moveset per determinization from the
  marginals** and letting the already-live per-iteration ISMCTS average over
  samples = **B in expectation**. Explicit matrix-column weighting is neither
  needed nor expressible inside the engine step — do not build it.

## Codebase reality (investigated 2026-07-23)

The substrate is already built and duel-tested; the user's unification is the
*implemented* design, not something to construct.

- **`Belief`** (`crates/bot/src/belief.rs`): `pinned_from_battle` (belief.rs:248,
  = 100% / open-sheet) and `with_fallback_policy` (belief.rs:186, = hidden /
  custom team) **both feed the same `determinize` → search**. `BlindAgent`
  (hidden) and `OpenAgent` (pinned) share `search_choose` bit-identically
  (`crates/bot/src/blind.rs:158`).
- **`determinize`** overwrites *only hidden fields* → open-sheet is a no-op, so
  "100% known" is free. `≤4`-move clamp already present (belief.rs:636).
  Spreads are materialized at format-norm (`base_set`) — marginals govern
  **move identity only**; the spread stays the residual hidden layer, as it is
  in the real protocol (foe HP arrives at 1/48; DVs/EVs never revealed).
- **ISMCTS is live-wired**: `BlindSearch::step_one` calls `determinize` per
  iteration (`blind.rs:325`, `:384`); the search (`SkuctSearch`, `smmcts.rs`)
  is determinization-agnostic — it solves per-node RM+ matrix games on the
  concrete battle. Sampling many determinizations and averaging ≈ the
  expectation over the belief = Model B in expectation.
- **Reveal-integration** = `Belief::sync` (belief.rs:267) + "revealed moves
  first" in `fallback_set` (belief.rs:615). **This is the load-bearing
  reveal-dominance invariant, and the known "imputed moveset drops a revealed
  move (Spikes)" bug lived exactly here.**
- **Community-prior data slot exists**: `data/community-rentals-v0/teams.json`
  (full sets, uniform weight), consumed by the fallback path.

**The single gap:** `fallback_set` (belief.rs:584) is **deterministic MAP** — it
takes the nearest prior set's moves in fixed order (belief.rs:614-635). That is
the N=1 over-commitment. Model B = sample the *filler (unrevealed)* moves from
the per-move marginals, using `determinize`'s existing per-iteration rng, while
keeping revealed-first. Bounded to ~one function.

## The data format (specified 2026-07-25)

Work item 1 said "declarative JSON" and left the rest open. Concretely:

```json
{
  "format": "nc2000-belief-prior",
  "version": 1,
  "note": "free text; the code ignores it",
  "species": {
    "snorlax": {
      "moves": { "bodyslam": 0.82, "curse": 0.61, "rest": 0.58,
                 "sleeptalk": 0.55, "doubleedge": 0.35, "lovelykiss": 0.12 },
      "items": { "leftovers": 0.70, "brightpowder": 0.15 },
      "lead": 0.31
    }
  }
}
```

Keys are PS ids — the same lowercase, punctuation-free ids used everywhere
else in the repo, so a human copying a move name off a replay gets it right.
Values are marginal probabilities: "this fraction of Snorlax carry this move".
`items` and `lead` are optional; a species may give `moves` alone.

**The editing invariant, and it is the one worth telling a human: a species'
move probabilities should sum to about 4.0**, because every set has four
moves. A counter derived from replays produces that automatically, and a
hand-edit that pushes the sum to 6 is telling the bot that Snorlax has six
moves. The interpreter does **not** enforce it — enforcement would violate
totality — but the reference counter reports it and a `--check` mode warns.

**Sampling k unrevealed slots.** The doc said "sample the filler moves from
the marginals" without saying how, and independent Bernoulli draws are wrong
here: they do not yield exactly the k slots that are actually open. The rule
is a **weighted draw without replacement**: given k open slots, draw k
distinct moves from the species' legal, not-already-revealed moves with
probability proportional to their marginals, on `determinize`'s existing
per-iteration rng. Always produces exactly k, respects relative frequency,
and needs no normalization of the input numbers. It does not reproduce the
marginals exactly — without-replacement draws distort them — which is
accepted: that distortion lives entirely in the guess about unrevealed
slots, i.e. inside class B.

## Precedence: what an owner's file does and does not override

- **Per species, not global.** An owner file that mentions only Snorlax
  changes only Snorlax; every other species keeps the built-in prior. Editing
  one landmine must not silently blank the rest of the table.
- **Within a species, wholesale.** The owner's `moves` map replaces the
  built-in one for that species outright; it is not merged per move. A
  per-move merge would produce a hybrid distribution that neither the editor
  nor a reader could predict.
- **Species in neither file** → the existing `fallback_set` behaviour,
  unchanged. This resolves the doc's earlier "coarse legal default *or*
  unknown" either/or in favour of *unchanged*, which keeps M18 strictly
  additive: with no prior file loaded, the bot behaves exactly as it does
  today.
- Load path: `--belief-prior FILE` on the PS client and arena, plus a
  conventional `data/belief-prior-v0.json` picked up when present. No env var.

## Scope: this never touches the shipped web product

The shipped artifact is the open-sheet web app, where the opponent's sets are
pinned at p=1.0 and there is nothing unrevealed to sample — the prior is a
no-op on that path by construction, and item 2 keeps the sampling strictly on
the fallback path. So **M18 is native + PS-client only**, and it is incapable
of regressing the certified product. That is also why it can be developed in
parallel with an open M17 without competing for the same certification
surface.

## The responsibility split, stated for the owner

The point of the mechanism/policy split is that the owner takes on exactly one
thing and no more. Worth stating in those terms, because it is what makes
hand-edited data safe to accept without review:

**The code guarantees, whatever the data says:** it never crashes on
malformed input; revealed facts always dominate the prior; legality is always
enforced; a set never exceeds 4 moves; HP/PP/status imputation is untouched;
the open-sheet path is untouched.

**The owner owns exactly one thing: prior quality.** A wrong prior costs
games against teams it mispredicts. It cannot produce a tactical blunder, and
it cannot crash the bot. That is the whole contract — and it is why chasing
the metagame is the owner's call to make or decline, not a maintenance
obligation the developer inherited.

## Work items (in order)

1. **Data format + counting tool.** Per-`(species, move)` marginal probability
   table (declarative JSON), plus a reference tool that counts it from replay
   corpora (`nc2000stadium2_spectator_logs.zip` at repo root is the seed
   source; move marginals come straight from observed usage, so reveal-only
   spectator logs suffice — no hidden sheets needed). **The content, and its
   weighting/sourcing (whose games count — noisy ladder vs top free-play), is a
   downstream/community concern and is explicitly out of dev scope.** Dev owns
   only the *format* and a *reference* counter; the weighting is a value
   judgment the developer cannot make.
2. **`fallback_set` → marginal sampling.** Replace the fixed nearest-set filler
   with per-move Bernoulli sampling from the marginal table, threaded on
   `determinize`'s rng. Keep revealed-first; keep the `≤4` clamp; keep
   format-norm spreads. Species absent from the table → coarse legal default
   (unchanged) or "unknown". Do **not** apply sampling on the `pinned` /
   open-sheet path (there is nothing unrevealed to sample).
3. **Total, malformed-robust interpreter.** Clamp/normalize probabilities,
   ignore unknown keys, default on missing/garbage. No crash on a typo. Not
   adversarial hardening — the premise excludes malice.
4. **Certify the class-A surface (reveal-dominance).** A corpus-replay gate
   asserting (a) no imputed slot ever contradicts a revealed fact
   (move/item/species) and (b) the `%`-HP imputation path is the fixed one.
   This is the *finite, batch-checkable* class-A invariant — the thing that
   keeps community-data losses inside class-B. Reuses the conformance / corpus
   harness. Not "watch games and patch": the class-A surface is a short list of
   structural invariants (reveal-dominance, `%`-HP, `≤4`, no-crash), each fixed
   once and verified in batch.

## No search / engine changes

ISMCTS already samples determinizations and averages = B in expectation.
`SkuctSearch`, `eval.rs`, the RM+ root solve, and the endgame solver are
**untouched**. Open-sheet remains the `pinned` belief through the identical
path; keep the marginal sampling strictly on the fallback path so the certified
open-sheet play cannot be contaminated by determinization noise.

## Explicit non-goals (keep the scope thin)

- No adversarial-data hardening (local, well-intentioned user).
- No correlation / joint-set modeling (marginals only).
- No ensemble / matrix-column weighting / search rewrite (sampled
  determinization already yields B).
- **No prior tuning by watching games.** That is whack-a-mole and re-enters the
  parked full-hidden-party goal. The one part that cannot be solver-certified —
  prior *quality* / over-confidence — **is** the accepted class-B landmine
  loss. Do not chase it; if you find yourself hand-authoring priors or patching
  per-observation, you have left this thin scope.
- No opponent-*play* modeling / exploitation (unchanged from the README M17
  non-goal). This refines the belief *prior* — what the opponent *has* — not how
  they *play*.

## Relationship to the roadmap

Consistent with the README (M17+) non-goal "priors/tables/evaluation still
specialize on the meta pool": this generalizes the *prior data source*
(meta-pool full sets → community per-move marginals), not the search or eval.
It converts the one un-completable part of a hidden-info bot — tracking a
drifting metagame — from an owner maintenance treadmill into an externalized,
community-owned data artifact, so the shipped code stays finite and certifiable.
