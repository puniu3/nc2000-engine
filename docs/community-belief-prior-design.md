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

**The quantity is P(species *carries* move), not P(species *reveals* move.)**
This matters more than it looks, and it corrects an earlier line in this doc
("move marginals come straight from observed usage, so reveal-only spectator
logs suffice"). Reveal-only counting estimates
`P(carries) × P(uses | carries)`, and the second factor is strongly
move-dependent: Rest is used almost whenever it is carried, a situational
coverage move often is not. So spectator reveals are a **systematically
biased, per-move-varying** estimator of the quantity the sampler consumes —
not merely a noisy one, and rescaling cannot fix it, because rescaling
preserves the biased ratios.

Consequences, all of which keep the split intact:

- **Full-set sources give carry-marginals directly** (`data/community-rentals-v0`
  and the meta pool are complete 4-move sets, already aggregated by species by
  `load_sources`). Small sample, unbiased.
- **Spectator reveals give a large sample of the biased quantity.** Useful for
  detecting drift and for ranking, not as carry-marginals on their own.
- **The reference counter emits both and labels them.** It does not attempt a
  reveal-rate correction: inferring `P(uses | carries)` per move is exactly
  the kind of modelling this scope excludes, and choosing how to combine the
  two sources is a weighting judgement, which the doc already assigns to the
  community rather than the developer.

**Measured (2026-07-25, `count_belief_prior`).** Complete sets: 42 species,
192 sets, per-species probability sum **4.00** exactly — the invariant holds
by construction. Spectator reveals over all 570 battles: 128 species, 3,083
mon-appearances, mean sum **2.45**, i.e. a mon shows about 2.45 of its 4
moves per game.

The spread is what settles the method. Over the 124 `(species, move)` pairs
present in both tables at carry ≥ 0.3, the **reveal/carry ratio runs 0.06 to
1.70, median 0.41** — a 28x spread, so the under-count is emphatically not a
constant factor and no rescaling can undo it. Jolteon's Rest is carried in
every sampled set and revealed in 7% of games; Exeggutor's Sleep Powder is
carried 64% and revealed 60%. A move you only click when you need it stays
invisible; a move that is the reason you brought the mon shows every game.

Ratios **above 1.0** (Skarmory Toxic 1.70, Tauros Double-Edge 1.50) are the
more interesting signal: they are impossible if both sources describe the
same population, so they are evidence that the curated rental/meta-pool sets
and whoever plays in the spectator corpus are **different populations**.
Combining the two sources is therefore not a bias-correction problem with a
right answer — it is a choice about whose metagame the bot should expect.
That is precisely the judgement this design refuses to make on the owner's
behalf.

**The editing invariant, for a human: a species' *carry* probabilities should
sum to about 4.0**, because every set has four moves. Full-set counting
produces that automatically. A hand-edit that pushes the sum to 6 is telling
the bot Snorlax has six moves; a reveal-derived table summing to ~2.5 is
telling it Snorlax has two and a half. The interpreter does **not** enforce
the sum — enforcement would violate totality — but the counter reports it per
species, so the number doubles as the coverage diagnostic that reveals
whether a table was built from complete sets or from reveals.

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
   corpora. *(Superseded by "The data format" above: reveal-only spectator
   logs do NOT suffice on their own — they estimate P(reveals), a per-move
   biased proxy for the P(carries) the sampler needs. The counter reads
   full-set sources for carry-marginals and the spectator corpus for the
   reveal-side view, and labels which is which.)* **The content, and its
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

## Implementation status (2026-07-26)

Items 1-4 are in tree. The remaining work is content and reach, not mechanism.

**Shipped.**

- **Item 1** (86bb08b) — the format and `examples/count_belief_prior.rs`.
- **Item 3** (`crates/bot/src/prior.rs`) — the interpreter. Total by
  construction: any bytes yield a table, and bytes that make no sense yield
  the empty one, which the sampler reads as "no prior loaded". Probabilities
  are clamped to `[0, 1]` and deliberately not renormalised (marginals over
  four slots sum to ~4.0, and the weighted draw needs no normalisation);
  quoted numbers and a trailing `%` are accepted; non-finite, negative and
  non-numeric values are dropped with a warning rather than an error; unknown
  keys are ignored, which the counter's own per-species `n` is the first test
  of. `overlay` implements the Precedence section literally — per species,
  wholesale within one.
- **Item 2** (`belief.rs`, the `MoveDraw` region) — the sampler. The
  unrevealed slots are drawn per determinization by weighted draw without
  replacement, on `determinize`'s existing rng. Revealed slots are a separate
  prefix copied in first, and the pool has them subtracted, so
  reveal-dominance is structural rather than checked. The prefix is read from
  the observation, not from the built roster mon, which repairs the case where
  the roster's defensive second stage drops the reveals. Legality is still
  code: an entry the format bans or the level forbids is never drawn.
- **Item 4** (`examples/reveal_dominance_gate.rs`) — the class-A gate.

**Measured (2026-07-26, `reveal_dominance_gate`, 570 battles x 8
determinizations per decision).** 20,765 decisions, 20,700 of them in fallback
mode (99.7% — the corpus is human custom teams, so the M18 surface is
essentially the whole corpus). **4,316,214 assertions, 0 violations** in all
three arms: the shipped sample table (42 species from complete sets, 102,533
prior-governed roster slots), a reveal-derived table (128 species, 116,990
slots), and `--no-prior`. `det_hp_pct_drift` is 0 — imputing a candidate's max
HP never moved an announced percentage anywhere in the corpus.

The gate is falsifiable, not decorative: deleting the revealed prefix from the
draw makes it report 400 violations over two battles, with coordinates.

**The one finding, and it is upstream of M18.** Exactly one mon in 570 battles
has an unsatisfiable reveal set — battle 215 p1 Snorlax, credited with five
distinct moves for four slots. Cause: `Observer::scan_move` counts the plain
`|move|` *release* of a two-turn move that Metronome charged, because the
release line carries no `[from]`. `corpus.rs::collect_set_evidence` already
fixes exactly this for the offline evidence path by anchoring on `-prepare`,
with a test citing this battle; the live `Observer` does not have that rule.
Identical with and without a prior, so it predates M18 and is not the
sampler's. Two consequences worth recording:

- on the fallback path it is harmless — the `<=4` clamp drops the false
  reveal, and the imputed set is the true one;
- on the **pinned / open-sheet** path the same over-reveal would make
  `sync_checked` fail closed ("pinned opponent team contradicts public battle
  observations"), which is a live failure mode of the shipped product.

Left unfixed here deliberately: it is outside items 2-4 and the fix lands on
`observe.rs`, which the certified open-sheet product depends on.

**Not done.**

- `--belief-prior FILE` is wired on `examples/play.rs` (plus the conventional
  `data/belief-prior-v0.json` when present). The arena spec and the PS client
  are not wired.
- Item marginals and `lead` are parsed and exposed but unconsumed: item
  identity is already reveal-dominated and pick identity is resampled
  uniformly by the determinizer.
- No table ships at the auto-pickup path. The reference table sits beside it
  at `data/belief-prior-v0.sample.json`, and a test asserts
  `data/belief-prior-v0.json` stays absent, because with no file loaded the
  bot must behave exactly as it does today — verified as the stronger property
  that the no-prior path consumes no rng, so the determinization is
  bit-identical rather than merely equivalent.
- Prior *content* remains out of dev scope, unchanged from item 1.

## Reveal-channel audit (2026-07-27) — outcome and two accepted debts

The class-A gate above found one over-reveal, and the audit it triggered turned
into a full comparison of the project's two answers to "what did the opponent
publicly reveal": offline `corpus.rs::collect_set_evidence` versus the live
`Observer` that the shipped product runs. Method was a **prefix-wise differ** —
for every protocol prefix of every corpus battle, feed both channels the same
prefix and diff their per-slot revealed-move sets. 475,821 prefix-slot
comparisons; 0 divergences remain. The granularity mattered: a full-log-only
diff catches the Metronome over-reveal but misses the Pursuit under-reveal,
which self-heals later in the same log.

Fixed on the live channel: `db7d7cb` (the plain release of a caller-charged
two-turn move was credited as the mon's own — over-reveal, unsatisfiable, fails
`sync_checked` closed on the pinned/open-sheet path), `a5495f9` (a `[from]` tag
naming the executing move itself — Pursuit's intercept — was dropped, an
under-reveal of exactly the class-A shape), `3d34eaf` (Mimic suppressed the
whole mon instead of the one slot it overwrote — under-reveal for as long as
the mon stayed in).

Of 21 rules compared the rest resolved as: 2 offline-only rules that are dead
in gen 2 (`|replace|`, `|-end| Transform`), 4 legitimate asymmetries that must
NOT be equalised (offline's `-enditem` map is own-set fabrication input rather
than a knowledge channel; `species_names` exists only because offline keys by
nickname; live returning `None` on an ambiguous subject loses reveals on
purpose, and matching offline there would be an over-reveal), and 3 edges with
zero corpus occurrences (duplicate nicknames — all 570 battles use species
names verbatim; nicknames containing `:`; `[still]` substring vs exact match).

**Accepted debt 1 — the Mimic fix ships without gate coverage.** Mimic occurs
once in 570 battles and that use failed, so no counter in the gate moves. Unit
tests stand in. What would actually validate it is a PS-hosted game against a
Mimic user via `tools/ps-client.js`; until then the fix rests on the protocol
reading (PS names the copied move on the `-activate` line, and `import.rs`
already parses that field) rather than on measurement. Owner-accepted
2026-07-27 on the reasoning that the change only ever *narrows* suppression, so
it cannot introduce an over-reveal.

**Accepted debt 2 — offline has the mirror-image Mimic defect, deferred.**
`collect_set_evidence` credits a mimicked move to the submitted set, so a
reconstructed human team can carry a move its owner never had. It reaches
offline reconstruction only. It is deferred rather than fixed because
`corpus.rs`'s bytes feed `reconstruction_schema_fingerprint()`, so any edit
invalidates every proof artifact in the repo. **Trigger: fix it in the same
change that next regenerates the corpus artifacts**, where the invalidation is
being paid anyway. Owner-accepted 2026-07-27.

## Next session: wire the prior to blind play (the last M18 gap)

Items 2-4 shipped, and the prior is live **only in `examples/play.rs`** — via
`--belief-prior FILE`, or `data/belief-prior-v0.json` when the owner puts one
there. Traced 2026-07-27, the consumer that M18 exists for is not connected:

| consumer | reads the prior? | why |
|---|---|---|
| `examples/play.rs` | yes | `set_belief_prior` on the agent; auto-pickup at `prior::DEFAULT_PATH` |
| `tools/ps-client.js` (blind ladder) | **no** | no client flag, and more fundamentally **no wasm binding** — the file has no route to the searcher the client drives |
| the Web app | n/a by design | open-sheet pins the sets, and the prior only ever touches the fallback path |

So "drop a file in and the bot uses it" is true for the local CLI and false for
hidden-team play, which is the whole point. Two pieces close it.

**Piece 1 — wasm binding.** `crates/wasm/src/lib.rs` exposes
`WasmProtocolSearcher` (`js_class = ProtocolSearcher`, holding a
`ProtocolAgent`) and `WasmBlindSearcher`. Add a setter mirroring
`ProtocolAgent::set_belief_prior` / `BlindSearch::set_belief_prior`, taking the
file's **JSON text** rather than a path — wasm has no filesystem, and
`prior::BeliefPrior` already parses totally from text with warnings instead of
errors, so a typo degrades rather than throws. Suggested shape:

```rust
/// `setBeliefPrior(json)` — returns the interpreter's warnings as JSON so the
/// caller can surface a malformed table instead of silently ignoring it.
pub fn setBeliefPrior(&mut self, json: &str) -> String
```

Call it before the first `observe`; an empty or unparseable table must leave
today's fallback imputation exactly as it is.

**Piece 2 — client flag.** `tools/ps-client.js` already defaults to
`--mode blind` and constructs `new wasm.ProtocolSearcher(...)` at
`ps-client.js:508`, right beside its existing `setOwnTeam` call. Add
`--belief-prior FILE`: read once at startup, pass the text through the new
setter at construction, print the returned warnings. No other change — do not
touch mode semantics.

**Constraints that must survive.**

- Default off: no flag and no file at `DEFAULT_PATH` ⇒ byte-identical play. The
  existing test asserting `data/belief-prior-v0.json` stays absent is the guard;
  keep it.
- The prior must never reach the `pinned` path. `--mode open` pins the
  opponent's sets, and sampling there would contaminate the certified
  open-sheet configuration.
- Main-ladder botting needs PS staff permission (README M15b POLICY). The
  client has no default `--server`; this work targets the local clone or an
  explicitly configured self-hosted server.

**Verification.** The class-A gate cannot see this — it replays the corpus
through the offline path, not the websocket. What can:

1. `cargo test --release` green, plus a wasm-side unit test that a malformed
   table yields warnings and an unchanged determinization.
2. A local-clone game (`node pokemon-showdown start --skip-build --no-security
   8123`) in `--mode blind` with and without a table, checking the client logs
   the species count and that games complete with 0 choice rejections — the
   same gate-a shape M15b used.
3. Optional strength read: the M15b harness plays the client against its
   `--random` driver; a prior that helps should not be *needed* to show up
   there, and the design already accepts class-B exploitability, so treat a
   null as expected rather than as a failure.

**Not in scope** (unchanged from the original list): arena spec wiring, item
marginals and `lead` (parsed, exposed, unconsumed), and prior *content*, which
stays a community concern.
