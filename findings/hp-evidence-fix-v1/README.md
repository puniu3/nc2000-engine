# Hidden Power public-evidence correction

**Research only: automatic inference is disabled by default.** Enable the
`hp-type-evidence` Cargo feature to run the candidate. The paired stress
pilot scored 16/24 for the ablated baseline and 13/24 for the candidate
(4 improved, 7 worsened, 13 tied seed pairs). All games ended without caps.
This does not establish a statistically reliable regression, but it fails
to support adoption as a strength improvement. Both control pairs were
bit-identical.

With the research feature enabled, the observer intersects possible Hidden Power types using prior public
super-effective/resisted messages. Belief filtering rejects contradictory
known sets. If fallback templates contradict the evidence, their type and
associated DVs, power and stats are projected to a compatible genotype.
Compatible templates keep their existing type and RNG consumption.

Evidence survives position serialization. Old positions default to all
16 types. Empty masks, masks without a revealed Hidden Power, masks that
contradict an explicit typed reveal, and gender-infeasible masks are rejected.
Copied moves and targets with known changed typing are conservatively
excluded. Neutral hits and immunities are not used for inference.

## Measurements

The frozen historical audit reconstructs 101 of 127 eligible decisions.
All 64 samples contradicted prior public evidence in 35 decisions before
the change; none of the 101 decisions has a contradictory sample afterward.
These selected cases are not a prevalence estimate. The remaining 26
positions could not be reconstructed and are not included in the denominator.

The four-case root probe uses actual inferred masks rather than injected
typed moves. Its 16 ablated baseline searches exactly reproduce the earlier
saved results, including visits and means. The correction also changes
DV-dependent power and stats when needed, so this is not the earlier
power-fixed, type-only intervention. Root changes alone are not evidence
of stronger play.

`duel-protocol.json` was fixed before match outcomes, with only team-input
spacing canonicalized when prefix validation rejected the input spelling.
Three explicit stress scenarios have eight paired trials each; an on-pool
control has two. Every full team passes the repository's team validator;
this validator uses flat learnsets and is not a complete cross-move legality
oracle. All forced prefix actions are checked against engine legal choices
for every seed before launching any match. Typed Hidden Power aliases are
converted to plain Hidden Power while retaining the intended DV type, so
the native observer cannot learn the type from an internal typed move ID.

The evaluated side alone retains or discards public type constraints.
Both arms use the same corrected blind opponent, 27,000 iterations, c=0.4,
paired battle/search seeds and alternating sides. The constructed teams
are not reconstructions of complete historical opponents. Forced exposures
are retained even when a move misses or another event prevents useful
information. Caps remain unknown. This is an exploratory screen, not a
significance-gated strength claim or a measurement against expert humans.

`summary.json` reports paired outcomes and controls. `records.json.gz`
contains anonymized historical model/root rows and complete generated match
traces. `duel-verification.json` records replay checks of every state key,
legal action, search seed, public type mask and terminal result. The
report also checks that the first action divergence follows public type
evidence. Verification replays the battle independently of the search;
it does not re-run every search.

## Validation

- The gated candidate passed 150 bot unit tests and 10 position tests;
  the default configuration passed 149 unit tests and 10 position tests.
- Two protocol import tests passed, including corpus replay in both modes.
- Workspace checking with all targets and web TypeScript checking passed.
- A 16-game, 64-iteration driver smoke passed 238 replayed decisions,
  exercised all scenarios from both sides, and preserved both control traces.
- The legacy no-prior reveal gate passed 4,316,484 assertions over 570 games
  and 20,765 decisions with zero violations. Its 165 HP percentage drift
  diagnostics are present; this gate does not use the operational v3 prior.
- Feature tests cover public target changes, calls/copies, revision changes,
  candidate pruning, spent PP, hidden-state perturbation invariance,
  position round trips, feasible type sampling, and genotype reconstruction
  independently through the engine.

`tools/hp-evidence-fix` rebuilds the examples and runs fresh model/root,
reveal-gate and paired-duel measurements into a new local directory. It
requires the private spectator corpus. It does not publish or overwrite
this frozen report. No raw human logs or player handles are included here.

The archived pilot ran before adding the default-off gate.
`verification.json` preserves the measured source hashes separately from
subsequent gate validation. The gated candidate reproduces the archived
model and root outputs; the ordinary default is separately checked against
the original audit.

At the first changed action in 23 paired games, 21 modeled active-opponent
loadouts omitted at least one move from the constructed actual set.
`first-divergence.json` retains all changed pairs, not just losses. For
example, Articuno fallback templates can include Sky Attack/Blizzard while
omitting actual Rest/Sleep Talk. This is a diagnostic of these constructed
fixtures, not proof of the losses' cause or a human-corpus coverage estimate.
The next causal test should separate type evidence from hidden-loadout
uncertainty using fresh trials, not tune these 24 outcomes.
