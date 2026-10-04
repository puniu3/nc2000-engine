# Hidden Power public-evidence audit

The current model can contradict Hidden Power effectiveness already shown
to the player. This is a concrete information-use defect, not evidence of
being near the optimal ceiling. No product code changes in this audit.

## Corpus and observation constraints

The local 570-game spectator corpus contains 503 direct Hidden Power uses
after excluding three games with Transform, Conversion, type changes, or
Foresight. Of these, 255 have informative super-effective/resisted messages.
Intersecting the compatible types for each user yields 152 reductions;
127 have a following turn with that user still active. The deployed
ProtocolAgent path reconstructed 101 such decisions; 26 have no ordinary
decision in the extracted log (for example, a `cant` event).

Thirty-five decisions from 35 games produced 64/64 sampled types that
contradict those past messages. The files retain each evidence chain, its
intermediate compatible-type sets, and the final intersection. No future
opponent move or hidden set is used to derive the constraints. Sampling
counts are correlated diagnostics, not independent prevalence estimates.
This selected subset is not representative of all decisions or players.

Removing only the opponent's Hidden Power move announcements in an
observation-channel ablation shows that 30 of the 35 bad cases had no
modeled Hidden Power beforehand. The other five retain their previous
modeled type unchanged after the informative observation. Thus this is not
a demonstrated bug where the reveal replaces a previously correct typed
prior with Dark. The sampler simply does not consume the type evidence;
plain Hidden Power on default DVs commonly becomes Dark.

## Decision sensitivity

Four inconsistent cases have a singleton type intersection. All four were
selected before the root probes. Both arms recover the three own picks
from the complete own-side log and keep the same imputed own sets. The
comparison changes only the plain Hidden Power reveal to the publicly
deducible typed reveal. Both imputed models have power 70, checked at run
time. Actual historical power, IVs, and unrevealed sets remain unknown.

Each case uses four fixed paired search seeds, 27,000 iterations, c=0.4,
and frozen belief pool v3. Type correction changes 12/16 seed-case choices.

| Anonymous game | Deduced type | Baseline choices | Corrected choices |
| --- | --- | --- | --- |
| 10, Aerodactyl | Rock | Toxic 3, Earthquake 1 | Earthquake 3, Toxic 1 |
| 346, Dragonair | Fighting | Soft-Boiled 4 | Ice Beam 2, switch 2 |
| 395, Articuno | Fighting | Rest 3, Double-Edge 1 | switch 4 |
| 559, Arcanine | Fighting | Thunder 3, Drill Peck 1 | Thunder 3, Drill Peck 1 |

For game 10, super-effective damage against Zapdos restricts the type to
Ice/Rock; a later resisted hit against Steelix leaves only Rock. The other
three cases use super-effective damage against a pure Normal target, which
uniquely implies Fighting. In game 559 the active defender at the analyzed
decision is Zapdos, after the earlier hit against Snorlax. Its aggregate
choice distribution stays unchanged despite two paired seed differences.

These probes establish action sensitivity, not that a particular new action
is better. No continuation wins, regret, exploitability, or expert-human
strength are measured. The next implementation should preserve public type
constraints through observation and imputation, with special handling for
transformed/type-changed targets, copied moves, and serialized positions.
Independent continuation trials remain required for a strength claim.

## Validation and reproduction

`tools/hp-evidence` rebuilds the two research examples, scans the private
local corpus, runs the sampler audit and frozen four-case probes, and
writes a fresh report under `tmp/hp-evidence-v1/run-*`. It never changes
the bot. The corpus is required and is not committed; its content digest
and the engine data digest are recorded in `summary.json`.

Six declarative parser controls are stored in `calibration.json`: a Normal
target singleton, critical/weather invariance, exclusion of called moves,
exclusion of type-changed games, rejection of an unrelated move's effect,
and a two-turn Ice/Rock-to-Rock intersection. Every report reruns these
controls and verifies the evidence intersections, sample totals, root seed
pairs, and 27,000-visit totals. The examples resolve inputs relative to the
current project directory; stale compile-time conformance paths cannot
silently turn a missing corpus into zero reconstructed decisions.

`records.json.gz` contains anonymized evidence/model rows and root probes;
no player handles or raw human logs are included. `verification.json`
records checks and artifact hashes. The readable report is `index.html`.
