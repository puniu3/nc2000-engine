# Prefix-only support for newly selected moves

Current fallback beliefs often omit moves that a human opponent selects
for the first time. In 6,701 labels from 560 spectator games, the production
EarlyPerish policy includes the subsequently selected move in at least one
of 64 samples in 3,548 cases (52.95%). The previous full-loadout sampling
candidate includes it in 4,304 (64.23%). The Layered ablation, without the
current early Perish protection, includes it in 3,443 (51.38%). No bot
policy is changed by this audit.

This is a positive-label support diagnostic, not a strength metric or a
calibrated carry-probability estimate. Broadening the union of sampled kits
can improve it without better decisions. Average inclusion per sample is
52.63% for production and 55.30% for full-loadout sampling, much closer
than the union coverage. The existing full-loadout candidate already failed
to improve ordinary play in `operational-loadout-draw-v1`; this audit is
not a reason to adopt it or to repeat that unchanged experiment.

## Selection and information boundary

All 570 local spectator logs are considered. Ten entire games involving
Mimic, Mirror Move, Metronome or Transform are excluded conservatively.
Among 15,961 ordinary direct move decisions in the other 560 games,
9,200 previously used moves, 41 otherwise already revealed moves, and ten
synthetic Struggle/recharge actions are excluded. Nine fresh-move positions
fail reconstruction and remain separately recorded. The remaining 6,701
labels are unique per game, acting side, species and move.

The observer is on the opposite side of the actual move user. The public
state is reconstructed at the turn-open cut, before either turn action.
Every scored move must be absent from that observer's accumulated reveals.
Opponent imputation uses only that prefix. Complete own-side evidence is
used for the observer's submitted set, as in the existing reconstruction
library; no opponent future move is supplied to its belief. The future
selected move is used only as a label after sampling. HP aliases compare
as plain Hidden Power; the experimental HP-type inference is disabled.

No table is fitted here. The fixed v3 pool and existing policies may share
source material with the spectator corpus, so these are not asserted to
be out-of-training-sample estimates. Never-used moves are not negative
labels, and many moves are selected because of the particular opponent
and position. Samples within a decision and decisions within a game are
not independent observations. Absence from 64 samples does not prove zero
probability in the sampler's full distribution.

## Concrete follow-up

Rest is absent from production samples in 313 of 436 first uses, including
156 Snorlax cases. Recover is absent in 61 of 74. A common Snorlax template
contains Body Slam, Earthquake, Self-Destruct and Curse, with no recovery.
Full-loadout sampling reduces Rest misses to 121, but also removes support
for other threats and has no demonstrated overall strength gain.

A narrow future test should preserve existing Perish protection and vary
only anticipation of unrevealed recovery. Its strength evaluation needs
fresh battle seeds and explicit non-recovery controls. This audit does
not establish the correct recovery probability, the value of that
intervention, distance from optimal play, or superiority to expert humans.

## Verification and reproduction

Every row is checked for prefix novelty, no duplicate label, preservation
of all revealed moves in every sampled kit, the four-slot limit, and
recomputed sample/inclusion totals. A revealed-only scoring control has
zero novel-label support; a diagnostic control with the answer appended
has full support. These validate the counting operation, not the relation
between this diagnostic and strength. A five-game smoke covered 40 labels.

`tools/next-move-support` checks input hashes, rebuilds the native audit,
measures the private corpus, and creates a fresh local report. It requires
the local corpus and does not publish. `records.json.gz` contains anonymous
coordinates and sampled move kits; no raw logs or player handles are saved.
