# Tactical exposure and conditional continuations

No product policy changed. This audit found no supported improvement among
three selected first-action alternatives. It does not establish that the bot
is near optimal or stronger than expert humans.

## Forced coverage

The previous 24-game self-play corpus never selected its Encore, Toxic, or
Baton Pass users. This probe retains the full legal catalog teams and forces
the relevant user to lead. The other picks maximize overlap with the bot's
ordinary selection; the opponent selects normally. Each of three families
faces `sample-07` and `sample-08`, with ordinary and scripted arms sharing
battle and agent seeds. Both sides use native blind search, prior v3,
c=0.4, and 27,000 iterations per decision.

All six pairs had identical terminal scores. Ordinary play used Toxic in
both matchups and Substitute followed by Baton Pass twice against
`sample-07`. Encore had no applicable opportunity and was never used.
The script is only an exposure device: unconditional Toxic can override
a sensible switch while asleep, and Substitute can be immediately broken.
Its failure is not evidence against competent tactical play.

At the recorded decisions where the source had the relevant kit available,
the defender's diagnostic samples included the target move in 712/712
draws across both arms. These correlated draws describe these particular
on-pool matchups, not population coverage or protection against off-pool
opponents. Diagnostic sampling does not guide the scripts or searches.

## Selected action comparisons

Two apparently questionable ordinary decisions were selected after replay
inspection. Each archived search choice was reproduced from its original
observer history and search seed. The first action was then varied, with
eight fresh paired future battle seeds and paired per-decision search
seeds. Later decisions use unchanged blind search on both sides.

| Position | Archived action | Alternative | Terminal wins, archived / alternative |
| --- | --- | --- | --- |
| Alakazam facing damaged Machamp | Recover | Psychic | 6/8 / 5/8 |
| Substitute Baton Pass replacement | Lapras | Marowak | 2/8 / 1/8 |

The archived Recover was followed by a critical hit that killed Alakazam.
That single outcome did not predict the fresh-seed comparison. The actual
opponent team is held fixed in these trials; they measure a conditional
effect under the current continuation policies, not minimax action loss
or an information-set average.

Four fresh root seeds were also searched at nested 27k, 108k, and 432k
budgets. Recover was chosen 3/4 times at 27k and 4/4 at larger budgets;
Lapras was chosen 4/4 at every budget. Those preferences are not ground
truth about action quality.

## Constructed Encore opportunity

An additional pair uses an explicit legal prefix: Alakazam uses Psychic
while Miltank uses Curse. Both full teams and both arms share that prefix;
no state fields are edited. At the next decision the specialist overrides
Psychic with Encore. The opponent switches to Snorlax, so Encore fails.
The specialist nevertheless wins that original full game while ordinary
play loses. The downstream histories and random-number consumption differ,
so this is only a candidate for fresh trials.

Those eight fresh paired continuations produced 3/8 wins for Psychic and
2/8 for Encore. All four root seeds selected Psychic at all three budgets.
The single specialist win did not turn into a positive pilot effect.

All three alternatives scored one fewer win than the archived action in
their respective eight-pair screens. These samples neither establish
statistical superiority of the archived actions nor exclude small gains.
No independent confirmation run or product patch is justified by this
screen. The separate 2-vs-2 reference-solver uncertainty remains open.

## Scope and next measurement

This is 14 exposure games plus 48 conditional continuation games, not a
representative strength benchmark. All 62 ended normally; a capped game
would retain an unknown outcome. A separate six-game, 300-iteration smoke
run is excluded from strength observations. Fixed 27k iterations are not
a 10-second native or 60-second oracle benchmark. The native truth-backed
observer is not an end-to-end test of deployed protocol observation.

The measured coverage hole is narrower now: these on-pool examples do not
support a blanket claim that Toxic or Baton Pass is missing from play.
They still do not test competent adversarial use across a broad range of
positions. Before more heuristic tuning, the next distinct evidence should
come from opponents or hidden sets outside the self-play distribution,
with any candidate failure confirmed on fresh continuation seeds. Prior
failures of PP shaping, generic rollout policies, and equal-time learned
value replacement remain separate negative evidence, not reasons to
repeat those experiments without a new causal case.

## Verification and reproduction

`tools/tactical-exposure` builds the research examples and repeats both
exposure protocols and both continuation protocols into a new timestamped
directory under `tmp/tactical-exposure-v1`. It restores the selected root
games from the exact hashed files in `sources/`, so continuation roots
remain frozen even if a future policy changes the new exposure games.
Each saved root choice must reproduce before a continuation run starts.
Elapsed times and binary hashes are machine-dependent.

Exposure reports verify paired states, observations, ordinary choices, and
seeds through the first intervention, then replay every game with the
existing `selfplay_endgames` verifier. The separate continuation verifier
replays each source prefix, resets only future battle RNG at the fork, and
checks every continuation state key, legal action, final turn, and outcome.
The report also checks paired future seeds and identical uninformed first
opponent replies.

The exposure runner gained its optional legal-prefix input after the main
run. Its default behavior was checked against the earlier six-game smoke:
all complete records match exactly except elapsed time. The continuation
runner source is unchanged since its runs. `verification.json` records
checks and artifact hashes. Compressed records retain the full observations,
search distributions, and replay decisions; `index.html` is the readable
report, with the Encore follow-up under `encore/`.
