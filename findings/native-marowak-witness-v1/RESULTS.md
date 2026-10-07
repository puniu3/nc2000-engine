# Marowak first-action witnesses

At actual game 10/frame 32, the bot selected Bug Hidden Power over Earthquake.
The higher nominal power of Earthquake does not establish a first-turn advantage:
Porygon2 has only 15 HP, and its two Curse boosts lower effective Speed to 61,
versus Marowak's 105. Both attacks win immediately with probability 0.91796875.

Complete engine enumeration of both attacks against all four opponent replies
used 488 executions. For each reply, the probabilities of either player winning,
a tie, and continuation are identical. The nonterminal HP/status/boost/item
projections also match. PP and move histories are not equated, and full successor
or optimal-action equivalence is not claimed.

Two explicit own policies, against fully unrestricted optimal opponent play,
were bounded to width at most 1e-6:

| First action, then Earthquake priority | Guaranteed score interval |
| --- | --- |
| Earthquake | [0.9306855194487832, 0.9306864968633306] |
| Actual Hidden Power | [0.930685499324752, 0.9306864968633307] |

The security values of these two fixed policies differ by at most
0.0000009975385787 (less than 0.0001 percentage points). Only each lower endpoint
transfers into an unrestricted continuation guarantee. Their upper endpoints do
not upper-bound optimal play after those first actions.

Separately fixing the opponent's first reply and then prioritizing Ice Beam,
while retaining all own continuation actions, gives these upper bounds on the
actual Hidden Power first action:

| Opponent first reply | Target upper bound |
| --- | ---: |
| Return | 0.9885644912719727 |
| Ice Beam | 0.9823436293663694 |
| Curse | 0.9985525116609576 |
| Recover | 0.9676788507621585 |

The last witness narrows the old target upper bound of about 0.98234. It is still
above the alternative's 0.9306855 lower guarantee, so no positive pure-action
loss is certified. A claim that the actual move is a mistake is unsupported.

Every policy/reply run reached WidthMet for its restricted game and reproduced
all fields except elapsed time on repeat. Neither work nor node budget bound.
The matrix auditor independently computes values by rational LP and checks bound
orientation. The continuation auditor also checks the one-row minimum directly.
Each rejects five corrupted-record cases. Existing eight fixed-policy/reply
release tests passed at the unchanged solver source in the preceding study;
these cover exhaustive continuations and first-action overrides.

This is exploratory analysis of the fixed actual hidden state using independent
chance classes. It is not an evaluation of the full blind policy, equivalence
of the two optimal first actions, or proof of proximity to the whole-game ceiling.
No normal bot behavior, engine rule, main solver, or public deployment changed.
Future candidate selection should check actual immediate effects before treating
nominal damage differences as evidence of a decision error.
