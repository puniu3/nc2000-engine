# Fixed opponent replies at game 104/frame 52

The four static opponent priorities did not tighten the existing Rest upper bound
of 0.08124596542620566. Their upper bounds were 0.130924898848 (Cross Chop),
1.0 (Earthquake), 0.10546875 (Rock Slide), and 0.986948422212 (Hidden Power).
Drill Peck retains an independently established lower bound of 0.045156087695040474.
No actual mistake was certified. Work/node caps remain unresolved mass, not evidence
that these priorities or the bot are optimal. No ordinary bot changed.

Selection used the closest alternative-lower versus selected-upper gap after
intersecting archived cell intervals. This exploratory selection is reproduced
by the auditor and is not representative sampling. All four proposed priorities
were retained. Three hit node limits and one exhausted engine work. Timing ran
alongside compilation and is not a performance comparison.

The already-bad Self-Destruct control was refuted (upper zero, ten executions).
The previously small-loss Ice Beam control was not refuted (upper one).
Eight existing fixed-policy/reply release tests passed, including exhaustive
finite continuations, both side orientations, first-action overrides, and the
prohibition on treating a restricted first-reply value as a whole-game guarantee.
The audit checks prior payoffs with independent rational LP, reproduces selection,
and rejects five record corruptions. It does not promote fixed-opponent lower
bounds into guarantees against unrestricted opponents.

The follow-up holds the strongest first reply, Rock Slide, fixed while changing
the continuation priority to Cross Chop and increasing the bound-search budget.
This remains a full-information diagnostic at the actual hidden state, with the
independent chance model. It does not establish blind-policy regret or strength.
