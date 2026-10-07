# Probability-first enumeration increases known chance mass

At the same4096-execution per-cell cap, all10 previously incomplete joint-action
cells gained resolved probability mass. The six previously complete cells
remain complete with matching event probabilities. Both methods used47,872
engine executions over all16 cells; there is no execution-count reduction.

Examples of resolved probability mass:

| Own / opponent | Frozen eager partial | Largest pending mass first |
| --- | ---: | ---: |
| Thunder / Thunderbolt | 0.07197% | 30.059% |
| Thunder / Hidden Power | 0.18544% | 35.269% |
| Hidden Power / Hidden Power | 0.75293% | 61.213% |
| Sleep Talk / Thunderbolt | 75.4407% | 99.5845% |

The candidate performs exact scripted transitions and prioritizes the largest
remaining prefix mass. It does not merge damage ranges. All newly enumerated
leaves in this target are nonterminal. More known live probability therefore
does not itself tighten final game values or the selected-action loss.
There is no graph-solver integration, timed comparison or ordinary-bot change.

A fresh engine replay checked all47,872 leaves and66,680 pending prefixes,
including trace agreement, exact prefix probabilities and leaf outcomes, with
maximum probability error zero. An independent prefix-free partition audit
checked all16 cells, unit total mass, aggregate outcomes, agreement with the
frozen baseline, and compatible outcome intervals. Five deliberate corruptions
were rejected. The114,552 verification executions are excluded from the
comparison. Saved records preserve all candidate leaf and pending scripts.

This one-root diagnostic justifies an opt-in integration experiment, not a
speedup or strength claim. Future comparison must include priority-queue and
retained-state costs and judge actual game-value/loss bounds under equal budgets.
