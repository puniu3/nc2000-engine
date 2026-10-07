# Actual waking Sleep Talk choice remains unresolved

The target is game62/frame92, the only actual Sleep Talk choice among24 waking
states in the360-root ordinary-bot corpus. Both Raikou wake before acting; own
HP is215/215 and opposing HP117/215, with Rest available to both. The analysis
retains all future legal choices and fixes the true hidden state.

At250,000 engine executions and50,000 retained-state ceiling, all four first-action
value intervals and losses remain[0,1]. Default enumeration retained682 states,
spent233,940 executions in eager enumeration, and exhausted49 eager cells.
Retaining exhausted prefixes increased retained states to6,677 and kept217,797
previously enumerated leaves, but all first-action intervals still remain[0,1]
at the same250,000 executions. Elapsed4.24s versus6.73s is a single descriptive
measurement, not a stable timing comparison. No error or near-optimality bound
was established, and no default change is justified.

Both selected-action calculations pass independent rational matrix checks and
five corruption rejection controls each. The initial nominal root index117
referred to input file order, while the frozen solver sorts by game/frame, and
therefore ran game62/frame172. That attempt is preserved and excluded; the
corrected index77 was registered by game/frame before either reported run.
Its250,000 executions are additional discarded work, not included in either arm.

One-step failure of Sleep Talk does not by itself prove long-term regret because
attacking consumes different PP and the opponent can heal. The previous complete
first-step census still had substantial unenumerated mass in several attack
cells here. Future computation should address probability coverage rather than
blindly adding depth to the same incomplete prefixes. Ordinary-bot behavior,
sleep rules and public deployment are unchanged.
