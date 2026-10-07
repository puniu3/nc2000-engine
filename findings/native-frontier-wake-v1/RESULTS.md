# Waking-state fixed-policy weakness

All 11 sleeping states with zero immediate-win guarantee in the completed
64-state sample have Time=1. The existing engine wakes the user before resolving
its move in these states, so the research policy's unconditional Sleep Talk
choice fails. Selecting the first legal registered priority attack instead gives
immediate-win guarantees approximately 0.8076923076 or 1 against every legal
opponent reply. In five states, every enumerated outcome under every reply is
an own terminal win. The other 53 sampled states are unchanged.

All 256 cells completed, using 41,430 enumeration executions plus 64 policy
validation executions. A separate replay selected actions through the new
FixedPolicySolver::with_sleep_talk_until_wake API, confirming the manually chosen
probe actions, then reproduced all 256 cell probabilities with zero error.
Four deliberate record corruptions were rejected. All 63 bound-solver tests
passed; sleeping-policy exhaustive tests now include both sides, wake/still-asleep
cases, exhausted Sleep Talk PP, and first-action override precedence.

This is a weakness in a research fixed policy, not an ordinary-bot mistake.
Higher immediate-win probabilities alone do not rank long-term policy values.
Root-search fields are copied from the frozen input, not a new graph result.
Sleep-counter semantics and ordinary bot behavior are unchanged. The new API is
opt-in; the old fixed policy retains its previous behavior. A separate registered
root-level comparison will test whether the new witness improves usable bounds.
