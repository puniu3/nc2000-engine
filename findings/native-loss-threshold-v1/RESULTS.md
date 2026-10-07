# Directed loss-lower-bound pilot

The new analysis-only entry point alternates between raising the optimal-value lower bound and lowering the target first-policy guarantee upper bound. For a policy refutation, one opponent reply is sufficient; reply selection prioritizes low upper bounds. The ordinary search, existing loss solver and default settings are unchanged.

Two new tests cover reply orientation and 24 combinations of two finite chance games, both sides, pure/mixed first policies and resumed budgets of one or 17 engine executions. Every intermediate interval encloses the exhaustive result and every threshold decision agrees. All 66 release bounds tests passed. The six-root legacy arm reproduces the frozen quotient-only results exactly except elapsed time and added metadata.

Three certified bad alternative actions were detected: game 7 frame 21 Self-Destruct (946 runs), game 14 frame 180 Explosion (76), and game 37 frame 23 Hidden Power (43,326). These are counterfactual alternatives, not bot mistakes. Three previously certified small-loss actual actions were correctly accepted (2,205, 64 and 54 runs). The audit independently checks payoff matrices and rejects a false AboveThreshold label.

Among the six actual diagnostic actions, game 1 frame 37 was certified within 0.02 after 53,759 runs; the other five remained unresolved at 250,000 runs. No new bad actual decision was proved. Next is the complete previously fixed 70-root set, with no outcome-based exclusions. All results fix actual hidden state and optimal full-information continuation; pure first-action loss is not blind-policy exploitability.
